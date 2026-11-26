"""Orquestador conversacional: une canal -> sesión -> NLU -> política -> (RAG | Siebel | handoff) -> respuesta.

Flujo de un turno (EP2 §6, diagrama de secuencia):
  1. Recupera o crea la sesión (Redis) y asigna cohorte del piloto.
  2. NLU: normaliza español chileno y clasifica la intención (SageMaker o local).
  3. Sentimiento (léxico chileno / Amazon Comprehend).
  4. PolicyRouter decide la resolución.
  5. Ejecuta: RAG (pgvector + LLM/extractivo), consulta Siebel asíncrona con espera
     acotada, creación de caso en Siebel, o handoff a un agente por WebSocket.
  6. Persiste mensajes (PII enmascarada + cifrados) y métricas.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

from app.channels.normalizer import InboundMessage
from app.conversation import responses as R
from app.conversation.router import PolicyRouter
from app.conversation.session_service import SessionService
from app.handoff.service import HandoffService
from app.integrations.siebel_publisher import SiebelPublisher, notifications_key
from app.nlp.intents import get_intent
from app.observability.telemetry import (INTENTS, LOW_CONF, RESOLUTIONS, SIEBEL_RESULTS, STAGE_LATENCY, TURN_LATENCY,
                                         correlation_id)
from app.privacy import MessageCipher, mask_pii

log = logging.getLogger(__name__)


@dataclass
class BotReply:
    session_id: str
    reply: str
    resolution: str
    intent: str | None
    confidence: float | None
    top_intents: list = field(default_factory=list)
    explanation: list = field(default_factory=list)
    sentiment: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    handoff_id: str | None = None
    siebel: dict | None = None
    cohort: str = "piloto"
    nlu_source: str | None = None
    generator: str | None = None
    correlation_id: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationEngine:
    def __init__(self, *, settings, sessions: SessionService, classifier, sentiment, rag, publisher: SiebelPublisher,
                 handoff: HandoffService, store, kv, cipher: MessageCipher):
        self.s = settings
        self.sessions, self.classifier, self.sentiment, self.rag = sessions, classifier, sentiment, rag
        self.publisher, self.handoff, self.store, self.kv, self.cipher = publisher, handoff, store, kv, cipher
        self.router = PolicyRouter(settings.intent_threshold, settings.max_low_confidence_turns)

    # ------------------------------------------------------------------ helpers
    def persist_message(self, sess: dict, role: str, text: str, **meta) -> None:
        self.sessions.add_turn(sess, role, text, **{k: v for k, v in meta.items() if k in ("intent",)})
        self.store.add_message({"session_id": sess["session_id"], "rol": role,
                                "texto_cifrado": self.cipher.encrypt(mask_pii(text)), **meta})

    def _crm_params(self, sess: dict) -> dict | None:
        if sess.get("customer_id"):
            return {"customer_id": sess["customer_id"]}
        if sess.get("phone"):
            return {"phone": sess["phone"]}
        return None

    # --------------------------------------------------------------------- turno
    def handle(self, msg: InboundMessage) -> BotReply:
        t0 = time.perf_counter()
        corr = correlation_id.get() if correlation_id.get() != "-" else uuid.uuid4().hex[:16]
        sess = self.sessions.get_or_create(msg)
        self.store.upsert_session({"session_id": sess["session_id"], "user_hash": sess["user_hash"],
                                   "canal": sess["channel"], "cohorte": sess["cohort"], "estado": sess["state"]})
        self.persist_message(sess, "cliente", msg.text, correlation_id=corr)

        t = time.perf_counter()
        nlu = self.classifier.predict(msg.text)
        STAGE_LATENCY.labels("nlu").observe(time.perf_counter() - t)
        sent = self.sentiment.analyze(msg.text)
        decision = self.router.decide(sess, nlu, sent)
        intent = get_intent(nlu.intent)
        reply = BotReply(sess["session_id"], "", decision.action, nlu.intent, nlu.confidence,
                         [{"intent": i, "probabilidad": p} for i, p in nlu.top], sentiment=sent,
                         cohort=sess["cohort"], nlu_source=nlu.source, correlation_id=corr)
        resolved = False

        if decision.action == "agent":
            self.handoff.forward_customer_message(sess, mask_pii(msg.text))
            reply.reply, reply.handoff_id = "Tu mensaje fue enviado al ejecutivo que te está atendiendo.", sess["handoff_id"]
        elif decision.action == "pilot_excluded":
            h = self.handoff.create(sess, decision.reason, nlu, sent)
            reply.reply, reply.handoff_id = R.PILOT_EXCLUDED.format(wait=self.handoff.wait_estimate()), h["id"]
        elif decision.action == "handoff":
            if intent.crm_command and (params := self._crm_params(sess)):
                self.publisher.command(intent.crm_command, {**params, "motivo": intent.name,
                                                            "detalle": mask_pii(msg.text)}, sess["session_id"], corr)
            h = self.handoff.create(sess, decision.reason, nlu, sent)
            reply.reply, reply.handoff_id = R.HANDOFF.format(wait=self.handoff.wait_estimate()), h["id"]
        elif decision.action == "clarify":
            LOW_CONF.inc()
            reply.reply = R.CLARIFY
        elif decision.action == "smalltalk":
            reply.reply = R.SMALLTALK.get(nlu.intent, R.SMALLTALK["greeting"])
        elif decision.action == "crm":
            params = self._crm_params(sess)
            if params is None:
                reply.reply, reply.resolution = R.NEED_ID, "crm_sin_identificar"
            else:
                t = time.perf_counter()
                res = self.publisher.request(intent.crm_operation, params, sess["session_id"], corr)
                STAGE_LATENCY.labels("siebel").observe(time.perf_counter() - t)
                SIEBEL_RESULTS.labels(res.status).inc()
                reply.siebel = {"operacion": intent.crm_operation, "estado": res.status,
                                "correlation_id": res.correlation_id}
                if res.status == "done":
                    reply.reply, resolved = R.format_crm(intent.crm_operation, res.data), True
                elif res.status == "pending":
                    reply.reply = R.CRM_PENDING
                elif res.status == "circuit_open":
                    reply.reply = R.CRM_DEGRADED
                else:
                    log.warning("Siebel %s falló: %s", intent.crm_operation, res.error)
                    reply.reply = R.CRM_FAILED
        else:  # rag
            t = time.perf_counter()
            ans = self.rag.answer(msg.text, nlu.intent)
            STAGE_LATENCY.labels("rag_total").observe(time.perf_counter() - t)
            reply.reply, reply.sources, reply.generator = ans.text, ans.sources, ans.generator
            resolved = ans.grounded
            if ans.grounded and intent.crm_command and (params := self._crm_params(sess)):
                ref = self.publisher.command(intent.crm_command, {**params, "motivo": intent.name,
                                                                  "detalle": mask_pii(msg.text)},
                                             sess["session_id"], corr)
                reply.reply += R.TICKET_CREATED.format(ref=ref.split(":")[-1].upper())
            if not ans.grounded:
                reply.resolution = "rag_sin_respaldo"

        if decision.action not in ("agent", "pilot_excluded"):
            reply.explanation = self.classifier.explain(nlu.normalized, nlu.intent)
        reply.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        sess["last_intent"] = nlu.intent
        self.sessions.save(sess)
        self.store.upsert_session({"session_id": sess["session_id"], "user_hash": sess["user_hash"],
                                   "canal": sess["channel"], "cohorte": sess["cohort"], "estado": sess["state"],
                                   "resuelta_por_bot": resolved})
        self.persist_message(sess, "bot", reply.reply, intent=nlu.intent, confianza=nlu.confidence,
                      resolucion=reply.resolution, latencia_ms=reply.latency_ms, correlation_id=corr)
        INTENTS.labels(nlu.intent).inc()
        RESOLUTIONS.labels(reply.resolution, msg.channel).inc()
        TURN_LATENCY.labels(msg.channel, reply.resolution).observe(reply.latency_ms / 1000)
        log.info("turno sesion=%s canal=%s intent=%s conf=%.2f resolucion=%s latencia=%.0fms", sess["session_id"],
                 msg.channel, nlu.intent, nlu.confidence, reply.resolution, reply.latency_ms)
        return reply

    # ------------------------------------------------------- notificaciones async
    def pending_updates(self, session_id: str) -> list[dict]:
        """Resultados de Siebel que llegaron después de responder 'pendiente' + mensajes del agente."""
        sess = self.sessions.get(session_id)
        out = []
        for n in self.kv.pop_all(notifications_key(session_id)):
            if n.get("type") == "agent":
                out.append({"from": "agente", "text": n["text"], "ts": n.get("ts")})
                continue
            if n.get("status") == "done":
                text = R.format_crm(n.get("operation"), n.get("data"))
            elif n.get("operation") == "create_ticket":
                continue
            else:
                text = "Sigo sin poder consultar tu información. Te derivo con un ejecutivo si lo prefieres."
            out.append({"from": "bot", "text": text, "correlation_id": n.get("correlation_id")})
            if sess:
                self.persist_message(sess, "bot", text, resolucion="crm_asincrono", correlation_id=n.get("correlation_id"))
        if sess and out:
            self.sessions.save(sess)
        return out
