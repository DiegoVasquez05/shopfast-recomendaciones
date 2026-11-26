"""API Controllers (EP2: api/controllers.py): endpoints REST, webhooks y WebSockets."""
from __future__ import annotations

import json
import time

from fastapi import (APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, Request, Response, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api import schemas as S
from app.api.security import authenticate_client, decode_ws_token, issue_token, require_scope
from app.channels.normalizer import InboundMessage, from_whatsapp
from app.channels.whatsapp import verify_signature
from app.integrations.circuit_breaker import read_state
from app.integrations.siebel_publisher import notifications_key
from app.nlp.intents import INTENTS
from app.privacy import pseudonymize

router = APIRouter()
STATIC = __import__("pathlib").Path(__file__).resolve().parent.parent / "static"


def st(request_or_ws):
    return request_or_ws.app.state


# --------------------------------------------------------------------- Sistema
@router.get("/health", tags=["Sistema"], summary="Liveness probe")
def health(request: Request):
    s = st(request)
    return {"status": "ok", "service": s.settings.service_name, "model_version": s.classifier.version,
            "uptime_s": round(time.time() - s.started, 1)}


@router.get("/ready", tags=["Sistema"], summary="Readiness probe: estado de dependencias")
def ready(request: Request):
    s = st(request)
    deps = {"kv": s.kv.ping(), "store": s.store.ping(), "queue": s.queue.ping(),
            "vector_store": s.vector_store.name, "nlu": s.classifier.source,
            "siebel_circuit": read_state(s.kv).get("state")}
    ok = deps["kv"] and deps["store"] and deps["queue"]
    return JSONResponse({"ready": ok, "dependencies": deps}, status_code=200 if ok else 503)


@router.get("/metrics", tags=["Sistema"], summary="Métricas Prometheus")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------- OAuth2
@router.post("/oauth/token", response_model=S.TokenResponse, tags=["Autenticación"],
             summary="OAuth2 client credentials (canales y servicios)")
def oauth_token(request: Request, grant_type: str = Form(...), client_id: str = Form(...),
                client_secret: str = Form(...)):
    if grant_type != "client_credentials":
        raise HTTPException(400, "unsupported_grant_type")
    s = st(request).settings
    scope = authenticate_client(s, client_id, client_secret)
    return S.TokenResponse(access_token=issue_token(s, client_id, scope), expires_in=s.jwt_expire_minutes * 60,
                           scope=scope)


@router.post("/api/v1/channels/web/token", response_model=S.TokenResponse, tags=["Autenticación"],
             summary="Token anónimo de corta duración para el Web Widget (cliente público)")
def web_token(request: Request):
    s = st(request).settings
    return S.TokenResponse(access_token=issue_token(s, "web-widget", "chat:write", minutes=30), expires_in=1800,
                           scope="chat:write")


# --------------------------------------------------------------------- Chat
@router.post("/api/v1/chat/messages", response_model=S.ChatResponse, tags=["Conversación"],
             summary="Enviar un mensaje al chatbot (Web Widget / App móvil)")
def chat(body: S.ChatRequest, request: Request, _=Depends(require_scope("chat:write"))):
    msg = InboundMessage(channel=body.channel, user_id=body.user_id, text=body.text, session_id=body.session_id,
                         customer_id=body.customer_id)
    return st(request).engine.handle(msg).to_dict()


@router.get("/api/v1/chat/sessions/{session_id}/updates", tags=["Conversación"],
            summary="Mensajes asíncronos pendientes (respuestas tardías de Siebel o del agente)")
def updates(session_id: str, request: Request, _=Depends(require_scope("chat:write"))):
    return {"session_id": session_id, "updates": st(request).engine.pending_updates(session_id)}


@router.get("/api/v1/chat/sessions/{session_id}", tags=["Conversación"], summary="Estado e historial de la sesión")
def session_state(session_id: str, request: Request, _=Depends(require_scope("chat:write"))):
    sess = st(request).sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Sesión no encontrada o expirada")
    return {k: v for k, v in sess.items() if k not in ("user_id", "phone")}


# -------------------------------------------------------------------- Modelo
@router.post("/api/v1/nlu/predict", tags=["Modelo IA"], summary="Inferencia directa del clasificador de intenciones")
def nlu_predict(body: S.NluRequest, request: Request, _=Depends(require_scope("chat:write"))):
    s = st(request)
    r = s.classifier.predict(body.text)
    intent = INTENTS.get(r.intent)
    return {"texto_normalizado": r.normalized, "intent": r.intent,
            "descripcion": intent.descripcion if intent else None,
            "resolucion": intent.resolution if intent else None, "confidence": r.confidence,
            "sobre_umbral": r.confidence >= s.settings.intent_threshold,
            "top": [{"intent": i, "probabilidad": p} for i, p in r.top],
            "explicacion": s.classifier.explain(r.normalized, r.intent), "sentimiento": s.sentiment.analyze(body.text),
            "source": r.source, "latency_ms": r.latency_ms, "model_version": s.classifier.version}


@router.post("/api/v1/rag/search", tags=["Modelo IA"], summary="Búsqueda semántica en la base de conocimiento")
def rag_search(body: S.RagRequest, request: Request, _=Depends(require_scope("chat:write"))):
    t = time.perf_counter()
    hits = st(request).retriever.search(body.query, intent=body.intent, k=body.k)
    return {"latency_ms": round((time.perf_counter() - t) * 1000, 2), "vector_store": st(request).vector_store.name,
            "results": [{"id": h.doc.id, "titulo": h.doc.titulo, "score": round(h.score, 4),
                         "similitud": round(h.similarity, 4), "intents": h.doc.intents, "contenido": h.doc.contenido}
                        for h in hits]}


@router.get("/api/v1/model", tags=["Modelo IA"], summary="Versión, métricas e intenciones del modelo")
def model_info(request: Request, _=Depends(require_scope("metrics:read"))):
    s = st(request)
    return {"model_version": s.classifier.version, "nlu_source": s.classifier.source,
            "sagemaker_endpoint": s.settings.sagemaker_endpoint or None, "llm_provider": s.settings.llm_provider,
            "intent_threshold": s.settings.intent_threshold, "metrics": s.model_metrics,
            "intents": {k: {"categoria": v.category, "descripcion": v.descripcion, "resolucion": v.resolution}
                        for k, v in INTENTS.items()}}


@router.get("/api/v1/metrics/summary", tags=["Operación"], summary="KPIs de negocio y operación")
def summary(request: Request, _=Depends(require_scope("metrics:read"))):
    s = st(request)
    return {"kpis": s.store.stats(), "siebel_circuit": read_state(s.kv), "cola_siebel_pendientes": s.queue.depth(),
            "agentes_conectados_en_esta_replica": len(s.ws.agents), "piloto_porcentaje": s.settings.pilot_percentage,
            "objetivos": {"automatizacion": 0.60, "latencia_p95_ms": 3000, "espera_min": 2}}


# ------------------------------------------------------------------- Handoff
@router.get("/api/v1/handoffs", tags=["Handoff"], summary="Casos derivados a agentes")
def handoffs(request: Request, estado: str | None = Query(None), _=Depends(require_scope("agent"))):
    return {"handoffs": st(request).store.handoffs(estado)}


def _agent_reply(state, session_id: str, text: str, agent: str) -> None:
    sess = state.sessions.get(session_id)
    if not sess:
        return
    event = {"type": "agent", "text": text, "agent": agent, "ts": time.time()}
    state.engine.persist_message(sess, "agente", text)
    state.sessions.save(sess)
    if sess["channel"] == "whatsapp" and sess.get("phone"):
        state.whatsapp.send_text(sess["phone"], text)
    elif not state.ws.emit_customer(session_id, event):
        state.kv.rpush(notifications_key(session_id), event, 3600)


def _close_handoff(state, session_id: str, agent: str) -> None:
    sess = state.sessions.get(session_id)
    if not sess or not sess.get("handoff_id"):
        return
    state.store.update_handoff(sess["handoff_id"], estado="cerrado", agente=agent)
    sess["state"], sess["handoff_id"], sess["low_conf_streak"] = "bot", None, 0
    state.sessions.save(sess)


@router.post("/api/v1/handoffs/{session_id}/reply", tags=["Handoff"], summary="Respuesta del agente (alternativa REST)")
def agent_reply(session_id: str, request: Request, text: str = Form(...), claims=Depends(require_scope("agent"))):
    _agent_reply(st(request), session_id, text, claims["sub"])
    return {"ok": True}


@router.post("/api/v1/handoffs/{session_id}/close", tags=["Handoff"], summary="Cerrar caso y devolver al bot")
def close_handoff(session_id: str, request: Request, claims=Depends(require_scope("agent"))):
    _close_handoff(st(request), session_id, claims["sub"])
    return {"ok": True}


@router.websocket("/ws/agents")
async def ws_agents(ws: WebSocket, token: str | None = None):
    state = st(ws)
    claims = decode_ws_token(state.settings, token, "agent")
    if claims is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    state.ws.bind_loop()
    state.ws.add_agent(ws)
    await ws.send_json({"type": "pending", "handoffs": state.store.handoffs("pendiente")})
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "reply":
                _agent_reply(state, data["session_id"], data["text"], claims["sub"])
            elif data.get("type") == "close":
                _close_handoff(state, data["session_id"], claims["sub"])
    except WebSocketDisconnect:
        pass
    finally:
        state.ws.remove_agent(ws)


@router.websocket("/ws/chat/{session_id}")
async def ws_chat(ws: WebSocket, session_id: str, token: str | None = None):
    state = st(ws)
    if decode_ws_token(state.settings, token, "chat:write") is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    state.ws.bind_loop()
    state.ws.add_customer(session_id, ws)
    try:
        while True:
            await ws.receive_text()  # canal solo de bajada; los mensajes del cliente van por REST
    except WebSocketDisconnect:
        pass
    finally:
        state.ws.remove_customer(session_id, ws)


# ------------------------------------------------------------------ WhatsApp
@router.get("/webhooks/whatsapp", tags=["Canales"], summary="Verificación del webhook (Meta)")
def whatsapp_verify(request: Request):
    q = request.query_params
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == st(request).settings.whatsapp_verify_token:
        return PlainTextResponse(q.get("hub.challenge", ""))
    raise HTTPException(403, "verify_token inválido")


@router.post("/webhooks/whatsapp", tags=["Canales"], summary="Mensajes entrantes de WhatsApp Business (firmados)")
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    s = st(request)
    raw = await request.body()
    if not verify_signature(raw, request.headers.get("X-Hub-Signature-256"), s.settings.whatsapp_app_secret):
        raise HTTPException(401, "Firma X-Hub-Signature-256 inválida")
    payload = json.loads(raw)
    msgs = [m for e in payload.get("entry", []) for c in e.get("changes", []) for m in from_whatsapp(c.get("value", {}))]

    def process():
        for m in msgs:
            reply = s.engine.handle(m)
            s.whatsapp.send_text(m.phone, reply.reply)

    background.add_task(process)  # Meta exige responder 200 rápido; el turno se procesa en segundo plano
    return {"received": len(msgs)}


# ----------------------------------------------------------- Privacidad (GDPR)
@router.get("/api/v1/privacy/users/{user_id}", tags=["Privacidad"], summary="Exportar datos del usuario (acceso)")
def export_user(user_id: str, request: Request, _=Depends(require_scope("admin"))):
    s = st(request)
    data = s.store.export_user(pseudonymize(user_id, s.settings.pii_salt))
    for m in data.get("mensajes", []):
        m["texto"] = s.cipher.decrypt(m.pop("texto_cifrado"))
    return data


@router.delete("/api/v1/privacy/users/{user_id}", tags=["Privacidad"], summary="Derecho al olvido (supresión)")
def delete_user(user_id: str, request: Request, _=Depends(require_scope("admin"))):
    s = st(request)
    uh = pseudonymize(user_id, s.settings.pii_salt)
    deleted = s.store.delete_user(uh)
    deleted["sesiones_cache"] = s.sessions.delete_user_sessions(uh)
    return {"eliminado": deleted}


# ---------------------------------------------------------------- Interfaces
@router.get("/widget", include_in_schema=False)
def widget():
    return FileResponse(STATIC / "widget.html")


@router.get("/agent", include_in_schema=False)
def agent_console():
    return FileResponse(STATIC / "agent.html")
