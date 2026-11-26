"""Intent/Policy Router + Confidence Evaluator (EP2: conversation/router.py).

Decide si el turno se resuelve con Siebel (crm), con la base de conocimiento (rag),
con small talk, pidiendo aclaración, o derivando a un humano (handoff).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.nlp.classifier import NluResult
from app.nlp.intents import get_intent


@dataclass
class Decision:
    action: str          # agent | pilot_excluded | handoff | clarify | crm | rag | smalltalk
    reason: str


class PolicyRouter:
    def __init__(self, threshold: float, max_low_conf: int):
        self.threshold, self.max_low_conf = threshold, max_low_conf

    def decide(self, session: dict, nlu: NluResult, sentiment: dict) -> Decision:
        if session.get("state") == "handoff":
            return Decision("agent", "conversacion_con_ejecutivo")
        if session.get("cohort") != "piloto":
            return Decision("pilot_excluded", "fuera_del_piloto_10")
        if sentiment.get("label") == "NEGATIVE" and sentiment.get("negativos", 0) >= 3:
            return Decision("handoff", "cliente_molesto")
        if nlu.confidence < self.threshold:
            session["low_conf_streak"] = session.get("low_conf_streak", 0) + 1
            if session["low_conf_streak"] >= self.max_low_conf:
                return Decision("handoff", "baja_confianza")
            return Decision("clarify", "baja_confianza")
        session["low_conf_streak"] = 0
        intent = get_intent(nlu.intent)
        if intent.resolution == "handoff":
            return Decision("handoff", f"intencion_{intent.name}")
        return Decision(intent.resolution, f"intencion_{intent.name}")
