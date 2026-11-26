"""Human Handoff (EP2: handoff/service.py): derivación fluida a agentes por WebSockets.

- Los agentes se conectan a /ws/agents y reciben en tiempo real los casos derivados
  con el resumen de la conversación (no tienen que preguntar todo de nuevo).
- El cliente (web widget) se conecta a /ws/chat/{session_id} y recibe las
  respuestas del agente. Si no está conectado (o es WhatsApp/App), el mensaje
  se entrega por la cola de notificaciones de la sesión o por el canal.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import WebSocket

from app.observability.telemetry import ACTIVE_WS, HANDOFFS

log = logging.getLogger(__name__)

WAIT_ESTIMATE_MIN_PER_CASE = 2


class ConnectionManager:
    def __init__(self) -> None:
        self.agents: set[WebSocket] = set()
        self.customers: dict[str, set[WebSocket]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self) -> None:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

    async def _send_all(self, sockets, event: dict) -> None:
        for ws in list(sockets):
            try:
                await ws.send_json(event)
            except Exception:
                sockets.discard(ws)

    def emit_agents(self, event: dict) -> None:
        self._emit(self._send_all(self.agents, event))

    def emit_customer(self, session_id: str, event: dict) -> bool:
        socks = self.customers.get(session_id)
        if not socks:
            return False
        self._emit(self._send_all(socks, event))
        return True

    def _emit(self, coro) -> None:
        """Permite emitir desde código síncrono (threadpool de FastAPI) hacia el event loop."""
        if self.loop is None:
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self.loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    def add_agent(self, ws: WebSocket) -> None:
        self.agents.add(ws)
        ACTIVE_WS.labels("agent").set(len(self.agents))

    def remove_agent(self, ws: WebSocket) -> None:
        self.agents.discard(ws)
        ACTIVE_WS.labels("agent").set(len(self.agents))

    def add_customer(self, session_id: str, ws: WebSocket) -> None:
        self.customers.setdefault(session_id, set()).add(ws)
        ACTIVE_WS.labels("customer").set(sum(len(v) for v in self.customers.values()))

    def remove_customer(self, session_id: str, ws: WebSocket) -> None:
        self.customers.get(session_id, set()).discard(ws)
        ACTIVE_WS.labels("customer").set(sum(len(v) for v in self.customers.values()))


class HandoffService:
    def __init__(self, store, manager: ConnectionManager):
        self.store, self.manager = store, manager

    def wait_estimate(self) -> str:
        open_cases = len(self.store.handoffs("pendiente"))
        minutes = max(1, open_cases * WAIT_ESTIMATE_MIN_PER_CASE // max(1, len(self.manager.agents) or 1))
        return f"{minutes} min aprox"

    def create(self, session: dict, reason: str, nlu=None, sentiment=None) -> dict:
        resumen = {
            "canal": session["channel"], "cliente": session.get("customer_id") or "no identificado",
            "ultima_intencion": getattr(nlu, "intent", None), "confianza": getattr(nlu, "confidence", None),
            "sentimiento": (sentiment or {}).get("label"),
            "historial": [f"{t['role']}: {t['text']}" for t in session["history"][-6:]],
        }
        h = {"id": f"HO-{uuid.uuid4().hex[:8]}", "session_id": session["session_id"], "motivo": reason,
             "estado": "pendiente", "resumen": str(resumen)}
        self.store.create_handoff(h)
        session["state"] = "handoff"
        session["handoff_id"] = h["id"]
        HANDOFFS.labels(reason).inc()
        self.manager.emit_agents({"type": "handoff_created", "handoff": {**h, "resumen": resumen},
                                  "ts": time.time()})
        log.info("Handoff %s creado para sesión %s (motivo=%s)", h["id"], session["session_id"], reason)
        return h

    def forward_customer_message(self, session: dict, text: str) -> None:
        self.manager.emit_agents({"type": "customer_message", "handoff_id": session.get("handoff_id"),
                                  "session_id": session["session_id"], "text": text, "ts": time.time()})
