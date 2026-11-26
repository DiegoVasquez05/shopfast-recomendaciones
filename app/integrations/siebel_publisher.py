"""Siebel Command Publisher (EP2 §5.2): publica operaciones asíncronas con correlation_id.

Flujo de lectura (p. ej. saldo):
  1. Se publica el comando en la cola y se registra la operación como 'pendiente'.
  2. El backend espera el resultado hasta SIEBEL_SYNC_WAIT_MS (por defecto 1,5 s)
     para responder dentro del SLA de 3 s.
  3. Si Siebel está lento o caído, se responde "te aviso en un momento" y el
     resultado llega después como notificación de la sesión (sin perder el mensaje).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from app.integrations.circuit_breaker import read_state

RESULT_TTL = 3600


def result_key(corr: str) -> str:
    return f"siebel:result:{corr}"


def notifications_key(session_id: str) -> str:
    return f"session:{session_id}:notifications"


@dataclass
class OperationResult:
    correlation_id: str
    status: str          # done | pending | failed | circuit_open
    data: dict | None = None
    error: str | None = None


class SiebelPublisher:
    def __init__(self, queue, kv, store, sync_wait_ms: int = 1500):
        self.queue, self.kv, self.store = queue, kv, store
        self.sync_wait_ms = sync_wait_ms

    def circuit_state(self) -> str:
        return read_state(self.kv).get("state", "closed")

    def _publish(self, operation: str, params: dict, session_id: str, correlation_id: str, kind: str) -> str:
        corr = f"{correlation_id}:{uuid.uuid4().hex[:8]}"
        self.store.upsert_operation({"correlation_id": corr, "session_id": session_id, "operacion": operation,
                                     "estado": "pendiente", "intentos": 0, "error": None})
        self.queue.publish({"correlation_id": corr, "operation": operation, "params": params, "kind": kind,
                            "session_id": session_id, "attempt": 0, "published_at": time.time()})
        return corr

    def request(self, operation: str, params: dict, session_id: str, correlation_id: str) -> OperationResult:
        """Lectura con espera acotada."""
        if self.circuit_state() == "open":
            # no se satura Siebel: igual se encola (RPO 0) pero se responde altiro en modo degradado
            corr = self._publish(operation, params, session_id, correlation_id, "read")
            self.kv.set(f"{result_key(corr)}:notify", True, RESULT_TTL)
            return OperationResult(corr, "circuit_open")
        corr = self._publish(operation, params, session_id, correlation_id, "read")
        deadline = time.monotonic() + self.sync_wait_ms / 1000
        while time.monotonic() < deadline:
            res = self.kv.get(result_key(corr))
            if res is not None:
                return OperationResult(corr, res["status"], res.get("data"), res.get("error"))
            time.sleep(0.03)
        # se acabó el tiempo: el worker notificará a la sesión cuando tenga el resultado
        self.kv.set(f"{result_key(corr)}:notify", True, RESULT_TTL)
        return OperationResult(corr, "pending")

    def command(self, operation: str, params: dict, session_id: str, correlation_id: str) -> str:
        """Escritura fire-and-forget (p. ej. crear ticket)."""
        params = {**params, "idempotency_key": f"{session_id}:{operation}:{correlation_id}"}
        return self._publish(operation, params, session_id, correlation_id, "write")
