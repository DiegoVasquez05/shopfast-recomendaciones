"""Circuit Breaker para el CRM Siebel (patrón exigido por el caso).

CLOSED    -> las llamadas pasan; se cuentan fallas consecutivas.
OPEN      -> tras N fallas, se rechazan llamadas de inmediato durante `recovery_s`
             (protege a Siebel de sobrecarga y al chatbot de esperar timeouts).
HALF_OPEN -> pasado el tiempo, UNA llamada de prueba (candado en Redis):
             si funciona se cierra, si falla se vuelve a abrir.

El estado y el contador de fallas viven en Redis (operaciones atómicas), así
todas las réplicas del worker comparten el MISMO circuito: con 2 workers y 4
hilos cada uno, 5 fallas en total abren el circuito para todos. El backend
conversacional y el dashboard leen ese mismo estado.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")
STATE_KEY = "siebel:circuit"
FAILURES_KEY = "siebel:circuit:failures"
PROBE_KEY = "siebel:circuit:probe"


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_s: float = 30.0, kv=None,
                 on_change: Callable[[str], None] | None = None, clock: Callable[[], float] = time.time):
        if kv is None:
            from app.persistence.kv import MemoryKV

            kv = MemoryKV()
        self.failure_threshold, self.recovery_s = failure_threshold, recovery_s
        self.kv, self.on_change, self.clock = kv, on_change, clock
        if self.kv.get(STATE_KEY) is None:
            self._set("closed")

    @property
    def state(self) -> str:
        return (self.kv.get(STATE_KEY) or {}).get("state", "closed")

    def _set(self, state: str, **extra) -> None:
        previous = self.state if self.kv.get(STATE_KEY) else None
        self.kv.set(STATE_KEY, {"state": state, "ts": self.clock(), "failures": extra.pop("failures", 0), **extra})
        if self.on_change and previous and previous != state:
            self.on_change(state)

    def _before_call(self) -> None:
        st = self.kv.get(STATE_KEY) or {"state": "closed"}
        if st["state"] == "open":
            if self.clock() - st.get("opened_at", 0) < self.recovery_s:
                raise CircuitOpenError("Circuit Breaker abierto: Siebel no disponible")
            # tiempo cumplido: solo un proceso hace la llamada de prueba
            if not self.kv.set_nx(PROBE_KEY, 1, ttl=max(5, int(self.recovery_s))):
                raise CircuitOpenError("Circuit Breaker semiabierto: llamada de prueba en curso")
            self._set("half_open")
        elif st["state"] == "half_open":
            raise CircuitOpenError("Circuit Breaker semiabierto: llamada de prueba en curso")

    def call(self, fn: Callable[[], T]) -> T:
        self._before_call()
        try:
            result = fn()
        except Exception:
            failures = self.kv.incr(FAILURES_KEY, ttl=600)
            if self.state == "half_open" or failures >= self.failure_threshold:
                self.kv.delete(FAILURES_KEY)
                self.kv.delete(PROBE_KEY)
                self._set("open", opened_at=self.clock(), failures=failures)
            else:
                st = self.kv.get(STATE_KEY) or {}
                self.kv.set(STATE_KEY, {**st, "failures": failures})
            raise
        self.kv.delete(FAILURES_KEY)
        if self.state != "closed":
            self.kv.delete(PROBE_KEY)
            self._set("closed")
        return result


def read_state(kv) -> dict:
    return kv.get(STATE_KEY) or {"state": "closed", "failures": 0}
