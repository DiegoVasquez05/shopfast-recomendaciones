"""Circuit Breaker para el CRM Siebel (patrón exigido por el caso).

CLOSED    -> las llamadas pasan; se cuentan fallas consecutivas.
OPEN      -> tras N fallas, se rechazan llamadas de inmediato durante `recovery_s`
             (protege a Siebel de sobrecarga y al chatbot de esperar timeouts).
HALF_OPEN -> pasado el tiempo, se deja pasar una llamada de prueba:
             si funciona se cierra, si falla se vuelve a abrir.

El estado se publica en el KV compartido para que el backend conversacional
y el dashboard lo conozcan sin llamar a Siebel.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")
STATE_KEY = "siebel:circuit"


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_s: float = 30.0, kv=None,
                 on_change: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self.kv = kv
        self.on_change = on_change
        self.clock = clock
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self._lock = threading.Lock()
        self._publish()

    def _set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self._publish()
            if self.on_change:
                self.on_change(state)

    def _publish(self) -> None:
        if self.kv is not None:
            self.kv.set(STATE_KEY, {"state": self.state, "failures": self.failures, "ts": time.time()})

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self.state == "open":
                if self.clock() - self.opened_at >= self.recovery_s:
                    self._set("half_open")
                else:
                    raise CircuitOpenError("Circuit Breaker abierto: Siebel no disponible")
        try:
            result = fn()
        except Exception:
            with self._lock:
                self.failures += 1
                if self.state == "half_open" or self.failures >= self.failure_threshold:
                    self.opened_at = self.clock()
                    self._set("open")
                else:
                    self._publish()
            raise
        with self._lock:
            self.failures = 0
            self._set("closed")
        return result


def read_state(kv) -> dict:
    return kv.get(STATE_KEY) or {"state": "closed", "failures": 0}
