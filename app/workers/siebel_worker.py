"""Integrador Siebel (Worker): consume la cola y aísla las caídas del CRM legacy.

    python -m app.workers.siebel_worker

- Lecturas: se ejecutan con Circuit Breaker + reintentos; el resultado se deja en Redis.
- Escrituras: si Siebel falla se re-encolan con backoff exponencial hasta
  SIEBEL_MAX_RETRIES; luego van a la Dead-Letter Queue (reprocesables).
- Si el backend ya respondió "pendiente", el resultado se agrega como
  notificación de la sesión (el widget/WhatsApp lo entrega al cliente).
"""
from __future__ import annotations

import logging
import threading
import time

from prometheus_client import Counter, Gauge, start_http_server

from app.config import get_settings
from app.integrations.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.integrations.queue import build_queue
from app.integrations.siebel_client import SiebelClient
from app.integrations.siebel_publisher import RESULT_TTL, notifications_key, result_key
from app.observability.telemetry import setup_logging
from app.persistence.kv import build_kv
from app.persistence.store import build_store

log = logging.getLogger("siebel-worker")
OPS = Counter("telecombot_siebel_operations_total", "Operaciones Siebel procesadas", ["operation", "status"])
CB_STATE = Gauge("telecombot_siebel_circuit_open", "1 si el Circuit Breaker está abierto")


class SiebelWorker:
    def __init__(self, client: SiebelClient, queue, kv, store, max_retries: int = 3, sleep=time.sleep):
        self.client, self.queue, self.kv, self.store = client, queue, kv, store
        self.max_retries = max_retries
        self.sleep = sleep

    def _finish(self, msg: dict, status: str, data=None, error=None) -> None:
        corr = msg["correlation_id"]
        payload = {"status": status, "data": data, "error": error, "operation": msg["operation"]}
        self.kv.set(result_key(corr), payload, RESULT_TTL)
        self.store.upsert_operation({"correlation_id": corr, "session_id": msg.get("session_id"),
                                     "operacion": msg["operation"], "estado": status,
                                     "intentos": msg.get("attempt", 0) + 1, "error": error})
        if self.kv.get(f"{result_key(corr)}:notify") and msg.get("session_id"):
            self.kv.rpush(notifications_key(msg["session_id"]), {"correlation_id": corr, **payload}, RESULT_TTL)
        OPS.labels(msg["operation"], status).inc()

    def handle(self, msg: dict) -> None:
        op = msg["operation"]
        try:
            data = self.client.execute(op, msg.get("params", {}))
            self._finish(msg, "done", data=data)
            log.info("Siebel %s ok corr=%s", op, msg["correlation_id"])
        except Exception as exc:
            open_ = isinstance(exc, CircuitOpenError)
            if msg.get("kind") == "write" and msg.get("attempt", 0) + 1 < self.max_retries:
                delay = min(2 ** msg.get("attempt", 0), 30)
                log.warning("Siebel %s falló (%s); reintento en %ss", op, exc, delay)
                self.sleep(delay)
                self.queue.publish({**msg, "attempt": msg.get("attempt", 0) + 1})
                return
            self._finish(msg, "circuit_open" if open_ else "failed", error=str(exc)[:300])
            if msg.get("kind") == "write":
                raise  # agotó reintentos -> DLQ
        finally:
            CB_STATE.set(1 if self.client.breaker.state == "open" else 0)


def main() -> None:
    s = get_settings()
    setup_logging(s.service_name.replace("backend", "siebel-worker"))
    kv = build_kv(s.redis_url)
    breaker = CircuitBreaker(s.cb_failure_threshold, s.cb_recovery_s, kv=kv,
                             on_change=lambda st: log.warning("Circuit Breaker Siebel -> %s", st.upper()))
    worker = SiebelWorker(SiebelClient(s, breaker), build_queue(s), kv, build_store(s.database_url),
                          max_retries=s.siebel_max_retries)
    start_http_server(9101)  # /metrics del worker para Prometheus
    n = int(__import__("os").getenv("WORKER_THREADS", "4"))
    threads = [threading.Thread(target=build_queue(s).consume, args=(worker.handle,), daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    log.info("Siebel worker iniciado con %s consumidores", n)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
