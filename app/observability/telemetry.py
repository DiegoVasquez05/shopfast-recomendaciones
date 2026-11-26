"""Observabilidad (EP2 §8.2): métricas Prometheus, logs JSON estructurados con
correlation_id y sanitización de PII en TODOS los logs."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

from prometheus_client import Counter, Gauge, Histogram

from app.privacy import mask_pii

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

HTTP_LATENCY = Histogram("telecombot_http_request_seconds", "Latencia HTTP por ruta",
                         ["method", "route", "status"], buckets=(.01, .025, .05, .1, .25, .5, 1, 2, 3, 5))
TURN_LATENCY = Histogram("telecombot_turn_latency_seconds", "Latencia de un turno conversacional (SLA <= 3 s)",
                         ["channel", "resolution"], buckets=(.05, .1, .25, .5, 1, 1.5, 2, 3, 5))
STAGE_LATENCY = Histogram("telecombot_stage_latency_seconds", "Latencia por etapa (nlu, rag, llm, siebel)",
                          ["stage"], buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2))
INTENTS = Counter("telecombot_intents_total", "Intenciones detectadas", ["intent"])
RESOLUTIONS = Counter("telecombot_resolutions_total", "Resolución por turno", ["resolution", "channel"])
HANDOFFS = Counter("telecombot_handoffs_total", "Derivaciones a agente humano", ["reason"])
SIEBEL_RESULTS = Counter("telecombot_siebel_requests_total", "Resultado de consultas a Siebel", ["status"])
ACTIVE_WS = Gauge("telecombot_websocket_connections", "Conexiones WebSocket activas", ["kind"])
LOW_CONF = Counter("telecombot_low_confidence_total", "Turnos bajo el umbral de confianza")


class PiiFilter(logging.Filter):
    """Enmascara RUT, correos, teléfonos y tarjetas en cualquier log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(str(record.msg))
        if record.args:
            record.args = tuple(mask_pii(str(a)) if isinstance(a, str) else a for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
               "service": self.service, "logger": record.name, "correlation_id": correlation_id.get(),
               "msg": record.getMessage()}
        if record.exc_info:
            out["exc"] = mask_pii(self.formatException(record.exc_info))
        return json.dumps(out, ensure_ascii=False)


def setup_logging(service: str, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter(service))
    h.addFilter(PiiFilter())
    root.addHandler(h)
    root.setLevel(level)
    for noisy in ("httpx", "pika", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
