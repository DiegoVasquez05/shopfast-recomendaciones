"""TelecomBot AI — Backend conversacional (FastAPI).

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Swagger/OpenAPI: /docs  ·  ReDoc: /redoc  ·  Web Widget: /widget  ·  Consola de agentes: /agent
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.controllers import router
from app.api.ratelimit import RateLimiter, client_key
from app.channels.whatsapp import WhatsAppSender
from app.config import Settings, get_settings
from app.conversation.engine import ConversationEngine
from app.conversation.session_service import SessionService
from app.handoff.service import ConnectionManager, HandoffService
from app.integrations.circuit_breaker import CircuitBreaker
from app.integrations.queue import MemoryQueue, build_queue
from app.integrations.siebel_client import SiebelClient
from app.integrations.siebel_publisher import SiebelPublisher
from app.nlp.classifier import build_classifier
from app.nlp.sentiment import SentimentAnalyzer
from app.observability.telemetry import HTTP_LATENCY, correlation_id, setup_logging
from app.persistence.kv import build_kv
from app.persistence.store import build_store
from app.privacy import MessageCipher
from app.rag.knowledge import load_kb
from app.rag.orchestrator import RagOrchestrator
from app.rag.retriever import Retriever, build_vector_store
from app.workers.siebel_worker import SiebelWorker

log = logging.getLogger("telecombot")

TAGS = [
    {"name": "Conversación", "description": "Turnos del chatbot (Web Widget y App móvil)"},
    {"name": "Modelo IA", "description": "Inferencia NLU (SageMaker/local) y búsqueda RAG"},
    {"name": "Canales", "description": "Webhooks de WhatsApp Business"},
    {"name": "Handoff", "description": "Derivación a agentes humanos"},
    {"name": "Operación", "description": "KPIs y estado del sistema"},
    {"name": "Privacidad", "description": "Ley 19.628 / GDPR: acceso y supresión de datos"},
    {"name": "Autenticación", "description": "OAuth2 client credentials y token del widget"},
    {"name": "Sistema", "description": "Health checks y métricas Prometheus"},
]


def _inprocess_siebel_http(settings):
    """Para correr sin Docker: el mock de Siebel se monta en el mismo proceso."""
    from fastapi.testclient import TestClient

    from app.siebel_mock.main import app as siebel_app

    c = TestClient(siebel_app)
    c.auth = (settings.siebel_user, settings.siebel_password)
    return c


def create_app(settings: Settings | None = None, *, kv=None, store=None, queue=None, siebel_http=None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.service_name)
    app = FastAPI(title="TelecomBot AI — API", version="1.0.0", openapi_tags=TAGS,
                  description="Backend conversacional de TelcoAustral: NLU en español chileno, RAG con pgvector, "
                              "integración asíncrona con Siebel (Circuit Breaker) y handoff a agentes.\n\n"
                              "**Autenticación:** `POST /oauth/token` (client credentials) y luego "
                              "`Authorization: Bearer <token>`.")
    s = app.state
    s.settings, s.started = settings, time.time()
    s.kv = kv or build_kv(settings.redis_url)
    s.store = store or build_store(settings.database_url)
    s.queue = queue or build_queue(settings)
    s.cipher = MessageCipher(settings.message_encryption_key, settings.pii_salt)

    # --- NLP / RAG
    s.classifier = build_classifier(settings)
    s.sentiment = SentimentAnalyzer(settings.sentiment_provider, settings.aws_region)
    embedder = joblib.load(Path(settings.models_dir) / "embedder.joblib")
    s.vector_store = build_vector_store(settings.database_url, embedder.dim)
    s.retriever = Retriever(embedder, s.vector_store, load_kb(settings.kb_path))
    s.rag = RagOrchestrator(s.retriever, settings.llm_provider, settings.rag_min_score, settings.rag_top_k,
                            settings.bedrock_model_id, settings.aws_region, settings.llm_timeout_s)
    metrics_file = Path(settings.models_dir) / "metrics.json"
    s.model_metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

    # --- Siebel / handoff / canales
    s.sessions = SessionService(s.kv, settings)
    s.publisher = SiebelPublisher(s.queue, s.kv, s.store, settings.siebel_sync_wait_ms, settings.cb_recovery_s)
    if isinstance(s.queue, MemoryQueue):
        # sin broker externo, el worker corre en el mismo proceso con la MISMA lógica
        http = siebel_http or (_inprocess_siebel_http(settings) if settings.siebel_base_url == "inprocess" else None)
        breaker = CircuitBreaker(settings.cb_failure_threshold, settings.cb_recovery_s, kv=s.kv)
        s.worker = SiebelWorker(SiebelClient(settings, breaker, http=http), s.queue, s.kv, s.store,
                                settings.siebel_max_retries, sleep=lambda _: None)
        s.queue.consume(s.worker.handle)
    s.ws = ConnectionManager(settings.redis_url)
    s.handoff = HandoffService(s.store, s.ws)
    s.whatsapp = WhatsAppSender(settings.whatsapp_token, settings.whatsapp_phone_number_id)
    s.engine = ConversationEngine(settings=settings, sessions=s.sessions, classifier=s.classifier,
                                  sentiment=s.sentiment, rag=s.rag, publisher=s.publisher, handoff=s.handoff,
                                  store=s.store, kv=s.kv, cipher=s.cipher)
    s.limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def correlation_and_limits(request: Request, call_next):
        corr = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex[:16]
        token = correlation_id.set(corr)
        t0 = time.perf_counter()
        try:
            if settings.rate_limit_enabled and request.url.path.startswith(("/api/", "/webhooks/", "/oauth/")):
                ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip() or
                      (request.client.host if request.client else "anon"))
                key, quota = client_key(request.headers.get("authorization"), ip, settings.rate_limit_per_minute)
                ok, retry = s.limiter.check(key, min(quota, settings.rate_limit_per_minute)
                                            if key.startswith("ip:") else quota)
                if not ok:
                    return JSONResponse({"detail": "Rate limit excedido"}, status_code=429,
                                        headers={"Retry-After": str(retry), "X-Correlation-ID": corr})
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", "other")
            HTTP_LATENCY.labels(request.method, route, response.status_code).observe(time.perf_counter() - t0)
            response.headers["X-Correlation-ID"] = corr
            response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
            return response
        finally:
            correlation_id.reset(token)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    app.include_router(router)
    log.info("TelecomBot iniciado: nlu=%s vector_store=%s cola=%s llm=%s", s.classifier.source, s.vector_store.name,
             type(s.queue).__name__, settings.llm_provider)
    return app


def __getattr__(name: str):  # `uvicorn app.main:app` crea la app al importarla
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
