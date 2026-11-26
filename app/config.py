"""Configuración centralizada (12-factor): todo se lee desde variables de entorno.

Los mismos valores funcionan en local (docker-compose), en EC2 y en Kubernetes
(ConfigMap/Secret). Sin REDIS_URL / colas / DATABASE_URL el sistema corre en
modo memoria (útil para tests y demo local).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "si", "sí"}


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


def _pairs(name: str, default: str) -> dict[str, str]:
    """'cliente1:secreto1,cliente2:secreto2' -> dict"""
    out = {}
    for item in _list(name, default):
        k, _, v = item.partition(":")
        out[k] = v
    return out


@dataclass
class Settings:
    service_name: str = field(default_factory=lambda: _env("SERVICE_NAME", "telecombot-backend"))
    environment: str = field(default_factory=lambda: _env("ENVIRONMENT", "dev"))

    # ---------------- Modelos / NLP ----------------
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "models")))
    kb_path: Path = field(default_factory=lambda: Path(_env("KB_PATH", "data/kb/telcoaustral_kb.json")))
    sagemaker_endpoint: str = field(default_factory=lambda: _env("SAGEMAKER_ENDPOINT"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-1")))
    intent_threshold: float = field(default_factory=lambda: float(_env("INTENT_THRESHOLD", "0.45")))
    rag_min_score: float = field(default_factory=lambda: float(_env("RAG_MIN_SCORE", "0.20")))
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "3")))
    max_low_confidence_turns: int = field(default_factory=lambda: int(_env("MAX_LOW_CONFIDENCE_TURNS", "2")))

    # LLM: "extractive" (sin LLM: responde con el fragmento recuperado) o "bedrock" (Amazon Bedrock vía LangChain)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "extractive"))
    bedrock_model_id: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "2.0")))
    # Sentimiento: "lexicon" (léxico chileno local) o "comprehend" (Amazon Comprehend)
    sentiment_provider: str = field(default_factory=lambda: _env("SENTIMENT_PROVIDER", "lexicon"))

    # ---------------- Infraestructura ----------------
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL"))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    # Cola: "rabbitmq", "sqs" o "memory"
    queue_backend: str = field(default_factory=lambda: _env("QUEUE_BACKEND", "memory"))
    rabbitmq_url: str = field(default_factory=lambda: _env("RABBITMQ_URL"))
    sqs_queue_url: str = field(default_factory=lambda: _env("SQS_QUEUE_URL"))
    siebel_queue: str = field(default_factory=lambda: _env("SIEBEL_QUEUE", "siebel.commands"))

    # ---------------- Siebel CRM ----------------
    siebel_base_url: str = field(default_factory=lambda: _env("SIEBEL_BASE_URL", "http://siebel-mock:9000"))
    siebel_secret_arn: str = field(default_factory=lambda: _env("SIEBEL_SECRET_ARN"))  # AWS Secrets Manager (Vault)
    siebel_user: str = field(default_factory=lambda: _env("SIEBEL_USER", "telecombot"))
    siebel_password: str = field(default_factory=lambda: _env("SIEBEL_PASSWORD", "dev-password"))
    siebel_timeout_s: float = field(default_factory=lambda: float(_env("SIEBEL_TIMEOUT_S", "1.0")))
    siebel_sync_wait_ms: int = field(default_factory=lambda: int(_env("SIEBEL_SYNC_WAIT_MS", "1500")))
    siebel_max_retries: int = field(default_factory=lambda: int(_env("SIEBEL_MAX_RETRIES", "3")))
    cb_failure_threshold: int = field(default_factory=lambda: int(_env("CB_FAILURE_THRESHOLD", "5")))
    cb_recovery_s: float = field(default_factory=lambda: float(_env("CB_RECOVERY_S", "30")))

    # ---------------- Seguridad ----------------
    # Clientes OAuth2 (client credentials) por canal / servicio
    oauth_clients: dict[str, str] = field(default_factory=lambda: _pairs(
        "OAUTH_CLIENTS",
        "mobile-app:dev-mobile-secret,dashboard:dev-dashboard-secret,agent-console:dev-agent-secret,"
        "loadtest:dev-loadtest-secret"))
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", "dev-jwt-secret-cambiar-en-produccion-32b"))
    jwt_expire_minutes: int = field(default_factory=lambda: int(_env("JWT_EXPIRE_MINUTES", "60")))
    oidc_issuer: str = field(default_factory=lambda: _env("OIDC_ISSUER"))          # ej. Amazon Cognito
    oidc_audience: str = field(default_factory=lambda: _env("OIDC_AUDIENCE"))
    pii_salt: str = field(default_factory=lambda: _env("PII_SALT", "dev-salt-cambiar"))
    message_encryption_key: str = field(default_factory=lambda: _env("MESSAGE_ENCRYPTION_KEY"))
    rate_limit_per_minute: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MINUTE", "120")))
    rate_limit_enabled: bool = field(default_factory=lambda: _bool("RATE_LIMIT_ENABLED", True))
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS", "*"))

    # ---------------- Canales ----------------
    whatsapp_verify_token: str = field(default_factory=lambda: _env("WHATSAPP_VERIFY_TOKEN", "dev-verify-token"))
    whatsapp_app_secret: str = field(default_factory=lambda: _env("WHATSAPP_APP_SECRET", "dev-whatsapp-secret"))
    whatsapp_token: str = field(default_factory=lambda: _env("WHATSAPP_TOKEN"))
    whatsapp_phone_number_id: str = field(default_factory=lambda: _env("WHATSAPP_PHONE_NUMBER_ID"))

    # ---------------- Piloto controlado ----------------
    pilot_percentage: int = field(default_factory=lambda: int(_env("PILOT_PERCENTAGE", "10")))
    pilot_allowlist: list[str] = field(default_factory=lambda: _list("PILOT_ALLOWLIST", "demo"))

    session_ttl_s: int = field(default_factory=lambda: int(_env("SESSION_TTL_S", "1800")))


def get_settings() -> Settings:
    return Settings()
