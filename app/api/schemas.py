from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str


class ChatRequest(BaseModel):
    user_id: str = Field(..., examples=["demo"], description="Identificador del usuario en el canal")
    text: str = Field(..., min_length=1, max_length=1000, examples=["wena, cuánto debo de la boleta?"])
    session_id: str | None = Field(None, description="Omitir en el primer mensaje")
    customer_id: str | None = Field(None, examples=["C-1001"], description="Cliente autenticado (OIDC)")
    channel: Literal["web", "app"] = "app"


class IntentProb(BaseModel):
    intent: str
    probabilidad: float


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    resolution: str
    intent: str | None
    confidence: float | None
    top_intents: list[IntentProb]
    explanation: list[dict]
    sentiment: dict
    sources: list[dict]
    handoff_id: str | None
    siebel: dict | None
    cohort: str
    nlu_source: str | None
    generator: str | None
    correlation_id: str
    latency_ms: float


class NluRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["no tengo señal en el celu po"])


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["cómo activo el roaming"])
    intent: str | None = None
    k: int = Field(3, ge=1, le=10)
