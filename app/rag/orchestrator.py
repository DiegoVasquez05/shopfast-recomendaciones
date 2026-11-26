"""RAG Orchestrator (EP2: rag/orchestrator.py): recupera contexto, construye el prompt,
invoca el LLM y aplica guardrails.

Proveedores de generación:
  extractive -> sin LLM: responde con el fragmento más relevante (determinista, ~1 ms, costo 0).
  bedrock    -> Amazon Bedrock vía LangChain (ChatBedrockConverse). Si excede LLM_TIMEOUT_S o
                falla, cae a extractive para no romper el SLA de 3 s.

Guardrails:
  - Si la similitud del mejor documento es menor a RAG_MIN_SCORE -> no se responde
    "inventando": se ofrece derivar a un ejecutivo.
  - La consulta se envía al LLM con PII enmascarada.
  - El prompt obliga a responder SOLO con el contexto recuperado.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass, field

from app.observability.telemetry import STAGE_LATENCY
from app.privacy import mask_pii
from app.rag.retriever import Retrieved, Retriever

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres TelecomBot, el asistente virtual de TelcoAustral (Chile). Responde en español de Chile, "
    "cercano y respetuoso (tuteo), en máximo 3 oraciones. Usa EXCLUSIVAMENTE la información del CONTEXTO. "
    "Si el contexto no responde la pregunta, di que no tienes esa información y ofrece derivar con un ejecutivo. "
    "Nunca pidas claves, números de tarjeta ni datos sensibles."
)
USER_TEMPLATE = "CONTEXTO:\n{context}\n\nPREGUNTA DEL CLIENTE: {question}"


@dataclass
class RagAnswer:
    text: str
    grounded: bool
    generator: str
    sources: list[dict] = field(default_factory=list)
    top_score: float = 0.0


class BedrockGenerator:
    """Generación con Amazon Bedrock usando LangChain (decisión ADR-004 de la EP2)."""
    name = "bedrock"

    def __init__(self, model_id: str, region: str, timeout_s: float):
        from langchain_aws import ChatBedrockConverse
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatBedrockConverse(model=model_id, region_name=region, temperature=0.2, max_tokens=300)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)])
        self.chain = prompt | llm | StrOutputParser()
        self.timeout_s = timeout_s
        self.pool = cf.ThreadPoolExecutor(max_workers=16)

    def generate(self, question: str, context: str) -> str:
        fut = self.pool.submit(self.chain.invoke, {"context": context, "question": question})
        return fut.result(timeout=self.timeout_s).strip()


class RagOrchestrator:
    def __init__(self, retriever: Retriever, provider: str = "extractive", min_score: float = 0.2, top_k: int = 3,
                 bedrock_model_id: str = "", region: str = "us-east-1", timeout_s: float = 2.0):
        self.retriever, self.min_score, self.top_k = retriever, min_score, top_k
        self.llm = None
        if provider == "bedrock":
            try:
                self.llm = BedrockGenerator(bedrock_model_id, region, timeout_s)
            except Exception as exc:  # p. ej. Bedrock no habilitado en AWS Academy
                log.warning("Bedrock no disponible (%s); se usa generación extractiva", exc)

    @staticmethod
    def _context(hits: list[Retrieved]) -> str:
        return "\n".join(f"[{h.doc.id}] {h.doc.titulo}: {h.doc.contenido}" for h in hits)

    def answer(self, question: str, intent: str | None) -> RagAnswer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, intent=intent, k=self.top_k)
        STAGE_LATENCY.labels("rag_retrieval").observe(time.perf_counter() - t0)
        sources = [{"id": h.doc.id, "titulo": h.doc.titulo, "score": round(h.score, 4),
                    "similitud": round(h.similarity, 4)} for h in hits]
        if not hits or hits[0].score < self.min_score:
            return RagAnswer("No encontré información confiable sobre eso en nuestra base de conocimiento. "
                             "¿Quieres que te derive con un ejecutivo?", False, "guardrail", sources,
                             hits[0].score if hits else 0.0)
        if self.llm is not None:
            t1 = time.perf_counter()
            try:
                text = self.llm.generate(mask_pii(question), self._context(hits))
                STAGE_LATENCY.labels("llm").observe(time.perf_counter() - t1)
                return RagAnswer(text, True, self.llm.name, sources, hits[0].score)
            except Exception as exc:
                log.warning("LLM falló o excedió el timeout (%s); respuesta extractiva", exc)
        return RagAnswer(hits[0].doc.contenido, True, "extractive", sources, hits[0].score)
