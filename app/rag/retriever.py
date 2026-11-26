"""Retriever del RAG (EP2: rag/retriever.py) sobre una base vectorial.

- PgVectorStore: PostgreSQL + extensión pgvector con índice HNSW (distancia coseno).
- MemoryVectorStore: producto punto con numpy (tests/demo).

Ranking híbrido (re-ranking de los candidatos de la base vectorial):
  score = 0,65 * similitud semántica (LSA, pgvector)
        + 0,35 * similitud léxica (TF-IDF de palabras normalizadas, captura términos exactos
                 como "router", "luz roja", "decodificador")
        + 0,15 si el documento está etiquetado con la intención detectada por el NLU.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.rag.embeddings import LsaEmbedder
from app.nlp.normalizer_cl import normalize
from app.rag.knowledge import KnowledgeDoc

log = logging.getLogger(__name__)
INTENT_BOOST = 0.15
W_SEMANTIC, W_LEXICAL = 0.65, 0.35


@dataclass
class Retrieved:
    doc: KnowledgeDoc
    score: float
    similarity: float


class MemoryVectorStore:
    name = "memory"

    def __init__(self):
        self.docs: list[KnowledgeDoc] = []
        self.vecs = np.zeros((0, 1), dtype=np.float32)

    def upsert(self, docs: list[KnowledgeDoc], vecs: np.ndarray) -> None:
        self.docs, self.vecs = list(docs), vecs

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[KnowledgeDoc, float]]:
        sims = self.vecs @ qvec
        order = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in order]

    def count(self) -> int:
        return len(self.docs)


class PgVectorStore:
    name = "pgvector"

    def __init__(self, url: str, dim: int):
        from psycopg_pool import ConnectionPool

        self.dim = dim
        self.pool = ConnectionPool(url, min_size=1, max_size=10, open=True, timeout=5, kwargs={"autocommit": True})
        with self.pool.connection() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""CREATE TABLE IF NOT EXISTS fuente_conocimiento (
                id TEXT PRIMARY KEY, titulo TEXT, intents JSONB, contenido TEXT, fuente TEXT, actualizado TEXT,
                embedding vector({dim}))""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_kb_hnsw ON fuente_conocimiento "
                      "USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def _lit(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    def upsert(self, docs: list[KnowledgeDoc], vecs: np.ndarray) -> None:
        with self.pool.connection() as c:
            for d, v in zip(docs, vecs):
                c.execute("""INSERT INTO fuente_conocimiento VALUES (%s,%s,%s,%s,%s,%s,%s::vector)
                             ON CONFLICT (id) DO UPDATE SET titulo=EXCLUDED.titulo, intents=EXCLUDED.intents,
                             contenido=EXCLUDED.contenido, fuente=EXCLUDED.fuente, actualizado=EXCLUDED.actualizado,
                             embedding=EXCLUDED.embedding""",
                          (d.id, d.titulo, json.dumps(d.intents), d.contenido, d.fuente, d.actualizado, self._lit(v)))
        log.info("Ingestados %s documentos en pgvector", len(docs))

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[KnowledgeDoc, float]]:
        with self.pool.connection() as c:
            rows = c.execute("""SELECT id, titulo, intents, contenido, fuente, actualizado,
                                1 - (embedding <=> %s::vector) AS sim
                                FROM fuente_conocimiento ORDER BY embedding <=> %s::vector LIMIT %s""",
                             (self._lit(qvec), self._lit(qvec), k)).fetchall()
        return [(KnowledgeDoc(r[0], r[1], r[2] if isinstance(r[2], list) else json.loads(r[2]), r[3], r[4], r[5]),
                 float(r[6])) for r in rows]

    def count(self) -> int:
        with self.pool.connection() as c:
            return c.execute("SELECT count(*) FROM fuente_conocimiento").fetchone()[0]


class Retriever:
    def __init__(self, embedder: LsaEmbedder, store, docs: list[KnowledgeDoc]):
        self.embedder, self.store = embedder, store
        self.store.upsert(docs, embedder.encode([d.text_for_embedding for d in docs]))
        self.lexical = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self.lex_docs = self.lexical.fit_transform([normalize(d.text_for_embedding) for d in docs])
        self.lex_index = {d.id: i for i, d in enumerate(docs)}

    def search(self, query: str, intent: str | None = None, k: int = 3) -> list[Retrieved]:
        q = self.embedder.encode([query])[0]
        cands = self.store.search(q, k=max(k * 4, 12))
        lex_q = self.lexical.transform([normalize(query)])
        ranked = []
        for d, sim in cands:
            i = self.lex_index.get(d.id)
            lex = float((self.lex_docs[i] @ lex_q.T).toarray()[0, 0]) if i is not None else 0.0
            score = W_SEMANTIC * sim + W_LEXICAL * lex + (INTENT_BOOST if intent and intent in d.intents else 0.0)
            ranked.append(Retrieved(d, score, sim))
        ranked.sort(key=lambda r: -r.score)
        return ranked[:k]


def build_vector_store(database_url: str, dim: int):
    return PgVectorStore(database_url, dim) if database_url else MemoryVectorStore()
