"""Embeddings semánticos para el RAG.

Se usa LSA (TF-IDF de caracteres y palabras + SVD truncado a 128 dimensiones,
vectores normalizados L2) porque:
  - corre en CPU en < 5 ms y no requiere GPU ni descargar modelos externos,
  - capta similitud de forma robusta en español chileno con errores de tipeo,
  - las dimensiones son fijas, lo que permite indexarlas en pgvector (HNSW).

La interfaz `encode()` permite reemplazarlo por Amazon Titan Embeddings
(Bedrock) sin cambiar el retriever (ver docs/decisiones.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from app.nlp.normalizer_cl import normalize


class LsaEmbedder:
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.features = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])
        self.svd: TruncatedSVD | None = None

    def fit(self, corpus: list[str]) -> "LsaEmbedder":
        X = self.features.fit_transform([normalize(t) for t in corpus])
        dim = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed).fit(X)
        self.dim = dim
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.svd.transform(self.features.transform([normalize(t) for t in texts]))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.where(norms == 0, 1, norms)).astype(np.float32)
