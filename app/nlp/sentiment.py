"""Análisis de sentimiento para detectar clientes molestos (gatillo de handoff).

- lexicon: léxico de español chileno (rápido, sin costo, sin red).
- comprehend: Amazon Comprehend (servicio AIaaS) con fallback al léxico.
"""
from __future__ import annotations

import logging

from app.nlp.normalizer_cl import normalize

log = logging.getLogger(__name__)

NEGATIVE = {"pesimo": 2, "pesima": 2, "malo": 1, "mala": 1, "malisimo": 2, "estafa": 3, "estafador": 3,
            "sernac": 3, "subtel": 2, "reclamo": 2, "reclamar": 2, "molesto": 2, "molesta": 2, "enojado": 2,
            "indignado": 3, "aburrido": 2, "chato": 2, "harto": 1, "insoportable": 2, "verguenza": 2,
            "colmo": 2, "insulto": 3, "nunca": 1, "nadie": 1, "abuso": 3, "ladrones": 3, "tusa": 2,
            "lamentablemente": 1, "demanda": 3, "denuncia": 3, "terrible": 2, "horrible": 2}
POSITIVE = {"gracias": 1, "excelente": 2, "buena": 1, "bueno": 1, "genial": 2, "perfecto": 2, "feliz": 2}


def lexicon_score(text: str) -> dict:
    words = normalize(text).split()
    neg = sum(NEGATIVE.get(w, 0) for w in words)
    pos = sum(POSITIVE.get(w, 0) for w in words)
    raw = text.strip()
    if raw.count("!") >= 2 or (len(raw) > 8 and raw.isupper()):
        neg += 1
    score = (pos - neg) / max(1.0, pos + neg)
    label = "NEGATIVE" if neg >= 3 or score <= -0.6 else ("POSITIVE" if score > 0.3 else "NEUTRAL")
    return {"label": label, "score": round(score, 3), "negativos": neg, "provider": "lexicon"}


class SentimentAnalyzer:
    def __init__(self, provider: str = "lexicon", region: str = "us-east-1"):
        self.provider = provider
        self.client = None
        if provider == "comprehend":
            import boto3
            from botocore.config import Config

            self.client = boto3.client("comprehend", region_name=region,
                                       config=Config(connect_timeout=0.5, read_timeout=0.8, retries={"max_attempts": 1}))

    def analyze(self, text: str) -> dict:
        lex = lexicon_score(text)
        if self.client is None:
            return lex
        try:
            r = self.client.detect_sentiment(Text=text[:4500], LanguageCode="es")
            label = r["Sentiment"]
            # el léxico chileno complementa a Comprehend con modismos que no conoce
            if lex["label"] == "NEGATIVE":
                label = "NEGATIVE"
            return {"label": label, "score": round(r["SentimentScore"].get("Negative", 0) * -1, 3),
                    "negativos": lex["negativos"], "provider": "comprehend"}
        except Exception as exc:
            log.warning("Comprehend no disponible (%s); usando léxico", exc)
            return lex
