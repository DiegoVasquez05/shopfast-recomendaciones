"""Reconocimiento de intención (NLU).

- LocalIntentClassifier: modelo scikit-learn cargado en memoria.
- SageMakerIntentClassifier: invoca el endpoint de AWS SageMaker; si falla o
  excede el timeout, usa el modelo local (fallback para el uptime de 99,9%).

La explicación (interpretabilidad) muestra los n-gramas del mensaje que más
pesaron a favor de la intención elegida (coeficiente x TF-IDF).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from app.nlp.model_def import top_k
from app.nlp.normalizer_cl import normalize

log = logging.getLogger(__name__)


@dataclass
class NluResult:
    intent: str
    confidence: float
    top: list[tuple[str, float]]
    normalized: str
    source: str
    latency_ms: float
    explanation: list[dict] = field(default_factory=list)


class LocalIntentClassifier:
    source = "local"

    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.version = bundle.get("version", "desconocida")
        feats = self.pipeline.named_steps["features"]
        self._names = np.concatenate([
            np.array([f"{name}:{t}" for t in tr.get_feature_names_out()]) for name, tr in feats.transformer_list])

    def _predict(self, normalized: str) -> list[tuple[str, float]]:
        return top_k(self.pipeline, [normalized], k=3)[0]

    def predict(self, text: str) -> NluResult:
        t0 = time.perf_counter()
        norm = normalize(text)
        top = self._predict(norm)
        return NluResult(top[0][0], round(top[0][1], 4), [(i, round(p, 4)) for i, p in top], norm, self.source,
                         round((time.perf_counter() - t0) * 1000, 2))

    def explain(self, normalized: str, intent: str, n: int = 6) -> list[dict]:
        X = self.pipeline.named_steps["features"].transform([normalized])
        clf = self.pipeline.named_steps["clf"]
        ci = list(clf.classes_).index(intent) if intent in clf.classes_ else 0
        contrib = X.multiply(clf.coef_[ci]).tocsr()
        idx = contrib.indices[np.argsort(-contrib.data)][:n * 3]
        out, seen = [], set()
        for i in idx:
            kind, term = self._names[i].split(":", 1)
            term = term.strip()
            if not term or term in seen or contrib[0, i] <= 0:
                continue
            seen.add(term)
            out.append({"ngrama": term, "tipo": "palabra" if kind == "word" else "caracteres",
                        "peso": round(float(contrib[0, i]), 4)})
            if len(out) >= n:
                break
        return out


class SageMakerIntentClassifier(LocalIntentClassifier):
    source = "sagemaker"

    def __init__(self, model_path, endpoint: str, region: str, timeout_s: float = 1.0):
        super().__init__(model_path)
        import boto3
        from botocore.config import Config

        self.endpoint = endpoint
        self.rt = boto3.client("sagemaker-runtime", region_name=region,
                               config=Config(connect_timeout=timeout_s, read_timeout=timeout_s,
                                             retries={"max_attempts": 1}))

    def predict(self, text: str) -> NluResult:
        t0 = time.perf_counter()
        norm = normalize(text)
        try:
            r = self.rt.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
                                        Accept="application/json", Body=json.dumps({"text": text}))
            pred = json.loads(r["Body"].read())["predictions"][0]
            top = [(p["intent"], round(float(p["probability"]), 4)) for p in pred["top"]]
            src = "sagemaker"
        except Exception as exc:
            log.warning("SageMaker no disponible (%s); usando modelo local", exc)
            top, src = [(i, round(p, 4)) for i, p in self._predict(norm)], "local-fallback"
        return NluResult(top[0][0], top[0][1], top, norm, src, round((time.perf_counter() - t0) * 1000, 2))


def build_classifier(settings) -> LocalIntentClassifier:
    path = Path(settings.models_dir) / "intent_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta: python -m training.train --bitext <csv> --out models")
    if settings.sagemaker_endpoint:
        return SageMakerIntentClassifier(path, settings.sagemaker_endpoint, settings.aws_region)
    return LocalIntentClassifier(path)
