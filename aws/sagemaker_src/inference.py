"""Handler de inferencia del endpoint SageMaker (contenedor scikit-learn, Python 3.8).

Request  (application/json): {"text": "wena, cuánto debo?"}   o   {"instances": ["...", "..."]}
Response (application/json): {"predictions": [{"intent": "...", "confidence": 0.93,
                               "normalized": "...", "top": [{"intent": "...", "probability": 0.93}, ...]}]}
"""
import json
import os

import joblib

from model_def import top_k
from normalizer_cl import normalize


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError("Content-Type no soportado: %s" % content_type)
    data = json.loads(body)
    texts = data.get("instances") or [data.get("text", "")]
    return [str(t) for t in texts]


def predict_fn(texts, model):
    norm = [normalize(t) for t in texts]
    out = []
    for n, top in zip(norm, top_k(model, norm, k=3)):
        out.append({"intent": top[0][0], "confidence": round(top[0][1], 4), "normalized": n,
                    "top": [{"intent": i, "probability": round(p, 4)} for i, p in top]})
    return out


def output_fn(predictions, accept):
    return json.dumps({"predictions": predictions}), "application/json"
