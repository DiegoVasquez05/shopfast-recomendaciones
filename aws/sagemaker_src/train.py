"""Script de entrenamiento para el Training Job de SageMaker (contenedor scikit-learn 1.2-1, Python 3.8).

SageMaker monta los datos en /opt/ml/input/data/<canal> y guarda /opt/ml/model en S3.
Las métricas se imprimen con formato `nombre=valor` para que SageMaker las capture
en CloudWatch (MetricDefinitions en aws/sagemaker_deploy.py).
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model_def import HYPERPARAMS, build_pipeline
from normalizer_cl import normalize


def main():
    ap = argparse.ArgumentParser()
    for k, v in HYPERPARAMS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", ""))
    a, _ = ap.parse_known_args()
    hp = {k: getattr(a, k) for k in HYPERPARAMS}
    if hp["class_weight"] in ("None", "none", ""):
        hp["class_weight"] = None

    train = pd.read_csv(os.path.join(a.train, "train.csv"))
    print("filas_entrenamiento=%d intents=%d" % (len(train), train["intent"].nunique()))
    model = build_pipeline(hp)
    model.fit([normalize(t) for t in train["text"]], train["intent"])

    metrics = {}
    if a.test and os.path.exists(os.path.join(a.test, "test.csv")):
        test = pd.read_csv(os.path.join(a.test, "test.csv"))
        pred = model.predict([normalize(t) for t in test["text"]])
        metrics = {"test_accuracy": accuracy_score(test["intent"], pred),
                   "test_precision": precision_score(test["intent"], pred, average="macro", zero_division=0),
                   "test_recall": recall_score(test["intent"], pred, average="macro", zero_division=0),
                   "test_f1": f1_score(test["intent"], pred, average="macro", zero_division=0)}
        for k, v in metrics.items():
            print("%s=%.4f" % (k, v))

    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(a.model_dir, "model.joblib"))
    with open(os.path.join(a.model_dir, "metrics.json"), "w") as f:
        json.dump({"hyperparameters": hp, "metrics": metrics}, f)


if __name__ == "__main__":
    main()
