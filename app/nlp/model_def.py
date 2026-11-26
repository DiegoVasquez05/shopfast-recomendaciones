"""Definición del clasificador de intenciones (compartida entre entrenamiento local y SageMaker).

Compatible con Python 3.8 / scikit-learn 1.2 (contenedor SageMaker sklearn 1.2-1).

Modelo: TF-IDF de palabras (1-2 gramas) + TF-IDF de caracteres (2-5 gramas,
robusto a faltas de ortografía y modismos) -> Regresión Logística multinomial.
Se eligió un modelo lineal por: latencia < 10 ms en CPU, probabilidades
calibrables para el umbral de confianza (handoff), interpretabilidad (pesos
por n-grama) y bajo costo de entrenamiento semanal.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

HYPERPARAMS = {
    "C": 8.0,
    "word_ngram_max": 2,
    "char_ngram_min": 2,
    "char_ngram_max": 5,
    "max_features_word": 60000,
    "max_features_char": 150000,
    "class_weight": "balanced",
    "max_iter": 3000,
}


def build_pipeline(hp=None):
    hp = dict(HYPERPARAMS, **(hp or {}))
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, int(hp["word_ngram_max"])), sublinear_tf=True,
                                 max_features=int(hp["max_features_word"]))),
        ("char", TfidfVectorizer(analyzer="char_wb",
                                 ngram_range=(int(hp["char_ngram_min"]), int(hp["char_ngram_max"])),
                                 sublinear_tf=True, min_df=2, max_features=int(hp["max_features_char"]))),
    ])
    clf = LogisticRegression(C=float(hp["C"]), max_iter=int(hp["max_iter"]),
                             class_weight=hp["class_weight"] or None)
    return Pipeline([("features", features), ("clf", clf)])


def top_k(pipeline, texts, k=3):
    """Devuelve, por texto, la lista [(intent, probabilidad), ...] ordenada."""
    proba = pipeline.predict_proba(texts)
    classes = pipeline.classes_
    out = []
    for row in proba:
        idx = row.argsort()[::-1][:k]
        out.append([(str(classes[i]), float(row[i])) for i in idx])
    return out
