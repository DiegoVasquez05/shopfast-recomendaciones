"""Entrenamiento del clasificador de intenciones + índice RAG.

    python -m training.train --bitext data/bitext.csv --out models

Genera en --out:
  intent_model.joblib   pipeline TF-IDF + Regresión Logística
  embedder.joblib       embeddings LSA para el RAG
  metrics.json          accuracy, precision, recall, F1 (macro y por intención), cobertura por umbral, RAG
  sagemaker/train.csv, sagemaker/test.csv  (datos para el Training Job de SageMaker)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from app.nlp.intents import INTENTS
from app.nlp.model_def import HYPERPARAMS, build_pipeline, top_k
from app.nlp.normalizer_cl import normalize
from app.rag.embeddings import LsaEmbedder
from app.rag.knowledge import load_kb
from app.rag.retriever import MemoryVectorStore, Retriever
from training import spanish_cl
from training.bitext import dataset_summary, load_bitext


def cls_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"n": int(len(y_true)), "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_macro": round(p, 4), "recall_macro": round(r, 4), "f1_macro": round(f, 4),
            "precision_weighted": round(pw, 4), "recall_weighted": round(rw, 4), "f1_weighted": round(fw, 4)}


def coverage_curve(y_true, preds) -> list[dict]:
    """Para cada umbral: % de mensajes que el bot resolvería solo y su accuracy (base del umbral de handoff)."""
    conf = np.array([p[0][1] for p in preds])
    hit = np.array([p[0][0] == t for p, t in zip(preds, y_true)])
    out = []
    for th in (0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8):
        m = conf >= th
        out.append({"umbral": th, "cobertura": round(float(m.mean()), 4),
                    "accuracy_sobre_cubiertos": round(float(hit[m].mean()), 4) if m.any() else None})
    return out


def top_confusions(y_true, y_pred, labels, n=8) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:n]
    return [{"real": labels[i], "predicho": labels[j], "casos": int(cm[i, j])} for i, j in idx if cm[i, j] > 0]


def train(bitext_path: str, out_dir: str, kb_path: str = "data/kb/telcoaustral_kb.json",
          eval_path: str = "data/eval_es_cl.csv", es_per_intent: int = 150, seed: int = 42) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    (out / "sagemaker").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Cargando Bitext desde {bitext_path}")
    bt = load_bitext(bitext_path)
    unknown = sorted(set(bt["intent"]) - set(INTENTS))
    if unknown:
        print(f"   aviso: intents no catalogados (se entrenan igual): {unknown}")
    bt_summary = dataset_summary(bt)
    print(json.dumps(bt_summary, ensure_ascii=False))
    es = spanish_cl.generate(n_per_intent=es_per_intent, seed=seed)

    bt_tr, bt_te = train_test_split(bt, test_size=0.15, stratify=bt["intent"], random_state=seed)
    es_tr, es_te = train_test_split(es, test_size=0.15, stratify=es["intent"], random_state=seed)
    train_df = pd.concat([bt_tr[["instruction", "intent"]].assign(lang="en"),
                          es_tr[["instruction", "intent"]].assign(lang="es-CL")], ignore_index=True)
    print(f"   train={len(train_df)} (bitext={len(bt_tr)}, es-CL={len(es_tr)})  test bitext={len(bt_te)} es-CL={len(es_te)}")

    print("[2/5] Entrenando clasificador (TF-IDF palabras+caracteres -> Regresión Logística)")
    t1 = time.time()
    model = build_pipeline()
    model.fit([normalize(t) for t in train_df["instruction"]], train_df["intent"])
    train_s = time.time() - t1

    print("[3/5] Evaluando")
    def evaluate(texts, y):
        preds = top_k(model, [normalize(t) for t in texts], k=3)
        y_pred = [p[0][0] for p in preds]
        m = cls_metrics(list(y), y_pred)
        m["top3_accuracy"] = round(float(np.mean([t in [i for i, _ in p] for p, t in zip(preds, y)])), 4)
        return m, preds, y_pred

    m_bt, preds_bt, ypred_bt = evaluate(bt_te["instruction"], bt_te["intent"])
    m_es, _, _ = evaluate(es_te["instruction"], es_te["intent"])
    ev = pd.read_csv(eval_path)
    m_ev, preds_ev, ypred_ev = evaluate(ev["text"], ev["intent"])
    labels = sorted(set(bt_te["intent"]))
    per_intent = classification_report(bt_te["intent"], ypred_bt, output_dict=True, zero_division=0)
    per_intent = {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                      "f1": round(v["f1-score"], 3), "soporte": int(v["support"])}
                  for k, v in per_intent.items() if k in labels}
    for name, m in (("Bitext test", m_bt), ("es-CL plantillas test", m_es), ("es-CL manual (eval_es_cl.csv)", m_ev)):
        print(f"   {name:32s} acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  top3={m['top3_accuracy']:.4f}")

    # Latencia de inferencia local (1 mensaje)
    lat = []
    for t in ev["text"].tolist()[:50]:
        s = time.perf_counter()
        model.predict_proba([normalize(t)])
        lat.append((time.perf_counter() - s) * 1000)

    print("[4/5] Construyendo índice RAG (embeddings LSA)")
    kb = load_kb(kb_path)
    embedder = LsaEmbedder(dim=128).fit([d.text_for_embedding for d in kb] + es["instruction"].tolist())
    retriever = Retriever(embedder, MemoryVectorStore(), kb)
    ev_kb = ev[ev["intent"].isin({i for d in kb for i in d.intents})]

    def rag_eval(use_intent: bool) -> dict:
        h1 = h3 = rr = 0.0
        for text, intent in zip(ev_kb["text"], ev_kb["intent"]):
            hits = retriever.search(text, intent=intent if use_intent else None, k=10)
            ranks = [r for r, h in enumerate(hits) if intent in h.doc.intents]
            h1 += 1.0 if ranks and ranks[0] == 0 else 0.0
            h3 += 1.0 if ranks and ranks[0] < 3 else 0.0
            rr += 1.0 / (ranks[0] + 1) if ranks else 0.0
        n = max(len(ev_kb), 1)
        return {"hit_rate_at_1": round(h1 / n, 4), "recall_at_3": round(h3 / n, 4), "mrr_at_10": round(rr / n, 4)}

    rag_metrics = {"documentos": len(kb), "dim_embedding": embedder.dim, "consultas_eval": int(len(ev_kb)),
                   "solo_busqueda_hibrida": rag_eval(False), "con_intencion_detectada": rag_eval(True)}
    print(f"   RAG sin intención: {rag_metrics['solo_busqueda_hibrida']}  con intención: {rag_metrics['con_intencion_detectada']}")

    print("[5/5] Guardando artefactos")
    version = time.strftime("%Y%m%d-%H%M%S")
    joblib.dump({"pipeline": model, "version": version, "hyperparams": HYPERPARAMS}, out / "intent_model.joblib",
                compress=3)
    joblib.dump(embedder, out / "embedder.joblib", compress=3)
    metrics = {
        "model_version": version,
        "dataset_bitext": bt_summary,
        "corpus_es_cl": {"filas": int(len(es)), "intents": int(es["intent"].nunique())},
        "split": {"train": int(len(train_df)), "test_bitext": int(len(bt_te)), "test_es_cl": int(len(es_te)),
                  "eval_manual_es_cl": int(len(ev))},
        "hiperparametros": HYPERPARAMS,
        "clasificador": {"bitext_test": m_bt, "es_cl_plantillas_test": m_es, "es_cl_manual": m_ev},
        "por_intencion_bitext": per_intent,
        "confusiones_principales": top_confusions(list(bt_te["intent"]), ypred_bt, labels),
        "cobertura_por_umbral_es_cl": coverage_curve(list(ev["intent"]), preds_ev),
        "cobertura_por_umbral_bitext": coverage_curve(list(bt_te["intent"]), preds_bt),
        "errores_es_cl_manual": [{"texto": t, "real": y, "predicho": p} for t, y, p in
                                 zip(ev["text"], ev["intent"], ypred_ev) if y != p],
        "rag": rag_metrics,
        "latencia_inferencia_local_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                                         "p95": round(float(np.percentile(lat, 95)), 2)},
        "tiempo_entrenamiento_s": round(train_s, 1),
        "tiempo_total_s": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_df[["instruction", "intent"]].rename(columns={"instruction": "text"}).to_csv(
        out / "sagemaker" / "train.csv", index=False)
    pd.concat([bt_te[["instruction", "intent"]].rename(columns={"instruction": "text"}), ev]).to_csv(
        out / "sagemaker" / "test.csv", index=False)
    print(f"Listo en {metrics['tiempo_total_s']}s -> {out.resolve()}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena el clasificador de intenciones y el índice RAG")
    ap.add_argument("--bitext", default="data/bitext.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--kb", default="data/kb/telcoaustral_kb.json")
    ap.add_argument("--eval", default="data/eval_es_cl.csv")
    a = ap.parse_args()
    train(a.bitext, a.out, kb_path=a.kb, eval_path=a.eval)


if __name__ == "__main__":
    main()
