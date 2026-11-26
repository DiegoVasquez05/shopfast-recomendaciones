"""Pruebas unitarias: NLP en español chileno, privacidad y calidad del modelo."""
import logging

from app.nlp.normalizer_cl import normalize
from app.nlp.sentiment import lexicon_score
from app.observability.telemetry import PiiFilter
from app.privacy import MessageCipher, mask_pii


def test_tc01_normalizador_modismos_chilenos():
    assert normalize("Wena po, kiero cachar cuánto debo altiro!!") == "hola quiero saber cuanto debo ahora"
    assert normalize("se me fue la señal del celu") == "se me fue la senal del celular"
    assert normalize("holaaaa") == "hola"
    assert "insulto" in normalize("weon no funciona")


def test_tc02_enmascarado_pii_en_texto_y_logs():
    txt = "mi rut es 12.345.678-9, correo juan@mail.cl y fono +56 9 8765 4321"
    out = mask_pii(txt)
    assert "12.345.678-9" not in out and "juan@mail.cl" not in out and "8765" not in out
    assert "[RUT]" in out and "[EMAIL]" in out and "[TELEFONO]" in out
    rec = logging.LogRecord("t", logging.INFO, "", 0, "cliente %s escribió", ("juan@mail.cl",), None)
    PiiFilter().filter(rec)
    assert "juan@mail.cl" not in rec.getMessage()


def test_tc03_cifrado_de_mensajes_en_reposo():
    c = MessageCipher(salt="s")
    token = c.encrypt("no tengo internet")
    assert "internet" not in token and c.decrypt(token) == "no tengo internet"


def test_tc04_sentimiento_detecta_cliente_molesto():
    assert lexicon_score("esto es una estafa voy a ir al sernac, pésimo servicio")["label"] == "NEGATIVE"
    assert lexicon_score("muchas gracias, excelente atención")["label"] == "POSITIVE"


def test_tc05_metricas_del_clasificador(trained):
    _, m = trained
    for split in ("bitext_test", "es_cl_manual"):
        r = m["clasificador"][split]
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0 <= r[k] <= 1
    # el set manual en español chileno NO se generó con plantillas: mide generalización real
    assert m["clasificador"]["es_cl_manual"]["accuracy"] >= 0.80
    assert m["rag"]["con_intencion_detectada"]["recall_at_3"] >= 0.85
    assert m["latencia_inferencia_local_ms"]["p95"] < 100
