"""Fixtures: entrena un modelo real (dataset pequeño con esquema Bitext) una vez por sesión de pruebas."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.persistence.kv import MemoryKV
from app.persistence.store import MemoryStore
from app.siebel_mock import main as siebel
from training.bitext import generate_sample
from training.train import train

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    data = tmp_path_factory.mktemp("data") / "bitext.csv"
    generate_sample(n_per_intent=60, seed=1).to_csv(data, index=False)
    out = tmp_path_factory.mktemp("models")
    metrics = train(str(data), str(out), kb_path=str(ROOT / "data/kb/telcoaustral_kb.json"),
                    eval_path=str(ROOT / "data/eval_es_cl.csv"), es_per_intent=80)
    return out, metrics


def make_settings(models_dir, **kw) -> Settings:
    s = Settings()
    s.models_dir = models_dir
    s.kb_path = ROOT / "data/kb/telcoaustral_kb.json"
    s.redis_url = s.database_url = s.sagemaker_endpoint = ""
    s.queue_backend = "memory"
    s.llm_provider, s.sentiment_provider = "extractive", "lexicon"
    s.jwt_secret = "secreto-de-pruebas-con-mas-de-32-caracteres"
    s.rate_limit_per_minute = 100_000
    s.pilot_allowlist = ["demo", "wa"]
    s.cb_failure_threshold, s.cb_recovery_s = 3, 30
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture(autouse=True)
def siebel_normal():
    siebel.CHAOS.update(failure_rate=0.0, latency_ms=0)
    yield
    siebel.CHAOS.update(failure_rate=0.0, latency_ms=0)


@pytest.fixture()
def app_factory(trained):
    def _make(**kw):
        s = make_settings(trained[0], **kw)
        http = TestClient(siebel.app)
        http.auth = (s.siebel_user, s.siebel_password)
        return create_app(s, kv=MemoryKV(), store=MemoryStore(), siebel_http=http)
    return _make


@pytest.fixture()
def client(app_factory):
    return TestClient(app_factory())


def token(client, client_id="dashboard", secret="dev-dashboard-secret") -> dict:
    r = client.post("/oauth/token", data={"grant_type": "client_credentials", "client_id": client_id,
                                          "client_secret": secret})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def auth(client):
    return token(client)


def chat(client, auth, text, user="demo", session_id=None, customer_id="C-1001"):
    r = client.post("/api/v1/chat/messages", headers=auth,
                    json={"user_id": user, "text": text, "session_id": session_id, "customer_id": customer_id})
    assert r.status_code == 200, r.text
    return r.json()
