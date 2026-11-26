"""Pruebas funcionales y de integración: API, Siebel (EDA + Circuit Breaker), handoff, canales y seguridad."""
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from app.integrations.queue import MemoryQueue
from app.integrations.siebel_publisher import SiebelPublisher
from app.conversation.session_service import pilot_cohort
from app.persistence.kv import MemoryKV
from app.persistence.store import MemoryStore
from app.siebel_mock import main as siebel
from tests.conftest import chat, token


def test_tc06_oauth2_y_scopes(client):
    assert client.post("/api/v1/chat/messages", json={"user_id": "x", "text": "hola"}).status_code == 401
    bad = client.post("/oauth/token", data={"grant_type": "client_credentials", "client_id": "dashboard",
                                            "client_secret": "malo"})
    assert bad.status_code == 401
    agent = token(client, "agent-console", "dev-agent-secret")
    # la consola de agentes no puede chatear como cliente (mínimo privilegio)
    assert client.post("/api/v1/chat/messages", headers=agent, json={"user_id": "x", "text": "hola"}).status_code == 403
    assert client.get("/api/v1/model", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401


def test_tc07_nlu_endpoint_con_explicacion(client, auth):
    r = client.post("/api/v1/nlu/predict", headers=auth, json={"text": "no tengo señal en el celu po"}).json()
    assert r["intent"] == "technical_support" and r["sobre_umbral"]
    assert r["texto_normalizado"] == "no tengo senal en el celular"
    assert len(r["top"]) == 3 and r["explicacion"]


def test_tc08_rag_recupera_documento_correcto(client, auth):
    r = client.post("/api/v1/rag/search", headers=auth,
                    json={"query": "el router tiene una luz roja y no hay internet", "intent": "technical_support"})
    assert r.json()["results"][0]["id"] == "KB-013"


def test_tc09_flujo_crm_consulta_siebel(client, auth):
    r = chat(client, auth, "cuánto debo? quiero saber mi saldo")
    assert r["intent"] == "check_balance" and r["resolution"] == "crm"
    assert r["siebel"]["estado"] == "done"
    assert "deuda" in r["reply"] or "al día" in r["reply"]
    sin_id = chat(client, auth, "cuánto debo?", user="demo:anonimo", customer_id=None)
    assert sin_id["resolution"] == "crm_sin_identificar"


def test_tc10_circuit_breaker_y_entrega_asincrona(client, auth):
    siebel.CHAOS["failure_rate"] = 1.0       # Siebel cae
    first = chat(client, auth, "cuánto debo de saldo")
    sid = first["session_id"]
    assert first["siebel"]["estado"] == "failed"
    for _ in range(3):
        chat(client, auth, "cuánto debo de saldo", session_id=sid)
    # tras 3 fallas el circuito está abierto: no se llama a Siebel y se responde en modo degradado
    calls_before = siebel.CALLS["total"]
    r = chat(client, auth, "cuánto debo de saldo", session_id=sid)
    assert r["siebel"]["estado"] == "circuit_open" and "intermitencia" in r["reply"]
    assert siebel.CALLS["total"] == calls_before
    ready = client.get("/ready").json()
    assert ready["dependencies"]["siebel_circuit"] == "open"


def test_tc11_respuesta_pendiente_llega_como_notificacion():
    """Siebel lento: el bot responde 'pendiente' y el resultado llega después (consistencia eventual)."""
    kv, store, q = MemoryKV(), MemoryStore(), MemoryQueue()
    pub = SiebelPublisher(q, kv, store, sync_wait_ms=50)
    res = pub.request("get_balance", {"customer_id": "C-1001"}, "s1", "corr1")
    assert res.status == "pending" and len(q.messages) == 1
    from app.workers.siebel_worker import SiebelWorker

    class FakeClient:
        breaker = type("B", (), {"state": "closed"})()
        def execute(self, op, params):
            return {"nombre": "Ana", "deuda": 0, "datos_disponibles_gb": 10}
    SiebelWorker(FakeClient(), q, kv, store).handle(q.messages[0])
    notes = kv.pop_all("session:s1:notifications")
    assert notes and notes[0]["status"] == "done"


def test_tc12_escrituras_fallidas_reintentan_y_van_a_dlq():
    kv, store, q = MemoryKV(), MemoryStore(), MemoryQueue()
    from app.workers.siebel_worker import SiebelWorker

    class Down:
        breaker = type("B", (), {"state": "closed"})()
        calls = 0
        def execute(self, op, params):
            Down.calls += 1
            raise RuntimeError("Siebel 503")
    w = SiebelWorker(Down(), q, kv, store, max_retries=3, sleep=lambda s: None)
    q.consume(w.handle)
    q.publish({"correlation_id": "c1", "operation": "create_ticket", "params": {}, "kind": "write",
               "session_id": "s", "attempt": 0})
    assert Down.calls == 3 and len(q.dlq) == 1       # 3 intentos con backoff y luego Dead-Letter Queue
    assert store.ops["c1"]["estado"] == "failed"


def test_tc13_handoff_por_reclamo_y_agente_por_websocket(app_factory):
    c = TestClient(app_factory())
    auth = token(c)
    agent = token(c, "agent-console", "dev-agent-secret")["Authorization"].split()[1]
    with c.websocket_connect(f"/ws/agents?token={agent}") as ws:
        assert ws.receive_json()["type"] == "pending"
        r = chat(c, auth, "esto es una estafa, quiero poner un reclamo al sernac")
        assert r["resolution"] == "handoff" and r["handoff_id"]
        ev = ws.receive_json()
        assert ev["type"] == "handoff_created" and ev["handoff"]["session_id"] == r["session_id"]
        ws.send_json({"type": "reply", "session_id": r["session_id"], "text": "Hola, soy Carla, te ayudo."})
        time.sleep(0.2)
    upd = c.get(f"/api/v1/chat/sessions/{r['session_id']}/updates", headers=auth).json()["updates"]
    assert upd and upd[0]["from"] == "agente"
    # el reclamo también generó un caso en Siebel de forma asíncrona
    assert any(t["motivo"] == "complaint" for t in siebel.TICKETS.values())


def test_tc14_baja_confianza_dos_veces_deriva(client, auth):
    r1 = chat(client, auth, "qwerty zxcv")
    assert r1["resolution"] == "clarify"
    r2 = chat(client, auth, "asdf ghjk", session_id=r1["session_id"])
    assert r2["resolution"] == "handoff"


def test_tc15_piloto_10_por_ciento(client, auth):
    cohorts = [pilot_cohort(f"user-{i}", 10, [], "sal") for i in range(5000)]
    share = cohorts.count("piloto") / len(cohorts)
    assert 0.08 <= share <= 0.12
    # usuario que con la sal de la app queda en el grupo de control (90%)
    s = client.app.state.settings
    outsider = next(f"u{i}" for i in range(500) if pilot_cohort(f"u{i}", 10, [], s.pii_salt) == "control")
    r = chat(client, auth, "cuánto debo", user=outsider)
    assert r["resolution"] == "pilot_excluded" and r["cohort"] == "control"


def test_tc16_webhook_whatsapp(client):
    s = client.app.state.settings
    v = client.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": s.whatsapp_verify_token,
                                                  "hub.challenge": "123"})
    assert v.text == "123"
    body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
        {"from": "56980000001", "id": "wamid.1", "type": "text", "text": {"body": "wena, cuánto debo?"}}]}}]}]}).encode()
    assert client.post("/webhooks/whatsapp", content=body, headers={"X-Hub-Signature-256": "sha256=00"}).status_code == 401
    sig = "sha256=" + hmac.new(s.whatsapp_app_secret.encode(), body, hashlib.sha256).hexdigest()
    assert client.post("/webhooks/whatsapp", content=body, headers={"X-Hub-Signature-256": sig}).status_code == 200
    out = client.app.state.whatsapp.outbox
    nombre = siebel.CUSTOMERS["C-1001"]["nombre"].split()[0]  # el teléfono identifica al cliente en Siebel
    assert out and out[-1]["to"] == "56980000001" and nombre in out[-1]["text"]["body"]


def test_tc17_rate_limit_429(app_factory):
    c = TestClient(app_factory(rate_limit_per_minute=5))
    # por IP: /oauth/token sin token queda limitado a 5 req/min
    codes = [c.post("/oauth/token", data={"grant_type": "client_credentials", "client_id": "dashboard",
                                          "client_secret": "dev-dashboard-secret"}).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429
    # por token: cada web widget tiene una cuota de 30 req/min
    c2 = TestClient(app_factory())
    w = {"Authorization": "Bearer " + c2.post("/api/v1/channels/web/token").json()["access_token"]}
    codes = [c2.post("/api/v1/nlu/predict", headers=w, json={"text": "hola"}).status_code for _ in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429


def test_tc18_privacidad_cifrado_exportacion_y_olvido(client, auth):
    r = chat(client, auth, "mi correo es ana@correo.cl y no tengo internet", user="demo-gdpr")
    store = client.app.state.store
    raw = [m["texto_cifrado"] for m in store.msgs if m["session_id"] == r["session_id"]]
    assert raw and all("internet" not in t for t in raw)          # cifrado en reposo
    exp = client.get("/api/v1/privacy/users/demo-gdpr", headers=auth).json()
    assert any("[EMAIL]" in m["texto"] for m in exp["mensajes"])     # PII enmascarada
    d = client.delete("/api/v1/privacy/users/demo-gdpr", headers=auth).json()["eliminado"]
    assert d["sesiones"] == 1 and d["mensajes"] >= 2
    assert client.get("/api/v1/privacy/users/demo-gdpr", headers=auth).json()["mensajes"] == []


def test_tc19_latencia_turno_bajo_sla_3s(client, auth):
    lat = []
    for text in ["hola", "cuánto debo", "no tengo internet", "cómo activo el roaming", "quiero cambiar mi plan"] * 10:
        t = time.perf_counter()
        chat(client, auth, text)
        lat.append(time.perf_counter() - t)
    lat.sort()
    assert lat[int(len(lat) * 0.95) - 1] < 3.0


def test_tc20_openapi_y_kpis(client, auth):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/chat/messages", "/api/v1/nlu/predict", "/webhooks/whatsapp", "/oauth/token",
              "/api/v1/privacy/users/{user_id}"):
        assert p in paths
    chat(client, auth, "no tengo internet desde ayer")
    k = client.get("/api/v1/metrics/summary", headers=auth).json()["kpis"]
    assert k["sesiones"] >= 1 and "tasa_automatizacion" in k and "tasa_handoff" in k


def test_tc21_circuit_breaker_compartido_entre_workers():
    """Dos workers con el mismo Redis comparten el circuito: las fallas de ambos se suman."""
    from app.integrations.circuit_breaker import CircuitBreaker, CircuitOpenError

    now = [1000.0]
    kv = MemoryKV()
    w1, w2 = (CircuitBreaker(4, recovery_s=30, kv=kv, clock=lambda: now[0]) for _ in range(2))

    def boom():
        raise RuntimeError("503")

    for i in range(4):  # 2 fallas en cada worker = 4 en total
        try:
            (w1 if i % 2 else w2).call(boom)
        except RuntimeError:
            pass
    assert w1.state == w2.state == "open"
    try:
        w2.call(lambda: "ok")
        raise AssertionError("debió rechazar la llamada")
    except CircuitOpenError:
        pass
    now[0] += 31  # pasó el tiempo de recuperación: una sola llamada de prueba cierra el circuito
    assert w1.call(lambda: "ok") == "ok" and w2.state == "closed"
