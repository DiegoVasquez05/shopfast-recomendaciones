"""Dashboard de TelecomBot AI (Streamlit): resultados del modelo y KPIs para la toma de decisiones.

    streamlit run dashboard/app.py

Consume la API REST con OAuth2 (client credentials, cliente "dashboard").
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLIENT_ID = os.getenv("DASHBOARD_CLIENT_ID", "dashboard")
CLIENT_SECRET = os.getenv("DASHBOARD_CLIENT_SECRET", "dev-dashboard-secret")
SIEBEL_ADMIN = os.getenv("SIEBEL_ADMIN_URL", "")
VERIFY = os.getenv("VERIFY_TLS", "true").lower() == "true"

st.set_page_config(page_title="TelecomBot AI · Dashboard", page_icon="📶", layout="wide")


@st.cache_data(ttl=3000)
def get_token() -> str:
    r = requests.post(f"{API}/oauth/token", data={"grant_type": "client_credentials", "client_id": CLIENT_ID,
                                                  "client_secret": CLIENT_SECRET}, timeout=5, verify=VERIFY)
    r.raise_for_status()
    return r.json()["access_token"]


def api(method: str, path: str, **kw):
    t = time.perf_counter()
    r = requests.request(method, f"{API}{path}", headers={"Authorization": f"Bearer {get_token()}"}, timeout=10,
                         verify=VERIFY, **kw)
    ms = (time.perf_counter() - t) * 1000
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.text[:300]}")
        return None, ms
    return r.json(), ms


def pct(x) -> str:
    return f"{(x or 0) * 100:.1f}%"


st.title("📶 TelecomBot AI · TelcoAustral")
try:
    h = requests.get(f"{API}/health", timeout=5, verify=VERIFY).json()
    rd = requests.get(f"{API}/ready", timeout=5, verify=VERIFY).json()
    deps = rd["dependencies"]
    st.caption(f"API {API} · modelo {h['model_version']} · NLU: **{deps['nlu']}** · base vectorial: "
               f"**{deps['vector_store']}** · Circuit Breaker Siebel: **{deps['siebel_circuit']}**")
except Exception as exc:
    st.error(f"API no disponible: {exc}")
    st.stop()

tab_kpi, tab_chat, tab_model, tab_ops = st.tabs(["📊 KPIs de negocio", "💬 Probar el chatbot",
                                                  "🧠 Calidad del modelo", "⚙️ Operación y resiliencia"])

with tab_kpi:
    data, _ = api("GET", "/api/v1/metrics/summary")
    if data:
        k = data["kpis"]
        c = st.columns(5)
        c[0].metric("Tasa de automatización", pct(k["tasa_automatizacion"]), help="Objetivo del caso: 60%")
        c[1].metric("Tasa de handoff", pct(k["tasa_handoff"]))
        c[2].metric("Latencia P95 por turno", f"{k['latencia_ms']['p95']:.0f} ms", help="SLA: ≤ 3.000 ms")
        c[3].metric("Sesiones (7 días)", k["sesiones"], help=f"Piloto: {k['sesiones_piloto']}")
        c[4].metric("Confianza promedio NLU", f"{k['confianza_promedio']:.2f}")
        a, b = st.columns(2)
        with a:
            st.subheader("Intenciones más consultadas")
            if k["por_intencion"]:
                st.bar_chart(pd.Series(k["por_intencion"]).sort_values(), horizontal=True)
            st.subheader("Cómo se resolvió cada turno")
            if k["por_resolucion"]:
                st.bar_chart(pd.Series(k["por_resolucion"]).sort_values(), horizontal=True)
        with b:
            st.subheader("Motivos de derivación a ejecutivo")
            if k["handoffs"]:
                st.bar_chart(pd.Series(k["handoffs"]).sort_values(), horizontal=True)
            st.subheader("Sesiones por canal y cohorte del piloto")
            st.write(pd.DataFrame({"canal": pd.Series(k["por_canal"]), }).T if k["por_canal"] else "Sin datos")
            st.write(pd.DataFrame({"cohorte": pd.Series(k["por_cohorte"])}).T if k["por_cohorte"] else "")
        st.info("**Lectura para la gerencia:** cada punto de automatización equivale a consultas repetitivas que "
                "no llegan al call center. Si la tasa de handoff por *baja_confianza* sube, hay que ampliar el "
                "corpus de esa intención; si sube por *cliente_molesto*, revisar la causa operacional (fallas).")

with tab_chat:
    if "sid" not in st.session_state:
        st.session_state.sid, st.session_state.log = None, []
    c = st.columns([2, 1, 1])
    user = c[0].text_input("Usuario", "demo:dashboard")
    customer = c[1].selectbox("Cliente Siebel (simula login)", ["C-1001", "C-1002", "C-1003", "C-1004", "(anónimo)"])
    if c[2].button("Nueva conversación"):
        st.session_state.sid, st.session_state.log = None, []
    for m in st.session_state.log:
        st.chat_message(m["role"]).write(m["text"])
    if prompt := st.chat_input("Escribe como cliente, p. ej.: wena, cuánto debo de la boleta?"):
        st.session_state.log.append({"role": "user", "text": prompt})
        d, ms = api("POST", "/api/v1/chat/messages", json={
            "user_id": user, "text": prompt, "session_id": st.session_state.sid, "channel": "web",
            "customer_id": None if customer == "(anónimo)" else customer})
        if d:
            st.session_state.sid = d["session_id"]
            st.session_state.log.append({"role": "assistant", "text": d["reply"]})
            st.session_state.last = d
        st.rerun()
    if st.session_state.sid and st.button("Buscar respuestas pendientes (Siebel / agente)"):
        d, _ = api("GET", f"/api/v1/chat/sessions/{st.session_state.sid}/updates")
        for u in (d or {}).get("updates", []):
            st.session_state.log.append({"role": "assistant", "text": f"[{u['from']}] {u['text']}"})
        st.rerun()
    last = st.session_state.get("last")
    if last:
        st.divider()
        st.subheader("¿Por qué respondió eso? (interpretabilidad)")
        c = st.columns(4)
        c[0].metric("Intención", last["intent"])
        c[1].metric("Confianza", f"{last['confidence']:.2f}")
        c[2].metric("Resolución", last["resolution"])
        c[3].metric("Latencia", f"{last['latency_ms']:.0f} ms", help=f"NLU: {last['nlu_source']}")
        a, b = st.columns(2)
        with a:
            st.write("**Top-3 intenciones**")
            st.bar_chart(pd.DataFrame(last["top_intents"]).set_index("intent"), horizontal=True)
            if last["explanation"]:
                st.write("**Palabras/fragmentos que más pesaron**")
                st.dataframe(pd.DataFrame(last["explanation"]), hide_index=True, width="stretch")
        with b:
            st.write("**Documentos recuperados (RAG)**")
            st.dataframe(pd.DataFrame(last["sources"]) if last["sources"] else pd.DataFrame(),
                         hide_index=True, width="stretch")
            st.write("**Sentimiento:**", last["sentiment"])
            if last["siebel"]:
                st.write("**Siebel:**", last["siebel"])

with tab_model:
    info, _ = api("GET", "/api/v1/model")
    if info and info.get("metrics"):
        m = info["metrics"]
        clf = m["clasificador"]
        st.subheader("Clasificador de intenciones")
        st.dataframe(pd.DataFrame(clf).T[["n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                                         "top3_accuracy"]], width="stretch")
        st.caption(f"Entrenado con {m['dataset_bitext']['filas']} ejemplos de Bitext + "
                   f"{m['corpus_es_cl']['filas']} de adaptación al español chileno · versión {m['model_version']}")
        a, b = st.columns(2)
        with a:
            st.write("**Cobertura vs. precisión por umbral de confianza (español chileno)**")
            cov = pd.DataFrame(m["cobertura_por_umbral_es_cl"]).set_index("umbral")
            st.line_chart(cov)
            st.caption(f"Umbral en producción: {info['intent_threshold']}. Por debajo, el bot pide aclaración y "
                       "luego deriva a un ejecutivo.")
        with b:
            st.write("**F1 por intención (Bitext test)**")
            st.bar_chart(pd.DataFrame(m["por_intencion_bitext"]).T["f1"].sort_values(), horizontal=True)
        st.write("**Errores en el set manual de español chileno**")
        st.dataframe(pd.DataFrame(m["errores_es_cl_manual"]), hide_index=True, width="stretch")
        st.subheader("RAG")
        st.json(m["rag"])

with tab_ops:
    data, _ = api("GET", "/api/v1/metrics/summary")
    if data:
        c = st.columns(4)
        cb = data["siebel_circuit"]
        c[0].metric("Circuit Breaker Siebel", cb.get("state", "?").upper(), help=f"fallas: {cb.get('failures')}")
        c[1].metric("Comandos en cola", data["cola_siebel_pendientes"] if data["cola_siebel_pendientes"] is not None
                    else "n/d")
        c[2].metric("Handoffs abiertos", data["kpis"]["handoffs_abiertos"])
        c[3].metric("Piloto", f"{data['piloto_porcentaje']}% de usuarios")
        st.write("**Operaciones Siebel (7 días)**", data["kpis"]["operaciones_siebel"])
    if SIEBEL_ADMIN:
        st.subheader("Simular falla del CRM Siebel (demo del Circuit Breaker)")
        c = st.columns(3)
        if c[0].button("🔴 Siebel caído"):
            requests.post(f"{SIEBEL_ADMIN}/admin/chaos", json={"failure_rate": 1.0}, timeout=5)
        if c[1].button("🟡 Siebel lento (2,5 s)"):
            requests.post(f"{SIEBEL_ADMIN}/admin/chaos", json={"failure_rate": 0, "latency_ms": 2500}, timeout=5)
        if c[2].button("🟢 Siebel normal"):
            requests.post(f"{SIEBEL_ADMIN}/admin/chaos", json={"failure_rate": 0, "latency_ms": 80}, timeout=5)
        try:
            st.caption(f"Estado del simulador: {requests.get(f'{SIEBEL_ADMIN}/health', timeout=3).json()}")
        except Exception:
            pass
    hs, _ = api("GET", "/api/v1/handoffs")
    if hs and hs["handoffs"]:
        st.subheader("Casos derivados")
        st.dataframe(pd.DataFrame(hs["handoffs"]), hide_index=True, width="stretch")
