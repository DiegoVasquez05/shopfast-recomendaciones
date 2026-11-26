"""Simulador del CRM legacy Siebel (REST + SOAP) para desarrollo, pruebas y demo.

    uvicorn app.siebel_mock.main:app --port 9000

Modo caos (para demostrar el Circuit Breaker en la defensa):
    POST /admin/chaos {"failure_rate": 1.0}          -> Siebel "caído"
    POST /admin/chaos {"latency_ms": 2500}           -> Siebel lento (supera el timeout)
    POST /admin/chaos {"failure_rate": 0, "latency_ms": 0}  -> normal
"""
from __future__ import annotations

import os
import random
import re
import secrets
import time
import uuid
from datetime import date, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

app = FastAPI(title="Siebel CRM (simulador)", version="8.1-mock")
basic = HTTPBasic()
USER = os.getenv("SIEBEL_USER", "telecombot")
PASSWORD = os.getenv("SIEBEL_PASSWORD", "dev-password")

CHAOS = {"failure_rate": float(os.getenv("SIEBEL_FAILURE_RATE", "0")),
         "latency_ms": int(os.getenv("SIEBEL_LATENCY_MS", "40"))}
CALLS = {"total": 0, "fallidas": 0}

NOMBRES = ["Camila Rojas", "Matías González", "Valentina Muñoz", "Benjamín Díaz", "Catalina Soto",
           "Joaquín Contreras", "Fernanda Silva", "Tomás Martínez", "Javiera Sepúlveda", "Vicente Morales",
           "Antonia Fuentes", "Agustín Torres", "Isidora Araya", "Diego Castillo", "Florencia Espinoza"]
PLANES_MOVIL = ["Plan Libre 30 GB", "Plan Libre 80 GB", "Plan Ilimitado"]
PLANES_HOGAR = ["Fibra 600 Mbps", "Fibra 940 Mbps", "Dúo Fibra 940 + TV"]


def _customer(i: int) -> dict:
    rnd = random.Random(i)
    cid = f"C-{1000 + i}"
    servicios = [{"tipo": "Móvil", "plan": rnd.choice(PLANES_MOVIL), "linea": f"+56 9 {rnd.randint(1000, 9999)} "
                                                                             f"{rnd.randint(1000, 9999)}"}]
    if rnd.random() < 0.6:
        servicios.append({"tipo": "Hogar", "plan": rnd.choice(PLANES_HOGAR), "linea": None})
    hoy = date(2026, 11, 26)
    boletas = []
    for m in range(3):
        periodo = (hoy.replace(day=1) - timedelta(days=30 * m))
        monto = rnd.choice([9990, 14990, 19990]) + (rnd.choice([19990, 24990, 32990]) if len(servicios) > 1 else 0)
        boletas.append({"numero": f"B{rnd.randint(10**7, 10**8 - 1)}", "periodo": periodo.strftime("%m-%Y"),
                        "monto": monto + rnd.choice([0, 0, 3990, 7990]),
                        "vencimiento": (periodo.replace(day=20)).isoformat(),
                        "estado": "pendiente de pago" if m == 0 and rnd.random() < 0.6 else "pagada",
                        "url_pdf": f"https://sucursal.telcoaustral.cl/boletas/{cid}/{periodo:%Y%m}.pdf"})
    deuda = sum(b["monto"] for b in boletas if b["estado"] != "pagada")
    return {"customer_id": cid, "nombre": NOMBRES[i % len(NOMBRES)], "telefono": f"569{80000000 + i}",
            "segmento": rnd.choice(["Residencial", "Residencial", "PYME"]), "servicios": servicios,
            "boletas": boletas, "deuda": deuda, "vencimiento": boletas[0]["vencimiento"],
            "datos_disponibles_gb": round(rnd.uniform(0.5, 60), 1)}


CUSTOMERS = {c["customer_id"]: c for c in (_customer(i) for i in range(1, 16))}
BY_PHONE = {c["telefono"]: c["customer_id"] for c in CUSTOMERS.values()}
TICKETS: dict[str, dict] = {}


def auth(creds: HTTPBasicCredentials = Depends(basic)) -> None:
    if not (secrets.compare_digest(creds.username, USER) and secrets.compare_digest(creds.password, PASSWORD)):
        raise HTTPException(401, "Credenciales Siebel inválidas")


def chaos() -> None:
    CALLS["total"] += 1
    if CHAOS["latency_ms"]:
        time.sleep(CHAOS["latency_ms"] / 1000)
    if random.random() < CHAOS["failure_rate"]:
        CALLS["fallidas"] += 1
        raise HTTPException(503, "SBL-DAT-00500: Siebel Object Manager no disponible")


def find(key: str, by_phone: bool = False) -> dict:
    cid = BY_PHONE.get(re.sub(r"\D", "", key)) if by_phone else key
    if not cid or cid not in CUSTOMERS:
        raise HTTPException(404, "Cliente no existe")
    return CUSTOMERS[cid]


def view(c: dict, what: str) -> dict:
    base = {"customer_id": c["customer_id"], "nombre": c["nombre"]}
    if what == "balance":
        return {**base, "deuda": c["deuda"], "vencimiento": c["vencimiento"],
                "datos_disponibles_gb": c["datos_disponibles_gb"]}
    if what == "invoices":
        return {**base, "boletas": c["boletas"]}
    return {**base, "segmento": c["segmento"], "servicios": c["servicios"]}


@app.get("/health")
def health():
    return {"status": "ok", "chaos": CHAOS, "calls": CALLS}


@app.get("/siebel/customers/{cid}/{what}", dependencies=[Depends(auth), Depends(chaos)])
def by_id(cid: str, what: str):
    if what not in ("balance", "invoices", "profile"):
        raise HTTPException(404)
    return view(find(cid), what)


@app.get("/siebel/customers/by-phone/{phone}/{what}", dependencies=[Depends(auth), Depends(chaos)])
def by_phone(phone: str, what: str):
    if what not in ("balance", "invoices", "profile"):
        raise HTTPException(404)
    return view(find(phone, by_phone=True), what)


class TicketIn(BaseModel):
    motivo: str
    detalle: str = ""
    idempotency_key: str = ""


def _ticket(c: dict, body: TicketIn) -> dict:
    for t in TICKETS.values():  # idempotencia: el mismo comando no crea dos casos
        if body.idempotency_key and t["idempotency_key"] == body.idempotency_key:
            return t
    t = {"ticket_id": f"SR-{uuid.uuid4().hex[:6].upper()}", "customer_id": c["customer_id"], "motivo": body.motivo,
         "detalle": body.detalle, "estado": "Abierto", "idempotency_key": body.idempotency_key}
    TICKETS[t["ticket_id"]] = t
    return t


@app.post("/siebel/customers/{cid}/tickets", dependencies=[Depends(auth), Depends(chaos)])
def ticket_by_id(cid: str, body: TicketIn):
    return _ticket(find(cid), body)


@app.post("/siebel/customers/by-phone/{phone}/tickets", dependencies=[Depends(auth), Depends(chaos)])
def ticket_by_phone(phone: str, body: TicketIn):
    return _ticket(find(phone, by_phone=True), body)


@app.post("/siebel/soap", dependencies=[Depends(auth), Depends(chaos)])
async def soap(request: Request):
    """Endpoint SOAP legado: <GetCustomerBalance><CustomerId>C-1001</CustomerId></GetCustomerBalance>"""
    body = (await request.body()).decode()
    m = re.search(r"<CustomerId>([^<]+)</CustomerId>", body)
    if not m:
        raise HTTPException(400, "SOAP Fault: CustomerId requerido")
    b = view(find(m.group(1)), "balance")
    xml = (f'<?xml version="1.0"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
           f"<GetCustomerBalanceResponse><CustomerId>{b['customer_id']}</CustomerId><Deuda>{b['deuda']}</Deuda>"
           f"<Vencimiento>{b['vencimiento']}</Vencimiento></GetCustomerBalanceResponse></soap:Body></soap:Envelope>")
    return Response(xml, media_type="text/xml")


class ChaosIn(BaseModel):
    failure_rate: float | None = None
    latency_ms: int | None = None


@app.post("/admin/chaos")
def set_chaos(body: ChaosIn):
    if body.failure_rate is not None:
        CHAOS["failure_rate"] = max(0.0, min(1.0, body.failure_rate))
    if body.latency_ms is not None:
        CHAOS["latency_ms"] = max(0, body.latency_ms)
    return CHAOS


@app.get("/admin/customers")
def list_customers():
    """Clientes de prueba para la demo (id, nombre y teléfono)."""
    return [{"customer_id": c["customer_id"], "nombre": c["nombre"], "telefono": c["telefono"]}
            for c in CUSTOMERS.values()]
