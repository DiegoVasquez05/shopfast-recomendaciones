"""Persistencia del modelo de dominio (EP2 §8.4): Usuario, SesionConversacional, Mensaje,
OperacionSiebel, Handoff y TraceMetric (latencia/confianza por mensaje).

- Los identificadores de usuario se guardan seudonimizados (HMAC).
- El texto de los mensajes se guarda con PII enmascarada y CIFRADO (Fernet).
- PostgreSQL (RDS) en producción; memoria para tests/demo.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    user_hash   TEXT PRIMARY KEY,
    canal       TEXT,
    consentimiento BOOLEAN DEFAULT TRUE,
    creado      TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sesiones (
    session_id  TEXT PRIMARY KEY,
    user_hash   TEXT REFERENCES usuarios(user_hash) ON DELETE CASCADE,
    canal       TEXT,
    cohorte     TEXT,
    estado      TEXT,
    resuelta_por_bot BOOLEAN DEFAULT FALSE,
    iniciada    TIMESTAMPTZ DEFAULT now(),
    actualizada TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS mensajes (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT REFERENCES sesiones(session_id) ON DELETE CASCADE,
    rol         TEXT NOT NULL,
    texto_cifrado TEXT NOT NULL,
    intent      TEXT,
    confianza   REAL,
    resolucion  TEXT,
    latencia_ms REAL,
    correlation_id TEXT,
    creado      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mensajes_sesion ON mensajes(session_id, id);
CREATE TABLE IF NOT EXISTS operaciones_siebel (
    correlation_id TEXT PRIMARY KEY,
    session_id  TEXT,
    operacion   TEXT,
    estado      TEXT,
    intentos    INT DEFAULT 0,
    error       TEXT,
    creado      TIMESTAMPTZ DEFAULT now(),
    actualizado TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS handoffs (
    id          TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES sesiones(session_id) ON DELETE CASCADE,
    motivo      TEXT,
    estado      TEXT,
    agente      TEXT,
    resumen     TEXT,
    creado      TIMESTAMPTZ DEFAULT now(),
    cerrado     TIMESTAMPTZ
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store(Protocol):
    def upsert_session(self, s: dict) -> None: ...
    def add_message(self, m: dict) -> None: ...
    def messages(self, session_id: str) -> list[dict]: ...
    def upsert_operation(self, op: dict) -> None: ...
    def create_handoff(self, h: dict) -> None: ...
    def update_handoff(self, hid: str, **fields) -> None: ...
    def handoffs(self, estado: str | None = None) -> list[dict]: ...
    def delete_user(self, user_hash: str) -> dict: ...
    def export_user(self, user_hash: str) -> dict: ...
    def stats(self) -> dict: ...
    def ping(self) -> bool: ...


def _kpis(sessions: list[dict], msgs: list[dict], ops: list[dict], handoffs: list[dict]) -> dict:
    bot = [m for m in msgs if m["rol"] == "bot"]
    lat = np.array([m["latencia_ms"] for m in bot if m.get("latencia_ms") is not None] or [0.0])
    with_handoff = {h["session_id"] for h in handoffs}
    piloto = [s for s in sessions if s.get("cohorte") == "piloto"]
    resolved = [s for s in piloto if s.get("resuelta_por_bot") and s["session_id"] not in with_handoff]
    return {
        "sesiones": len(sessions),
        "sesiones_piloto": len(piloto),
        "mensajes": len(msgs),
        "tasa_automatizacion": round(len(resolved) / len(piloto), 4) if piloto else 0.0,
        "tasa_handoff": round(len(with_handoff & {s["session_id"] for s in piloto}) / len(piloto), 4) if piloto else 0.0,
        "latencia_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                        "p99": round(float(np.percentile(lat, 99)), 1)},
        "confianza_promedio": round(float(np.mean([m["confianza"] for m in bot if m.get("confianza") is not None]
                                                  or [0])), 4),
        "por_intencion": dict(Counter(m["intent"] for m in bot if m.get("intent")).most_common(15)),
        "por_resolucion": dict(Counter(m.get("resolucion") for m in bot if m.get("resolucion"))),
        "por_canal": dict(Counter(s.get("canal") for s in sessions)),
        "por_cohorte": dict(Counter(s.get("cohorte") for s in sessions)),
        "operaciones_siebel": dict(Counter(o.get("estado") for o in ops)),
        "handoffs": dict(Counter(h.get("motivo") for h in handoffs)),
        "handoffs_abiertos": sum(1 for h in handoffs if h.get("estado") != "cerrado"),
    }


class MemoryStore:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.msgs: list[dict] = []
        self.ops: dict[str, dict] = {}
        self.hand: dict[str, dict] = {}
        self._lock = threading.Lock()

    def upsert_session(self, s: dict) -> None:
        with self._lock:
            self.users.setdefault(s["user_hash"], {"user_hash": s["user_hash"], "canal": s.get("canal"),
                                                   "creado": _now()})
            cur = self.sessions.get(s["session_id"], {"iniciada": _now()})
            self.sessions[s["session_id"]] = {**cur, **s, "actualizada": _now()}

    def add_message(self, m: dict) -> None:
        with self._lock:
            self.msgs.append({**m, "creado": _now()})

    def messages(self, session_id: str) -> list[dict]:
        return [m for m in self.msgs if m["session_id"] == session_id]

    def upsert_operation(self, op: dict) -> None:
        with self._lock:
            self.ops[op["correlation_id"]] = {**self.ops.get(op["correlation_id"], {}), **op, "actualizado": _now()}

    def create_handoff(self, h: dict) -> None:
        with self._lock:
            self.hand[h["id"]] = {**h, "creado": _now()}

    def update_handoff(self, hid: str, **fields) -> None:
        with self._lock:
            if hid in self.hand:
                self.hand[hid].update(fields)

    def handoffs(self, estado: str | None = None) -> list[dict]:
        return [h for h in self.hand.values() if estado is None or h["estado"] == estado]

    def delete_user(self, user_hash: str) -> dict:
        with self._lock:
            sids = {k for k, s in self.sessions.items() if s["user_hash"] == user_hash}
            n_msgs = sum(1 for m in self.msgs if m["session_id"] in sids)
            self.msgs = [m for m in self.msgs if m["session_id"] not in sids]
            self.hand = {k: h for k, h in self.hand.items() if h["session_id"] not in sids}
            for s in sids:
                self.sessions.pop(s, None)
            self.users.pop(user_hash, None)
            return {"sesiones": len(sids), "mensajes": n_msgs}

    def export_user(self, user_hash: str) -> dict:
        sids = [k for k, s in self.sessions.items() if s["user_hash"] == user_hash]
        return {"usuario": self.users.get(user_hash), "sesiones": [self.sessions[s] for s in sids],
                "mensajes": [m for m in self.msgs if m["session_id"] in sids]}

    def stats(self) -> dict:
        return _kpis(list(self.sessions.values()), self.msgs, list(self.ops.values()), list(self.hand.values()))

    def ping(self) -> bool:
        return True


class PostgresStore:
    def __init__(self, url: str) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(url, min_size=1, max_size=10, open=True, timeout=5,
                                   kwargs={"row_factory": dict_row, "autocommit": True})
        for attempt in range(20):
            try:
                with self.pool.connection() as c:
                    c.execute(SCHEMA)
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(1.5)

    def _x(self, sql: str, params=()):
        with self.pool.connection() as c:
            return c.execute(sql, params)

    def _all(self, sql: str, params=()) -> list[dict]:
        with self.pool.connection() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def upsert_session(self, s: dict) -> None:
        with self.pool.connection() as c:
            c.execute("INSERT INTO usuarios (user_hash, canal) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                      (s["user_hash"], s.get("canal")))
            c.execute(
                "INSERT INTO sesiones (session_id, user_hash, canal, cohorte, estado, resuelta_por_bot)"
                " VALUES (%(session_id)s,%(user_hash)s,%(canal)s,%(cohorte)s,%(estado)s,%(resuelta_por_bot)s)"
                " ON CONFLICT (session_id) DO UPDATE SET estado=EXCLUDED.estado,"
                " resuelta_por_bot=sesiones.resuelta_por_bot OR EXCLUDED.resuelta_por_bot, actualizada=now()",
                {"canal": None, "cohorte": None, "estado": "activa", "resuelta_por_bot": False, **s})

    def add_message(self, m: dict) -> None:
        self._x("INSERT INTO mensajes (session_id, rol, texto_cifrado, intent, confianza, resolucion, latencia_ms,"
                " correlation_id) VALUES (%(session_id)s,%(rol)s,%(texto_cifrado)s,%(intent)s,%(confianza)s,"
                " %(resolucion)s,%(latencia_ms)s,%(correlation_id)s)",
                {"intent": None, "confianza": None, "resolucion": None, "latencia_ms": None,
                 "correlation_id": None, **m})

    def messages(self, session_id: str) -> list[dict]:
        return self._all("SELECT * FROM mensajes WHERE session_id=%s ORDER BY id", (session_id,))

    def upsert_operation(self, op: dict) -> None:
        self._x("INSERT INTO operaciones_siebel (correlation_id, session_id, operacion, estado, intentos, error)"
                " VALUES (%(correlation_id)s,%(session_id)s,%(operacion)s,%(estado)s,%(intentos)s,%(error)s)"
                " ON CONFLICT (correlation_id) DO UPDATE SET estado=EXCLUDED.estado, intentos=EXCLUDED.intentos,"
                " error=EXCLUDED.error, actualizado=now()",
                {"session_id": None, "operacion": None, "intentos": 0, "error": None, **op})

    def create_handoff(self, h: dict) -> None:
        self._x("INSERT INTO handoffs (id, session_id, motivo, estado, resumen) VALUES"
                " (%(id)s,%(session_id)s,%(motivo)s,%(estado)s,%(resumen)s)", h)

    def update_handoff(self, hid: str, **fields) -> None:
        sets = ", ".join(f"{k}=%({k})s" for k in fields)
        if fields.get("estado") == "cerrado":
            sets += ", cerrado=now()"
        self._x(f"UPDATE handoffs SET {sets} WHERE id=%(id)s", {**fields, "id": hid})

    def handoffs(self, estado: str | None = None) -> list[dict]:
        if estado:
            return self._all("SELECT * FROM handoffs WHERE estado=%s ORDER BY creado DESC", (estado,))
        return self._all("SELECT * FROM handoffs ORDER BY creado DESC LIMIT 200")

    def delete_user(self, user_hash: str) -> dict:
        with self.pool.connection() as c:
            n_s = c.execute("SELECT count(*) AS n FROM sesiones WHERE user_hash=%s", (user_hash,)).fetchone()["n"]
            n_m = c.execute("SELECT count(*) AS n FROM mensajes m JOIN sesiones s USING (session_id)"
                            " WHERE s.user_hash=%s", (user_hash,)).fetchone()["n"]
            c.execute("DELETE FROM usuarios WHERE user_hash=%s", (user_hash,))  # ON DELETE CASCADE
        return {"sesiones": int(n_s), "mensajes": int(n_m)}

    def export_user(self, user_hash: str) -> dict:
        ses = self._all("SELECT * FROM sesiones WHERE user_hash=%s", (user_hash,))
        msgs = self._all("SELECT m.* FROM mensajes m JOIN sesiones s USING (session_id) WHERE s.user_hash=%s"
                         " ORDER BY m.id", (user_hash,))
        user = self._all("SELECT * FROM usuarios WHERE user_hash=%s", (user_hash,))
        return {"usuario": user[0] if user else None, "sesiones": ses, "mensajes": msgs}

    def stats(self) -> dict:
        since = "now() - interval '7 days'"
        return _kpis(self._all(f"SELECT * FROM sesiones WHERE iniciada > {since}"),
                     self._all(f"SELECT session_id, rol, intent, confianza, resolucion, latencia_ms FROM mensajes"
                               f" WHERE creado > {since}"),
                     self._all(f"SELECT estado FROM operaciones_siebel WHERE creado > {since}"),
                     self._all(f"SELECT session_id, motivo, estado FROM handoffs WHERE creado > {since}"))

    def ping(self) -> bool:
        try:
            self._x("SELECT 1")
            return True
        except Exception:
            return False


def build_store(url: str) -> Store:
    return PostgresStore(url) if url else MemoryStore()
