"""Session Manager (EP2: conversation/session_service.py): contexto e historial de la conversación en Redis."""
from __future__ import annotations

import hashlib
import time
import uuid

from app.channels.normalizer import InboundMessage
from app.privacy import mask_pii, pseudonymize

MAX_HISTORY = 10


def pilot_cohort(user_id: str, percentage: int, allowlist: list[str], salt: str) -> str:
    """Asignación determinista y estable al piloto (10% de usuarios, EP2 ADR-005)."""
    if user_id in allowlist or any(user_id.startswith(a + ":") for a in allowlist):
        return "piloto"
    bucket = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:8], 16) % 100
    return "piloto" if bucket < percentage else "control"


class SessionService:
    def __init__(self, kv, settings):
        self.kv, self.s = kv, settings

    @staticmethod
    def key(session_id: str) -> str:
        return f"session:{session_id}"

    def get(self, session_id: str) -> dict | None:
        return self.kv.get(self.key(session_id))

    def get_or_create(self, msg: InboundMessage) -> dict:
        sid = msg.session_id or f"{msg.channel}-{uuid.uuid4().hex[:12]}"
        sess = self.get(sid)
        if sess is None:
            sess = {"session_id": sid, "channel": msg.channel, "user_id": msg.user_id,
                    "user_hash": pseudonymize(msg.user_id, self.s.pii_salt),
                    "cohort": pilot_cohort(msg.user_id, self.s.pilot_percentage, self.s.pilot_allowlist,
                                           self.s.pii_salt),
                    "customer_id": msg.customer_id, "phone": msg.phone, "state": "bot",
                    "low_conf_streak": 0, "handoff_id": None, "history": [], "created": time.time()}
        if msg.customer_id:
            sess["customer_id"] = msg.customer_id
        return sess

    def add_turn(self, sess: dict, role: str, text: str, **meta) -> None:
        sess["history"] = (sess["history"] + [{"role": role, "text": mask_pii(text), "ts": time.time(), **meta}])[
            -MAX_HISTORY:]

    def save(self, sess: dict) -> None:
        self.kv.set(self.key(sess["session_id"]), sess, self.s.session_ttl_s)

    def delete_user_sessions(self, user_hash: str) -> int:
        n = 0
        for k in self.kv.keys("session:"):
            v = self.kv.get(k)
            if isinstance(v, dict) and v.get("user_hash") == user_hash:
                self.kv.delete(k)
                n += 1
        return n
