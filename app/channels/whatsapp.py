"""Adaptador WhatsApp Business (Cloud API): verificación del webhook, firma y envío de respuestas."""
from __future__ import annotations

import hashlib
import hmac
import logging

import httpx

log = logging.getLogger(__name__)
GRAPH_URL = "https://graph.facebook.com/v20.0"


def verify_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    """Meta firma cada webhook con HMAC-SHA256 en el header X-Hub-Signature-256: 'sha256=<hex>'."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


class WhatsAppSender:
    def __init__(self, token: str, phone_number_id: str):
        self.token, self.phone_id = token, phone_number_id
        self.outbox: list[dict] = []   # sin credenciales (demo/tests) los mensajes quedan aquí

    def send_text(self, to: str, text: str) -> None:
        payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text[:4096]}}
        if not self.token:
            self.outbox.append(payload)
            log.info("WhatsApp (modo demo) -> respuesta encolada en outbox")
            return
        try:
            r = httpx.post(f"{GRAPH_URL}/{self.phone_id}/messages", json=payload, timeout=5,
                           headers={"Authorization": f"Bearer {self.token}"})
            r.raise_for_status()
        except Exception as exc:
            log.error("Error enviando a WhatsApp: %s", exc)
