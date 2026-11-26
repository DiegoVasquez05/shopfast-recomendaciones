"""Protección de datos personales (Ley N°19.628 Chile + GDPR para roaming).

- mask_pii(): enmascara RUT, correos, teléfonos y tarjetas en texto libre
  ANTES de loguear, persistir o enviar a un LLM.
- pseudonymize(): seudónimo estable HMAC-SHA256 para identificadores.
- MessageCipher: cifrado simétrico (Fernet: AES-128-CBC + HMAC-SHA256) del
  contenido de los mensajes en reposo.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re

PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("RUT", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dkK]\b")),
    ("TARJETA", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("TELEFONO", re.compile(r"(?:\+?56\s?)?\b9\s?\d{4}\s?\d{4}\b")),
]


def mask_pii(text: str) -> str:
    if not text:
        return text
    for label, rx in PII_PATTERNS:
        text = rx.sub(f"[{label}]", text)
    return text


def contains_pii(text: str) -> bool:
    return any(rx.search(text or "") for _, rx in PII_PATTERNS)


def pseudonymize(value: object, salt: str) -> str:
    return hmac.new(salt.encode(), str(value).encode(), hashlib.sha256).hexdigest()[:32]


class MessageCipher:
    """Cifra/descifra texto. Sin clave configurada deriva una de la sal (solo para dev)."""

    def __init__(self, key: str = "", salt: str = "dev"):
        from cryptography.fernet import Fernet

        if not key:
            key = base64.urlsafe_b64encode(hashlib.sha256(salt.encode()).digest()).decode()
        self.f = Fernet(key.encode())

    def encrypt(self, text: str) -> str:
        return self.f.encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self.f.decrypt(token.encode()).decode()
