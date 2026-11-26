"""Channel Normalizer (EP2 §5.2): homogeneiza Web Widget, WhatsApp y App móvil a un contrato interno."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

CHANNELS = {"web", "whatsapp", "app"}


@dataclass
class InboundMessage:
    channel: str
    user_id: str               # id lógico del canal (id de app, teléfono WhatsApp, id anónimo web)
    text: str
    session_id: str | None = None
    customer_id: str | None = None   # cliente autenticado (OIDC en la app / web)
    phone: str | None = None         # WhatsApp: se usa para identificar al cliente en Siebel
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict = field(default_factory=dict)


def from_api(body: dict, channel: str) -> InboundMessage:
    return InboundMessage(channel=channel, user_id=str(body["user_id"]), text=body["text"].strip(),
                          session_id=body.get("session_id"), customer_id=body.get("customer_id"),
                          metadata={"locale": body.get("locale", "es-CL")})


def from_whatsapp(value: dict) -> list[InboundMessage]:
    """Payload de WhatsApp Cloud API: entry[].changes[].value.messages[]"""
    out = []
    for m in value.get("messages", []):
        if m.get("type") != "text":
            continue
        phone = m["from"]
        out.append(InboundMessage(channel="whatsapp", user_id=f"wa:{phone}", text=m["text"]["body"].strip(),
                                  session_id=f"wa-{phone}", phone=phone, message_id=m.get("id", uuid.uuid4().hex),
                                  metadata={"wa_message_id": m.get("id")}))
    return out
