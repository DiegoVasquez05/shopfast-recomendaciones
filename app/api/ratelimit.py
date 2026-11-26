"""Rate limiting por cliente (ventana deslizante de 60 s). Complementa el limit_req por IP de Nginx/API Gateway.

La cuota depende del cliente OAuth2: un canal servidor-a-servidor (app móvil) agrupa
tráfico de miles de usuarios, mientras que cada token del web widget es de un solo usuario.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import jwt

CLIENT_QUOTAS = {"mobile-app": 6000, "loadtest": 100_000, "dashboard": 600, "agent-console": 600,
                 "web-widget": 30}


def client_key(authorization: str | None, fallback: str, default_quota: int) -> tuple[str, int]:
    """Clave y cuota a partir del JWT (la firma se valida después, en la dependencia de seguridad)."""
    if authorization and authorization.lower().startswith("bearer "):
        try:
            claims = jwt.decode(authorization[7:], options={"verify_signature": False})
            sub = str(claims.get("sub", "anon"))
            return f"{sub}:{claims.get('jti', '')}", CLIENT_QUOTAS.get(sub, default_quota)
        except jwt.InvalidTokenError:
            pass
    return f"ip:{fallback}", default_quota


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str, limit: int | None = None) -> tuple[bool, int]:
        limit = limit or self.per_minute
        now = time.monotonic()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
            return True, 0
