"""Autenticación OAuth2 (EP2 §8.1).

- Canales y servicios (App móvil, dashboard, consola de agentes, pruebas de carga)
  usan el flujo OAuth2 *client credentials* en POST /oauth/token -> JWT con scopes.
- El Web Widget es un cliente público: obtiene un token anónimo de corta duración
  (scope chat:write) en POST /api/v1/channels/web/token.
- Si se configura OIDC_ISSUER (p. ej. Amazon Cognito), también se aceptan tokens
  RS256 firmados por ese proveedor (validación por JWKS).
"""
from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False, description="JWT obtenido en POST /oauth/token")
ALG = "HS256"

CLIENT_SCOPES = {
    "mobile-app": "chat:write",
    "dashboard": "chat:write metrics:read admin",
    "agent-console": "agent metrics:read",
    "loadtest": "chat:write metrics:read",
}
_jwks_clients: dict = {}


def issue_token(settings, subject: str, scope: str, minutes: int | None = None) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": subject, "scope": scope, "iat": now, "jti": uuid.uuid4().hex,
                       "exp": now + timedelta(minutes=minutes or settings.jwt_expire_minutes),
                       "iss": "telecombot"}, settings.jwt_secret, algorithm=ALG)


def authenticate_client(settings, client_id: str, client_secret: str) -> str:
    expected = settings.oauth_clients.get(client_id)
    if not expected or not secrets.compare_digest(expected, client_secret):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid_client")
    return CLIENT_SCOPES.get(client_id, "chat:write")


def _decode(settings, token: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") == "RS256" and settings.oidc_issuer:
            url = settings.oidc_issuer.rstrip("/") + "/.well-known/jwks.json"
            client = _jwks_clients.setdefault(url, jwt.PyJWKClient(url))
            key = client.get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=["RS256"], issuer=settings.oidc_issuer,
                              audience=settings.oidc_audience or None)
        return jwt.decode(token, settings.jwt_secret, algorithms=[ALG], issuer="telecombot")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token inválido")


def require_scope(scope: str):
    def dep(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        if creds is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Falta Authorization: Bearer <token>",
                                headers={"WWW-Authenticate": "Bearer"})
        claims = _decode(request.app.state.settings, creds.credentials)
        scopes = set(str(claims.get("scope", "")).split())
        if scope not in scopes and "admin" not in scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"El token no tiene el scope '{scope}'")
        return claims
    return dep


def decode_ws_token(settings, token: str | None, scope: str) -> dict | None:
    if not token:
        return None
    try:
        claims = _decode(settings, token)
    except HTTPException:
        return None
    scopes = set(str(claims.get("scope", "")).split())
    return claims if (scope in scopes or "admin" in scopes) else None
