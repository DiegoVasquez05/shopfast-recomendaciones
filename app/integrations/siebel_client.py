"""Cliente del CRM legacy Siebel (REST; el mock también expone un endpoint SOAP).

- Timeout explícito por llamada (EP2 §8.3).
- Reintentos con backoff exponencial SOLO en operaciones idempotentes (lecturas).
- Todas las llamadas pasan por el Circuit Breaker.
- Credenciales desde AWS Secrets Manager (equivalente gestionado de Vault) si
  SIEBEL_SECRET_ARN está definido; si no, desde variables de entorno.
"""
from __future__ import annotations

import json
import logging
import time

import httpx

from app.integrations.circuit_breaker import CircuitBreaker

log = logging.getLogger(__name__)

READ_OPS = {"get_balance": "balance", "get_invoices": "invoices", "get_profile": "profile"}
WRITE_OPS = {"create_ticket": "tickets"}


def load_credentials(settings) -> tuple[str, str]:
    if settings.siebel_secret_arn:
        import boto3

        sm = boto3.client("secretsmanager", region_name=settings.aws_region)
        secret = json.loads(sm.get_secret_value(SecretId=settings.siebel_secret_arn)["SecretString"])
        return secret["username"], secret["password"]
    return settings.siebel_user, settings.siebel_password


class SiebelError(RuntimeError):
    pass


class SiebelClient:
    def __init__(self, settings, breaker: CircuitBreaker, http: httpx.Client | None = None):
        user, pwd = load_credentials(settings)
        self.http = http or httpx.Client(base_url=settings.siebel_base_url, auth=(user, pwd),
                                         timeout=settings.siebel_timeout_s)
        self.breaker = breaker
        self.max_retries = settings.siebel_max_retries

    def _customer_path(self, params: dict) -> str:
        if params.get("customer_id"):
            return f"/siebel/customers/{params['customer_id']}"
        if params.get("phone"):
            return f"/siebel/customers/by-phone/{params['phone']}"
        raise SiebelError("Se requiere customer_id o phone")

    def _get(self, path: str) -> dict:
        delay = 0.1
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.http.get(path)
                if r.status_code == 404:
                    raise SiebelError("Cliente no encontrado en Siebel")
                r.raise_for_status()
                return r.json()
            except SiebelError:
                raise
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("Siebel GET %s falló (intento %s/%s): %s", path, attempt, self.max_retries, exc)
                time.sleep(delay)
                delay *= 2
        raise SiebelError("inalcanzable")

    def execute(self, operation: str, params: dict) -> dict:
        if operation in READ_OPS:
            path = f"{self._customer_path(params)}/{READ_OPS[operation]}"
            return self.breaker.call(lambda: self._get(path))
        if operation in WRITE_OPS:
            def post():
                r = self.http.post(f"{self._customer_path(params)}/{WRITE_OPS[operation]}", json=params,
                                   headers={"Idempotency-Key": params.get("idempotency_key", "")})
                r.raise_for_status()
                return r.json()
            return self.breaker.call(post)  # escritura: sin reintento inmediato (lo maneja la cola)
        raise SiebelError(f"Operación desconocida: {operation}")
