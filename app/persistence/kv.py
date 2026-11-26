"""Almacenamiento clave-valor de baja latencia (Redis / ElastiCache).

Se usa para: estado de sesión, resultados de operaciones Siebel, notificaciones
pendientes por sesión y estado compartido del Circuit Breaker.
Sin REDIS_URL se usa una implementación en memoria con la misma interfaz.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any


class MemoryKV:
    def __init__(self) -> None:
        self._d: dict[str, tuple[float | None, Any]] = {}
        self._lock = threading.Lock()

    def _alive(self, key: str):
        item = self._d.get(key)
        if item is None:
            return None
        exp, val = item
        if exp is not None and exp < time.time():
            self._d.pop(key, None)
            return None
        return val

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._alive(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            self._d[key] = (time.time() + ttl if ttl else None, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)

    def rpush(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            lst = list(self._alive(key) or [])
            lst.append(value)
            self._d[key] = (time.time() + ttl if ttl else None, lst)

    def pop_all(self, key: str) -> list:
        with self._lock:
            lst = list(self._alive(key) or [])
            self._d.pop(key, None)
            return lst

    def incr(self, key: str, ttl: int | None = None) -> int:
        with self._lock:
            val = int(self._alive(key) or 0) + 1
            self._d[key] = (time.time() + ttl if ttl else None, val)
            return val

    def set_nx(self, key: str, value: Any, ttl: int) -> bool:
        with self._lock:
            if self._alive(key) is not None:
                return False
            self._d[key] = (time.time() + ttl, value)
            return True

    def keys(self, prefix: str) -> list[str]:
        with self._lock:
            return [k for k in list(self._d) if k.startswith(prefix) and self._alive(k) is not None]

    def ping(self) -> bool:
        return True


class RedisKV:
    def __init__(self, url: str) -> None:
        import redis

        self.r = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=1, decode_responses=True)

    def get(self, key: str):
        v = self.r.get(key)
        return None if v is None else json.loads(v)

    def set(self, key: str, value, ttl: int | None = None) -> None:
        self.r.set(key, json.dumps(value, default=str), ex=ttl)

    def delete(self, key: str) -> None:
        self.r.delete(key)

    def rpush(self, key: str, value, ttl: int | None = None) -> None:
        p = self.r.pipeline()
        p.rpush(key, json.dumps(value, default=str))
        if ttl:
            p.expire(key, ttl)
        p.execute()

    def pop_all(self, key: str) -> list:
        p = self.r.pipeline()
        p.lrange(key, 0, -1)
        p.delete(key)
        items, _ = p.execute()
        return [json.loads(x) for x in items]

    def incr(self, key: str, ttl: int | None = None) -> int:
        p = self.r.pipeline()
        p.incr(key)
        if ttl:
            p.expire(key, ttl)
        return int(p.execute()[0])

    def set_nx(self, key: str, value, ttl: int) -> bool:
        return bool(self.r.set(key, json.dumps(value), nx=True, ex=ttl))

    def keys(self, prefix: str) -> list[str]:
        return list(self.r.scan_iter(match=f"{prefix}*", count=500))

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


def build_kv(url: str):
    return RedisKV(url) if url else MemoryKV()
