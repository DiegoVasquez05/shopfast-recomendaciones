"""Cola de mensajes para los comandos hacia Siebel (Event-Driven, EP2 ADR-003).

Backends intercambiables por configuración (QUEUE_BACKEND):
  rabbitmq -> RabbitMQ (cola durable + dead-letter queue). Usado en el despliegue EC2/docker-compose.
  sqs      -> Amazon SQS (opción de la EP2 para EKS; DLQ por redrive policy).
  memory   -> en proceso (tests y demo local): el handler se ejecuta en el mismo hilo.

Los mensajes son persistentes (delivery_mode=2 / SQS), por eso RPO = 0:
si el worker o Siebel caen, los comandos esperan en la cola.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)
Handler = Callable[[dict], None]


class MessageQueue(Protocol):
    def publish(self, message: dict) -> None: ...
    def consume(self, handler: Handler) -> None: ...
    def ping(self) -> bool: ...
    def depth(self) -> int | None: ...


class MemoryQueue:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.handler: Handler | None = None
        self.dlq: list[dict] = []

    def publish(self, message: dict) -> None:
        self.messages.append(message)
        if self.handler:
            try:
                self.handler(message)
            except Exception as exc:  # igual que en RabbitMQ: el mensaje fallido va a la DLQ
                log.error("handler falló, mensaje a DLQ: %s", exc)
                self.dlq.append(message)

    def consume(self, handler: Handler) -> None:
        self.handler = handler

    def ping(self) -> bool:
        return True

    def depth(self) -> int | None:
        return 0


class RabbitQueue:
    def __init__(self, url: str, queue: str) -> None:
        self.url, self.queue = url, queue
        self._lock = threading.Lock()
        self._conn = None
        self._ch = None

    def _declare(self, ch) -> None:
        ch.exchange_declare(exchange=f"{self.queue}.dlx", exchange_type="fanout", durable=True)
        ch.queue_declare(queue=f"{self.queue}.dlq", durable=True)
        ch.queue_bind(queue=f"{self.queue}.dlq", exchange=f"{self.queue}.dlx")
        ch.queue_declare(queue=self.queue, durable=True,
                         arguments={"x-dead-letter-exchange": f"{self.queue}.dlx"})

    def _channel(self):
        import pika

        if self._conn is None or self._conn.is_closed or self._ch is None or self._ch.is_closed:
            params = pika.URLParameters(self.url)
            params.heartbeat = 30
            params.blocked_connection_timeout = 5
            self._conn = pika.BlockingConnection(params)
            self._ch = self._conn.channel()
            self._declare(self._ch)
        return self._ch

    def publish(self, message: dict) -> None:
        import pika

        with self._lock:
            for attempt in range(2):
                try:
                    self._channel().basic_publish(
                        exchange="", routing_key=self.queue, body=json.dumps(message).encode(),
                        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2,
                                                        correlation_id=message.get("correlation_id")))
                    return
                except Exception as exc:
                    log.warning("publish RabbitMQ falló (intento %s): %s", attempt + 1, exc)
                    self._conn = None
            raise RuntimeError("RabbitMQ no disponible")

    def consume(self, handler: Handler) -> None:
        import pika

        while True:
            try:
                conn = pika.BlockingConnection(pika.URLParameters(self.url))
                ch = conn.channel()
                self._declare(ch)
                ch.basic_qos(prefetch_count=20)

                def on_msg(channel, method, _props, body):
                    try:
                        handler(json.loads(body))
                        channel.basic_ack(method.delivery_tag)
                    except Exception:
                        log.exception("Error procesando comando; se envía a la DLQ")
                        channel.basic_nack(method.delivery_tag, requeue=False)

                ch.basic_consume(queue=self.queue, on_message_callback=on_msg)
                log.info("Consumiendo cola RabbitMQ '%s'", self.queue)
                ch.start_consuming()
            except Exception as exc:
                log.warning("Conexión RabbitMQ perdida (%s); reintentando en 3 s", exc)
                time.sleep(3)

    def ping(self) -> bool:
        try:
            with self._lock:
                self._channel()
            return True
        except Exception:
            return False

    def depth(self) -> int | None:
        try:
            with self._lock:
                return self._channel().queue_declare(queue=self.queue, passive=True).method.message_count
        except Exception:
            return None


class SqsQueue:
    def __init__(self, queue_url: str, region: str) -> None:
        import boto3

        self.url = queue_url
        self.sqs = boto3.client("sqs", region_name=region)

    def publish(self, message: dict) -> None:
        self.sqs.send_message(QueueUrl=self.url, MessageBody=json.dumps(message),
                              MessageAttributes={"correlation_id": {
                                  "DataType": "String", "StringValue": message.get("correlation_id", "-")}})

    def consume(self, handler: Handler) -> None:
        log.info("Consumiendo SQS %s", self.url)
        while True:
            resp = self.sqs.receive_message(QueueUrl=self.url, MaxNumberOfMessages=10, WaitTimeSeconds=20)
            for m in resp.get("Messages", []):
                try:
                    handler(json.loads(m["Body"]))
                    self.sqs.delete_message(QueueUrl=self.url, ReceiptHandle=m["ReceiptHandle"])
                except Exception:
                    # no se borra: vuelve a la cola y tras maxReceiveCount va a la DLQ (redrive policy)
                    log.exception("Error procesando mensaje SQS")

    def ping(self) -> bool:
        try:
            self.sqs.get_queue_attributes(QueueUrl=self.url, AttributeNames=["QueueArn"])
            return True
        except Exception:
            return False

    def depth(self) -> int | None:
        try:
            a = self.sqs.get_queue_attributes(QueueUrl=self.url, AttributeNames=["ApproximateNumberOfMessages"])
            return int(a["Attributes"]["ApproximateNumberOfMessages"])
        except Exception:
            return None


def build_queue(settings) -> MessageQueue:
    backend = (settings.queue_backend or "memory").lower()
    if backend == "rabbitmq":
        return RabbitQueue(settings.rabbitmq_url, settings.siebel_queue)
    if backend == "sqs":
        return SqsQueue(settings.sqs_queue_url, settings.aws_region)
    return MemoryQueue()
