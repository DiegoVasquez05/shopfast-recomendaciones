"""Prueba de carga de TelecomBot AI (Locust).

Simula clientes conversando: cada usuario virtual mantiene una sesión y envía
mensajes con pausas de lectura/escritura humanas (3-8 s), como en un chat real.

    locust -f loadtest/locustfile.py --host https://<IP-EC2> --headless -u 300 -r 30 -t 5m \
           --csv loadtest/results/carga --html loadtest/results/carga.html

Para 1.000 conversaciones simultáneas: -u 1000 -r 50 (ejecutar Locust desde otra máquina/EC2).
"""
import os
import random

from locust import HttpUser, between, events, task

CLIENT_ID = os.getenv("LOAD_CLIENT_ID", "loadtest")
CLIENT_SECRET = os.getenv("LOAD_CLIENT_SECRET", "dev-loadtest-secret")
CUSTOMERS = [f"C-{1000 + i}" for i in range(1, 16)]

MESSAGES = {
    "saldo": ["wena, cuánto debo?", "cuantos gigas me quedan", "tengo deuda pendiente?"],
    "boleta": ["cuánto sale mi boleta este mes", "mándame la boleta en pdf", "por qué la cuenta vino tan cara"],
    "soporte": ["no tengo internet desde ayer", "el modem tiene luz roja", "la tele no tiene señal",
                "se cae el wifi a cada rato"],
    "info": ["cómo activo el roaming", "qué planes tienen con más gigas", "cómo me porto a telcoaustral",
             "dónde puedo pagar la cuenta", "a qué hora atienden"],
    "cierre": ["gracias", "eso era todo, chao"],
    "humano": ["quiero hablar con un ejecutivo"],
}


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--verify-tls", default="false")


_TOKEN: dict = {}


class Cliente(HttpUser):
    wait_time = between(3, 8)

    def on_start(self):
        self.client.verify = self.environment.parsed_options.verify_tls == "true"
        if "value" not in _TOKEN:  # un token por proceso de Locust, como un canal real (client credentials)
            r = self.client.post("/oauth/token", data={"grant_type": "client_credentials", "client_id": CLIENT_ID,
                                                       "client_secret": CLIENT_SECRET}, name="/oauth/token")
            _TOKEN["value"] = r.json()["access_token"]
        self.client.headers["Authorization"] = "Bearer " + _TOKEN["value"]
        self.user_id = f"demo:load-{random.randint(0, 10**9)}"
        self.customer = random.choice(CUSTOMERS)
        self.session_id = None

    def say(self, topic: str):
        with self.client.post("/api/v1/chat/messages", name=f"/api/v1/chat/messages [{topic}]", catch_response=True,
                              json={"user_id": self.user_id, "text": random.choice(MESSAGES[topic]),
                                    "session_id": self.session_id, "customer_id": self.customer}) as r:
            if r.status_code == 200:
                d = r.json()
                self.session_id = d["session_id"]
                if d["latency_ms"] > 3000:
                    r.failure(f"SLA excedido: {d['latency_ms']} ms")
            elif r.status_code == 429:
                r.success()  # el rate limit es comportamiento esperado, no error del servicio

    @task(5)
    def saldo(self):
        self.say("saldo")

    @task(3)
    def boleta(self):
        self.say("boleta")

    @task(4)
    def soporte(self):
        self.say("soporte")

    @task(5)
    def info(self):
        self.say("info")

    @task(1)
    def humano(self):
        self.say("humano")
        self.session_id = None  # el cliente pasó a un ejecutivo; la próxima es una conversación nueva

    @task(2)
    def cierre(self):
        self.say("cierre")
        self.session_id = None
