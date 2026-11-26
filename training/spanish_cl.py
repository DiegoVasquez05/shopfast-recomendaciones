"""Corpus de adaptación al español chileno para el dominio telco (TelcoAustral).

El dataset Bitext está en inglés y orientado a e-commerce. Para cumplir el
requisito "NLP adaptado al español chileno (modismos y contexto de servicio)"
se genera un corpus por plantillas con:
  - frases base por intención (redactadas para telecomunicaciones),
  - variaciones de registro chileno (wena, po, altiro, cachai, lucas...),
  - errores de tipeo y abreviaturas de chat (q, xq, kiero, porfa).

El conjunto de EVALUACIÓN (data/eval_es_cl.csv) está escrito a mano y NO se
genera con estas plantillas, para medir generalización real.
"""
from __future__ import annotations

import random

import pandas as pd

BASE = {
    "check_balance": [
        "cuanto saldo me queda", "quiero saber mi saldo", "cuanto debo", "cuanta plata debo",
        "cuantos megas me quedan", "cuantos gigas me quedan en el plan", "revisar mi saldo",
        "tengo deuda pendiente", "consultar saldo de mi prepago", "me queda saldo para llamar",
        "cuanto tengo que pagar este mes", "estoy al dia con los pagos", "cuanto es mi deuda",
        "se me acabaron los datos", "saldo disponible de mi linea",
    ],
    "check_invoice": [
        "cuanto sale mi boleta", "cuanto es la boleta de este mes", "por que mi boleta viene tan cara",
        "me llego la cuenta muy alta", "revisar mi ultima boleta", "detalle de la boleta",
        "cuando vence mi boleta", "que me estan cobrando en la boleta", "monto de la cuenta del mes",
        "me cobraron de mas en la boleta", "no entiendo el cobro de mi boleta", "fecha de vencimiento de la cuenta",
    ],
    "get_invoice": [
        "necesito una copia de la boleta", "mandame la boleta en pdf", "donde descargo mi boleta",
        "quiero la boleta del mes pasado", "envienme la boleta al correo", "como bajo la boleta electronica",
        "no me llego la boleta", "necesito la boleta para rendir en la pega", "segunda copia de mi cuenta",
    ],
    "check_plan": [
        "que plan tengo", "cual es mi plan actual", "que incluye mi plan", "cuantos gigas trae mi plan",
        "mi plan tiene minutos ilimitados", "que servicios tengo contratados", "ver mi plan de hogar",
        "cuantos canales tiene mi plan de tele", "velocidad de internet contratada",
    ],
    "change_plan": [
        "quiero cambiar de plan", "quiero un plan con mas gigas", "me quiero subir de plan",
        "hay planes mas baratos", "quiero bajar mi plan", "que planes tienen", "quiero agregar canales",
        "quiero mas velocidad de internet", "ofertas de planes moviles", "planes con redes sociales libres",
        "mejorar mi plan de hogar", "contratar fibra optica",
    ],
    "technical_support": [
        "no tengo senal", "se me corto el internet", "el internet anda lento", "no funciona la tele",
        "el modem tiene luz roja", "no puedo llamar", "no me llegan los mensajes", "se cae el wifi a cada rato",
        "la television no tiene senal", "el decodificador no prende", "internet muy lento en la noche",
        "no tengo internet desde ayer", "el celular dice sin servicio", "no carga ninguna pagina",
        "falla tecnica en mi casa", "se pega la imagen de la tele",
    ],
    "portability": [
        "quiero portarme a telcoaustral", "como me cambio de compania con mi mismo numero",
        "portabilidad de mi numero", "me quiero ir a otra compania", "quiero mantener mi numero y cambiarme",
        "cuanto se demora la portabilidad", "requisitos para la portabilidad", "vengo de otra empresa",
    ],
    "roaming": [
        "me voy de viaje que hago con el celular", "como activo el roaming", "cuanto cuesta el roaming",
        "voy a argentina funciona mi plan", "uso de datos en el extranjero", "roaming en europa",
        "me voy a peru y quiero usar whatsapp", "bolsa de roaming", "llamadas desde el extranjero",
    ],
    "check_payment_methods": [
        "como puedo pagar", "donde pago la cuenta", "puedo pagar con tarjeta", "se puede pagar con transferencia",
        "aceptan webpay", "medios de pago", "puedo dejar el pago automatico", "pagar en el supermercado",
        "pac o pat como lo activo",
    ],
    "payment_issue": [
        "pague y me sigue apareciendo la deuda", "me cobraron dos veces", "el pago no se reflejo",
        "me rechazaron la tarjeta", "pague la boleta y me cortaron igual", "error al pagar en la pagina",
        "me descontaron la plata pero no aparece el pago", "pago duplicado",
    ],
    "complaint": [
        "quiero poner un reclamo", "estoy muy molesto con el servicio", "esto es una estafa",
        "voy a reclamar al sernac", "pesimo servicio", "llevo una semana sin internet y nadie me ayuda",
        "estoy chato de llamar", "quiero hacer un reclamo formal", "el servicio es malisimo",
        "me tienen aburrido con los cortes",
    ],
    "contact_human_agent": [
        "quiero hablar con una persona", "comunicame con un ejecutivo", "pasame con un humano",
        "no quiero hablar con un robot", "necesito un ejecutivo", "derivame con alguien",
        "hablar con un operador", "atiendeme una persona real",
    ],
    "contact_customer_service": [
        "cual es el horario de atencion", "numero del call center", "donde hay una sucursal",
        "como los contacto", "telefono de atencion al cliente", "atienden los domingos",
        "direccion de la oficina comercial",
    ],
    "delete_account": [
        "quiero dar de baja el servicio", "quiero cancelar mi plan", "quiero terminar el contrato",
        "como doy de baja internet", "quiero cortar la tele", "cancelar mi linea", "renunciar al servicio",
    ],
    "check_cancellation_fee": [
        "cuanto me cobran por terminar el contrato", "hay multa por cancelar", "cobran por dar de baja",
        "costo por termino anticipado", "me cobran si me cambio antes de tiempo",
    ],
    "recover_password": [
        "se me olvido la clave", "no me acuerdo de la contrasena de la app", "recuperar clave",
        "cambiar la clave de la sucursal virtual", "me bloquearon la cuenta por la clave",
    ],
    "edit_account": [
        "quiero cambiar mi correo", "actualizar mi direccion", "cambiar el numero de contacto",
        "modificar mis datos personales", "cambiar el titular de la cuenta",
    ],
    "get_refund": [
        "quiero que me devuelvan la plata", "solicito devolucion de un cobro", "me deben devolver dinero",
        "reembolso por los dias sin servicio", "quiero la devolucion del cobro mal hecho",
    ],
    "track_order": [
        "cuando llega mi celular", "estado de mi pedido", "cuando vienen a instalar", "seguimiento del despacho",
        "no ha llegado el tecnico", "a que hora llega el instalador",
    ],
    "place_order": [
        "quiero comprar un celular", "quiero contratar internet", "contratar plan hogar", "comprar un equipo",
        "quiero un chip nuevo", "quiero contratar tele",
    ],
    "review": [
        "quiero dejar una opinion", "quiero evaluar la atencion", "felicitar al tecnico", "dejar un comentario",
    ],
    "greeting": [
        "hola", "buenas", "buenos dias", "buenas tardes", "hola como estas", "alo", "hola necesito ayuda",
    ],
    "goodbye": [
        "gracias", "muchas gracias", "chao", "eso era todo", "gracias por la ayuda", "adios", "listo gracias",
    ],
}

PREFIX = ["", "", "", "hola ", "wena ", "oye ", "buenas, ", "hola, una consulta: ", "porfa ", "oiga ", "disculpe "]
SUFFIX = ["", "", "", " po", " porfa", " altiro", " por favor", "?", "??", " cachai", " gracias", " urgente"]
TYPOS = [("que", "q"), ("quiero", "kiero"), ("porque", "xq"), ("por favor", "porfa"), ("necesito", "nesecito"),
         ("celular", "celu"), ("telefono", "fono"), ("dinero", "plata"), ("mucho", "harto")]


def _typo(text: str, rnd: random.Random) -> str:
    for a, b in TYPOS:
        if a in text and rnd.random() < 0.3:
            text = text.replace(a, b)
    return text


def generate(n_per_intent: int = 150, seed: int = 42) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for intent, phrases in BASE.items():
        seen = set()
        for _ in range(n_per_intent * 5):
            t = rnd.choice(PREFIX) + _typo(rnd.choice(phrases), rnd) + rnd.choice(SUFFIX)
            t = t.strip()
            if t in seen:
                continue
            seen.add(t)
            rows.append({"instruction": t, "intent": intent, "lang": "es-CL", "source": "plantillas_telco"})
            if len(seen) >= n_per_intent:
                break
    return pd.DataFrame(rows)
