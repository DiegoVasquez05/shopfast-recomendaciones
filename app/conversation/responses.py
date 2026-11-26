"""Plantillas de respuesta en español chileno para datos del CRM y small talk."""
from __future__ import annotations


def clp(n) -> str:
    return "$" + f"{int(round(float(n))):,}".replace(",", ".")


def format_crm(operation: str, data: dict | None) -> str:
    if not data:
        return "No pude obtener tu información en este momento."
    name = data.get("nombre", "").split(" ")[0]
    hi = f"{name}, " if name else ""
    if operation == "get_balance":
        deuda = data.get("deuda", 0)
        txt = (f"{hi}tu deuda actual es de {clp(deuda)} con vencimiento el {data.get('vencimiento')}."
               if deuda else f"{hi}estás al día, no tienes deuda pendiente.")
        if data.get("datos_disponibles_gb") is not None:
            txt += f" Te quedan {data['datos_disponibles_gb']} GB de datos móviles de tu plan."
        return txt.strip()
    if operation == "get_invoices":
        inv = (data.get("boletas") or [None])[0]
        if not inv:
            return f"{hi}no encontré boletas emitidas en tu cuenta."
        return (f"{hi}tu última boleta ({inv['periodo']}) es por {clp(inv['monto'])}, vence el {inv['vencimiento']}"
                f" y está {inv['estado']}. Puedes descargarla aquí: {inv['url_pdf']}")
    if operation == "get_profile":
        s = ", ".join(f"{x['tipo']} {x['plan']}" for x in data.get("servicios", []))
        return f"{hi}tienes contratado: {s}." if s else f"{hi}no encontré servicios activos."
    return "Listo, ya tengo tu información."


SMALLTALK = {
    "greeting": "¡Hola! Soy TelecomBot de TelcoAustral 👋 Te puedo ayudar con tu saldo, boletas, planes, "
                "fallas técnicas y más. ¿En qué te ayudo?",
    "goodbye": "¡Gracias por escribirnos! Que tengas un buen día. Si necesitas algo más, aquí estaré.",
}

CLARIFY = ("No estoy seguro de haberte entendido 🤔. ¿Me lo puedes decir con otras palabras? "
           "Por ejemplo: \"cuánto debo\", \"no tengo internet\" o \"quiero cambiar mi plan\".")
HANDOFF = ("Te voy a derivar con un ejecutivo para que te ayude mejor. Ya le compartí el resumen de "
           "nuestra conversación, así no tienes que repetir nada. Tiempo de espera estimado: {wait}.")
PILOT_EXCLUDED = ("Gracias por escribir a TelcoAustral. En este momento tu atención será gestionada por "
                  "un ejecutivo. Tiempo de espera estimado: {wait}.")
NEED_ID = ("Para revisar esa información necesito validar tu identidad. Ingresa a la App TelcoAustral "
           "o escríbenos desde el WhatsApp asociado a tu línea.")
CRM_PENDING = ("Estoy consultando tu información en nuestros sistemas ⏳. Te la envío en un momento "
               "por este mismo canal.")
CRM_DEGRADED = ("Nuestro sistema de clientes está con intermitencia en este momento. Dejé tu consulta en cola y "
                "te enviaré la respuesta apenas se restablezca. Mientras tanto, puedes revisar tu saldo en la App.")
TICKET_CREATED = " Además, registré un caso en nuestro sistema (referencia {ref}) para hacerle seguimiento."
