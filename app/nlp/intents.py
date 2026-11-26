"""Catálogo de intenciones y política de resolución (Intent/Policy Router, EP2 §5.2).

Las 27 intenciones del dataset Bitext (nombres originales en inglés) se
conservan y se reinterpretan para el dominio telco; se agregan 8 intenciones
propias de TelcoAustral (saldo, plan, soporte técnico, portabilidad, etc.).

resolution:
  crm       -> requiere datos del cliente desde Siebel (operación asíncrona)
  rag       -> se responde con la base de conocimiento (RAG)
  handoff   -> se deriva a un agente humano
  smalltalk -> saludo / despedida
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    name: str
    category: str
    descripcion: str
    resolution: str
    crm_operation: str | None = None       # operación de lectura en Siebel
    crm_command: str | None = None         # operación de escritura (fire-and-forget)
    repetitive: bool = True                # consulta repetitiva automatizable (KPI 60%)
    examples: tuple = field(default_factory=tuple)


INTENTS: dict[str, Intent] = {i.name: i for i in [
    # ----------------- Bitext (27) -----------------
    Intent("create_account", "ACCOUNT", "Crear cuenta en la sucursal virtual", "rag"),
    Intent("delete_account", "ACCOUNT", "Dar de baja un servicio / cerrar cuenta", "handoff", repetitive=False),
    Intent("edit_account", "ACCOUNT", "Actualizar datos de contacto", "rag"),
    Intent("recover_password", "ACCOUNT", "Recuperar clave de la app o sucursal virtual", "rag"),
    Intent("registration_problems", "ACCOUNT", "Problemas para registrarse", "rag"),
    Intent("switch_account", "ACCOUNT", "Cambiar de cuenta / titular", "rag"),
    Intent("check_cancellation_fee", "CANCEL", "Costo por término anticipado de contrato", "rag"),
    Intent("contact_customer_service", "CONTACT", "Canales y horarios de atención", "rag"),
    Intent("contact_human_agent", "CONTACT", "Hablar con un ejecutivo", "handoff", repetitive=False),
    Intent("delivery_options", "DELIVERY", "Opciones de despacho de equipos y SIM", "rag"),
    Intent("delivery_period", "DELIVERY", "Plazos de despacho", "rag"),
    Intent("complaint", "FEEDBACK", "Reclamo formal", "handoff", crm_command="create_ticket", repetitive=False),
    Intent("review", "FEEDBACK", "Dejar una opinión del servicio", "rag"),
    Intent("check_invoice", "INVOICE", "Consultar boleta y monto a pagar", "crm", crm_operation="get_invoices"),
    Intent("get_invoice", "INVOICE", "Obtener copia de la boleta", "crm", crm_operation="get_invoices"),
    Intent("newsletter_subscription", "NEWSLETTER", "Suscripción a ofertas", "rag"),
    Intent("cancel_order", "ORDER", "Anular compra de equipo", "rag"),
    Intent("change_order", "ORDER", "Modificar compra de equipo", "rag"),
    Intent("place_order", "ORDER", "Comprar equipo o contratar servicio", "rag"),
    Intent("track_order", "ORDER", "Estado de pedido / instalación", "rag"),
    Intent("check_payment_methods", "PAYMENT", "Medios de pago disponibles", "rag"),
    Intent("payment_issue", "PAYMENT", "Problema con un pago", "rag", crm_command="create_ticket"),
    Intent("check_refund_policy", "REFUND", "Política de devoluciones", "rag"),
    Intent("get_refund", "REFUND", "Solicitar devolución de dinero", "rag", crm_command="create_ticket"),
    Intent("track_refund", "REFUND", "Estado de una devolución", "rag"),
    Intent("change_shipping_address", "SHIPPING", "Cambiar dirección de instalación/despacho", "rag"),
    Intent("set_up_shipping_address", "SHIPPING", "Registrar dirección de despacho", "rag"),
    # ----------------- TelcoAustral (propias) -----------------
    Intent("check_balance", "TELCO", "Consultar saldo, deuda o bolsa de datos", "crm", crm_operation="get_balance"),
    Intent("check_plan", "TELCO", "Ver el plan contratado", "crm", crm_operation="get_profile"),
    Intent("change_plan", "TELCO", "Cambiar o mejorar el plan", "rag"),
    Intent("technical_support", "TELCO", "Falla técnica: sin señal, internet lento, TV", "rag",
           crm_command="create_ticket"),
    Intent("portability", "TELCO", "Portabilidad numérica", "rag"),
    Intent("roaming", "TELCO", "Roaming internacional", "rag"),
    Intent("greeting", "SMALLTALK", "Saludo", "smalltalk", repetitive=False),
    Intent("goodbye", "SMALLTALK", "Despedida / agradecimiento", "smalltalk", repetitive=False),
]}

BITEXT_INTENTS = [n for n, i in INTENTS.items() if i.category not in {"TELCO", "SMALLTALK"}]


def get_intent(name: str) -> Intent:
    return INTENTS.get(name) or Intent(name, "UNKNOWN", name, "rag")
