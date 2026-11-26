# Guion de la demostración en vivo (≈ 7 minutos)

Antes de la presentación:

- [ ] Learner Lab iniciado, endpoint `telecombot-nlu` *InService*, EC2 encendida.
- [ ] `docker compose ps` con todos los servicios *Up*; `https://<IP>/ready` con `"nlu": "sagemaker"`.
- [ ] Pestañas abiertas: widget, dashboard, consola de agentes (conectada), Swagger, consola SageMaker.
- [ ] Postman con la colección `docs/postman_collection.json` y la variable `base_url` configurada.

## 1. Flujo completo en el Web Widget (entrada → procesamiento → modelo → resultado)

Con "ver detalle técnico" activado:

1. `wena po` → saludo.
2. `cuánto debo de la boleta?` → intención `check_invoice`, consulta a Siebel, monto y link de la boleta.
3. `el internet anda lento y el modem tiene luz roja` → `technical_support`: respuesta RAG
   (artículo KB-013) y caso creado en Siebel.
4. `cómo activo el roaming pa ir a perú` → RAG.

En el dashboard (pestaña *Probar el chatbot*) mostrar la explicación: top-3 intenciones, n-gramas que
pesaron y documentos recuperados.

## 2. Consumo de la API (Postman o curl)

1. `POST /oauth/token` → token.
2. `POST /api/v1/nlu/predict` con `"no tengo señal en el celu po"`: el modelo de SageMaker predice
   (`"source": "sagemaker"`).
3. `POST /api/v1/chat/messages` → respuesta completa con `correlation_id` y `latency_ms`.
4. Mostrar `/docs` (Swagger) brevemente.

## 3. Handoff a un humano

1. En el widget: `esto es una estafa, voy a reclamar al sernac`.
2. La consola de agentes recibe el caso al instante, con el resumen y el sentimiento.
3. El agente responde y el cliente lo ve en el widget.

## 4. Resiliencia: Siebel se cae

1. Dashboard > *Operación y resiliencia* > **Siebel caído**.
2. En el widget, preguntar el saldo varias veces: tras 5 fallas el circuito pasa a **OPEN** y el bot
   responde en modo degradado, sin esperar timeouts.
3. **Siebel normal**: a los 30 s el circuito pasa a semiabierto y luego a cerrado, y las consultas
   encoladas se entregan.

## 5. Métricas

- Dashboard > *KPIs*: tasa de automatización, handoff, P95.
- Dashboard > *Calidad del modelo*: accuracy, precision, recall y F1, más la curva de umbral.
- CloudWatch: invocaciones y `ModelLatency` del endpoint.

## Preguntas probables

- **¿Por qué SageMaker y no solo Bedrock?** El NLU es un modelo propio entrenado con Bitext + español
  chileno; SageMaker lo sirve con auto-scaling y métricas a ~USD 41/mes. Bedrock se usa solo para redactar.
- **¿Qué pasa si SageMaker cae?** El backend usa el modelo local como respaldo (`local-fallback`).
- **¿Cómo se cumple el SLA de 3 s si Siebel es lento?** Espera acotada de 1,5 s; si no llega, se
  responde "pendiente" y se notifica después. El Circuit Breaker evita esperar timeouts cuando Siebel está caído.
- **¿Cómo se protege la PII?** Enmascarado antes de loguear o guardar, cifrado en reposo, seudónimos
  y endpoints de acceso/supresión.
- **¿Cómo se llega a 1.000 conversaciones?** Backend sin estado (sesión en Redis), HPA de 3 a 10 réplicas,
  workers independientes y la prueba de carga como evidencia.
