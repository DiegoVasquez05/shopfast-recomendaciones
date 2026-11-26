# Pruebas y calidad

```bash
make test                                   # 20 casos (pytest)
make load HOST=https://<IP> USERS=300       # carga con Locust + resumen p50/p95/p99
```

## 1. Casos de prueba

| ID | Tipo | Qué verifica | Archivo |
|---|---|---|---|
| TC01 | Unitaria | Normalización de modismos chilenos, letras repetidas e insultos | `tests/test_nlp.py` |
| TC02 | Unitaria | Enmascarado de RUT, correo y teléfono en texto y en logs | `tests/test_nlp.py` |
| TC03 | Unitaria | Cifrado/descifrado de mensajes en reposo | `tests/test_nlp.py` |
| TC04 | Unitaria | Sentimiento: cliente molesto vs. satisfecho | `tests/test_nlp.py` |
| TC05 | Modelo | Accuracy/precision/recall/F1 en rango, accuracy ≥ 80% en español chileno manual, Recall@3 del RAG ≥ 85%, p95 de inferencia < 100 ms | `tests/test_nlp.py` |
| TC06 | Seguridad | Sin token → 401, secreto inválido → 401, scope insuficiente → 403, JWT falso → 401 | `tests/test_api.py` |
| TC07 | Funcional | `/nlu/predict` devuelve intención, top-3 y explicación | `tests/test_api.py` |
| TC08 | Funcional | El RAG recupera el artículo correcto ("router con luz roja" → KB-013) | `tests/test_api.py` |
| TC09 | Integración | Consulta de saldo end-to-end con Siebel; cliente no identificado pide validación | `tests/test_api.py` |
| TC10 | Resiliencia | Siebel caído: tras 3 fallas se abre el Circuit Breaker y se responde en modo degradado sin llamar a Siebel | `tests/test_api.py` |
| TC11 | Integración | Siebel lento: respuesta "pendiente" y resultado entregado luego como notificación | `tests/test_api.py` |
| TC12 | Resiliencia | Escritura fallida: 3 intentos con backoff y luego Dead-Letter Queue | `tests/test_api.py` |
| TC13 | Integración | Reclamo → handoff → evento WebSocket al agente → respuesta al cliente → caso en Siebel | `tests/test_api.py` |
| TC14 | Funcional | Dos turnos con baja confianza → derivación | `tests/test_api.py` |
| TC15 | Funcional | Piloto: ~10% de usuarios (8–12% en 5.000 IDs); usuario de control va al canal tradicional | `tests/test_api.py` |
| TC16 | Interoperabilidad | Webhook WhatsApp: verificación, firma inválida → 401, respuesta al número correcto | `tests/test_api.py` |
| TC17 | Seguridad | Rate limit por IP (OAuth) y por token del widget → 429 | `tests/test_api.py` |
| TC18 | Cumplimiento | Mensajes cifrados en BD, exportación con PII enmascarada y derecho al olvido | `tests/test_api.py` |
| TC19 | Rendimiento | p95 de 50 turnos < 3 s | `tests/test_api.py` |
| TC20 | Documentación | OpenAPI expone los endpoints principales y los KPIs se calculan | `tests/test_api.py` |

Resultado actual: **20/20 aprobadas** (también corren en GitHub Actions en cada push).

## 2. Métricas del modelo

`python -m training.train` genera `models/metrics.json` con:

- accuracy, precision, recall y F1 (macro y ponderado) en tres conjuntos: test de Bitext (15%
  estratificado), test del corpus es-CL y el **set manual en español chileno** (67 frases escritas a mano);
- F1 por intención, confusiones más frecuentes y errores del set manual;
- curva **cobertura vs. accuracy por umbral de confianza**, que justifica el umbral de handoff (0,45);
- Hit@1, Recall@3 y MRR del RAG, con y sin la intención detectada;
- latencia de inferencia local.

> Completar esta tabla con los valores de `models/metrics.json` del entrenamiento con el dataset real.

| Conjunto | n | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| Bitext test | | | | | |
| es-CL manual | 67 | | | | |

Referencia con la muestra sintética: el set manual en español chileno obtuvo accuracy 0,955 y
F1 macro 0,965. El RAG llegó a Recall@3 de 0,82 solo con búsqueda híbrida y de 1,00 sumando la
intención detectada (MRR 0,93). La inferencia local tarda 1,2 ms (p50) y 1,6 ms (p95).

## 3. Pruebas de carga (Locust)

Perfil: cada usuario virtual mantiene una conversación con pausas de 3 a 8 s (lectura/escritura) y una
mezcla realista de consultas: saldo 25%, información 25%, soporte técnico 20%, boleta 15%, cierre 10%
y ejecutivo 5%.

**Ejecución local de referencia** (1 contenedor backend con 2 workers, Redis, Siebel simulado con
80 ms de latencia, 4 vCPU):

| Usuarios | Requests | p50 | p95 | p99 | Throughput | Errores |
|---|---|---|---|---|---|---|
| 300 | 4.205 | 93 ms | 130 ms | 240 ms | 46,7 req/s | 0% en chat |
| 500 | 10.590 | 96 ms | 210 ms | 360 ms | 88,9 req/s | 0,03% |

Las consultas que pasan por Siebel (saldo, boleta) dominan la latencia (~100 ms, por la latencia del
CRM); las de RAG y small talk responden en 15-20 ms (p50).

> Repetir contra la EC2 (`make load HOST=https://<IP> USERS=300`) y pegar la tabla que imprime
> `loadtest/summarize.py`, junto con capturas de CloudWatch durante la prueba.

## 4. Oportunidades de optimización

1. **Cache de respuestas RAG** por intención + texto normalizado en Redis: las preguntas frecuentes
   son muy repetitivas y se evitaría el retrieval y la llamada al LLM.
2. **Prefetch del perfil de Siebel** al iniciar sesión de un cliente identificado: la primera consulta
   de saldo se respondería desde Redis.
3. **Calibrar el umbral** de confianza con los datos del piloto del 10% (curva cobertura/accuracy).
4. **Reentrenamiento** semanal con las conversaciones mal clasificadas que corrigen los agentes
   (aprendizaje activo).
5. **Escalado del worker por profundidad de cola** (KEDA con SQS) en vez de CPU.
6. **Embeddings más ricos** (Amazon Titan Embeddings) si crece el corpus del RAG, sin cambiar el retriever.
