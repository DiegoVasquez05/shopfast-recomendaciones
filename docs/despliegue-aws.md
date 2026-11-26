# Despliegue en AWS Academy (Learner Lab)

Arquitectura del piloto: **SageMaker** (NLU) + **S3** (datos y modelos) + **una EC2** con Docker Compose
(Nginx, backend ×2, Siebel worker ×2, simulador Siebel, dashboard, Redis, RabbitMQ y PostgreSQL+pgvector).

> Los créditos de AWS Academy son limitados: **elimina el endpoint de SageMaker** al terminar cada
> sesión de trabajo (`make delete-sm`) y detén la EC2.

## 1. Preparar credenciales

1. Iniciar el Learner Lab (**Start Lab**) y esperar el indicador verde.
2. En **AWS Details > AWS CLI**, copiar las credenciales en `~/.aws/credentials` (expiran cada ~4 h).
3. Región: `us-east-1`.

## 2. Entrenar con el dataset real

```bash
git clone https://github.com/DiegoVasquez05/shopfast-recomendaciones.git telecombot && cd telecombot
python -m venv .venv && source .venv/bin/activate && make install
# descargar Bitext según data/README.md -> data/bitext.csv
make train BITEXT=data/bitext.csv
cat models/metrics.json | head -60       # métricas para el informe
```

## 3. Desplegar el modelo en SageMaker

```bash
make deploy-sm BUCKET=telecombot-equipo<N>
```

El script:

1. Crea el bucket con cifrado SSE-KMS y acceso público bloqueado.
2. Sube el código de entrenamiento/inferencia y los CSV.
3. Lanza un **Training Job** (`ml.m5.large`, contenedor oficial scikit-learn 1.2-1). Las métricas
   `test_accuracy`, `test_precision`, `test_recall` y `test_f1` quedan en CloudWatch.
4. Crea el **endpoint** `telecombot-nlu` y configura **auto-scaling** de 1 a 2 instancias según
   invocaciones por instancia. AWS Academy prohíbe `ml.t2.medium`, así que el script prueba otros tipos
   (en nuestra cuenta quedó en `ml.m5.large`) y, si ninguno está permitido, usa Serverless Inference.
   Si el entrenamiento ya terminó, `make endpoint-sm BUCKET=...` recrea solo el endpoint.

Verificación:

```bash
make test-sm        # 200 invocaciones: accuracy, p50/p95/p99 de latencia
aws sagemaker-runtime invoke-endpoint --endpoint-name telecombot-nlu \
  --content-type application/json --body '{"text":"wena cuanto debo po"}' \
  --cli-binary-format raw-in-base64-out /dev/stdout
```

Evidencias para el informe: consola de SageMaker (Training job *Completed*, endpoint *InService*,
pestaña *Monitor*), y `python -m aws.cloudwatch_report --endpoint telecombot-nlu`.

## 4. Lanzar la EC2

| Parámetro | Valor |
|---|---|
| AMI | Amazon Linux 2023 |
| Tipo | `t3.large` (2 vCPU, 8 GB), la misma familia que en la EP2 |
| Disco | 30 GB gp3 |
| Perfil IAM | `LabInstanceProfile` (permite invocar SageMaker sin copiar llaves) |
| Security group | 22 (solo tu IP), 80 y 443 (0.0.0.0/0) |

```bash
ssh -i labsuser.pem ec2-user@<IP>
curl -fsSL https://raw.githubusercontent.com/DiegoVasquez05/shopfast-recomendaciones/main/infra/ec2-setup.sh | bash
cd ~/telecombot && nano .env       # secretos + SAGEMAKER_ENDPOINT=telecombot-nlu
# copiar models/ desde tu PC (scp -r models ec2-user@<IP>:~/telecombot/) o entrenar en la EC2
newgrp docker
docker compose up -d --build
docker compose --profile monitoring up -d      # Prometheus + Grafana (opcional)
docker compose ps
```

Con el perfil `LabInstanceProfile` no hace falta montar `~/.aws`: boto3 toma las credenciales del
rol de la instancia.

## 5. Verificar

```bash
curl -k https://<IP>/health
curl -k https://<IP>/ready        # dependencies.nlu debe decir "sagemaker"
```

- `https://<IP>/widget` · `https://<IP>/docs` · `https://<IP>/dashboard/` · `https://<IP>/agent`
- RabbitMQ y Grafana quedan solo en localhost. Para verlos, usar un túnel SSH:
  `ssh -i labsuser.pem -L 15672:localhost:15672 -L 3000:localhost:3000 ec2-user@<IP>`

## 6. Prueba de carga contra AWS

Desde tu PC u otra EC2 (no desde la misma máquina):

```bash
make load HOST=https://<IP> USERS=300
```

Durante la prueba, tomar capturas de CloudWatch (CPU de la EC2, invocaciones y `ModelLatency` del
endpoint) y del dashboard de Grafana.

## 7. Costos

**Piloto en AWS Academy (lo desplegado):**

| Recurso | Tipo | USD/hora | USD/mes 24×7 |
|---|---|---|---|
| EC2 | t3.large | 0,0832 | ~60,7 |
| Endpoint SageMaker | ml.m5.large (AWS Academy no permite ml.t2.medium) | 0,115 | ~84 |
| Training Job semanal | ml.m5.large, ~5 min | 0,115 | ~0,04 |
| S3 + CloudWatch | < 1 GB, logs básicos | — | ~3 |
| **Total** | | | **~148 USD/mes** |

**Producción (estimación de la EP2 con AWS Pricing Calculator):** USD 2.940,62/mes, dentro del
presupuesto del caso (USD 2.500–3.500). El piloto valida la arquitectura con ~5% de ese costo; los
mayores ítems en producción son ELB y RDS Multi-AZ.

Optimización: usar `ml.t2.medium` (el modelo pesa < 10 MB y responde en ~2 ms), escalar el endpoint a
1 instancia fuera de horario, cache de respuestas RAG frecuentes en Redis, y apagar recursos al terminar.

## 8. Limpieza

```bash
make delete-sm                      # endpoint y endpoint-config
docker compose down                 # en la EC2
# detener o terminar la EC2 desde la consola; vaciar el bucket si ya no se usa
```
