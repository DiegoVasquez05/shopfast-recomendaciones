"""Entrena y despliega el clasificador de intenciones en AWS SageMaker (pensado para AWS Academy Learner Lab).

    python -m aws.sagemaker_deploy deploy --bucket telecombot-<tu-nombre>    # training job + endpoint + auto-scaling
    python -m aws.sagemaker_deploy test                                       # invoca el endpoint y mide latencia/accuracy
    python -m aws.sagemaker_deploy delete                                     # ¡eliminar al terminar para no gastar créditos!

Pasos de `deploy`:
  1. Empaqueta aws/sagemaker_src + normalizador + definición del modelo (sourcedir.tar.gz) y lo sube a S3.
  2. Sube models/sagemaker/{train,test}.csv (generados por `python -m training.train`) a S3 (cifrado SSE-KMS).
  3. Training Job con el contenedor oficial scikit-learn 1.2-1 (ml.m5.large), métricas a CloudWatch.
  4. Model + EndpointConfig + Endpoint (ml.t2.medium). Si el endpoint existe se actualiza sin downtime.
  5. Auto-scaling 1..N instancias por invocaciones/instancia (target tracking).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
SKLEARN_ACCOUNTS = {"us-east-1": "683313688378", "us-east-2": "257758044811", "us-west-1": "746614075791",
                    "us-west-2": "246618743249"}
METRICS = [{"Name": n, "Regex": f"{n}=([0-9\\.]+)"} for n in ("test_accuracy", "test_precision", "test_recall",
                                                              "test_f1")]


def image_uri(region: str) -> str:
    if region not in SKLEARN_ACCOUNTS:
        raise SystemExit(f"Región {region} no configurada; usa --image-uri")
    return f"{SKLEARN_ACCOUNTS[region]}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def sourcedir_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in [ROOT / "aws/sagemaker_src/train.py", ROOT / "aws/sagemaker_src/inference.py",
                  ROOT / "app/nlp/normalizer_cl.py", ROOT / "app/nlp/model_def.py"]:
            tar.add(f, arcname=f.name)
    return buf.getvalue()


def wait(describe, key, ok, bad, every=30):
    while True:
        status = describe()[key]
        print(f"   {time.strftime('%H:%M:%S')} estado: {status}")
        if status in ok:
            return
        if status in bad:
            raise SystemExit(f"Terminó con estado {status}: {describe().get('FailureReason', '')}")
        time.sleep(every)


ENDPOINT_FALLBACKS = ["ml.t2.medium", "ml.m5.large", "ml.c5.large", "ml.t2.large", "ml.m4.xlarge"]


def last_training_job(sm) -> str:
    jobs = sm.list_training_jobs(NameContains="telecombot-nlu", StatusEquals="Completed", SortBy="CreationTime",
                                 SortOrder="Descending", MaxResults=1)["TrainingJobSummaries"]
    if not jobs:
        raise SystemExit("No hay Training Jobs completados: ejecuta deploy sin --reuse-training")
    return jobs[0]["TrainingJobName"]


def train_job(a, sm, s3, role, image, stamp) -> str:
    prefix = f"{a.prefix}/{stamp}"
    print(f"[1/5] Bucket s3://{a.bucket} (cifrado SSE-KMS)")
    try:
        s3.head_bucket(Bucket=a.bucket)
    except Exception:
        kw = {} if a.region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": a.region}}
        s3.create_bucket(Bucket=a.bucket, **kw)
    s3.put_bucket_encryption(Bucket=a.bucket, ServerSideEncryptionConfiguration={"Rules": [
        {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}}]})
    s3.put_public_access_block(Bucket=a.bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})

    print("[2/5] Subiendo código y datos")
    src_key = f"{prefix}/code/sourcedir.tar.gz"
    s3.put_object(Bucket=a.bucket, Key=src_key, Body=sourcedir_tar())
    data_dir = ROOT / a.models / "sagemaker"
    for ch in ("train", "test"):
        s3.upload_file(str(data_dir / f"{ch}.csv"), a.bucket, f"{prefix}/data/{ch}/{ch}.csv")
    src_uri = f"s3://{a.bucket}/{src_key}"

    job = f"telecombot-nlu-{stamp}"
    print(f"[3/5] Training Job {job} ({a.train_instance})")
    channel = lambda ch: {"ChannelName": ch, "ContentType": "text/csv", "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": f"s3://{a.bucket}/{prefix}/data/{ch}/",
        "S3DataDistributionType": "FullyReplicated"}}}
    sm.create_training_job(
        TrainingJobName=job, RoleArn=role,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File", "MetricDefinitions": METRICS},
        HyperParameters={"sagemaker_program": json.dumps("train.py"),
                         "sagemaker_submit_directory": json.dumps(src_uri),
                         "sagemaker_region": json.dumps(a.region)},
        InputDataConfig=[channel("train"), channel("test")],
        OutputDataConfig={"S3OutputPath": f"s3://{a.bucket}/{prefix}/output"},
        ResourceConfig={"InstanceType": a.train_instance, "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        Tags=[{"Key": "proyecto", "Value": "telecombot"}])
    wait(lambda: sm.describe_training_job(TrainingJobName=job), "TrainingJobStatus", {"Completed"},
         {"Failed", "Stopped"})
    return job


def deploy(a) -> None:
    from botocore.exceptions import ClientError

    sess = boto3.Session(region_name=a.region)
    sm, s3 = sess.client("sagemaker"), sess.client("s3")
    acct = sess.client("sts").get_caller_identity()["Account"]
    role = a.role or f"arn:aws:iam::{acct}:role/LabRole"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    image = a.image_uri or image_uri(a.region)

    if a.reuse_training:
        job = last_training_job(sm)
        print(f"[1-3/5] Reutilizando el Training Job {job}")
    else:
        job = train_job(a, sm, s3, role, image, stamp)
    desc = sm.describe_training_job(TrainingJobName=job)
    print("   métricas:", {m["MetricName"]: round(m["Value"], 4) for m in desc.get("FinalMetricDataList", [])})
    model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
    src_uri = json.loads(desc["HyperParameters"]["sagemaker_submit_directory"])

    print(f"[4/5] Endpoint {a.endpoint}")
    model_name = f"telecombot-nlu-{stamp}"
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer={
        "Image": image, "ModelDataUrl": model_data,
        "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": src_uri,
                        "SAGEMAKER_REGION": a.region, "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}})
    # AWS Academy restringe algunos tipos de instancia: se prueban en orden y, si ninguno está
    # permitido, se usa SageMaker Serverless Inference (escala automáticamente, se paga por uso).
    candidates = [a.endpoint_instance] + [t for t in ENDPOINT_FALLBACKS if t != a.endpoint_instance] + ["serverless"]
    chosen = None
    for i, itype in enumerate(candidates):
        cfg = f"telecombot-nlu-cfg-{stamp}-{i}"
        variant = {"VariantName": "AllTraffic", "ModelName": model_name}
        if itype == "serverless":
            variant["ServerlessConfig"] = {"MemorySizeInMB": 2048, "MaxConcurrency": 10}
        else:
            variant.update({"InitialInstanceCount": 1, "InstanceType": itype, "InitialVariantWeight": 1.0})
        try:
            sm.create_endpoint_config(EndpointConfigName=cfg, ProductionVariants=[variant])
            try:
                sm.describe_endpoint(EndpointName=a.endpoint)
                sm.update_endpoint(EndpointName=a.endpoint, EndpointConfigName=cfg)  # blue/green gestionado
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "ValidationException":
                    raise
                sm.create_endpoint(EndpointName=a.endpoint, EndpointConfigName=cfg)
            chosen = itype
            print(f"   tipo de endpoint: {itype}")
            break
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code not in ("AccessDeniedException", "ValidationException", "ResourceLimitExceeded"):
                raise
            print(f"   {itype} no permitido en esta cuenta ({code}); probando el siguiente")
            try:
                sm.delete_endpoint_config(EndpointConfigName=cfg)
            except ClientError:
                pass
    if chosen is None:
        raise SystemExit("Ningún tipo de endpoint está permitido en esta cuenta")
    wait(lambda: sm.describe_endpoint(EndpointName=a.endpoint), "EndpointStatus", {"InService"}, {"Failed"})

    print("[5/5] Auto-scaling")
    if chosen == "serverless":
        print("   Serverless: SageMaker escala automáticamente hasta 10 invocaciones concurrentes")
    else:
        try:
            aas = sess.client("application-autoscaling")
            rid = f"endpoint/{a.endpoint}/variant/AllTraffic"
            aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=rid,
                                         ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                                         MinCapacity=1, MaxCapacity=a.max_instances)
            aas.put_scaling_policy(
                PolicyName="telecombot-invocaciones", ServiceNamespace="sagemaker", ResourceId=rid,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount", PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": float(a.target_invocations), "ScaleInCooldown": 300, "ScaleOutCooldown": 60,
                    "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"}})
            print(f"   1..{a.max_instances} instancias, objetivo {a.target_invocations} invocaciones/min por instancia")
        except Exception as exc:
            print(f"   [aviso] no se pudo configurar auto-scaling (¿restricción del Learner Lab?): {exc}")
    print(f"\nListo. Agrega en .env:  SAGEMAKER_ENDPOINT={a.endpoint}")


def test(a) -> None:
    import pandas as pd

    rt = boto3.Session(region_name=a.region).client("sagemaker-runtime")
    df = pd.read_csv(ROOT / a.models / "sagemaker" / "test.csv").sample(n=a.n, random_state=0)
    lat, ok = [], 0
    for text, intent in zip(df["text"], df["intent"]):
        t = time.perf_counter()
        r = rt.invoke_endpoint(EndpointName=a.endpoint, ContentType="application/json", Accept="application/json",
                               Body=json.dumps({"text": text}))
        lat.append((time.perf_counter() - t) * 1000)
        ok += json.loads(r["Body"].read())["predictions"][0]["intent"] == intent
    lat.sort()
    p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)
    print(json.dumps({"invocaciones": len(lat), "accuracy": round(ok / len(lat), 4), "p50_ms": p(.5),
                      "p95_ms": p(.95), "p99_ms": p(.99)}, indent=2))


def delete(a) -> None:
    sm = boto3.Session(region_name=a.region).client("sagemaker")
    cfg = sm.describe_endpoint(EndpointName=a.endpoint)["EndpointConfigName"]
    sm.delete_endpoint(EndpointName=a.endpoint)
    sm.delete_endpoint_config(EndpointConfigName=cfg)
    print(f"Endpoint {a.endpoint} eliminado (modelos y datos en S3 se conservan)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Despliegue del NLU en AWS SageMaker")
    ap.add_argument("action", choices=["deploy", "test", "delete"])
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--bucket")
    ap.add_argument("--prefix", default="telecombot")
    ap.add_argument("--models", default="models")
    ap.add_argument("--endpoint", default="telecombot-nlu")
    ap.add_argument("--role", default=None, help="ARN del rol de ejecución (por defecto LabRole)")
    ap.add_argument("--image-uri", default=None)
    ap.add_argument("--train-instance", default="ml.m5.large")
    ap.add_argument("--endpoint-instance", default="ml.t2.medium")
    ap.add_argument("--max-instances", type=int, default=2)
    ap.add_argument("--target-invocations", type=int, default=200)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--reuse-training", action="store_true",
                    help="no entrena de nuevo: usa el último Training Job completado")
    a = ap.parse_args()
    if a.action == "deploy" and not a.bucket:
        ap.error("--bucket es obligatorio para deploy")
    {"deploy": deploy, "test": test, "delete": delete}[a.action](a)


if __name__ == "__main__":
    main()
