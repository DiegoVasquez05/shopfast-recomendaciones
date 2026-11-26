"""Extrae métricas de CloudWatch del endpoint SageMaker y de la EC2 (evidencia para el informe).

    python -m aws.cloudwatch_report --endpoint telecombot-nlu --instance-id i-0abc123 --hours 3
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import boto3


def stat(cw, ns, metric, dims, start, end, stats, period=300):
    pts = cw.get_metric_statistics(Namespace=ns, MetricName=metric, Dimensions=dims, StartTime=start, EndTime=end,
                                   Period=period, Statistics=list(stats))["Datapoints"]
    if not pts:
        return None
    out = {}
    for s in stats:
        vals = [p[s] for p in pts]
        out[s] = round(max(vals) if s == "Maximum" else sum(vals) if s == "Sum" else sum(vals) / len(vals), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--endpoint", default="telecombot-nlu")
    ap.add_argument("--instance-id")
    ap.add_argument("--hours", type=int, default=3)
    a = ap.parse_args()
    cw = boto3.client("cloudwatch", region_name=a.region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=a.hours)
    ep = [{"Name": "EndpointName", "Value": a.endpoint}, {"Name": "VariantName", "Value": "AllTraffic"}]
    rep = {"ventana_utc": f"{start:%Y-%m-%d %H:%M} -> {end:%H:%M}", "sagemaker_endpoint": {
        "Invocations": stat(cw, "AWS/SageMaker", "Invocations", ep, start, end, ["Sum"]),
        "ModelLatency_microseg": stat(cw, "AWS/SageMaker", "ModelLatency", ep, start, end, ["Average", "Maximum"]),
        "OverheadLatency_microseg": stat(cw, "AWS/SageMaker", "OverheadLatency", ep, start, end, ["Average", "Maximum"]),
        "Invocation4XXErrors": stat(cw, "AWS/SageMaker", "Invocation4XXErrors", ep, start, end, ["Sum"]),
        "Invocation5XXErrors": stat(cw, "AWS/SageMaker", "Invocation5XXErrors", ep, start, end, ["Sum"]),
        "CPUUtilization": stat(cw, "/aws/sagemaker/Endpoints", "CPUUtilization", ep, start, end, ["Average", "Maximum"]),
        "MemoryUtilization": stat(cw, "/aws/sagemaker/Endpoints", "MemoryUtilization", ep, start, end,
                                  ["Average", "Maximum"]),
    }}
    if a.instance_id:
        d = [{"Name": "InstanceId", "Value": a.instance_id}]
        rep["ec2"] = {m: stat(cw, "AWS/EC2", m, d, start, end, ["Average", "Maximum"])
                      for m in ("CPUUtilization", "NetworkIn", "NetworkOut")}
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
