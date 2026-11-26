"""Resume el CSV de Locust (p50/p95/p99, throughput, tasa de errores) como tabla Markdown para el informe.

    python loadtest/summarize.py loadtest/results/carga_stats.csv
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
cols = {"Name": "Endpoint", "Request Count": "Requests", "Failure Count": "Errores", "50%": "p50 (ms)",
        "95%": "p95 (ms)", "99%": "p99 (ms)", "Requests/s": "RPS"}
out = df[list(cols)].rename(columns=cols)
out["Error %"] = (out["Errores"] / out["Requests"].clip(lower=1) * 100).round(2)
out["RPS"] = out["RPS"].round(2)
print(out.to_markdown(index=False))
agg = df[df["Name"] == "Aggregated"].iloc[0]
print(f"\nSLA p95 <= 3000 ms: {'CUMPLE' if agg['95%'] <= 3000 else 'NO CUMPLE'} (p95 = {agg['95%']} ms)")
print(f"Tasa de error: {agg['Failure Count'] / max(agg['Request Count'], 1) * 100:.2f}%")
