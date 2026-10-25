"""Log 5 seeded runs in MLflow and report mean ± SD across seeds.

Run:  PYTHONHASHSEED=0 python -m src.run_seeds
Writes results_summary.csv (committed) and mlflow.db (local, git-ignored).
"""
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from src.train import main

SEEDS = [0, 1, 2, 3, 42]
ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() or "unknown"


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("d05-repro-pipeline")
    commit, rows = git_commit(), []
    for seed in SEEDS:
        with mlflow.start_run(run_name=f"seed-{seed}"):
            r = main(seed)
            mlflow.log_params({"model": "logistic_regression", "scaler": "StandardScaler (in Pipeline)",
                               "cv": "StratifiedKFold(5, shuffle)", "seed": seed,
                               "git_commit": commit, "data_md5": r["data_md5"]})
            mlflow.log_metrics({"accuracy_mean": r["accuracy_mean"], "accuracy_sd": r["accuracy_sd"]})
            rows.append(r)
    table = pd.DataFrame(rows)[["seed", "accuracy_mean", "accuracy_sd"]]
    table.to_csv(ROOT / "results_summary.csv", index=False)
    print(table.to_string(index=False))
    m, s = table["accuracy_mean"].mean(), table["accuracy_mean"].std(ddof=1)
    print(f"\nAccuracy across 5 seeds: {m:.4f} ± {s:.4f} (mean ± sample SD, ddof=1)")
