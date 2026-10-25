"""Train and evaluate a model reproducibly.

Run:  PYTHONHASHSEED=0 python -m src.train --seed 42
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main(seed: int = 42, data_path: Path = DATA) -> dict:
    set_seed(seed)
    df = pd.read_csv(data_path)
    X, y = df.drop(columns="target").values, df["target"].values
    # Scaling lives INSIDE the pipeline: in each CV fold the scaler is fit on the
    # training folds only, so no information from the test fold leaks into training.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=1)
    return {
        "seed": seed,
        "data_md5": hashlib.md5(Path(data_path).read_bytes()).hexdigest(),
        "accuracy_mean": round(float(scores.mean()), 5),
        "accuracy_sd": round(float(scores.std(ddof=1)), 5),  # sample SD across the 5 folds
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    print(json.dumps(main(ap.parse_args().seed)))
