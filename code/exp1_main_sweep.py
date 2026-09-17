"""Experiment 1: main parametric sweep.

Produces exp1_main.csv, which drives Table I and Figures 1 and 2.

Design (fully crossed):
    18 synthetic models, number of concepts stratified over 5 ... 20
    2 activation functions   x 3 matrix norms
    3 values of t_alpha      x 5 values of rho
  = 1620 runs.

The number of concepts is assigned systematically rather than drawn at
random, so it is not confounded with the model identity.
"""

import csv
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_qfcm import (make_inputs, make_model, sign_agreement, transform)

SEED = 42
N_MODELS = 18
T = 20
PHI = 0.8
K = 100

ACTS = ["sigmoid", "tanh"]
METRICS = ["frobenius", "nuclear", "spectral"]
# t_alpha = 1 evaluates the whole trace, T // 2 the second half, T the last state.
TALPHAS = [("all", 1), ("half", T // 2), ("last", T)]
RHOS = [1.0, 10.0, 100.0, 1000.0, 10000.0]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main(m0=0, m1=N_MODELS):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exp1_main.csv")
    fields = [
        "model_id", "n_concepts", "activation", "metric", "steps_mode",
        "t_alpha", "rho", "divergence", "divergence_abs", "error",
        "sign_agreement", "n_iter", "seconds", "success",
    ]

    total = (m1 - m0) * len(ACTS) * len(METRICS) * len(TALPHAS) * len(RHOS)
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        pbar = tqdm(total=total, desc=f"exp1[{m0}:{m1}]", mininterval=10)

        for mid in range(m0, m1):
            n = 5 + (mid % 16)               # stratified over 5 ... 20
            W = make_model(n, seed=SEED + mid)
            X = make_inputs(K, n, seed=SEED + 10000 + mid)

            for act in ACTS:
                for metric in METRICS:
                    for label, ta in TALPHAS:
                        for rho in RHOS:
                            r = transform(
                                W, X, T=T, t_alpha=ta, rho=rho, metric=metric,
                                act=act, phi=PHI, seed=SEED + mid, maxiter=1000,
                            )
                            writer.writerow({
                                "model_id": mid,
                                "n_concepts": n,
                                "activation": act,
                                "metric": metric,
                                "steps_mode": label,
                                "t_alpha": ta,
                                "rho": rho,
                                "divergence": f"{r.divergence:.6f}",
                                "divergence_abs": f"{r.divergence_abs:.6f}",
                                "error": f"{r.error:.6e}",
                                "sign_agreement": f"{sign_agreement(W, r.Wp):.6f}",
                                "n_iter": r.n_iter,
                                "seconds": f"{r.seconds:.4f}",
                                "success": int(r.success),
                            })
                            fh.flush()
                            pbar.update(1)
        pbar.close()
    print("written", path)


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else N_MODELS
    main(a, b)
