"""Checks the three analytical claims the manuscript states without a table.

(a) Section IV-D, the analytic gradient of Eq. (12) against finite differences.
(b) Section IV-A, the closed-form recovery of Proposition 1.
(c) Section V-I, the transcription of the Lissos interconnection matrix.

"""

import csv
import os
import sys

import numpy as np
from scipy.optimize import approx_fprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lissos_data as LD
from privacy_qfcm import (ACTIVATIONS, loss_and_grad, make_inputs, make_model,
                          qfcm_states)

SEED = 42
T = 20
PHI = 0.8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

STEP = 1e-7          # forward-difference step
ACTS = ["sigmoid", "tanh"]
METRICS = ["frobenius", "nuclear", "spectral"]


def gradient_check(N=7, K=20, rho=100.0):
    """Maximum relative error between Eq. (12) and forward differences."""
    # An independent stream, since a start placed exactly at W would sit on a
    # point where both gradient terms vanish and the check would be vacuous.
    rng = np.random.default_rng([SEED, 20260724])
    W = make_model(N, seed=SEED)
    X = make_inputs(K, N, seed=SEED + 1)
    W0 = rng.uniform(-1.0, 1.0, (N, N))
    np.fill_diagonal(W0, 0.0)
    free = (~np.eye(N, dtype=bool)).ravel()

    out = []
    for act in ACTS:
        for metric in METRICS:
            def f(v):
                return loss_and_grad(v, W, X, T, 1, rho, metric, act, PHI)[0]
            g = loss_and_grad(W0.ravel(), W, X, T, 1, rho, metric, act, PHI)[1]
            gn = approx_fprime(W0.ravel(), f, STEP)
            assert np.max(np.abs(g[free])) > 0, "degenerate test point"
            rel = np.max(np.abs(g[free] - gn[free])) / np.max(np.abs(gn[free]))
            out.append({"claim": "gradient", "activation": act, "detail": metric,
                        "value": f"{rel:.3e}"})
    return out


def inverse(act, y):
    if act == "sigmoid":
        return np.log(y / (1.0 - y))
    return np.arctanh(y)


def proposition_check(sizes=(5, 10, 20)):
    """Exact recovery of W from one pre-t_alpha transition, Eq. (3)."""
    out = []
    for act in ACTS:
        for N in sizes:
            W = make_model(N, seed=SEED)
            X = make_inputs(3 * N, N, seed=SEED + 2)
            S = qfcm_states(W, X, 1, act, PHI)
            A0, A1 = S[0], S[1]
            B = (A1 - (1.0 - PHI) * A0) / PHI
            What = np.linalg.pinv(A0) @ inverse(act, B)
            out.append({"claim": "proposition1", "activation": act,
                        "detail": f"N={N}",
                        "value": f"{np.linalg.norm(W - What, 'fro'):.3e}"})
    return out


def transcription_check():
    W = LD.weight_matrix()
    od = np.max(np.abs(LD.out_degree(W) - LD.PUBLISHED_OD))
    idg = np.max(np.abs(LD.in_degree(W) - LD.PUBLISHED_ID))
    return [{"claim": "transcription", "activation": "-", "detail": "max centrality error",
             "value": f"{max(od, idg):.3e}"},
            {"claim": "transcription", "activation": "-", "detail": "causal links",
             "value": str(int(np.sum(W != 0)))}]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = gradient_check() + proposition_check() + transcription_check()
    with open(os.path.join(OUT, "verify_claims.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "activation", "detail", "value"])
        w.writeheader()
        w.writerows(rows)

    grad = [float(r["value"]) for r in rows if r["claim"] == "gradient"]
    prop = [float(r["value"]) for r in rows if r["claim"] == "proposition1"]
    print(f"gradient      max relative error {max(grad):.1e}")
    print(f"proposition 1 max Frobenius error {max(prop):.1e}")
    for r in rows:
        if r["claim"] == "transcription":
            print(f"transcription {r['detail']} = {r['value']}")


if __name__ == "__main__":
    main()
