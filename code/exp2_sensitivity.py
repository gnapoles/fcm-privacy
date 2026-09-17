"""Experiment 2: sensitivity analyses.

(a) t_alpha varied explicitly over its whole admissible range, which the first
    submission only probed through the coarse last / half / all proxy.
(b) phi varied over [0.2, 1.0], previously fixed at 0.8.
(c) the loss trade-off parameter rho, including its two degenerate endpoints.
    Because the objective has two terms whose relative weight is already a
    continuous parameter, removing a term is not an independent design choice
    but the limiting case of rho.  The endpoints are therefore reported inside
    the sensitivity curve rather than as a separate component ablation.
"""

import csv
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_qfcm import (make_inputs, make_model, sign_agreement,
                          structural_divergence, transform)

SEED = 42
N_MODELS = 18
T = 20
PHI = 0.8
K = 100
ACTS = ["sigmoid", "tanh"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# Optional model range, so a long sweep can be split across sessions.
M0 = int(os.environ.get("M0", 0))
M1 = int(os.environ.get("M1", N_MODELS))
MODE = "a" if M0 > 0 else "w"


def models():
    for mid in range(M0, M1):
        n = 5 + (mid % 16)   # stratified over 5 ... 20, as in exp1
        yield mid, n, make_model(n, seed=SEED + mid), make_inputs(K, n, seed=SEED + 10000 + mid)


def run_talpha():
    path = os.path.join(OUT, "exp2_talpha.csv")
    fields = ["model_id", "n_concepts", "activation", "t_alpha", "rho",
              "divergence", "error", "sign_agreement"]
    talphas = [1, 3, 5, 10, 15, 20]
    rhos = [10.0, 1000.0]
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=(M1 - M0) * len(ACTS) * len(talphas) * len(rhos), desc="t_alpha")
        for mid, n, W, X in models():
            for act in ACTS:
                for ta in talphas:
                    for rho in rhos:
                        r = transform(W, X, T=T, t_alpha=ta, rho=rho,
                                      metric="frobenius", act=act, phi=PHI,
                                      seed=SEED + mid, maxiter=1000)
                        w.writerow({
                            "model_id": mid, "n_concepts": n, "activation": act,
                            "t_alpha": ta, "rho": rho,
                            "divergence": f"{r.divergence:.6f}",
                            "error": f"{r.error:.6e}",
                            "sign_agreement": f"{sign_agreement(W, r.Wp):.6f}",
                        })
                        fh.flush()
                        pbar.update(1)
        pbar.close()
    print("written", path)


def run_phi():
    path = os.path.join(OUT, "exp2_phi.csv")
    fields = ["model_id", "n_concepts", "activation", "phi", "rho",
              "divergence", "error", "sign_agreement"]
    phis = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    rhos = [10.0, 1000.0]
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=(M1 - M0) * len(ACTS) * len(phis) * len(rhos), desc="phi")
        for mid, n, W, X in models():
            for act in ACTS:
                for phi in phis:
                    for rho in rhos:
                        r = transform(W, X, T=T, t_alpha=1, rho=rho,
                                      metric="frobenius", act=act, phi=phi,
                                      seed=SEED + mid, maxiter=1000)
                        w.writerow({
                            "model_id": mid, "n_concepts": n, "activation": act,
                            "phi": phi, "rho": rho,
                            "divergence": f"{r.divergence:.6f}",
                            "error": f"{r.error:.6e}",
                            "sign_agreement": f"{sign_agreement(W, r.Wp):.6f}",
                        })
                        fh.flush()
                        pbar.update(1)
        pbar.close()
    print("written", path)


def run_rho_endpoints():
    """rho = 0 (dissimilarity only) and the fidelity-only limit."""
    path = os.path.join(OUT, "exp2_endpoints.csv")
    fields = ["model_id", "n_concepts", "activation", "variant",
              "divergence", "error", "sign_agreement"]
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=(M1 - M0) * len(ACTS) * 2, desc="endpoints")
        for mid, n, W, X in models():
            for act in ACTS:
                for variant in ["dissimilarity-only", "fidelity-only"]:
                    if variant == "dissimilarity-only":
                        r = transform(W, X, T=T, t_alpha=1, rho=0.0,
                                      metric="frobenius", act=act, phi=PHI,
                                      seed=SEED + mid, maxiter=1000)
                    else:
                        r = transform(W, X, T=T, t_alpha=1, rho=1.0,
                                      metric="frobenius", act=act, phi=PHI,
                                      seed=SEED + mid, maxiter=1000,
                                      fidelity_only=True)
                    w.writerow({
                        "model_id": mid, "n_concepts": n, "activation": act,
                        "variant": variant,
                        "divergence": f"{r.divergence:.6f}",
                        "error": f"{r.error:.6e}",
                        "sign_agreement": f"{sign_agreement(W, r.Wp):.6f}",
                    })
                    fh.flush()
                    pbar.update(1)
        pbar.close()
    print("written", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    import sys as _s
    w = _s.argv[1] if len(_s.argv) > 1 else "all"
    if w in ("all", "talpha"): run_talpha()
    if w in ("all", "phi"): run_phi()
    if w in ("all", "endpoints"): run_rho_endpoints()
