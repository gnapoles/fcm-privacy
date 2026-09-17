"""Experiments 3 to 5.

Experiment 3  computational overhead and scalability in the number of
              concepts (latency, peak memory, iterations).
Experiment 4  comparison against three structural baselines, including two
              that do not preserve the spectrum of the original matrix.
Experiment 5  adversarial reconstruction of W under the two observation
              models of Section V-G.  The released-trace adversary sees the
              trajectories of W', which is what a deployment running the
              transformed model exposes.  The original-trace adversary sees the
              trajectories of W itself from t_alpha onward.
"""

import csv
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_qfcm import (baseline_edge_rewiring, baseline_random_perturbation,
                          baseline_similarity, edge_recovery, evaluate,
                          make_inputs, make_model, reconstruction_attack,
                          sign_agreement, structural_divergence, transform)

SEED = 42
T = 20
PHI = 0.8
K = 100
M0 = int(os.environ.get("M0", 0))   # optional model range, so a long sweep
M1 = int(os.environ.get("M1", 99))  # can be split across sessions
MODE = "a" if M0 > 0 else "w"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ---------------------------------------------------------------------------
def run_overhead():
    path = os.path.join(OUT, "exp3_overhead.csv")
    fields = ["n_concepts", "rep", "activation", "seconds", "peak_mib",
              "n_iter", "n_fev", "divergence", "error", "params"]
    sizes = [5, 10, 20, 30, 50, 80, 100]
    reps = 3
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=len(sizes) * reps * 2, desc="overhead")
        for n in sizes:
            for rep in range(reps):
                W = make_model(n, seed=SEED + rep)
                X = make_inputs(K, n, seed=SEED + 500 + rep)
                for act in ["sigmoid", "tanh"]:
                    r = transform(W, X, T=T, t_alpha=1, rho=100.0,
                                  metric="frobenius", act=act, phi=PHI,
                                  seed=SEED + rep, maxiter=1000,
                                  track_memory=True)
                    w.writerow({
                        "n_concepts": n, "rep": rep, "activation": act,
                        "seconds": f"{r.seconds:.4f}",
                        "peak_mib": f"{r.peak_mib:.3f}",
                        "n_iter": r.n_iter, "n_fev": r.n_fev,
                        "divergence": f"{r.divergence:.6f}",
                        "error": f"{r.error:.6e}",
                        "params": n * (n - 1),
                    })
                    fh.flush()
                    pbar.update(1)
        pbar.close()
    print("written", path)


# ---------------------------------------------------------------------------
def run_baselines():
    """Our method against three structural baselines at matched budgets."""
    path = os.path.join(OUT, "exp4_baselines.csv")
    fields = ["model_id", "n_concepts", "activation", "method",
              "divergence", "error", "sign_agreement", "spec_ratio", "trace_abs_diff"]
    n_models = 18
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=(min(M1, n_models) - max(M0, 0)) * 2 * 4, desc="baselines")
        for mid in range(max(M0, 0), min(M1, n_models)):
            n = 5 + (mid % 16)
            W = make_model(n, seed=SEED + mid)
            X = make_inputs(K, n, seed=SEED + 10000 + mid)
            ev_W = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]

            for act in ["sigmoid", "tanh"]:
                # Our proposal, at a fixed mid-range rho (no per-model
                # cherry-picking of the best configuration).
                r = transform(W, X, T=T, t_alpha=1, rho=100.0,
                              metric="frobenius", act=act, phi=PHI,
                              seed=SEED + mid, maxiter=1000)
                target = r.divergence
                cands = {
                    "proposal": r.Wp,
                    "similarity": baseline_similarity(W, seed=SEED + mid),
                    "perturbation": baseline_random_perturbation(
                        W, target, seed=SEED + mid),
                    "rewiring": baseline_edge_rewiring(W, seed=SEED + mid),
                }
                for name, Wp in cands.items():
                    ev = np.sort(np.abs(np.linalg.eigvals(Wp)))[::-1]
                    spec_ratio = float(np.mean(np.abs(ev - ev_W)) /
                                       (np.mean(ev_W) + 1e-12))
                    w.writerow({
                        "model_id": mid, "n_concepts": n, "activation": act,
                        "method": name,
                        "divergence": f"{structural_divergence(W, Wp):.6f}",
                        "error": f"{evaluate(W, Wp, X, T, 1, act, PHI):.6e}",
                        "sign_agreement": f"{sign_agreement(W, Wp):.6f}",
                        "spec_ratio": f"{spec_ratio:.6f}",
                        "trace_abs_diff": f"{abs(np.trace(W) - np.trace(Wp)):.6e}",
                    })
                    pbar.update(1)
                fh.flush()
        pbar.close()
    print("written", path)


# ---------------------------------------------------------------------------
def run_attack():
    """Reconstruction of W under the two observation models of Section V-G."""
    path = os.path.join(OUT, "exp5_attack.csv")
    fields = ["model_id", "n_concepts", "activation", "rho", "t_alpha",
              "divergence", "recon_err", "recon_rel", "sign_agree_recon",
              "edge_prec", "edge_rec", "naive_err",
              "orig_rel", "orig_sign"]
    n_models = 9
    rhos = [1.0, 100.0, 10000.0]
    talphas = [1, 20]
    with open(path, MODE, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if MODE == "w":
            w.writeheader()
        pbar = tqdm(total=(min(M1, n_models) - max(M0, 0)) * 2 * len(rhos) * len(talphas), desc="attack")
        for mid in range(max(M0, 0), min(M1, n_models)):
            n = 5 + (mid % 16)
            W = make_model(n, seed=SEED + mid)
            X = make_inputs(K, n, seed=SEED + 10000 + mid)
            base = np.linalg.norm(W, "fro")
            for act in ["sigmoid", "tanh"]:
                for ta in talphas:
                    # The original-trace adversary does not see the released
                    # model at all, so its result does not depend on rho and is
                    # computed once per (model, activation, t_alpha).
                    Worig = reconstruction_attack(
                        W, X, T, ta, act=act, phi=PHI,
                        seed=SEED + mid, maxiter=1000, n_restarts=2)
                    orig_rel = float(np.linalg.norm(W - Worig, "fro")) / base
                    orig_sign = sign_agreement(W, Worig)
                    for rho in rhos:
                        r = transform(W, X, T=T, t_alpha=ta, rho=rho,
                                      metric="frobenius", act=act, phi=PHI,
                                      seed=SEED + mid, maxiter=1000)
                        What = reconstruction_attack(
                            r.Wp, X, T, ta, act=act, phi=PHI,
                            seed=SEED + mid, maxiter=1000, n_restarts=2)
                        err = float(np.linalg.norm(W - What, "fro"))
                        prec, rec = edge_recovery(W, What)
                        w.writerow({
                            "model_id": mid, "n_concepts": n, "activation": act,
                            "rho": rho, "t_alpha": ta,
                            "divergence": f"{r.divergence:.6f}",
                            "recon_err": f"{err:.6f}",
                            "recon_rel": f"{err / base:.6f}",
                            "sign_agree_recon": f"{sign_agreement(W, What):.6f}",
                            "edge_prec": f"{prec:.6f}",
                            "edge_rec": f"{rec:.6f}",
                            # A trivial adversary that simply reads W'.
                            "naive_err": f"{np.linalg.norm(W - r.Wp, 'fro') / base:.6f}",
                            "orig_rel": f"{orig_rel:.6f}",
                            "orig_sign": f"{orig_sign:.6f}",
                        })
                        fh.flush()
                        pbar.update(1)
        pbar.close()
    print("written", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "overhead"):
        run_overhead()
    if which in ("all", "baselines"):
        run_baselines()
    if which in ("all", "attack"):
        run_attack()
