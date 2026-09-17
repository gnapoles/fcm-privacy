"""Experiment 6: the Lissos River basin case study.

Transforms the real quasi-nonlinear FCM elicited by Papadopoulos et al. and
measures what the released model keeps and what it destroys.

Approach
--------
The released model has to stay t_alpha-equivalent over the operating range of
the real system.  The K = 100 initial conditions are therefore relative
perturbations of the published initial state, each concept being rescaled
independently by a factor drawn uniformly from [0.8, 1.2].  The factor is a
multiplier, not an activation value.  The published initial state of the basin
model holds five negative entries, so its activations live in [-1, 1] rather
than in [0, 1], and a +/- 20 per cent perturbation of it stays inside
[-0.60, 0.96].  The clip below is a safeguard that never binds at this spread.
The three scenarios of the source article perturb the same entries by 10 and
20 per cent, so they lie inside the same family without being members of the
training sample.  Everything else follows the synthetic study, with T = 20,
phi = 0.8, t_alpha = 1, the Frobenius norm and an iteration budget of 1,000.

Outputs
-------
exp6_case_study.csv   -> Table IV
exp6_scenarios.csv    -> the scenario validation quoted in Section V-I
lissos_Wp.npy         -> the released matrix at the sigmoid, rho = 100 setting
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lissos_data as LD
from privacy_qfcm import qfcm_states, sign_agreement, structural_divergence, transform

SEED = 42
K = 100
SPREAD = 0.2          # the +/- 20 per cent band of the source scenario analysis
BOUND = 1.0           # admissible activation range of the basin model
T = 20
PHI = 0.8
T_ALPHA = 1
METRIC = "frobenius"
RHOS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
ACTS = ["sigmoid", "tanh"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# The configuration whose released matrix is stored and used in Section V-I.
REFERENCE = ("sigmoid", 100.0)


def edge_sign_agreement(W, Wp):
    """Sign agreement restricted to the causal links the experts elicited.

    The unrestricted figure also counts the many absent links of a sparse
    elicited map, which a dense released matrix can never match, so both are
    reported.
    """
    mask = (np.abs(W) > 0) & ~np.eye(W.shape[0], dtype=bool)
    return float(np.mean(np.sign(W[mask]) == np.sign(Wp[mask])))


def main():
    os.makedirs(OUT, exist_ok=True)
    LD.validate(verbose=True)

    W = LD.weight_matrix()
    rng = np.random.default_rng(SEED + 10000)
    factors = rng.uniform(1.0 - SPREAD, 1.0 + SPREAD, (K, W.shape[0]))
    X = np.clip(LD.INITIAL_STATE * factors, -BOUND, BOUND)
    scen = LD.scenarios()
    cent_W = LD.centrality(W)

    rows, scen_rows = [], []
    for act in ACTS:
        for rho in RHOS:
            r = transform(W, X, T=T, t_alpha=T_ALPHA, rho=rho, metric=METRIC,
                          act=act, phi=PHI, seed=SEED, maxiter=1000)
            Wp = r.Wp
            rows.append({
                "activation": "hyperbolic" if act == "tanh" else act,
                "rho": rho,
                "divergence": f"{r.divergence:.6f}",
                "error": f"{r.error:.6e}",
                "sign": f"{sign_agreement(W, Wp):.6f}",
                "sign_links": f"{edge_sign_agreement(W, Wp):.6f}",
                "cent_corr": f"{pearsonr(cent_W, LD.centrality(Wp)).statistic:.6f}",
                "seconds": f"{r.seconds:.4f}",
            })

            # Out-of-sample validation on the three published scenarios.
            A = qfcm_states(W, scen, T, act, PHI)[-1]
            B = qfcm_states(Wp, scen, T, act, PHI)[-1]
            scen_rows.append({
                "activation": "hyperbolic" if act == "tanh" else act,
                "rho": rho,
                "max_abs_dev": f"{np.max(np.abs(A - B)):.6e}",
                "spearman_min": f"{min(spearmanr(A[i], B[i]).statistic for i in range(len(scen))):.6f}",
                "spearman_mean": f"{np.mean([spearmanr(A[i], B[i]).statistic for i in range(len(scen))]):.6f}",
            })

            if (act, rho) == REFERENCE:
                np.save(os.path.join(OUT, "lissos_Wp.npy"), Wp)

    with open(os.path.join(OUT, "exp6_case_study.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "exp6_scenarios.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scen_rows[0]))
        w.writeheader()
        w.writerows(scen_rows)
    print("written exp6_case_study.csv, exp6_scenarios.csv and lissos_Wp.npy")


if __name__ == "__main__":
    main()
