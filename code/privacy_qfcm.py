"""
Privacy-preserving quasi-nonlinear Fuzzy Cognitive Maps through structural
transformation.

Core library supporting the manuscript.  It provides the qFCM reasoning rule,
the transformation loss of Eq. (6) with its analytic gradient, the
bound-constrained optimizer, the structural baselines and the reconstruction
attack.

Design notes
------------
1. The loss gradient is analytic, obtained by backpropagation through the qFCM
   recurrence.  Finite differences would cost N^2 + 1 function evaluations per
   gradient and exhaust a budget of 15000 evaluations before converging for
   N >= 20.
2. t_alpha is an explicit parameter rather than a coarse last / half / all
   proxy, so the temporal window can be swept over its whole range.
3. The zero diagonal is enforced on both W and W', consistent with the FCM
   definition used in the manuscript.
4. The privacy metric is the relative structural divergence
   delta = ||W - W'||_* / ||W||_*, which is comparable across model sizes.
5. Both loss terms are normalized, so rho keeps a stable meaning across
   temporal windows of different length.
6. Seeding is deterministic and per configuration, so a rerun reproduces every
   stored value and no two configurations of a model share a starting point.
7. reconstruction_attack takes the model whose reasoning trace the adversary
   observes as an explicit argument, so both observation models of Section V-G
   run through the same code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

EPS = 1e-10


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    t = np.tanh(x)
    return 1.0 - t * t


ACTIVATIONS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
}


# ---------------------------------------------------------------------------
# qFCM reasoning
# ---------------------------------------------------------------------------
def qfcm_states(W, X, T, act="sigmoid", phi=0.8):
    """Run the quasi-nonlinear reasoning rule and return every state.

    A^(t+1) = phi * f(A^(t) W) + (1 - phi) * A^(0)

    Returns an array of shape (T + 1, K, N) holding A^(0) ... A^(T).
    """
    f, _ = ACTIVATIONS[act]
    A0 = np.asarray(X, dtype=float)
    A = A0.copy()
    states = [A]
    for _ in range(T):
        A = phi * f(A @ W) + (1.0 - phi) * A0
        states.append(A)
    return np.asarray(states)


def qfcm_states_and_preacts(W, X, T, act="sigmoid", phi=0.8):
    """Same as `qfcm_states` but also returns the pre-activations Z^(t)."""
    f, _ = ACTIVATIONS[act]
    A0 = np.asarray(X, dtype=float)
    A = A0.copy()
    states = [A]
    preacts = []
    for _ in range(T):
        Z = A @ W
        preacts.append(Z)
        A = phi * f(Z) + (1.0 - phi) * A0
        states.append(A)
    return np.asarray(states), np.asarray(preacts)


# ---------------------------------------------------------------------------
# Matrix dissimilarity
# ---------------------------------------------------------------------------
def matrix_norm(M, metric="frobenius"):
    if metric == "frobenius":
        return float(np.linalg.norm(M, "fro"))
    if metric == "nuclear":
        return float(np.sum(svd(M, compute_uv=False)))
    if metric == "spectral":
        return float(np.max(svd(M, compute_uv=False)))
    raise ValueError(f"Unknown matrix norm: {metric}")


def matrix_norm_and_grad(M, metric="frobenius"):
    """Return (||M||_*, d||M||_* / dM)."""
    if metric == "frobenius":
        val = float(np.linalg.norm(M, "fro"))
        grad = M / (val + EPS)
        return val, grad
    U, s, Vt = svd(M, full_matrices=False)
    if metric == "nuclear":
        return float(np.sum(s)), U @ Vt
    if metric == "spectral":
        return float(s[0]), np.outer(U[:, 0], Vt[0, :])
    raise ValueError(f"Unknown matrix norm: {metric}")


def structural_divergence(W, Wp, metric="frobenius", relative=True):
    """Structural divergence between the original and the transformed model.

    With `relative=True` the value is ||W - W'||_* / ||W||_*, which is
    dimensionless and therefore comparable across different model sizes.
    """
    d = matrix_norm(W - Wp, metric)
    if relative:
        return d / (matrix_norm(W, metric) + EPS)
    return d


# ---------------------------------------------------------------------------
# Loss function and analytic gradient
# ---------------------------------------------------------------------------
def loss_and_grad(
    w_flat,
    W,
    X,
    T,
    t_alpha,
    rho,
    metric="frobenius",
    act="sigmoid",
    phi=0.8,
    fidelity_only=False,
    states_ref=None,
):
    """Objective of Eq. (6) in the revised manuscript, with its gradient.

    L(W') = 1 / (||W - W'||_* + eps) + rho * MSE_{t >= t_alpha}(A, A')

    The mean squared error is averaged over the evaluated iterations, the
    initial conditions and the concepts, so rho retains the same meaning
    regardless of t_alpha.
    """
    N = W.shape[0]
    Wp = w_flat.reshape(N, N)
    f, df = ACTIVATIONS[act]

    if states_ref is None:
        states_ref = qfcm_states(W, X, T, act, phi)

    states, preacts = qfcm_states_and_preacts(Wp, X, T, act, phi)

    K = X.shape[0]
    n_eval = T - t_alpha + 1
    scale = rho / (n_eval * K * N)

    diff = states[t_alpha:] - states_ref[t_alpha:]
    fidelity = scale * float(np.sum(diff ** 2))

    # Backpropagation through the recurrence.
    Gbar = np.zeros_like(states)          # dL / dA'^(t)
    Gbar[t_alpha:] = 2.0 * scale * diff

    grad_W = np.zeros_like(Wp)
    for t in range(T - 1, -1, -1):
        delta = Gbar[t + 1] * phi * df(preacts[t])   # (K, N)
        grad_W += states[t].T @ delta
        Gbar[t] += delta @ Wp.T

    if fidelity_only:
        total = fidelity
    else:
        D, dD = matrix_norm_and_grad(W - Wp, metric)
        total = 1.0 / (D + EPS) + fidelity
        # d/dW' [1 / (D + eps)] with D = ||W - W'||_*
        grad_W += (1.0 / (D + EPS) ** 2) * dD

    # The diagonal is pinned to zero, so no gradient is propagated there.
    np.fill_diagonal(grad_W, 0.0)
    return total, grad_W.ravel()


def evaluate(W, Wp, X, T, t_alpha, act="sigmoid", phi=0.8):
    """Mean squared simulation error over the evaluated temporal window."""
    s1 = qfcm_states(W, X, T, act, phi)
    s2 = qfcm_states(Wp, X, T, act, phi)
    return float(np.mean((s1[t_alpha:] - s2[t_alpha:]) ** 2))


# ---------------------------------------------------------------------------
# The privacy-preserving transformation
# ---------------------------------------------------------------------------
@dataclass
class TransformResult:
    Wp: np.ndarray
    divergence: float
    divergence_abs: float
    error: float
    n_iter: int
    n_fev: int
    seconds: float
    success: bool
    peak_mib: float = 0.0
    extra: dict = field(default_factory=dict)


def transform(
    W,
    X,
    T=20,
    t_alpha=1,
    rho=100.0,
    metric="frobenius",
    act="sigmoid",
    phi=0.8,
    seed=0,
    maxiter=2000,
    init="random",
    fidelity_only=False,
    track_memory=False,
):
    """Build a structurally divergent yet functionally equivalent qFCM."""
    N = W.shape[0]
    # An independent stream: seeding this the same way as `make_model` would
    # start the optimiser exactly at W, where both the fidelity gradient and
    # the dissimilarity gradient vanish and L-BFGS-B halts at iteration zero.
    rng = np.random.default_rng([int(seed), 20260724])

    if init == "random":
        W0 = rng.uniform(-1.0, 1.0, (N, N))
    elif init == "original":
        W0 = W.copy()
    elif init == "negated":
        W0 = -W.copy()
    else:
        raise ValueError(f"Unknown initialisation: {init}")
    np.fill_diagonal(W0, 0.0)

    # Guard against a degenerate start: at W' = W the objective is singular
    # and its gradient is identically zero.
    if np.linalg.norm(W - W0, "fro") < 1e-6:
        W0 = np.clip(W0 + 0.1 * rng.standard_normal((N, N)), -1.0, 1.0)
        np.fill_diagonal(W0, 0.0)

    # Box constraints; the diagonal is pinned to zero (no self-loops).
    lo = np.full((N, N), -1.0)
    hi = np.full((N, N), 1.0)
    np.fill_diagonal(lo, 0.0)
    np.fill_diagonal(hi, 0.0)
    bounds = list(zip(lo.ravel(), hi.ravel()))

    states_ref = qfcm_states(W, X, T, act, phi)

    if track_memory:
        import tracemalloc

        tracemalloc.start()

    t0 = time.perf_counter()
    res = minimize(
        loss_and_grad,
        W0.ravel(),
        args=(W, X, T, t_alpha, rho, metric, act, phi, fidelity_only, states_ref),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "maxfun": 10 * maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )
    seconds = time.perf_counter() - t0

    peak_mib = 0.0
    if track_memory:
        import tracemalloc

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak / (1024 ** 2)

    Wp = res.x.reshape(N, N)
    np.fill_diagonal(Wp, 0.0)

    return TransformResult(
        Wp=Wp,
        divergence=structural_divergence(W, Wp, "frobenius", relative=True),
        divergence_abs=structural_divergence(W, Wp, "frobenius", relative=False),
        error=evaluate(W, Wp, X, T, t_alpha, act, phi),
        n_iter=int(res.nit),
        n_fev=int(res.nfev),
        seconds=seconds,
        success=bool(res.success),
        peak_mib=peak_mib,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_similarity(W, seed=0):
    """Orthogonal similarity transformation W' = P^T W P.

    In exact arithmetic the conjugation preserves the spectrum of W.  The
    result must nevertheless be projected back onto the admissible FCM set,
    namely a zero diagonal and entries inside [-1, 1], and that projection
    destroys the preservation.  `run_baselines` measures the residual
    eigenvalue shift, so the claim is checked rather than assumed.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    Wp = Q.T @ W @ Q
    m = np.max(np.abs(Wp))
    if m > 1.0:
        Wp = Wp / m
    np.fill_diagonal(Wp, 0.0)
    return Wp


def baseline_random_perturbation(W, target_div, seed=0, tol=1e-3, max_steps=60):
    """Additive random perturbation calibrated to a target relative divergence.

    Unlike the similarity transformation this baseline does not preserve the
    spectrum of W, so it is a genuinely stronger structural competitor.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    E = rng.standard_normal((N, N))
    np.fill_diagonal(E, 0.0)
    E = E / (np.linalg.norm(E, "fro") + EPS)
    base = np.linalg.norm(W, "fro")

    lo, hi = 0.0, 8.0 * base + 1.0
    Wp = W.copy()
    for _ in range(max_steps):
        mid = 0.5 * (lo + hi)
        cand = np.clip(W + mid * E, -1.0, 1.0)
        np.fill_diagonal(cand, 0.0)
        d = np.linalg.norm(W - cand, "fro") / (base + EPS)
        Wp = cand
        if abs(d - target_div) < tol:
            break
        if d < target_div:
            lo = mid
        else:
            hi = mid
    return Wp


def baseline_edge_rewiring(W, seed=0, frac=1.0):
    """Degree-preserving edge rewiring, in the spirit of graph anonymisation.

    Non-zero causal weights are permuted among the existing edge slots, which
    preserves the multiset of weights and the total connection count while
    destroying the identity of individual causal links.
    """
    rng = np.random.default_rng(seed)
    Wp = W.copy()
    idx = np.argwhere(np.abs(W) > 0)
    idx = idx[idx[:, 0] != idx[:, 1]]
    if len(idx) < 2:
        return Wp
    n_sw = max(1, int(frac * len(idx)))
    sel = rng.choice(len(idx), size=n_sw, replace=False)
    vals = np.array([W[i, j] for i, j in idx[sel]])
    perm = rng.permutation(len(vals))
    for (i, j), v in zip(idx[sel], vals[perm]):
        Wp[i, j] = v
    np.fill_diagonal(Wp, 0.0)
    return Wp


# ---------------------------------------------------------------------------
# Adversarial reconstruction
# ---------------------------------------------------------------------------
def reconstruction_attack(
    W_obs,
    X,
    T,
    t_alpha,
    act="sigmoid",
    phi=0.8,
    seed=0,
    maxiter=2000,
    n_restarts=3,
):
    """Trajectory-matching attack against the traces generated by `W_obs`.

    The adversary knows the algorithm, the activation function and phi, and it
    observes a reasoning trace from iteration t_alpha onward.  It searches for
    a matrix that reproduces those observations, which is the strongest
    inference available without access to the pre-t_alpha transient.

    `W_obs` is the model whose trace is exposed.  Passing the released matrix
    W' models a deployment in which the transformed model is the artifact that
    runs, which is the threat model of Section IV-A.  Passing the original
    matrix W models a deployment that publishes the original trajectories
    while releasing W' as documentation, and Section V-G reports both.
    """
    N = W_obs.shape[0]
    target = qfcm_states(W_obs, X, T, act, phi)

    lo = np.full((N, N), -1.0)
    hi = np.full((N, N), 1.0)
    np.fill_diagonal(lo, 0.0)
    np.fill_diagonal(hi, 0.0)
    bounds = list(zip(lo.ravel(), hi.ravel()))

    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + 1000 * r)
        W0 = rng.uniform(-1.0, 1.0, (N, N))
        np.fill_diagonal(W0, 0.0)
        res = minimize(
            loss_and_grad,
            W0.ravel(),
            args=(W_obs, X, T, t_alpha, 1.0, "frobenius", act, phi, True, target),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "maxfun": 10 * maxiter, "ftol": 1e-14},
        )
        if best is None or res.fun < best.fun:
            best = res

    What = best.x.reshape(N, N)
    np.fill_diagonal(What, 0.0)
    return What


def sign_agreement(W, Wp, tol=1e-8):
    """Fraction of off-diagonal entries whose causal sign is preserved."""
    N = W.shape[0]
    mask = ~np.eye(N, dtype=bool)
    a = np.sign(np.where(np.abs(W) < tol, 0.0, W))[mask]
    b = np.sign(np.where(np.abs(Wp) < tol, 0.0, Wp))[mask]
    return float(np.mean(a == b))


def edge_recovery(W, What, thr=1e-2):
    """Precision and recall of the recovered causal edges."""
    N = W.shape[0]
    mask = ~np.eye(N, dtype=bool)
    true_e = (np.abs(W) > thr)[mask]
    pred_e = (np.abs(What) > thr)[mask]
    tp = float(np.sum(true_e & pred_e))
    prec = tp / max(float(np.sum(pred_e)), 1.0)
    rec = tp / max(float(np.sum(true_e)), 1.0)
    return prec, rec


# ---------------------------------------------------------------------------
# Synthetic model generation
# ---------------------------------------------------------------------------
def make_model(N, seed=0, density=1.0):
    """Random qFCM weight matrix with a zero diagonal."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1.0, 1.0, (N, N))
    if density < 1.0:
        drop = rng.random((N, N)) > density
        W[drop] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def make_inputs(K, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((K, N))
