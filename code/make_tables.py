

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

ACT = {"sigmoid": "sigmoid", "tanh": "hyperbolic"}
OUT_LINES, TXT = [], []


def load(name):
    return pd.read_csv(os.path.join(RES, name))


def w(line=""):
    OUT_LINES.append(line)


def note(line):
    TXT.append(line)


def e(x, d=1):
    """Format in the 1.2e-03 style used by the tables."""
    return f"{x:.{d}e}"


# ---------------------------------------------------------------------------
# Table I and the text of Sections V-A, V-B and V-E
# ---------------------------------------------------------------------------
def table_one():
    d = load("exp1_main.csv")
    note("== Section V, experimental setup ==")
    note(f"configurations          {len(d)}")
    note(f"models                  {d.model_id.nunique()}")
    note(f"concepts                {d.n_concepts.min()} to {d.n_concepts.max()}")
    note(f"converged runs          {int(d.success.sum())} of {len(d)} "
         f"({100 * d.success.mean():.0f} per cent)")
    note(f"converged at N<=10      {100 * d[d.n_concepts <= 10].success.mean():.0f} per cent")
    note(f"converged at N>=15      {100 * d[d.n_concepts >= 15].success.mean():.0f} per cent")

    w("% ---- Table I, upper block ----")
    up = d.groupby(["metric", "steps_mode", "activation"]).agg(
        dm=("divergence", "mean"), ds=("divergence", "std"), er=("error", "mean"))
    for metric in ["frobenius", "nuclear", "spectral"]:
        for steps in ["all", "half", "last"]:
            s = up.loc[(metric, steps, "sigmoid")]
            t = up.loc[(metric, steps, "tanh")]
            w(f"{metric} & {steps} & {s.dm:.2f}\\,$\\pm$\\,{s.ds:.2f} & {e(s.er)} "
              f"& {t.dm:.2f}\\,$\\pm$\\,{t.ds:.2f} & {e(t.er)} \\\\")
    w("% ---- Table I, endpoint rows ----")
    ep = load("exp2_endpoints.csv").groupby(["variant", "activation"]).mean(numeric_only=True)
    for var, lab in [("dissimilarity-only", "diss. only"), ("fidelity-only", "fid. only")]:
        sg = ep.loc[(var, "sigmoid")]
        tn = ep.loc[(var, "tanh")]
        w(f"\\multicolumn{{2}}{{l}}{{{lab}}} & {e(sg.divergence, 2)} & {e(sg.error, 2)}\\,/\\,"
          f"{sg.sign_agreement:.2f} & {e(tn.divergence, 2)} & {e(tn.error, 2)}\\,/\\,{tn.sign_agreement:.2f} \\\\")
    w("% ---- Table I, lower block ----")
    lo = d.groupby(["rho", "activation"]).agg(
        dm=("divergence", "mean"), ds=("divergence", "std"),
        er=("error", "mean"), sg=("sign_agreement", "mean"))
    for rho in sorted(d.rho.unique()):
        s = lo.loc[(rho, "sigmoid")]
        t = lo.loc[(rho, "tanh")]
        w(f"\\multicolumn{{2}}{{l}}{{{rho:.0e}}} & {s.dm:.2f}\\,$\\pm$\\,{s.ds:.2f} "
          f"& {e(s.er)}\\,/\\,{s.sg:.2f} & {t.dm:.2f}\\,$\\pm$\\,{t.ds:.2f} "
          f"& {e(t.er)}\\,/\\,{t.sg:.2f} \\\\")
    w()

    note("")
    note("== Section V-A ==")
    for a in ["tanh", "sigmoid"]:
        f = up.xs(a, level=2).loc["frobenius"]
        note(f"{ACT[a]:10s} frobenius divergence last {f.loc['last'].dm:.2f}, "
             f"all {f.loc['all'].dm:.2f}")
    cheap = up.xs("tanh", level=2).loc[("nuclear", "last")]
    fro = up.xs("tanh", level=2).loc[("frobenius", "last")]
    spe = up.xs("tanh", level=2).loc[("spectral", "last")]
    note(f"nuclear lowest divergence in all six groups: "
         f"{all(up.xs(a, level=2).loc[(m, s)].dm >= up.xs(a, level=2).loc[('nuclear', s)].dm for a in ACT for m in ['frobenius', 'spectral'] for s in ['all', 'half', 'last'])}")
    note(f"exception on error, hyperbolic/last: nuclear {e(cheap.er)} vs "
         f"frobenius {e(fro.er)} and spectral {e(spe.er)}")
    sig_high = sum(up.loc[(m, s, "sigmoid")].dm > up.loc[(m, s, "tanh")].dm
                   for m in ["frobenius", "nuclear", "spectral"] for s in ["all", "half", "last"])
    note(f"sigmoid divergence exceeds hyperbolic in {sig_high} of 9 cells; the "
         f"exceptions are nuclear/last and spectral/last")

    note("")
    note("== Section V-B ==")
    rho_lo, rho_hi = lo.loc[(1.0, "sigmoid")], lo.loc[(10000.0, "sigmoid")]
    note(f"sigmoid error {e(rho_lo.er)} at rho=1 down to {e(rho_hi.er)} at rho=1e4")
    ep = load("exp2_endpoints.csv").groupby(["variant", "activation"]).mean(numeric_only=True)
    note(f"dissimilarity only: divergence {ep.loc[('dissimilarity-only', 'sigmoid')].divergence:.2f}, "
         f"error {e(ep.loc[('dissimilarity-only', 'sigmoid')].error)} sigmoid and "
         f"{e(ep.loc[('dissimilarity-only', 'tanh')].error)} hyperbolic")
    for a in ["sigmoid", "tanh"]:
        r = ep.loc[("fidelity-only", a)]
        note(f"fidelity only {ACT[a]:10s}: divergence {e(r.divergence)}, error {e(r.error)}, "
             f"sign {r.sign_agreement:.3f}")

    note("")
    note("== Section V-E ==")
    note(f"mean sign agreement {d.sign_agreement.mean():.3f}")
    note(f"sigmoid sign agreement {lo.loc[(1.0, 'sigmoid')].sg:.3f} at rho=1 "
         f"to {lo.loc[(10000.0, 'sigmoid')].sg:.3f} at rho=1e4")
    note(f"hyperbolic sign agreement {lo.loc[(1.0, 'tanh')].sg:.2f} at rho=1 "
         f"to {lo.loc[(10000.0, 'tanh')].sg:.2f} at rho=1e4")


# ---------------------------------------------------------------------------
# Sections V-C and V-D
# ---------------------------------------------------------------------------
def sensitivity_text():
    note("")
    note("== Section V-C, t_alpha ==")
    t = load("exp2_talpha.csv")
    note(f"models {t.model_id.nunique()}, concepts {t.n_concepts.min()} to {t.n_concepts.max()}")
    g = t.groupby(["activation", "t_alpha"]).agg(dm=("divergence", "mean"), er=("error", "mean"))
    for a in ["tanh", "sigmoid"]:
        s = g.loc[a]
        note(f"{ACT[a]:10s} divergence {s.dm.iloc[0]:.2f} to {s.dm.iloc[-1]:.2f}, "
             f"error {e(s.er.iloc[0])} to {e(s.er.iloc[-1])}")

    note("")
    note("== Section V-D, phi ==")
    p = load("exp2_phi.csv")
    g = p.groupby(["activation", "phi"]).agg(dm=("divergence", "mean"), er=("error", "mean"))
    for a in ["sigmoid", "tanh"]:
        s = g.loc[a]
        note(f"{ACT[a]:10s} divergence {s.dm.min():.2f} to {s.dm.max():.2f}, "
             f"at phi=0.2 {s.dm.iloc[0]:.2f}, at phi=1.0 {s.dm.iloc[-1]:.2f}, "
             f"minimum {s.dm.min():.2f} at phi={s.dm.idxmin()}")
        note(f"{ACT[a]:10s} error {e(s.er.iloc[0])} at phi=0.2 to {e(s.er.iloc[-1])} at phi=1.0")


# ---------------------------------------------------------------------------
# Section V-F
# ---------------------------------------------------------------------------
def baselines_text():
    note("")
    note("== Section V-F, baselines ==")
    b = load("exp4_baselines.csv")
    note(f"models {b.model_id.nunique()}, concepts {b.n_concepts.min()} to {b.n_concepts.max()}")
    g = b.groupby(["activation", "method"]).agg(dm=("divergence", "mean"), er=("error", "mean"))
    for a in ["sigmoid", "tanh"]:
        for m in ["proposal", "similarity", "perturbation", "rewiring"]:
            r = g.loc[(a, m)]
            note(f"{ACT[a]:10s} {m:12s} divergence {r.dm:.2f} error {e(r.er)}")
    sp = b.groupby("method").spec_ratio.mean()
    for m in ["similarity", "proposal", "rewiring", "perturbation"]:
        note(f"eigenvalue shift {m:12s} {sp[m]:.2f}")


# ---------------------------------------------------------------------------
# Table II and Section V-G
# ---------------------------------------------------------------------------
def table_two():
    a = load("exp5_attack.csv")
    note("")
    note("== Table II and Section V-G ==")
    note(f"models {a.model_id.nunique()}, concepts {a.n_concepts.min()} to {a.n_concepts.max()}")
    g = a.groupby(["activation", "t_alpha", "rho"]).mean(numeric_only=True)

    w("% ---- Table II ----")
    for act in ["tanh", "sigmoid"]:
        for ta in [1, 20]:
            for rho in [1.0, 10000.0]:
                r = g.loc[(act, ta, rho)]
                w(f"{ACT[act]} & {ta} & {rho:.0e} & {r.divergence:.2f} & {r.recon_rel:.2f} "
                  f"& {r.sign_agree_recon:.2f} & {r.orig_rel:.2f} \\\\")
    w()

    ratio = (a.recon_rel / a.divergence)
    note(f"reconstruction error over divergence: min {ratio.min():.2f}, "
         f"mean {ratio.mean():.2f}, max {ratio.max():.2f}")
    note(f"never exceeds the divergence: {bool((ratio <= 1.0 + 1e-6).all())}")
    for ta in [1, 20]:
        note(f"t_alpha={ta:2d} mean ratio {ratio[a.t_alpha == ta].mean():.2f}")
    note(f"runs recovering W outright (recon_rel < 1e-3): {int((a.recon_rel < 1e-3).sum())} of {len(a)}")
    note(f"adversary advantage at t_alpha=20: {1.0 / ratio[a.t_alpha == 20].mean():.1f} times")
    r = g.loc[("tanh", 1, 10000.0)]
    note(f"hyperbolic t_alpha=1 rho=1e4: recon {r.recon_rel:.2f}, "
         f"sign recovered {100 * r.sign_agree_recon:.0f} per cent")
    note(f"original-trace adversary: relative error at most {a.orig_rel.max():.2e}, "
         f"sign agreement at least {a.orig_sign.min():.3f}")


# ---------------------------------------------------------------------------
# Table III and Section V-H
# ---------------------------------------------------------------------------
def table_three():
    o = load("exp3_overhead.csv")
    note("")
    note("== Table III and Section V-H ==")
    g = o.groupby("n_concepts").agg(
        params=("params", "first"), tm=("seconds", "mean"), ts=("seconds", "std"),
        mem=("peak_mib", "mean"), it=("n_iter", "mean"),
        dv=("divergence", "mean"), er=("error", "mean"), n=("seconds", "size"))
    w("% ---- Table III ----")
    for n, r in g.iterrows():
        w(f"{n} & {int(r.params)} & {r.tm:.2f}$\\pm${r.ts:.2f} & {r.mem:.1f} & "
          f"{int(round(r.it))} & {r.dv:.2f} & {e(r.er)} \\\\")
    w()
    note(f"runs per size {sorted(set(g.n.astype(int)))}")
    note(f"runtime {g.tm.iloc[0]:.2f} s at N={g.index[0]} to {g.tm.iloc[-1]:.2f} s at N={g.index[-1]}")
    note(f"peak memory at most {g.mem.max():.1f} MiB")
    small = g[g.index <= 10].er.max()
    note(f"error at most {e(small)} up to N=10, {e(g.er.max())} at the largest sizes")


# ---------------------------------------------------------------------------
# Table IV and Section V-I
# ---------------------------------------------------------------------------
def table_four():
    c = load("exp6_case_study.csv")
    sc = load("exp6_scenarios.csv").set_index(["activation", "rho"])
    note("")
    note("== Table IV and Section V-I ==")
    w("% ---- Table IV ----")
    for act in ["sigmoid", "hyperbolic"]:
        for rho in [1.0, 100.0, 10000.0]:
            r = c[(c.activation == act) & (c.rho == rho)].iloc[0]
            v = sc.loc[(act, rho)]
            w(f"{act} & {rho:.0e} & {r.divergence:.2f} & {e(r.error, 2)} & {r.sign:.2f} & "
              f"{r.sign_links:.2f} & {r.cent_corr:+.2f} & {e(v.max_abs_dev, 2)} & "
              f"{v.spearman_min:.3f} \\\\")
    w()
    ref = c[(c.activation == "sigmoid") & (c.rho == 100.0)].iloc[0]
    refs = sc.loc[("sigmoid", 100.0)]
    note(f"sigmoid rho=100: divergence {ref.divergence:.2f}, error {e(ref.error)}, "
         f"sign {ref.sign:.3f} over all entries and {ref.sign_links:.2f} over the elicited links")
    note(f"scenario check: max deviation {e(refs.max_abs_dev)}, "
         f"Spearman at least {refs.spearman_min:.3f}")
    note(f"centrality correlation range {c.cent_corr.min():+.2f} to {c.cent_corr.max():+.2f}")
    note(f"endpoints: see Table I rows diss. only and fid. only")


def main():
    table_one()
    sensitivity_text()
    baselines_text()
    table_two()
    table_three()
    table_four()
    with open(os.path.join(RES, "tables.tex"), "w") as fh:
        fh.write("\n".join(OUT_LINES) + "\n")
    with open(os.path.join(RES, "text_numbers.txt"), "w") as fh:
        fh.write("\n".join(TXT) + "\n")
    print("\n".join(TXT))
    print("\ntables written to results/tables.tex")


if __name__ == "__main__":
    main()
