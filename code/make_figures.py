"""Regenerates every figure of the manuscript.

Design constraints
------------------
* Every figure is rendered onto an identical fixed canvas (5.0 x 3.9 in) and
  saved WITHOUT `bbox_inches="tight"`.  Tight cropping trims each figure to
  its own content, so figures with a colour bar or a wider axis label end up
  with different aspect ratios and therefore different rendered heights once
  LaTeX scales them to a common width.  A fixed canvas guarantees that every
  sub-figure occupies exactly the same space on the page.
* `constrained_layout` keeps axis labels, tick labels and colour bars inside
  that canvas, so nothing is clipped and no padding has to be added.
* Type is sized for the final rendered size.  Each panel occupies about 0.485
  of a 3.5 in column, so the figure is reduced by a factor of roughly 0.34 and
  the sizes below land between 8 and 9 pt on the page.
* Error heat maps factor the common power of ten onto the colour bar, so the
  cell annotations are short and cannot collide.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)

PALETTE = {"hyperbolic": "tab:grey", "sigmoid": "tab:blue"}
FIGSIZE = (5.0, 3.9)
SZ = {"label": 26, "tick": 23, "legend": 21, "annot": 22}


def style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": SZ["tick"],
        "axes.labelsize": SZ["label"],
        "xtick.labelsize": SZ["tick"],
        "ytick.labelsize": SZ["tick"],
        "legend.fontsize": SZ["legend"],
        "axes.linewidth": 1.4,
        "grid.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.01,
        "figure.constrained_layout.h_pad": 0.01,
    })


def load(name):
    d = pd.read_csv(os.path.join(RES, name))
    if "activation" in d.columns:
        d["activation"] = d["activation"].replace({"tanh": "hyperbolic"})
    return d


def save(fig, name):
    # No tight bounding box: the canvas size must be identical for every panel.
    fig.savefig(os.path.join(FIG, name), format="pdf")
    plt.close(fig)


def new_fig():
    return plt.subplots(figsize=FIGSIZE, layout="constrained")


# ---------------------------------------------------------------------------
# Figure 1: heat maps
# ---------------------------------------------------------------------------
def heatmaps():
    df = load("exp1_main.csv")
    rows = ["all", "half", "last"]
    cols = ["frobenius", "nuclear", "spectral"]
    short = ["frob.", "nucl.", "spec."]
    for quantity in ["divergence", "error"]:
        piv = df.pivot_table(index=["activation", "steps_mode"],
                             columns="metric", values=quantity)
        for activation, tag in [("sigmoid", "sigmoid"), ("hyperbolic", "tanh")]:
            style()
            data = piv.loc[activation].reindex(rows)[cols]
            fig, ax = new_fig()
            if quantity == "divergence":
                sns.heatmap(data, cmap="Blues", annot=True, fmt=".2f", ax=ax,
                            annot_kws={"size": SZ["annot"]},
                            cbar_kws={"pad": 0.02})
            else:
                k = int(np.floor(np.log10(np.nanmax(data.values))))
                sns.heatmap(data / 10.0 ** k, cmap="Blues", annot=True,
                            fmt=".2f", ax=ax, annot_kws={"size": SZ["annot"]},
                            cbar_kws={"pad": 0.02,
                                      "label": rf"$\times 10^{{{k}}}$"})
                cb = ax.collections[0].colorbar
                cb.ax.yaxis.label.set_size(SZ["tick"])
            cb = ax.collections[0].colorbar
            cb.ax.tick_params(labelsize=SZ["tick"] - 4)
            ax.set_xticklabels(short, fontsize=SZ["tick"])
            ax.set_yticklabels(rows, fontsize=SZ["tick"], rotation=0)
            ax.set_xlabel("Dissimilarity Function")
            ax.set_ylabel("Hidden States")
            save(fig, f"{quantity}_{tag}_heatmap.pdf")
    print("heatmaps done")


# ---------------------------------------------------------------------------
# Generic two-panel line plot
# ---------------------------------------------------------------------------
def lineplot(data, x, y, ylabel, fname, xlabel, logx_ticks=None, legend="best"):
    style()
    fig, ax = new_fig()
    sns.lineplot(data=data, x=x, y=y, hue="activation", style="activation",
                 markers=["s", "s"], dashes=False, errorbar="sd",
                 markersize=14, palette=PALETTE, ax=ax)
    if logx_ticks is not None:
        ax.set_xticks(logx_ticks)
        ax.set_xticklabels([f"$10^{{{int(v)}}}$" for v in logx_ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    lg = ax.legend(loc=legend, fontsize=SZ["legend"], handlelength=1.2,
                   borderpad=0.2, labelspacing=0.2, framealpha=0.9,
                   borderaxespad=0.2)
    lg.set_title(None)
    save(fig, fname)


def rho_plots():
    df = load("exp1_main.csv")
    df["log_rho"] = np.log10(df["rho"])
    t = sorted(df["log_rho"].unique())
    lineplot(df, "log_rho", "divergence", "Divergence",
             "divergence_vs_beta.pdf", "Rho Value", t, "upper right")
    lineplot(df, "log_rho", "error", "Functional Error",
             "error_vs_beta.pdf", "Rho Value", t, "upper right")
    print("rho plots done")


def talpha_plots():
    d = load("exp2_talpha.csv")
    lineplot(d, "t_alpha", "divergence", "Divergence",
             "divergence_vs_talpha.pdf", r"$t_\alpha$", None, "lower right")
    lineplot(d, "t_alpha", "error", "Functional Error",
             "error_vs_talpha.pdf", r"$t_\alpha$", None, "upper left")
    print("t_alpha plots done")


def phi_plots():
    d = load("exp2_phi.csv")
    lineplot(d, "phi", "divergence", "Divergence",
             "divergence_vs_phi.pdf", r"$\phi$", None, "upper right")
    lineplot(d, "phi", "error", "Functional Error",
             "error_vs_phi.pdf", r"$\phi$", None, "upper left")
    print("phi plots done")


# ---------------------------------------------------------------------------
# Figure 3: comparison against the structural baselines
# ---------------------------------------------------------------------------
def comparison():
    b = load("exp4_baselines.csv")
    order = ["proposal", "similarity", "perturbation", "rewiring"]
    labels = ["Prop.", "Simil.", "Pert.", "Rewir."]
    for quantity, ylabel, fname, logy in [
            ("divergence", "Divergence", "comparison_divergence.pdf", False),
            ("error", "Functional Error", "comparison_error.pdf", True)]:
        style()
        g = b.groupby(["method", "activation"])[quantity].agg(["mean", "std"])
        fig, ax = new_fig()
        xs = np.arange(len(order))
        w = 0.38
        for k, (act, col) in enumerate([("sigmoid", "tab:blue"),
                                        ("hyperbolic", "tab:grey")]):
            m = [g.loc[(o, act), "mean"] for o in order]
            e = [g.loc[(o, act), "std"] for o in order]
            ax.bar(xs + (k - 0.5) * w, m, w, yerr=e, capsize=3, label=act,
                   color=col, edgecolor="black", linewidth=1.1,
                   error_kw={"elinewidth": 1.6, "capthick": 1.6})
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=SZ["tick"] - 3, rotation=30,
                           ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Method")
        ax.legend(loc="upper left" if logy else "upper right",
                  fontsize=SZ["legend"], handlelength=1.1, borderpad=0.2,
                  labelspacing=0.2, framealpha=0.9, borderaxespad=0.2)
        save(fig, fname)
    print("comparison bar charts done")


if __name__ == "__main__":
    heatmaps(); rho_plots(); comparison(); talpha_plots(); phi_plots()
    print("all figures written to", FIG)
