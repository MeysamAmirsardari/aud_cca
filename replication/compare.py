"""Compare the paper (digitized Fig. 5), NoiseTools and this replication.

    python replication/compare.py

Needs replication/results/<subject>.npz (from replicate.py) and the NoiseTools
scores of the same models on the same recordings (nt_cca_crossvalidate, saved
by run_pc_revision_nt.m). Writes comparison figures and a table to OUT.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_replication/source/paper_figure5_digitized.csv"
NT = ROOT / "paper_replication/paper_text_2026_11_04/results"
PY = ROOT / "replication/results"
OUT = PY / "comparison"

SUBJECTS = ["EL", "AC", "BQ", "ChW", "JOS", "MC", "NM", "TT"]
MODELS = ["backward", "forward", "cca1", "cca2", "cca2plus", "cca3"]
LABELS = ["Backward", "Forward", "CCA 1", "CCA 2", "CCA 2+", "CCA 3"]
COLORS = {"paper": "#2b3240", "nt": "#16878c", "py": "#c8692d"}


def nt_curves(subject, model):
    bank = "script23" if model == "cca3" else "grid27"    # the other models do not use the bank
    d = loadmat(NT / bank / "matlab" / f"{subject}_{model}.mat", squeeze_me=True)
    return np.atleast_2d(d["mean_scores"])               # component x shift


def py_curves(subject, model):
    return np.load(PY / f"{subject}.npz")[model].mean(1).T


def load():
    with open(PAPER) as f:
        rows = {r["historical_subject"]: r for r in csv.DictReader(f)}
    paper = np.array([[float(rows[s][m]) for m in MODELS] for s in SUBJECTS])
    nt = np.array([[nt_curves(s, m)[0].max() for m in MODELS] for s in SUBJECTS])
    py = np.array([[py_curves(s, m)[0].max() for m in MODELS] for s in SUBJECTS])
    # worst disagreement anywhere on the curves: every shift and every component
    gap = np.array([[np.abs(py_curves(s, m) - nt_curves(s, m)).max() for m in MODELS] for s in SUBJECTS])
    return paper, nt, py, gap


def stats(x, y):
    d = y - x
    ccc = 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return (f"RMSE {np.sqrt((d ** 2).mean()):.4f} · r {np.corrcoef(x, y)[0, 1]:.3f}\n"
            f"bias {d.mean():+.4f} · CCC {ccc:.3f}")


def scatter(ax, x, y, xlabel, ylabel, colour):
    ax.plot([0, 0.4], [0, 0.4], "--", color="0.5", lw=1)
    ax.scatter(x.ravel(), y.ravel(), s=28, color=colour, alpha=0.85, edgecolors="none")
    ax.set(xlim=(0, 0.4), ylim=(0, 0.4), xlabel=xlabel, ylabel=ylabel, aspect="equal")
    ax.set_title(stats(x.ravel(), y.ravel()), fontsize=11, fontweight="bold")


def heatmap(ax, values, title, **kw):
    im = ax.imshow(values, aspect="auto", **kw)
    ax.set_xticks(range(6), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(8), SUBJECTS)
    ax.set_title(title, fontweight="bold")
    return im


def figure_agreement(paper, nt, py, gap):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw=dict(height_ratios=(1, 0.9)),
                             layout="constrained")
    scatter(axes[0, 0], paper, nt, "paper (digitized)", "NoiseTools", COLORS["nt"])
    scatter(axes[0, 1], paper, py, "paper (digitized)", "Python", COLORS["py"])
    scatter(axes[0, 2], nt, py, "NoiseTools", "Python", COLORS["py"])
    diff = dict(cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    heatmap(axes[1, 0], nt - paper, "NoiseTools minus paper", **diff)
    im = heatmap(axes[1, 1], py - paper, "Python minus paper", **diff)
    fig.colorbar(im, ax=axes[1, :2], fraction=0.03, pad=0.02, label="peak correlation difference")
    im = heatmap(axes[1, 2], np.log10(gap), "Python vs NoiseTools:\nworst |difference| over all shifts and CCs",
                 cmap="viridis", vmin=-16, vmax=-8)
    for i in range(8):
        for j in range(6):
            axes[1, 2].text(j, i, f"{gap[i, j]:.0e}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.05, pad=0.03, label="log10 |difference|")
    return fig


def figure_subjects(paper, nt, py):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True, layout="constrained")
    x = np.arange(6)
    for ax, s, p, n, q in zip(axes.flat, SUBJECTS, paper, nt, py):
        ax.plot(x, p, "o-", color=COLORS["paper"], label="paper (digitized)")
        ax.plot(x, n, "s-", color=COLORS["nt"], ms=7, label="NoiseTools")
        ax.plot(x, q, "^--", color=COLORS["py"], ms=5, label="Python")
        ax.set_xticks(x, LABELS, rotation=40, ha="right")
        ax.set(ylim=(0, 0.42), title=s)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("peak held-out correlation")
    axes[0, 0].legend(loc="upper left", frameon=False)
    return fig


def figure_curves(subject="EL"):
    """Every held-out correlation-vs-shift curve, NoiseTools under Python, for one subject."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 6.5), gridspec_kw=dict(height_ratios=(2, 1)), sharex=True,
                             layout="constrained")
    t = np.load(PY / f"{subject}.npz")["shifts"] / 64
    for k, (m, label) in enumerate(zip(MODELS, LABELS)):
        n, q = nt_curves(subject, m), py_curves(subject, m)
        axes[0, k].plot(t, n.T, color=COLORS["nt"], lw=2.5, alpha=0.5)
        axes[0, k].plot(t, q.T, color=COLORS["py"], lw=0.7)
        axes[0, k].set(title=label, ylim=(-0.03, 0.42))
        axes[1, k].semilogy(t, np.abs(q - n).max(0) + 1e-17, color="k", lw=0.8)
        axes[1, k].set(ylim=(1e-16, 1e-8), xlabel="shift (s)")
    axes[0, 0].set_ylabel(f"{subject}: held-out correlation\n(NoiseTools thick, Python thin)")
    axes[1, 0].set_ylabel("max |Python − NoiseTools|\nover components")
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paper, nt, py, gap = load()
    for name, fig in [("agreement", figure_agreement(paper, nt, py, gap)),
                      ("subjects", figure_subjects(paper, nt, py)), ("curves_EL", figure_curves())]:
        fig.savefig(OUT / f"{name}.png", dpi=200)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
    with open(OUT / "peaks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "model", "paper", "noisetools", "python", "worst_curve_difference"])
        for i, s in enumerate(SUBJECTS):
            for j, m in enumerate(MODELS):
                w.writerow([s, m, f"{paper[i, j]:.6f}", f"{nt[i, j]:.6f}", f"{py[i, j]:.6f}", f"{gap[i, j]:.2e}"])
    print(f"worst Python vs NoiseTools difference: {gap.max():.1e}")
    for who, v in [("NoiseTools", nt), ("Python", py)]:
        print(f"{who} vs paper:", stats(paper.ravel(), v.ravel()).replace("\n", " · "))


if __name__ == "__main__":
    main()
