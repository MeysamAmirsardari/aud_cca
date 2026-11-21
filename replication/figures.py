"""Figures 2-7, drawn as the paper's MATLAB scripts drew them.

Sizes are MATLAB figure pixels at 72 per inch, so a 14-point font takes the
same share of a panel as in the article. Axes sit at MATLAB's default
positions (Figs. 2-5), or where they sit in the article's composite images
(Figs. 6 and 7, and the scalp map of Fig. 2), measured from the PDF.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.interpolate import RBFInterpolator
from scipy.signal import freqz

RATE = 64
MATLAB = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
NAMES = ["backward", "forward", "cca1", "cca2", "cca2plus", "cca3"]
LABELS = ["backward", "forward", "CCA 1", "CCA 2", "CCA 2+", "CCA 3"]
DURATIONS = [1, 2, 4, 8, 16, 32, 64]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 15.4, "axes.titleweight": "bold", "axes.labelsize": 15.4,
    "axes.linewidth": 0.5, "axes.edgecolor": "#262626", "axes.prop_cycle": cycler(color=MATLAB),
    "axes.unicode_minus": False, "axes.titlepad": 4,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.color": "#262626", "ytick.color": "#262626",
    "lines.linewidth": 0.5, "grid.color": "#262626", "grid.alpha": 0.15, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 12.6, "legend.handlelength": 1.6,
    "savefig.dpi": 300,
})
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")     # MATLAB writes 0, 0.1, -2


def draw(folder, subjects, layout):
    res = {s: dict(np.load(folder / f"{s}.npz")) for s in subjects}
    xy = np.loadtxt(layout, usecols=(1, 2), max_rows=128)
    if "EL" in res:
        el = res["EL"]
        save(figure2(el, xy), folder / "Figure_2")
        save(figure3(el), folder / "Figure_3")
        save(figure4(el), folder / "Figure_4")
        save(figure6(el, xy), folder / "Figure_6")
        save(figure7(el, res), folder / "Figure_7")
    save(figure5(res), folder / "Figure_5")


def save(fig, stem):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(PLAIN)
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


def canvas(width, height):
    return plt.figure(figsize=(width / 72, height / 72))


def mean(res, name):
    return res[name].mean(1)            # shift x component


def shift_axis(ax, shifts, ylabel=True, zero=True):
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.set_yticks(np.arange(5) * 0.1)
    ax.set(xlim=(shifts[0] / RATE, shifts[-1] / RATE), ylim=(-0.02, 0.4), xlabel="shift (s)")
    if ylabel:
        ax.set_ylabel("correlation")
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y")
    ax.axvline(0, ls=":", color=MATLAB[0])
    if zero:
        ax.axhline(0, ls=":", color=MATLAB[0])


def topoplot(ax, values, xy, limit):
    """nt_topoplot: biharmonic interpolation inside the head, jet colours.
    The head has radius 0.5 in an axes spanning -0.6..0.6."""
    r = 0.5
    g = np.linspace(-r, r, 201)
    X, Y = np.meshgrid(g, g)
    inside = X ** 2 + Y ** 2 <= r ** 2
    Z = np.full(X.shape, np.nan)
    Z[inside] = RBFInterpolator(xy, values, kernel="thin_plate_spline")(np.c_[X[inside], Y[inside]])
    im = ax.imshow(Z, extent=(-r, r, -r, r), origin="lower", cmap="jet", vmin=-limit, vmax=limit)
    a = np.linspace(0, 2 * np.pi, 101)
    ear_x = np.array([.492, .510, .518, .5299, .5419, .54, .547, .532, .510, .484])
    ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932, -.1313, -.1384, -.1199])
    for x, y in [(r * np.cos(a), r * np.sin(a)), ([.09, 0, -.09], [r - .004, r * 1.15, r - .004]),
                 (ear_x, ear_y), (-ear_x, ear_y)]:
        ax.plot(x, y, "k", lw=1.5)
    ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), aspect="equal")
    ax.axis("off")
    return im


def head(center, diameter, aspect):
    """Axes for topoplot whose head has the given centre and diameter (figure units)."""
    w, h = 1.2 * diameter, 1.2 * diameter * aspect
    return [center[0] - w / 2, center[1] - h / 2, w, h]


def colorbar(fig, im, box, limit=None):
    bar = fig.colorbar(im, cax=fig.add_axes(box), ticks=None if limit is None else [-limit, 0, limit])
    if limit is not None:
        bar.ax.set_yticklabels([f"{v:g}".replace("0.", ".") for v in (-limit, 0, limit)])
    bar.ax.tick_params(direction="in", length=3, left=True, right=True)
    return bar


def figure2(res, xy):                    # Fig_A.m
    fig = canvas(600, 300)
    ax = fig.add_axes([0.13, 0.11, 0.33466, 0.815])
    t = res["shifts"] / RATE
    ax.plot(t, res["train"], ":r", label="train")
    ax.plot(t, mean(res, "backward")[:, 0], "r", lw=1, label="test")
    shift_axis(ax, res["shifts"])
    ax.legend(loc="upper right")
    c = res["topography"][:, 0]
    im = topoplot(fig.add_axes(head((0.6944, 0.4886), 0.2117, 2)), c, xy, np.abs(c).max())
    colorbar(fig, im, [0.8416, 0.2129, 0.034, 0.6098]).set_label("correlation")
    return fig


def figure3(res):                        # Fig_B.m
    fig = canvas(600, 300)
    ax = fig.add_axes([0.13, 0.11, 0.33466, 0.815])
    ax.plot(res["shifts"] / RATE, mean(res, "forward")[:, 0], "r", lw=1)
    shift_axis(ax, res["shifts"])
    lag = np.arange(80) / RATE
    ax = fig.add_axes([0.57034, 0.58384, 0.33466, 0.34116])
    ax.plot(lag, res["fir"], color="b")
    ax.plot(lag, res["fir"], "o", color="b", ms=6.67, mew=0)
    ax.plot([0, lag[-1]], [0, 0], "k", lw=0.1)
    ax.set(xlim=(0, lag[-1]), xticks=[0, 0.5, 1], yticks=[], xlabel="s")
    ax = fig.add_axes([0.57034, 0.11, 0.33466, 0.34116])
    f, h = freqz(res["fir"], 1, 512, fs=RATE)
    ax.plot(f, np.abs(h) / np.abs(h).max())
    ax.set(xlim=(0, f[-1]), ylim=(0, 1.1), yticks=[0, 1], xlabel="Hz")
    return fig


def figure4(res):                        # Fig_D.m
    fig = canvas(600, 300)
    t = res["shifts"] / RATE
    for k, name in enumerate(["cca1", "cca2", "cca2plus", "cca3"]):
        ax = fig.add_axes([0.13 + k * 0.20612, 0.11, 0.15665, 0.815])
        a = mean(res, name)
        # nt_linecolors([],[2 3 1]): CC1 orange and thick, CC2 yellow, CC3 blue, then the cycle
        for c in range(a.shape[1]):
            colour = MATLAB[[1, 2, 0][c]] if c < 3 else MATLAB[c % 7]
            ax.plot(t, a[:, c], color=colour, lw=2 if c == 0 else 0.5)
        shift_axis(ax, res["shifts"], ylabel=k == 0, zero=False)
        ax.set_title(["CCA model 1", "CCA model 2", "CCA model 2+", "CCA model 3"][k])
    return fig


def figure5(res):                        # Fig_F.m, figure 21
    fig = canvas(600, 450)
    ax = fig.add_axes([0.13, 0.11, 0.775, 0.815])
    peaks = np.array([[mean(r, m)[:, 0].max() for m in NAMES] for r in res.values()])
    x = np.arange(1, 7)
    for i, p in enumerate(peaks):
        colour, width = ("b", 2) if i == 0 else (MATLAB[(i - 1) % 7], 0.5)
        ax.plot(x, p, "o-", color=colour, lw=width, ms=10, mfc="none", label=f"s{i + 1}")
    top = max(0.4, np.ceil(peaks.max() * 10) / 10)
    ax.set_xticks(x, LABELS)
    ax.set_yticks(np.arange(0, top + 0.01, 0.1))
    ax.set(xlim=(0.5, 6.5), ylim=(0, top), xlabel="model", ylabel="correlation")
    ax.legend(loc="upper left")
    return fig


def figure6(res, xy):                    # Fig_E.m; positions from the article's image
    fig = canvas(584, 794)
    px = lambda x0, y0, x1, y1: [x0 / 1325, 1 - y1 / 1803, (x1 - x0) / 1325, (y1 - y0) / 1803]
    for k in range(12):
        row, col = divmod(k, 4)
        ax = fig.add_axes(px(74 + 314.3 * col, 42 + 273.2 * row, 306 + 314.7 * col, 242 + 273.4 * row))
        f, h = freqz(res["filters"][:, k], 1, 512, fs=RATE)
        ax.semilogx(f[1:], np.sqrt(np.abs(h[1:]) / np.abs(h).max()))
        ax.set(xlim=(0.05, 20), ylim=(0, 1.1), xticks=[0.1, 1, 10], title=f"CC {k + 1}")
        ax.set_xticklabels(["0.1", "1", "10"] if row == 2 else [])
        ax.set_yticks([0, 1] if col == 0 else [])
        ax.grid(axis="x", which="major")
        if k == 4:
            ax.set_ylabel("normalized amplitude")
        if row == 2:
            ax.set_xlabel("frequency (Hz)")

        c = res["maps"][:, k]
        # Fig_E.m's colour-scale ladder, with steps above 0.1 added
        limit = next(v for v in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] if np.abs(c).max() < v)
        centre = ((140 + 314.67 * col) / 1325, 1 - (1095 + 307 * row) / 1803)
        ax = fig.add_axes(head(centre, 170 / 1325, 1325 / 1803))
        im = topoplot(ax, c, xy, 1.1 * limit)
        ax.set_title(f"CC {k + 1}")
        colorbar(fig, im, px(272 + 314.67 * col, 996 + 307 * row, 301 + 314.67 * col, 1183 + 307 * row), limit)
    return fig


def figure7(el, res):                    # Fig_H/J/H5/H6.m; positions from the article's image
    fig = canvas(800, 301)
    dot = dict(marker="o", ms=6, mew=0)

    ax = fig.add_axes([0.0408, 0.1268, 0.1888, 0.8122])
    for c in range(6):
        ax.semilogx(DURATIONS, el["cca3_cc_dprime"][:, c], color="r" if c == 0 else MATLAB[(c - 1) % 7],
                    lw=2 if c == 0 else 0.5, label=f"CC{c + 1}", **dot)
    duration_axis(ax, range(5), "d-prime", "(a)")
    ax.legend(loc="upper left")

    ax = fig.add_axes([0.2891, 0.3387, 0.1894, 0.5987])
    r = np.concatenate([s["cca3"].mean(1)[s["cca3"].mean(1)[:, 0].argmax(), :12] for s in res.values()])
    d = np.concatenate([s["cca3_cc_dprime"][-1] for s in res.values()])
    centroid = np.concatenate([s["cca3_centroid"] for s in res.values()])
    dots = ax.scatter(r, d, s=60, c=np.log2(centroid), cmap="jet", edgecolors="k", linewidths=0.5)
    ax.set_xticks(np.arange(0, 0.41, 0.1))
    ax.set_yticks(range(4))
    ax.set(xlim=(0, 0.4), ylim=(0, 3), xlabel="correlation", ylabel="d-prime", title="(b)")
    bar = fig.colorbar(dots, cax=fig.add_axes([0.2891, 0.114, 0.1894, 0.0433]),
                       orientation="horizontal", ticks=[-1, 0, 1, 2])
    bar.ax.set_xticklabels(["0.5", "1", "2", "4"])
    bar.ax.invert_xaxis()
    bar.set_label("spectrum centroid (Hz)")

    for box, key, ticks, ylabel, title in [([0.5396, 0.1268, 0.1891, 0.8122], "dprime", range(5), "d-prime", "(c)"),
                                           ([0.8072, 0.1268, 0.1813, 0.8122], "correct", range(50, 101, 10),
                                            "% correct", "(d)")]:
        ax = fig.add_axes(box)
        ax.semilogx(DURATIONS, el[f"cca3_joint_{key}"], color="b", lw=2, label="CCA, joint", **dot)
        ax.semilogx(DURATIONS, el[f"cca3_best_{key}"], color=MATLAB[0], label="CCA, best CC", **dot)
        ax.semilogx(DURATIONS, el[f"backward_{key}"], color=MATLAB[1], label="backward", **dot)
        ax.semilogx(DURATIONS, el[f"forward_{key}"], color=MATLAB[2], label="forward", **dot)
        if key == "correct":
            ax.semilogx(DURATIONS, el["cca3_joint_chance"], "--k")
        else:
            ax.legend(loc="upper left")
        duration_axis(ax, ticks, ylabel, title)
    return fig


def duration_axis(ax, ticks, ylabel, title):
    ax.set_xticks(DURATIONS, [str(d) for d in DURATIONS])
    ax.set_yticks(ticks)
    ax.set(xlim=(1, 64), ylim=(ticks[0], ticks[-1]), xlabel="duration (s)", ylabel=ylabel, title=title)
    ax.minorticks_off()


if __name__ == "__main__":                 # redraw from saved results
    import replicate
    done = [s for s in replicate.SUBJECTS if (replicate.OUT / f"{s}.npz").exists()]
    draw(replicate.OUT, done, replicate.LAYOUT)
