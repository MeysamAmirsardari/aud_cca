"""Replicate the figures of de Cheveigne et al., NeuroImage 172 (2018) 206-216.

Reads the prepared recordings (one .mat per subject: 20 trials of 80 EEG PCs,
the compressed envelope and the 128 referenced sensors, all at 64 Hz), fits
the paper's six models with cca_models, leaving one trial out at every shift,
and writes the scores, the fitted models and Figures 2-7 to OUT.

    python replication/replicate.py              # all eight subjects
    python replication/replicate.py EL           # one subject

Model sizes are the paper's. The trial handling around the package (shifts,
lags, filterbank, PCA over all trials, per-trial centering) is NoiseTools',
so the scores can be checked against nt_cca_crossvalidate on the same inputs.
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cca_models import CCA, BackwardRegressionModel, ForwardRegressionModel
import figures

DATA = ROOT / "paper_replication/work/shared_inputs/common"
LAYOUT = ROOT / "paper_replication/source/biosemi128.lay"    # FieldTrip's, not shipped here
OUT = ROOT / "replication/results"

SUBJECTS = ["EL", "AC", "BQ", "ChW", "JOS", "MC", "NM", "TT"]  # s1..s8 in Fig. 5
SHIFTS = np.unique(np.r_[-192:449:30, -100:101:5, -40:41:2])    # audio delay, in samples
WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 19, 23, 27, 32, 38, 45, 54, 64, 76, 91, 108, 128]
DURATIONS = [1, 2, 4, 8, 16, 32, 64]                            # Fig. 7, seconds
RATE = 64
MODELS = {"backward": BackwardRegressionModel, "forward": ForwardRegressionModel,
          "cca1": CCA, "cca2": CCA, "cca2plus": CCA, "cca3": CCA}


def run(subject):
    eeg, env, sensors = load(subject)
    x = {m: features(m, eeg, env) for m in MODELS if m != "forward"}
    r = {m: crossvalidate(m, *x[m]) for m in x}
    topo = topography(sensors, env, best(r["backward"]))
    channel = np.abs(topo[:, 0]).argmax()           # Fig. 3 predicts the best sensor
    x["forward"] = [s[:, [channel]] for s in sensors], lags(env, 80)
    r["forward"] = crossvalidate("forward", *x["forward"])
    models = {m: fit(m, *at(best(r[m]), *x[m])) for m in MODELS}

    weights = models["cca3"].y_weights_
    results = dict(r, shifts=SHIFTS, topography=topo, channel=channel,
                   train=training_curve(*x["backward"]),
                   fir=fold_one(*x["forward"], best(r["backward"])),
                   filters=bank([np.eye(129, 1)])[0] @ weights,   # impulse responses, as Fig_E.m
                   maps=topography(sensors, [t @ weights for t in x["cca3"][1]], best(r["cca3"])),
                   **classification(x, r, permutations=1000 if subject == "EL" else 0))
    np.savez(OUT / f"{subject}.npz", **results)
    with open(OUT / f"{subject}_models.pkl", "wb") as f:
        pickle.dump(models, f)


def load(subject):
    d = loadmat(DATA / f"{subject}.mat", squeeze_me=True)
    out = []
    for key, tag in [("eeg", "eeg"), ("env", "env"), ("sensor_eeg", "sensor")]:
        trials = [np.asarray(t, float).reshape(len(t), -1) for t in d[key]]
        sums = [hashlib.sha256(t.astype("<f8").tobytes(order="F")).hexdigest() for t in trials]
        assert sums == list(d[f"{tag}_sha256"]), f"{subject}: {key} differs from the export"
        out.append(trials)
    return out


# The NoiseTools steps that build each model's inputs.

def pca(trials, n):                     # nt_pca: uncentred, all trials pooled
    v = np.linalg.eigh(sum(t.T @ t for t in trials))[1][:, ::-1]
    return [t @ v[:, :n] for t in trials]


def lags(trials, n):                    # nt_multishift(x, 0:n-1)
    return [np.hstack([t[k:len(t) - n + 1 + k] for k in range(n)]) for t in trials]


def smooth(x, w):                       # nt_smooth with an integer width
    m = x[:w + 1].mean(0)
    c = np.cumsum(x - m, 0)
    c[w:] = c[w:] - c[:-w]
    return c / w + m


def bank(trials):                       # differences of adjacent smoothers, per channel
    return [np.diff(np.stack([smooth(t, w) for w in WIDTHS], 2), axis=2).reshape(len(t), -1)
            for t in trials]


def features(name, eeg, env):
    if name == "cca1":
        return pca(eeg, 40), lags(env, 40)
    if name in ("cca2", "cca2plus"):
        k = 40 if name == "cca2" else 80
        return pca(lags(pca(eeg, 60), 10), k), lags(env, k)
    if name == "cca3":
        return pca(bank(pca(eeg, 60)), 139), bank(env)
    return eeg, env                     # backward: the 80 PCs


def at(shift, x, y):                    # nt_relshift, then remove each trial's mean
    xs, ys = [], []
    for a, b in zip(x, y):
        n = min(len(a), len(b))
        a, b = (a[shift:n], b[:n - shift]) if shift >= 0 else (a[:n + shift], b[-shift:n])
        xs.append(a - a.mean(0))
        ys.append(b - b.mean(0))
    return xs, ys


# Fitting and scoring are the package's.

def fit(name, x, y):
    model = MODELS[name](rcond=1e-12)   # nt_cca's eigenvalue threshold
    model.fit(x, y)
    return model


def crossvalidate(name, x, y):
    """Held-out correlation for every shift, trial and component."""
    scores = []
    for s in SHIFTS:
        xs, ys = at(s, x, y)
        scores.append([fit(name, xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:]).score(xs[i], ys[i])
                       for i in range(len(xs))])
    return np.array(scores)


def best(scores):
    return SHIFTS[scores.mean(1)[:, 0].argmax()]


def training_curve(x, y):               # Fig. 2, dotted: no cross-validation
    curve = []
    for s in SHIFTS:
        xs, ys = at(s, x, y)
        curve.append(fit("backward", xs, ys).score(xs, ys)[0])
    return np.array(curve)


def fold_one(x, y, shift):              # Fig. 3 shows the fold that leaves out trial 1
    xs, ys = at(shift, x, y)
    return fit("forward", xs[1:], ys[1:]).coef_[:, 0]


def topography(sensors, signals, shift):
    """Correlation of each sensor with each signal, pooled over trials."""
    s, w = at(shift, sensors, signals)
    unit = lambda t: t / np.sqrt((t ** 2).mean(0))
    return sum(unit(a).T @ unit(b) for a, b in zip(s, w)) / sum(len(a) for a in s)


# Fig. 7: match vs mismatch on each trial, from the model fitted without it.

def classification(x, r, permutations):
    out = {}
    for name in ("backward", "forward"):
        c = classify(held_out(name, *x[name], best(r[name])), [0])
        out.update({f"{name}_{k}": v for k, v in c.items()})
    trials = held_out("cca3", *x["cca3"], best(r["cca3"]))
    for tag, cc, n in [("joint", range(6), permutations), ("best", [0], 0)]:
        out.update({f"cca3_{tag}_{k}": v for k, v in classify(trials, cc, n).items()})
    out["cca3_cc_dprime"] = np.array([dprime(*pooled(trials, d))[:12] for d in DURATIONS])
    f, p = welch(np.vstack([a for _, a, _ in trials]), RATE, nperseg=1024, axis=0)
    out["cca3_centroid"] = (f @ p / p.sum(0))[:12]
    return out


def held_out(name, x, y, shift):
    xs, ys = at(shift, x, y)
    out = []
    for i in range(len(xs)):
        model = fit(name, xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        j = (i + 1) % len(xs)           # the mismatch is the next trial's audio
        out.append((*waves(name, model, xs[i], ys[i]), waves(name, model, xs[j], ys[j])[1]))
    return out


def waves(name, model, x, y):
    """The EEG-side and audio-side signals a model correlates."""
    if name == "backward":
        return (x - model.x_mean_) @ model.coef_, y
    if name == "forward":
        return x, (y - model.x_mean_) @ model.coef_
    return (x - model.x_mean_) @ model.x_weights_, (y - model.y_mean_) @ model.y_weights_


def segments(e, a, m, seconds):
    """Correlation of each EEG segment with its own audio and with the mismatch."""
    n = seconds * RATE
    k = min(len(e), len(a), len(m)) // n

    def unit(t):
        t = t[:k * n].reshape(k, n, -1)
        return (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)

    e, a, m = unit(e), unit(a), unit(m)
    return (e * a).mean(1), (e * m).mean(1)


def pooled(trials, seconds):
    match, other = zip(*[segments(*t, seconds) for t in trials])
    return np.vstack(match), np.vstack(other)


def dprime(a, b):                       # nt_dprime
    return np.abs(a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)


def classify(trials, components, permutations=0):
    """LDA on the segment correlations, one trial left out; d' and % correct per duration."""
    rng = np.random.default_rng(0)
    out = {"dprime": [], "correct": [], "chance": []}
    for d in DURATIONS:
        X = [np.vstack(segments(*t, d))[:, list(components)] for t in trials]
        L = [np.repeat([1, 0], len(t) // 2) for t in X]
        correct, held = lda(X, L)
        labels = np.concatenate(L)
        split = np.cumsum([len(t) for t in L])[:-1]
        shuffled = [lda(X, np.split(rng.permutation(labels), split))[0] for _ in range(permutations)]
        out["correct"].append(correct)
        out["dprime"].append(dprime(held[labels == 1], held[labels == 0]))
        out["chance"].append(np.percentile(shuffled, 95) if shuffled else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def lda(X, L):
    hits, held = 0, []
    for i in range(len(X)):
        x, l = np.vstack(X[:i] + X[i + 1:]), np.concatenate(L[:i] + L[i + 1:])
        a, b = x[l == 1], x[l == 0]
        w = np.linalg.solve(np.atleast_2d(np.cov(a.T) + np.cov(b.T)), a.mean(0) - b.mean(0))
        held.append((X[i] - (a.mean(0) + b.mean(0)) / 2) @ w)
        hits += ((held[-1] > 0) == (L[i] == 1)).sum()
    return 100 * hits / sum(map(len, L)), np.concatenate(held)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = sys.argv[1:] or SUBJECTS
    for subject in subjects:
        print(subject, flush=True)
        run(subject)
    figures.draw(OUT, subjects, LAYOUT)
