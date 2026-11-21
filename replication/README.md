# Replicating de Cheveigné et al. (2018)

`replicate.py` reads the prepared recordings, fits the paper's six models with
`cca_models` as it is on main, and saves the scores, the fitted models and
Figures 2–7. `figures.py` only draws.

    python replication/replicate.py            # all eight subjects
    python replication/replicate.py EL         # one subject
    python replication/figures.py              # redraw from saved results
    python replication/compare.py              # paper vs NoiseTools vs Python

Subjects can run in separate shells (about 7 GB each), then `figures.py`
redraws the cohort figures.

Inputs, set at the top of `replicate.py` and not in git:

    DATA     one .mat per subject: 20 trials of 80 EEG PCs, the compressed
             envelope and the 128 referenced sensors at 64 Hz; the SHA-256
             sums stored with them are checked on load
    LAYOUT   FieldTrip's biosemi128.lay, for the scalp maps

Outputs go to `replication/results/`: `<subject>.npz` (every held-out score
by shift, trial and component, and what the figures need),
`<subject>_models.pkl` (each model refitted on all trials at its best shift)
and `Figure_2..7.png/.pdf`.

## What comes from where

    model sizes      the paper: 80 PCs (backward), 80 lags (forward),
                     40 PCs + 40 lags (CCA 1), 60 PCs x 10 lags -> 40 PCs
                     + 40 lags (CCA 2) or 80 + 80 (CCA 2+), 60 PCs through the
                     filterbank -> 139 PCs, with the filterbank on the audio (CCA 3)
    filterbank       differences of boxcars of widths unique(round(2.^(0:0.25:7))),
                     23 bands: the bank that drew the published Figure 6
                     (the text says 21 bands; change WIDTHS to try another)
    shifts           -192:30:448, -100:5:100, -40:2:40 samples, as the scripts
    forward target   the sensor that best correlates with the envelope at
                     the backward model's best shift (Fig. 2 map, Fig. 3)
    Figure 6         CCA 3 refitted on all trials at its best shift; the
                     filters are drawn the way Fig_E.m drew them
    Figure 7         LDA on segment correlations, one trial left out; the
                     mismatch is the next trial's audio; every trial's
                     correlations come from the model fitted without it;
                     1000 label permutations give the dashed line
    trial handling   NoiseTools: nt_relshift, nt_multishift, nt_multismooth
                     within each channel, nt_pca over all trials, each
                     trial's mean removed (nt_cov_lags)

Only fitting and scoring use the package. It is called with its bases and
PCA switched off because main's `pre_pca` fails on a list of trials, and its
lags look back and pad where NoiseTools' look ahead and trim.
