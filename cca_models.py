"""Stimulus-response decoding with canonical correlation analysis.

Alain de Cheveigne's CCA (de Cheveigne et al., NeuroImage 2018)

The API copies scikit-learn (fit / score, fitted attributes ending in an underscore)

Data everywhere: `eeg` a (n_samples, n_channels) array or a list of such trial arrays;
`env` a (n_samples,) or (n_samples, 1) array or a list of them.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Model:
    """Base class for decoding models and shared utilities.

    This class holds common configuration options used by the concrete model
    implementations (`CCA` and `Regression`) and implements shared feature
    preparation and numerical helpers.

    Args:
        type (str): Model type identifier ("forward", "backward", or "cca").
        eeg_basis (callable or None): Function to transform EEG trials into features.
        stim_basis (callable or None): Function to transform stimulus trials into features.
        pre_pca (int or None): Number of principal components to keep when pre-reducing EEG.
        eeg_keep (int or None): Number of eigen-directions to retain in EEG whitener.
        stim_keep (int or None): Number of eigen-directions to retain in stimulus whitener.
        n_components (int or None): Number of canonical components to retain (CCA only).
        rcond (float): Relative cutoff for small eigenvalues when building whiteners.

    Attributes:
        type, eeg_basis, stim_basis, pre_pca, eeg_keep, stim_keep, n_components, rcond
            Stored initialization parameters used by subclasses.
    """

    def __init__(
        self,
        type: str,
        eeg_basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        stim_basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        pre_pca: Optional[int] = None,
        eeg_keep: Optional[int] = None,
        stim_keep: Optional[int] = None,
        n_components: Optional[int] = None,
        rcond: float = 1e-8,
    ) -> None:
        self.type = type
        self.eeg_basis = eeg_basis
        self.stim_basis = stim_basis
        self.pre_pca = pre_pca
        self.eeg_keep = eeg_keep
        self.stim_keep = stim_keep
        self.n_components = n_components
        self.rcond = rcond



    @staticmethod
    def _eeg(
        eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        pre_pca: Optional[int],
        basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]],
        pca: Optional[NDArray[Any]],
    ) -> Tuple[List[NDArray[Any]], Optional[NDArray[Any]]]:
        """Prepare EEG-side features for fit/score.

        The method converts the input into a list of trials, optionally fits or
        reuses a PCA reduction (when ``pca`` is None it is fitted on the provided
        trials) and then applies the provided feature ``basis``.

        Args:
            eeg (array or list): (n_samples, n_channels) array or list of trial arrays.
            pre_pca (int or None): Number of PCA components to apply before basis.
            basis (callable or None): Feature function to apply to each trial.
            pca (array or None): Pre-fitted PCA map to reuse (shape (n_channels, k)).

        Returns:
            tuple: ``(trials, pca)`` where ``trials`` is the list of transformed trial arrays
            and ``pca`` is the fitted or reused PCA map.
        """
        trials = Model._trials(eeg)
        if pre_pca:
            if pca is None:
                pca = Model._fit_pca(trials, pre_pca)
            trials = [t @ pca for t in trials]
        return Model._apply(basis, trials), pca

    @staticmethod
    def _apply(
        basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]],
        trials: List[NDArray[Any]],
    ) -> List[NDArray[Any]]:
        """Apply a feature basis to each trial.

        If ``basis`` is ``None``, the input ``trials`` list is returned unchanged.

        Args:
            basis (callable or None): Function to transform a single trial array.
            trials (list): List of (n_samples, n_features) arrays.

        Returns:
            list: List of transformed trial arrays.
        """
        return trials if basis is None else [basis(t) for t in trials]

    @staticmethod
    def _trials(view: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> List[NDArray[Any]]:
        """Normalize input into a list of 2-D trial arrays.

        Single arrays are wrapped in a list. 1-D signals are converted to
        (n_samples, 1) column arrays.

        Args:
            view (array or list): Array or list of arrays representing trials.

        Returns:
            list: List of 2-D NumPy arrays with shape (n_samples, n_features).
        """
        if isinstance(view, np.ndarray):
            view = [view]
        out = []
        for x in view:
            x = np.asarray(x, float)
            out.append(x[:, None] if x.ndim == 1 else x)
        return out

    # ---- numeric core (the algorithm) ---------------------------------------

    @staticmethod
    def _covariances(
        X: Union[NDArray[Any], Sequence[NDArray[Any]]],
        Y: Union[NDArray[Any], Sequence[NDArray[Any]]],
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Compute pooled, mean-removed covariances across trials.

        This routine computes the block covariances for two multivariate views
        without concatenating all trials into a single array, returning the
        covariances and per-view means.

        Args:
            X (array or list): (n_samples, n_features) array or list of trial arrays.
            Y (array or list): (n_samples, n_features) array or list of trial arrays.

        Returns:
            tuple: ``(Cxx, Cyy, Cxy, mx, my)`` where Cxx and Cyy are the auto-covariances,
            Cxy the cross-covariance, and ``mx``/``my`` the pooled means of each view.
        """
        Xs, Ys = Model._trials(X), Model._trials(Y)
        n = sum(len(x) for x in Xs)
        mx = sum(x.sum(0) for x in Xs) / n
        my = sum(y.sum(0) for y in Ys) / n
        p, q = Xs[0].shape[1], Ys[0].shape[1]
        Cxx, Cyy, Cxy = np.zeros((p, p)), np.zeros((q, q)), np.zeros((p, q))
        for x, y in zip(Xs, Ys):
            x, y = x - mx, y - my
            Cxx += x.T @ x
            Cyy += y.T @ y
            Cxy += x.T @ y
        return Cxx / n, Cyy / n, Cxy / n, mx, my

    @staticmethod
    def _whitener(cov: NDArray[Any], keep: Optional[int], rcond: float) -> NDArray[Any]:
        """Construct a whitening transform from a covariance matrix.

        The function diagonalizes the symmetric covariance, thresholds small
        eigenvalues using ``rcond`` and optionally keeps only the top ``keep``
        components. The returned matrix ``W`` satisfies that ``X @ W`` has
        approximately identity covariance when ``X`` has covariance ``cov``.

        Args:
            cov (ndarray): Square covariance matrix (n_features, n_features).
            keep (int or None): Number of principal directions to retain; ``None`` keeps
                all directions above the relative cutoff.
            rcond (float): Relative cutoff for small eigenvalues (multiplied by max).

        Returns:
            ndarray: Whitening matrix ``W`` with shape (n_features, n_kept).
        """
        ev, V = np.linalg.eigh(0.5 * (cov + cov.T))
        ev, V = ev[::-1], V[:, ::-1]
        mask = ev > rcond * ev[0]
        if keep is not None:
            mask[keep:] = False
        ev, V = ev[mask], V[:, mask]
        return V/np.sqrt(ev)

    @staticmethod
    def _fit_pca(trials: List[NDArray[Any]], k: int) -> NDArray[Any]:
        """Fit a PCA map that reduces channel dimensionality to ``k`` components.

        Args:
            trials (list): List of (n_samples, n_channels) arrays.
            k (int): Number of principal components to retain.

        Returns:
            ndarray: PCA map with shape (n_channels, k) whose columns are the top-k
            principal directions.
        """
        n = sum(len(t) for t in trials)
        mu = sum(t.sum(0) for t in trials) / n
        cov = sum((t - mu).T @ (t - mu) for t in trials) / n
        _ev, V = np.linalg.eigh(0.5 * (cov + cov.T))
        return V[:, ::-1][:, :k]

    @staticmethod
    def _correlate(A: NDArray[Any], B: NDArray[Any]) -> NDArray[Any]:
        """Compute per-column Pearson correlations between aligned matrices.

        Both inputs must have the same shape ``(n_samples, k)`` and are column-wise
        mean-centered before correlation is computed.

        Args:
            A (ndarray): Left matrix of shape (n_samples, k).
            B (ndarray): Right matrix of shape (n_samples, k).

        Returns:
            ndarray: 1-D array of length ``k`` containing Pearson correlation values
            for each column pair.
        """
        A, B = A - A.mean(0), B - B.mean(0)
        denom = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)
        return np.divide((A * B).sum(0), denom, out=np.zeros(A.shape[1]), where=denom > 0)

    # feature helpers:

    @staticmethod
    def _time_lag(x: Union[NDArray[Any], Sequence[Any]], n_lags: int) -> NDArray[Any]:
        """Build a time-lagged feature matrix for a signal.

        The output contains lagged copies of the input signal from lag 0 up to
        ``n_lags - 1`` arranged in lag-major order.

        Args:
            x (ndarray): Input array of shape (n_samples, n_channels) or (n_samples,).
            n_lags (int): Number of lagged copies to include.

        Returns:
            ndarray: Array with shape (n_samples, n_channels * n_lags).
        """
        x = Model._as_2d(x)
        n, c = x.shape
        out = np.zeros((n, n_lags, c))
        for lag in range(n_lags):
            out[lag:, lag, :] = x[:n - lag]
        return out.reshape(n, n_lags * c)

    @staticmethod
    def _smoother(
        x: Union[NDArray[Any], Sequence[Any]],
        n_bands: int = 21,
        min_samples: int = 2,
        max_samples: int = 128,
    ) -> NDArray[Any]:
        """Create a bank of causal moving-average filters of different widths.

        Each channel from ``x`` is replaced by its moving average computed over a set
        of log-spaced window lengths between ``min_samples`` and ``max_samples``.

        Args:
            x (ndarray): Input signal with shape (n_samples, n_channels) or (n_samples,).
            n_bands (int): Number of filter widths (bands) to generate.
            min_samples (int): Minimum window length in samples.
            max_samples (int): Maximum window length in samples.

        Returns:
            ndarray: Array with shape (n_samples, n_channels * n_widths) containing
            the filtered channels concatenated along the feature axis.
        """
        x = Model._as_2d(x)
        n, c = x.shape
        widths = sorted({int(round(w)) for w in np.geomspace(min_samples, max_samples, n_bands)})
        out = np.empty((n, len(widths), c))
        for i, w in enumerate(widths):
            out[:, i, :] = Model._moving_average(x, w)
        return out.reshape(n, len(widths) * c)

    @staticmethod
    def _moving_average(x: NDArray[Any], w: int) -> NDArray[Any]:
        """Compute a causal boxcar (moving average) of width ``w``.

        For the first ``w`` samples partial averages are used (averaging fewer
        than ``w`` points).

        Args:
            x (ndarray): Input array with shape (n_samples, n_channels).
            w (int): Window length in samples.

        Returns:
            ndarray: Smoothed array with the same shape as ``x``.
        """
        if w <= 1:
            return x.copy()
        csum = np.cumsum(x, axis=0)
        out = np.empty_like(x)
        out[:w] = csum[:w] / np.arange(1, w + 1)[:, None]
        out[w:] = (csum[w:] - csum[:-w]) / w
        return out

    @staticmethod
    def _as_2d(x: Union[NDArray[Any], Sequence[Any]]) -> NDArray[Any]:
        """Ensure an array is two-dimensional.

        A 1-D array is converted to shape ``(n_samples, 1)``; a 2-D array is
        returned unchanged.

        Args:
            x (array): Input array of 1 or 2 dimensions.

        Returns:
            ndarray: 2-D NumPy array.
        """
        x = np.asarray(x, float)
        return x if x.ndim == 2 else x[:, None]


class CCA(Model):
    """Canonical correlation analysis between EEG and stimulus feature views.

    The algorithm whitens both views, computes the SVD of the whitened
    cross-covariance and returns the singular vectors as weights and the
    singular values as canonical correlations.

    This class implements a scikit-learn-like ``fit`` / ``score`` API.
    """

    def fit(self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], env: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> "CCA":
        """Fit a CCA model to EEG and stimulus data.

        Args:
            eeg (array or list): EEG trials as an array or list of arrays.
            env (array or list): Stimulus/envelope trials as an array or list.

        Returns:
            CCA: ``self`` fitted in-place with attributes ``x_weights_``, ``y_weights_``
            and ``canonical_correlations_``.
        """

        E, self.pca_ = self._eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._trials(env))
        Cxx, Cyy, Cxy, self.x_mean_, self.y_mean_ = self._covariances(E, S)
        Wx = self._whitener(Cxx, self.eeg_keep, self.rcond)
        Wy = self._whitener(Cyy, self.stim_keep, self.rcond)
        U, s, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        k = self.n_components or len(s)
        self.x_weights_ = Wx @ U[:, :k]
        self.y_weights_ = Wy @ Vt[:k].T
        self.canonical_correlations_ = s[:k]
        return self

    def score(self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], env: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> NDArray[Any]:
        """Compute per-component correlations on new data using fitted weights.

        The method projects new EEG and stimulus trials using the fitted weights
        and returns the Pearson correlation per canonical component.

        Args:
            eeg (array or list): New EEG trials.
            env (array or list): New stimulus trials.

        Returns:
            ndarray: 1-D array of canonical correlations (one per component).
        """
        E, _ = self._eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._trials(env))
        sx = (np.vstack(E) - self.x_mean_) @ self.x_weights_
        sy = (np.vstack(S) - self.y_mean_) @ self.y_weights_
        return self._correlate(sx, sy)


class Regression(Model):
    """Regularized least-squares regression for encoding/decoding.

    This class implements both forward (predict EEG from stimulus) and backward
    (reconstruct stimulus from EEG) regression depending on the ``type``
    attribute of the instance.
    """

    def fit(self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], env: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> "Regression":
        """Fit a regularized least-squares map.

        For ``type=='backward'`` the model learns to predict stimulus from EEG.
        For ``type=='forward'`` the model learns to predict EEG channels from the
        stimulus feature representation.

        Args:
            eeg (array or list): EEG trials.
            env (array or list): Stimulus trials.

        Returns:
            Regression: ``self`` fitted in-place with attribute ``coef_``.
        """

        E, self.pca_ = self._eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._trials(env))
        X, Y, keep = ((E, self._trials(env), self.eeg_keep) if self.type == "backward"
                      else (S, self._trials(eeg), self.stim_keep))
        Cxx, _Cyy, Cxy, self.x_mean_, self.y_mean_ = self._covariances(X, Y)
        Wx = self._whitener(Cxx, keep, self.rcond)
        self.coef_ = (Wx @ Wx.T) @ Cxy
        return self

    def score(self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], env: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> NDArray[Any]:
        """Score predictions as per-output Pearson correlations.

        For forward models this returns correlations per EEG channel (caller may
        wish to take the maximum). For backward models it returns the
        reconstruction correlation of the stimulus.

        Args:
            eeg (array or list): EEG trials.
            env (array or list): Stimulus trials.

        Returns:
            ndarray: 1-D array of correlation values, one per target dimension.
        """
        E, _ = self._eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._trials(env))
        X, Y = (E, self._trials(env)) if self.type == "backward" else (S, self._trials(eeg))
        pred = (np.vstack(X) - self.x_mean_) @ self.coef_ + self.y_mean_
        return self._correlate(pred, np.vstack(Y))


MODEL_PRESETS = {
    "forward": dict(type="forward", stim_basis=partial(Model._time_lag, n_lags=80)),
    "backward": dict(type="backward", eeg_keep=80),
    "cca1": dict(type="cca", stim_basis=partial(Model._time_lag, n_lags=40),
                 eeg_keep=40, stim_keep=40, n_components=40),
    "cca2": dict(type="cca", eeg_basis=partial(Model._time_lag, n_lags=10),
                 stim_basis=partial(Model._time_lag, n_lags=40),
                 pre_pca=80, eeg_keep=40, stim_keep=40, n_components=40),
    "cca2plus": dict(type="cca", eeg_basis=partial(Model._time_lag, n_lags=10),
                     stim_basis=partial(Model._time_lag, n_lags=80),
                     pre_pca=80, eeg_keep=80, stim_keep=80, n_components=80),
    "cca3": dict(type="cca", eeg_basis=Model._smoother, stim_basis=Model._smoother,
                 pre_pca=60, eeg_keep=139, n_components=21),
}

MODEL_TYPES = {"cca": CCA, "forward": Regression, "backward": Regression}

def model(name: str) -> Model:
    """Instantiate a preset model by name.

    Args:
        name (str): Key name of a preset in ``_PRESETS`` (for example, "cca3").

    Returns:
        Model: An instance of the configured model class (``CCA`` or ``Regression``).
    """
    assert name in MODEL_PRESETS, f"Unknown model preset '{name}'"
    params = MODEL_PRESETS[name]
    assert params["type"] in MODEL_TYPES, f"Unknown model type '{params['type']}'"
    return MODEL_TYPES[params["type"]](**params)