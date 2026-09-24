"""Stimulus-response decoding with canonical correlation analysis.

Alain de Cheveigne's CCA (de Cheveigne et al., NeuroImage 2018)

The API copies scikit-learn (fit / score, fitted attributes ending in an underscore)

Data everywhere: `eeg` a (n_samples, n_channels) array or a list of such trial arrays;
`audio` a (n_samples,) or (n_samples, 1) array or a list of them.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Model:
    """Base class for decoding models and shared utilities.

    This class holds common configuration options used by the concrete model
    implementations (`CCA` and `Regression`) and implements shared feature
    preparation and numerical helpers.

    Args:
        eeg_basis (callable or None): Function to transform EEG trials into features.
        stim_basis (callable or None): Function to transform stimulus trials into features.
        pre_pca (int or None): Number of principal components to keep when pre-reducing EEG.
        eeg_keep (int or None): Number of eigen-directions to retain in EEG pre-whitener.
        stim_keep (int or None): Number of eigen-directions to retain in stimulus pre-whitener.
        n_components (int or None): Number of canonical components to retain (CCA only).
        rcond (float): Relative cutoff for small eigenvalues when building whiteners.

    Attributes:
        eeg_basis, stim_basis, pre_pca, eeg_keep, stim_keep, n_components, rcond
            Stored initialization parameters used by subclasses.
    """

    def __init__(
        self,
        eeg_basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        stim_basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        pre_pca: Optional[int] = None,
        eeg_keep: Optional[int] = None,
        stim_keep: Optional[int] = None,
        n_components: Optional[int] = None,
        rcond: float = 1e-8,
    ) -> None:
        self.eeg_basis = eeg_basis
        self.stim_basis = stim_basis
        self.pre_pca = pre_pca
        self.eeg_keep = eeg_keep
        self.stim_keep = stim_keep
        self.n_components = n_components
        self.rcond = rcond



    @staticmethod
    def _prepare_eeg(
        eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        pre_pca: Optional[int],
        basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]],
        pca: Optional[NDArray[Any]],
    ) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
        """Prepare EEG-side features for fit/score.

        The method optionally fits or reuses a PCA reduction (when ``pca`` is None
        it is fitted on the provided array) and then applies the provided feature
        ``basis``.

        Args:
            eeg (ndarray): (n_samples, n_channels) array.
            pre_pca (int or None): Number of PCA components to apply before basis.
            basis (callable or None): Feature function to apply to the array.
            pca (array or None): Pre-fitted PCA map to reuse (shape (n_channels, k)).

        Returns:
            tuple: ``(features, pca)`` where ``features`` is the transformed array
            and ``pca`` is the fitted or reused PCA map.
        """
        trials = Model._normalize_trials_as_list(eeg)
        if pre_pca:
            if pca is None:
                pca = Model._fit_pca(trials, pre_pca)
            trials = [t @ pca for t in trials]
        return Model._apply(basis, trials), pca

    @staticmethod
    def _apply(
        basis: Optional[Callable[[NDArray[Any]], NDArray[Any]]],
        x: NDArray[Any],
    ) -> NDArray[Any]:
        """Apply a feature basis to a data array.

        If ``basis`` is ``None``, the input array ``x`` is returned unchanged.

        Args:
            basis (callable or None): Function to transform the array.
            x (ndarray): (n_samples, n_features) array.

        Returns:
            ndarray: Transformed array.
        """
        return x if basis is None else np.vstack([basis(t) for t in x])

    @staticmethod
    def _normalize_trials_as_list(view: Union[NDArray[Any], Sequence[NDArray[Any]]]
                          ) -> List[NDArray[Any]]:
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
    def _compute_covariances(
        X: Union[NDArray[Any], Sequence[NDArray[Any]]],
        Y: Union[NDArray[Any], Sequence[NDArray[Any]]],
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Compute mean-removed covariances for two views.

        This routine computes the block covariances for two multivariate views
        using numpy's covariance estimator (``np.cov``), returning the
        covariances and per-view means.

        Args:
            X (ndarray): (n_samples, n_features) array.
            Y (ndarray): (n_samples, n_features) array.

        Returns:
            tuple: ``(Cxx, Cyy, Cxy, mx, my)`` where Cxx and Cyy are the auto-covariances,
            Cxy the cross-covariance, and ``mx``/``my`` the means of each view.
        """
        Xs, Ys = Model._normalize_trials_as_list(X), Model._normalize_trials_as_list(Y)
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
    def _compute_whitener(cov: NDArray[Any], keep: Optional[int], rcond: float) -> NDArray[Any]:
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
    def _fit_pca(x: NDArray[Any], k: int) -> NDArray[Any]:
        """Fit a PCA map that reduces channel dimensionality to ``k`` components.

        Args:
            x (ndarray): (n_samples, n_channels) array.
            k (int): Number of principal components to retain.

        Returns:
            ndarray: PCA map with shape (n_channels, k) whose columns are the top-k
            principal directions.
        """
        mu = x.mean(0)
        cov = (x - mu).T @ (x - mu) / len(x)
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
    def _add_time_lags(x: Union[NDArray[Any], Sequence[Any]], 
                       n_lags: int) -> NDArray[Any]:
        """Build a time-lagged feature matrix for a signal. This is used for
        adding temporal context to either the EEG or Audio Envelope signals.

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
        x: NDArray[Any],
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
    def _as_2d(x: NDArray[Any]) -> NDArray[Any]:
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

    def fit(self, 
            eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], 
            audio: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> "CCA":
        """Fit a CCA model to EEG and stimulus data.

        Args:
            eeg (array or list): EEG trials as an array or list of arrays.
            audio (array or list): Audio Stimulus (envelope) trials as an array or list.

        Returns:
            CCA: ``self`` fitted in-place with attributes ``x_weights_``, ``y_weights_``
            and ``canonical_correlations_``.
        """

        E, self.pca_ = self._prepare_eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._normalize_trials_as_list(audio))
        Cxx, Cyy, Cxy, self.x_mean_, self.y_mean_ = self._compute_covariances(E, S)
        Wx = self._compute_whitener(Cxx, self.eeg_keep, self.rcond)
        Wy = self._compute_whitener(Cyy, self.stim_keep, self.rcond)
        U, s, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        k = self.n_components or len(s)
        self.singular_values_ = s[:k]
        self.x_weights_ = Wx @ U[:, :k]  # num data channels x n_components
        self.y_weights_ = Wy @ Vt[:k].T
        self.canonical_correlations_ = s[:k]

    def fit_transform(self, 
                      eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], 
                      audio: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> "CCA":
        """Fit a CCA model and transform the input data.

        Args:
            eeg (array or list): EEG trials as an array or list of arrays.
            audio (array or list): Audio Stimulus (envelope) trials as an array or list.

        Returns:
            the transformed data.
        """
        self.fit(eeg, audio)
        sx = (np.vstack(eeg) - self.x_mean_) @ self.x_weights_
        sy = (np.vstack(audio) - self.y_mean_) @ self.y_weights_
        return sx, sy

    def score(self, 
              eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
              audio: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> NDArray[Any]:
        """Compute per-component correlations on new data using fitted weights.

        The method projects new EEG and stimulus trials using the fitted weights
        and returns the Pearson correlation per canonical component.

        Args:
            eeg (array or list): New EEG trials.
            audio (array or list): New stimulus trials.

        Returns:
            ndarray: 1-D array of canonical correlations (one per component).
        """
        E, _ = self._prepare_eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._normalize_trials_as_list(audio))
        sx = (np.vstack(E) - self.x_mean_) @ self.x_weights_
        sy = (np.vstack(S) - self.y_mean_) @ self.y_weights_
        return self._correlate(sx, sy)


class Regression(Model):
    """Regularized least-squares regression for encoding/decoding.

    This class implements both forward (predict EEG from stimulus) and backward
    (reconstruct stimulus from EEG) regression depending on the ``type``
    attribute of the instance.
    """
    def __init__(self, *args, **kwargs):
        self.type = kwargs.pop("type", None)
        assert self.type in ("forward", "backward"), "type must be 'forward' or 'backward'"
        super().__init__(*args, **kwargs)


    def fit(self, 
            eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], 
            audio: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> "Regression":
        """Fit a regularized least-squares map.

        For ``type=='backward'`` the model learns to predict stimulus from EEG.
        For ``type=='forward'`` the model learns to predict EEG channels from the
        stimulus feature representation.

        Args:
            eeg (array or list): EEG trials.
            audio (array or list): Stimulus trials.

        Returns:
            Regression: ``self`` fitted in-place with attribute ``coef_``.
        """

        E, self.pca_ = self._prepare_eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._normalize_trials_as_list(audio))
        X, Y, keep = ((E, self._normalize_trials_as_list(audio), self.eeg_keep) if self.type == "backward"
                      else (S, self._normalize_trials_as_list(eeg), self.stim_keep))
        Cxx, _Cyy, Cxy, self.x_mean_, self.y_mean_ = self._compute_covariances(X, Y)
        Wx = self._compute_whitener(Cxx, keep, self.rcond)
        self.coef_ = (Wx @ Wx.T) @ Cxy

    def score(self, 
              eeg: Union[NDArray[Any], Sequence[NDArray[Any]]], 
              audio: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> NDArray[Any]:
        """Score predictions as per-output Pearson correlations.

        For forward models this returns correlations per EEG channel (caller may
        wish to take the maximum). For backward models it returns the
        reconstruction correlation of the stimulus.

        Args:
            eeg (array or list): EEG trials.
            audio (array or list): Stimulus trials.

        Returns:
            ndarray: 1-D array of correlation values, one per target dimension.
        """
        E, _ = self._prepare_eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._normalize_trials_as_list(audio))
        X, Y = (E, self._normalize_trials_as_list(audio)) if self.type == "backward" else (S, self._normalize_trials_as_list(eeg))
        pred = (np.vstack(X) - self.x_mean_) @ self.coef_ + self.y_mean_
        return self._correlate(pred, np.vstack(Y))

class ForwardRegressionModel(Regression):
    def __init__(self, *args, **kwargs):
        kwargs["type"] = "forward"
        super().__init__(*args, **kwargs)

class BackwardRegressionModel(Regression):
    def __init__(self, *args, **kwargs):
        kwargs["type"] = "backward"
        super().__init__(*args, **kwargs)


def model(name: str) -> Model:
    """Create one of the standard models from de Cheveigne's paper based on the 
    given name.

    Args:
      name (str): Name of the model to create. Must be one of "forward", 
        "backward", "cca1", "cca2", "cca2plus", or "cca3".

    Returns:
      Model: An instance of the requested model.
    """

    if name == 'forward':
        return ForwardRegressionModel(stim_basis=partial(Model._add_time_lags, n_lags=80))
    elif name == 'backward':
        return BackwardRegressionModel(eeg_keep=80)
    elif name == 'cca1':
        return CCA(stim_basis=partial(Model._add_time_lags, n_lags=40),
                   eeg_keep=40, stim_keep=40, n_components=40)
    elif name == 'cca2':
        return CCA(eeg_basis=partial(Model._add_time_lags, n_lags=10),
                   stim_basis=partial(Model._add_time_lags, n_lags=40),
                   pre_pca=80, eeg_keep=40, stim_keep=40, n_components=40)
    elif name == 'cca2plus':
        return CCA(eeg_basis=partial(Model._add_time_lags, n_lags=10),
                   stim_basis=partial(Model._add_time_lags, n_lags=80),
                   pre_pca=80, eeg_keep=80, stim_keep=80, n_components=80)
    elif name == 'cca3':
        return CCA(eeg_basis=Model._smoother, stim_basis=Model._smoother,
                   pre_pca=60, eeg_keep=139, n_components=21)
    else:
        raise ValueError(f"Unknown model name '{name}'")