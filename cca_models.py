"""Stimulus-response decoding with canonical correlation analysis.

Alain de Cheveigne's CCA (de Cheveigne et al., NeuroImage 2018)

The API copies scikit-learn (fit / score, fitted attributes ending in an underscore)

Data everywhere: `eeg` a (n_samples, n_channels) array or a list of such trial arrays;
`audio` a (n_samples,) or (n_samples, n_features) array or a list of them. Paired
trials may have unequal lengths. Feature transforms never cross trial boundaries.
"""

from __future__ import annotations

from copy import copy
from functools import partial
import pickle
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

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
        eeg_pca (int or None): EEG components to retain after the feature basis.
        stim_pca (int or None): Stimulus components to retain after the feature basis.
        pca_center (bool): Remove the pooled mean when fitting PCA rotations.
        trial_center (bool): Center each aligned trial before fitting and transforming.
        feature_center (bool): Center each feature trial after its basis and PCA.
        eeg_offset (int or None): First feature sample's offset within the EEG
            trial. None infers it for built-in lag bases and otherwise uses zero.
        stim_offset (int or None): Corresponding stimulus feature sample offset.

    Attributes:
        eeg_basis, stim_basis, pre_pca, eeg_keep, stim_keep, n_components, rcond,
        eeg_pca, stim_pca, pca_center, trial_center, feature_center, eeg_offset,
        stim_offset: Stored initialization parameters used by subclasses.
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
        eeg_pca: Optional[int] = None,
        stim_pca: Optional[int] = None,
        pca_center: bool = True,
        trial_center: bool = False,
        feature_center: bool = False,
        eeg_offset: Optional[int] = None,
        stim_offset: Optional[int] = None,
    ) -> None:
        self.eeg_basis = eeg_basis
        self.stim_basis = stim_basis
        self.pre_pca = pre_pca
        self.eeg_keep = eeg_keep
        self.stim_keep = stim_keep
        self.n_components = n_components
        self.rcond = rcond
        self.eeg_pca = eeg_pca
        self.stim_pca = stim_pca
        self.pca_center = pca_center
        self.trial_center = trial_center
        self.feature_center = feature_center
        self.eeg_offset = eeg_offset
        self.stim_offset = stim_offset



    @staticmethod
    def _prepare_eeg(
        eeg: NDArray[Any],
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
            trials = [x @ pca for x in trials]
        x = trials[0] if isinstance(eeg, np.ndarray) else trials
        return Model._apply(basis, x), pca

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
        if basis is None:
            return x
        if isinstance(x, (list, tuple)):
            return [basis(trial) for trial in x]
        return basis(x)

    # ---- numeric core (the algorithm) ---------------------------------------

    @staticmethod
    def _compute_covariances(
        X: NDArray[Any],
        Y: NDArray[Any],
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Compute mean-removed covariances for two views.

        This routine pools centered trial moments for two multivariate views,
        returning biased covariances and per-view means without stacking trials.

        Args:
            X (ndarray): (n_samples, n_features) array.
            Y (ndarray): (n_samples, n_features) array.

        Returns:
            tuple: ``(Cxx, Cyy, Cxy, mx, my)`` where Cxx and Cyy are the auto-covariances,
            Cxy the cross-covariance, and ``mx``/``my`` the means of each view.
        """
        X, Y = Model._paired_trials(X, Y)
        moments = Model._sum_statistics(Model._statistics(x, y) for x, y in zip(X, Y))
        return Model._from_statistics(moments)

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
        return Model._pca_trials(Model._normalize_trials_as_list(x), k, True)

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
    def _add_time_lags(x: NDArray[Any], n_lags: int) -> NDArray[Any]:
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
        if x.ndim not in (1, 2):
            raise ValueError("Expected a one- or two-dimensional array.")
        return x if x.ndim == 2 else x[:, None]


    @staticmethod
    def _normalize_trials_as_list(
        x: Union[NDArray[Any], Sequence[NDArray[Any]]],
    ) -> List[NDArray[Any]]:
        """Return validated two-dimensional trials without stacking recordings."""
        trials = [x] if isinstance(x, np.ndarray) else list(x)
        if not trials:
            raise ValueError("At least one trial is required.")
        trials = [Model._as_2d(trial) for trial in trials]
        channels = trials[0].shape[1]
        for trial in trials:
            if len(trial) < 2 or channels == 0 or trial.shape[1] != channels:
                raise ValueError("Trials need at least two samples and matching feature counts.")
            if not np.isfinite(trial).all():
                raise ValueError("Trials must contain only finite values.")
        return trials

    @staticmethod
    def _paired_trials(eeg: Any, audio: Any) -> Tuple[List[NDArray[Any]], List[NDArray[Any]]]:
        """Check trial counts and sample alignment of the two input views."""
        eeg, audio = Model._normalize_trials_as_list(eeg), Model._normalize_trials_as_list(audio)
        if len(eeg) != len(audio) or any(len(e) != len(s) for e, s in zip(eeg, audio)):
            raise ValueError("EEG and stimulus need matching trial counts and trial lengths.")
        return eeg, audio

    @staticmethod
    def _statistics(x: NDArray[Any], y: NDArray[Any]) -> Tuple[Any, ...]:
        """Return sample count, means, and centered sums of products for one trial."""
        mx, my = x.mean(0), y.mean(0)
        x, y = x - mx, y - my
        return len(x), mx, my, x.T @ x, y.T @ y, x.T @ y

    @staticmethod
    def _sum_statistics(items: Iterable[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        """Pool centered trial moments, including between-trial mean differences."""
        total = None
        for item in items:
            if total is None:
                total = item
                continue
            n, mx, my, xx, yy, xy = total
            m, ux, uy, uu, vv, uv = item
            dx, dy, scale = ux - mx, uy - my, n * m / (n + m)
            total = (n + m, mx + dx * m / (n + m), my + dy * m / (n + m),
                     xx + uu + scale * np.outer(dx, dx),
                     yy + vv + scale * np.outer(dy, dy),
                     xy + uv + scale * np.outer(dx, dy))
        if total is None:
            raise ValueError("Cannot pool an empty set of trials.")
        return total

    @staticmethod
    def _loo_statistics(items: Sequence[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        """Pool each training fold using prefixes and suffixes of trial moments.

        This reuses the trial products without subtracting nearly equal large
        matrices when recordings have substantially different means.
        """
        prefixes, suffixes = [items[0]], [items[-1]]
        for item in items[1:]:
            prefixes.append(Model._sum_statistics((prefixes[-1], item)))
        for item in reversed(items[:-1]):
            suffixes.append(Model._sum_statistics((item, suffixes[-1])))
        suffixes.reverse()
        result = []
        for index in range(len(items)):
            parts = []
            if index:
                parts.append(prefixes[index - 1])
            if index + 1 < len(items):
                parts.append(suffixes[index + 1])
            result.append(Model._sum_statistics(parts))
        return result

    @staticmethod
    def _from_statistics(moments: Tuple[Any, ...]) -> Tuple[NDArray[Any], ...]:
        """Convert pooled moments to biased covariances and fitted means."""
        n, mx, my, xx, yy, xy = moments
        return xx / n, yy / n, xy / n, mx, my

    @staticmethod
    def _pca_trials(trials: Iterable[NDArray[Any]], keep: int, center: bool) -> NDArray[Any]:
        """Fit a PCA rotation from a stream of feature trials."""
        if not isinstance(keep, (int, np.integer)) or keep < 1:
            raise ValueError("PCA component counts must be positive integers.")
        count, mean, scatter = 0, None, None
        for trial in trials:
            m, u = len(trial), trial.mean(0)
            z = trial - u if center else trial
            c = z.T @ z
            if mean is None:
                count, mean, scatter = m, u, c
            else:
                delta = u - mean
                scatter += c
                if center:
                    scatter += count * m / (count + m) * np.outer(delta, delta)
                mean = mean + delta * m / (count + m)
                count += m
        if scatter is None:
            raise ValueError("Cannot fit PCA without trials.")
        _, vectors = np.linalg.eigh(0.5 * (scatter + scatter.T))
        return vectors[:, ::-1][:, :keep]

    @staticmethod
    def time_lags(
        x: NDArray[Any],
        n_lags: int,
        direction: str = "past",
        trim: bool = False,
    ) -> NDArray[Any]:
        """Build temporal features within one recording.

        Args:
            x (ndarray): Samples by channels.
            n_lags (int): Number of consecutive lags, including zero.
            direction (str): "past" or "future" relative to each sample.
            trim (bool): Remove samples without a complete lag window.

        Returns:
            ndarray: Lag-major features. Untrimmed boundaries are zero-padded.
            A trimmed past basis starts at sample n_lags - 1. When configured
            with ``partial``, its offset is inferred during paired fitting.
        """
        x = Model._as_2d(x)
        if not isinstance(n_lags, (int, np.integer)) or not 1 <= n_lags <= len(x):
            raise ValueError("n_lags must be an integer between one and the trial length.")
        if direction not in ("past", "future"):
            raise ValueError("direction must be 'past' or 'future'.")
        if direction == "future":
            result = Model._add_time_lags(x[::-1], n_lags)[::-1]
            return result[:len(x) - n_lags + 1] if trim else result
        result = Model._add_time_lags(x, n_lags)
        return result[n_lags - 1:] if trim else result

    @staticmethod
    def filterbank(
        x: NDArray[Any],
        widths: Optional[Sequence[int]] = None,
        difference: bool = False,
        channel_major: bool = False,
        onset: str = "partial",
        center: bool = False,
    ) -> NDArray[Any]:
        """Apply a configurable bank of moving-average filters to one trial.

        Args:
            x (ndarray): Samples by channels.
            widths (sequence of int or None): Filter widths in samples.
            difference (bool): Difference adjacent columns of the assembled bank.
            channel_major (bool): Group widths within each input channel.
            onset (str): "partial" uses partial averages at the onset; "mean"
                extends the onset with the mean of the first width + 1 samples.
            center (bool): Remove each output column's trial mean.

        Returns:
            ndarray: Filtered features. Differencing removes one output column.
        """
        x = Model._as_2d(x)
        if widths is None:
            widths = sorted({int(round(w)) for w in np.geomspace(2, 128, 21)})
        widths = list(widths)
        if not widths or any(not isinstance(w, (int, np.integer)) or w < 1 for w in widths):
            raise ValueError("Filter widths must be positive integers.")
        if onset not in ("partial", "mean"):
            raise ValueError("onset must be 'partial' or 'mean'.")
        bank = np.empty((len(x), len(widths), x.shape[1]))
        for i, width in enumerate(widths):
            if onset == "mean":
                if width >= len(x):
                    smoothed = np.broadcast_to(x.mean(0), x.shape)
                else:
                    start = x[:width + 1].mean(0)
                    smoothed = np.cumsum(x - start, axis=0)
                    smoothed[width:] -= smoothed[:-width].copy()
                    smoothed = smoothed / width + start
            else:
                sums = np.cumsum(x, axis=0)
                sums[width:] -= sums[:-width].copy()
                smoothed = sums / np.minimum(np.arange(1, len(x) + 1), width)[:, None]
            bank[:, i, :] = smoothed
        if channel_major:
            bank = bank.transpose(0, 2, 1)
        result = bank.reshape(len(x), -1)
        if difference:
            if result.shape[1] < 2:
                raise ValueError("Differencing requires at least two filter columns.")
            result = np.diff(result, axis=1)
        return result - result.mean(0) if center else result

    def _new(self) -> "Model":
        """Create an unfitted estimator with the same configuration."""
        names = ("eeg_basis", "stim_basis", "pre_pca", "eeg_keep", "stim_keep",
                 "n_components", "rcond", "eeg_pca", "stim_pca", "pca_center",
                 "trial_center", "feature_center", "eeg_offset", "stim_offset")
        params = {name: getattr(self, name) for name in names}
        if isinstance(self, Regression):
            params["type"] = self.type
        return self.__class__(**params)

    def _prepare_side(self, trials: List[NDArray[Any]], side: str, fit: bool) -> List[NDArray[Any]]:
        """Fit or reuse one side's PCA rotations and trial-local feature basis."""
        active = not ((side == "eeg" and self.type == "forward") or
                      (side == "stim" and self.type == "backward"))
        basis = getattr(self, side + "_basis") if active else None
        keep = getattr(self, side + "_pca") if active else None
        if fit:
            offset = getattr(self, side + "_offset") if active else 0
            if offset is None:
                options = basis.keywords if isinstance(basis, partial) and basis.func is Model.time_lags else {}
                offset = (options["n_lags"] - 1 if options.get("trim", False) and
                          options.get("direction", "past") == "past" else 0)
            setattr(self, side + "_offset_", offset)
        if fit and side == "eeg":
            self.pca_ = (self._pca_trials(trials, self.pre_pca, self.pca_center)
                         if self.pre_pca and active else None)
        pre = self.pca_ if side == "eeg" else None

        def features(trial):
            values = trial @ pre if pre is not None else trial
            return self._as_2d(self._apply(basis, values))

        attribute = side + "_pca_"
        if fit:
            rotation = self._pca_trials((features(t) for t in trials), keep, self.pca_center) if keep else None
            setattr(self, attribute, rotation)
        rotation = getattr(self, attribute)
        result = []
        for trial in trials:
            values = features(trial)
            if rotation is not None:
                values = values @ rotation
            if self.feature_center and active:
                values = values - values.mean(0)
            if len(values) < 2 or values.shape[1] == 0 or not np.isfinite(values).all():
                raise ValueError("A feature basis must produce at least two finite samples.")
            result.append(values)
        return result

    def _prepare_trials(self, eeg: Any, audio: Any, fit: bool) -> Tuple[List[NDArray[Any]], List[NDArray[Any]]]:
        """Validate input dimensions and transform both views one trial at a time."""
        eeg, audio = self._paired_trials(eeg, audio)
        if fit:
            self.n_eeg_features_in_, self.n_stim_features_in_ = eeg[0].shape[1], audio[0].shape[1]
        elif (eeg[0].shape[1] != self.n_eeg_features_in_ or
              audio[0].shape[1] != self.n_stim_features_in_):
            raise ValueError("Input feature counts differ from those used for fitting.")
        return self._prepare_side(eeg, "eeg", fit), self._prepare_side(audio, "stim", fit)

    def _align_trials(
        self, eeg: List[NDArray[Any]], audio: List[NDArray[Any]], shift: int,
    ) -> Tuple[List[NDArray[Any]], List[NDArray[Any]]]:
        """Align feature trials; positive shifts advance EEG relative to stimulus."""
        if not isinstance(shift, (int, np.integer)):
            raise ValueError("Shifts must be integer sample counts.")
        E, S = [], []
        for e, s in zip(eeg, audio):
            e_offset, s_offset = self.eeg_offset_, self.stim_offset_
            start = max(e_offset, s_offset)
            stop = min(e_offset + len(e), s_offset + len(s))
            e, s = e[start - e_offset:stop - e_offset], s[start - s_offset:stop - s_offset]
            length = stop - start
            if length - abs(shift) < 2:
                raise ValueError("A shift leaves fewer than two paired samples in a trial.")
            e, s = e[:length], s[:length]
            if shift > 0:
                e, s = e[shift:], s[:-shift]
            elif shift < 0:
                e, s = e[:shift], s[-shift:]
            if self.trial_center:
                e, s = e - e.mean(0), s - s.mean(0)
            E.append(e)
            S.append(s)
        return E, S

    def prepare(self, eeg: Any, audio: Any, shift: Optional[int] = None) -> Tuple[Any, Any]:
        """Apply fitted features and alignment without projecting onto model weights.

        Args:
            eeg (ndarray or sequence): EEG recording or trials.
            audio (ndarray or sequence): Corresponding stimulus recording or trials.
            shift (int or None): Override the fitted shift, in samples.

        Returns:
            tuple: EEG and stimulus features. Sequence inputs preserve trial lists.
            Custom bases that remove initial samples must declare their feature
            offsets using ``eeg_offset`` or ``stim_offset`` at construction.
        """
        if not hasattr(self, "shift_"):
            raise ValueError("Fit the model before preparing new data.")
        E, S = self._prepare_trials(eeg, audio, False)
        E, S = self._align_trials(E, S, self.shift_ if shift is None else shift)
        return (E[0], S[0]) if isinstance(eeg, np.ndarray) and isinstance(audio, np.ndarray) else (E, S)

    def _fit_trials(self, eeg: Any, audio: Any, shift: int) -> None:
        """Fit feature maps and solve from pooled aligned trial covariances."""
        self._validate_config()
        E, S = self._prepare_trials(eeg, audio, True)
        E, S = self._align_trials(E, S, shift)
        self._fit_covariances(*self._compute_covariances(E, S))
        self.shift_ = int(shift)

    def _validate_config(self) -> None:
        """Check dimensions, cutoffs, and explicit feature time offsets."""
        if self.rcond < 0 or not np.isfinite(self.rcond):
            raise ValueError("rcond must be finite and nonnegative.")
        for keep in (self.pre_pca, self.eeg_pca, self.stim_pca, self.eeg_keep,
                     self.stim_keep, self.n_components):
            if keep is not None and (not isinstance(keep, (int, np.integer)) or keep < 1):
                raise ValueError("Component counts must be positive integers or None.")
        for offset in (self.eeg_offset, self.stim_offset):
            if offset is not None and (not isinstance(offset, (int, np.integer)) or offset < 0):
                raise ValueError("Feature offsets must be nonnegative integers or None.")

    def _score_covariances(
        self, Cxx: NDArray[Any], Cyy: NDArray[Any], Cxy: NDArray[Any],
        mx: NDArray[Any], my: NDArray[Any],
    ) -> NDArray[Any]:
        """Score fixed weights from held-out centered covariance matrices."""
        if self.type == "forward":
            Cxx, Cyy, Cxy = Cyy, Cxx, Cxy.T
        if isinstance(self, Regression):
            A = self.coef_
            numerator = np.diag(A.T @ Cxy)
            vx, vy = np.diag(A.T @ Cxx @ A), np.diag(Cyy)
        else:
            A, B = self.x_weights_, self.y_weights_
            numerator = np.diag(A.T @ Cxy @ B)
            vx, vy = np.diag(A.T @ Cxx @ A), np.diag(B.T @ Cyy @ B)
        denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    def fit_cv(
        self,
        eeg: Sequence[NDArray[Any]],
        audio: Sequence[NDArray[Any]],
        shifts: Sequence[int],
        component: int = 0,
        pca_scope: str = "fold",
        keep_folds: bool = False,
        cv: str = "loo",
    ) -> None:
        """Select a temporal shift and refit the model on all available trials.

        Args:
            eeg (sequence of ndarray): EEG trials with possibly unequal lengths.
            audio (sequence of ndarray): Corresponding stimulus trials.
            shifts (sequence of int): Candidate shifts in samples. Positive shifts
                advance EEG after trial-local feature expansion.
            component (int): Output component used to select the best shift.
            pca_scope (str): "fold" fits PCA on training trials only; "all" fits
                PCA on all trials before evaluation. The latter is transductive.
            keep_folds (bool): Retain fold estimators at the selected shift.
            cv (str): "loo" for leave-one-trial-out selection; "none" selects
                by training correlation, without a held-out performance claim.

        Returns:
            None. Fits this estimator on all trials at ``best_shift_``. Stores
            ``fold_scores_`` (trial, output, shift), ``mean_scores_`` and
            ``training_scores_`` (output, shift), ``shifts_``, ``best_index_``,
            and ``best_score_``. ``fold_models_`` is retained only on request.
            Best-shift CV scores are selection scores, not nested-test estimates.
        """
        fitted = self._new()
        fitted._fit_cv(eeg, audio, shifts, component, pca_scope, keep_folds, cv)
        self.__dict__.clear()
        self.__dict__.update(fitted.__dict__)

    def _fit_cv(
        self, eeg: Sequence[NDArray[Any]], audio: Sequence[NDArray[Any]],
        shifts: Sequence[int], component: int, pca_scope: str, keep_folds: bool, cv: str,
    ) -> None:
        """Fit and evaluate a fresh candidate before committing fitted state."""
        eeg, audio = self._paired_trials(eeg, audio)
        shifts = np.asarray(shifts)
        if shifts.ndim != 1 or not len(shifts) or shifts.dtype.kind not in "iu":
            raise ValueError("shifts must be a nonempty sequence of integer sample counts.")
        if pca_scope not in ("fold", "all") or cv not in ("loo", "none"):
            raise ValueError("Use pca_scope='fold'/'all' and cv='loo'/'none'.")
        if cv == "loo" and len(eeg) < 2:
            raise ValueError("Leave-one-trial-out evaluation needs at least two trials.")
        if not isinstance(component, (int, np.integer)) or component < 0:
            raise ValueError("component must be a nonnegative integer.")
        self._validate_config()
        E, S = self._prepare_trials(eeg, audio, True)
        outputs = (S[0].shape[1] if self.type == "backward" else
                   E[0].shape[1] if self.type == "forward" else
                   min(E[0].shape[1], S[0].shape[1], self.n_components or E[0].shape[1]))
        if component >= outputs:
            raise ValueError("The selection component exceeds the model output count.")
        training = np.full((outputs, len(shifts)), np.nan)
        folds = np.full((len(eeg) if cv == "loo" else 0, outputs, len(shifts)), np.nan)
        shared = pca_scope == "all" or not any((self.pre_pca, self.eeg_pca, self.stim_pca))
        for index, shift in enumerate(shifts):
            a, b = self._align_trials(E, S, int(shift))
            moments = [self._statistics(x, y) for x, y in zip(a, b)]
            total = self._sum_statistics(moments)
            fitted = copy(self)
            fitted._fit_covariances(*self._from_statistics(total))
            scores = fitted._score_covariances(*self._from_statistics(total))
            training[:len(scores), index] = scores
            if cv == "loo" and shared:
                for fold, train in enumerate(self._loo_statistics(moments)):
                    held = moments[fold]
                    fitted._fit_covariances(*self._from_statistics(train))
                    scores = fitted._score_covariances(*self._from_statistics(held))
                    folds[fold, :len(scores), index] = scores
        if cv == "loo" and not shared:
            for fold in range(len(eeg)):
                fitted = self._new()
                e = [x for i, x in enumerate(eeg) if i != fold]
                s = [x for i, x in enumerate(audio) if i != fold]
                A, B = fitted._prepare_trials(e, s, True)
                X, Y = fitted._prepare_trials([eeg[fold]], [audio[fold]], False)
                for index, shift in enumerate(shifts):
                    a, b = fitted._align_trials(A, B, int(shift))
                    fitted._fit_covariances(*self._compute_covariances(a, b))
                    x, y = fitted._align_trials(X, Y, int(shift))
                    scores = fitted._score_covariances(*self._compute_covariances(x, y))
                    folds[fold, :len(scores), index] = scores
        means = folds.mean(0) if cv == "loo" else training.copy()
        valid = np.isfinite(means[component])
        if not valid.any():
            raise ValueError("The selection component is unavailable in every candidate fit.")
        best = int(np.argmax(np.where(valid, means[component], -np.inf)))
        shift = int(shifts[best])
        a, b = self._align_trials(E, S, shift)
        self._fit_covariances(*self._compute_covariances(a, b))
        self.shift_ = self.best_shift_ = shift
        self.shifts_, self.best_index_ = shifts.copy(), best
        self.fold_scores_, self.mean_scores_, self.training_scores_ = folds, means, training
        self.best_score_, self.pca_scope_, self.cv_ = float(means[component, best]), pca_scope, cv
        if keep_folds and cv == "loo":
            fitted_folds = []
            for fold in range(len(eeg)):
                if shared:
                    fitted = copy(self)
                    for name in ("shifts_", "fold_scores_", "mean_scores_", "training_scores_",
                                 "best_index_", "best_shift_", "best_score_", "fold_models_", "cv_", "pca_scope_"):
                        fitted.__dict__.pop(name, None)
                    train_e = [x for i, x in enumerate(a) if i != fold]
                    train_s = [x for i, x in enumerate(b) if i != fold]
                    fitted._fit_covariances(*self._compute_covariances(train_e, train_s))
                else:
                    fitted = self._new()
                    fitted.fit([x for i, x in enumerate(eeg) if i != fold],
                               [x for i, x in enumerate(audio) if i != fold], shift=shift)
                fitted_folds.append(fitted)
            self.fold_models_ = fitted_folds

    def save(self, path: str) -> None:
        """Save a fitted estimator, including feature maps and evaluation results.

        Args:
            path (str): Destination file. Custom basis functions must be importable
                module-level functions; built-in bases and partials are supported.

        Returns:
            None.
        """
        if not hasattr(self, "x_weights_") and not hasattr(self, "coef_"):
            raise ValueError("Fit the model before saving it.")
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, "wb") as stream:
            stream.write(payload)

    @staticmethod
    def load(path: str) -> "Model":
        """Load a fitted estimator saved by this package.

        Args:
            path (str): Model file. Only load files from trusted sources, using a
                compatible package version and any custom basis modules.

        Returns:
            Model: The fitted CCA or regression estimator.
        """
        with open(path, "rb") as stream:
            fitted = pickle.load(stream)
        if not isinstance(fitted, Model):
            raise TypeError("The file does not contain a Model.")
        return fitted


class CCA(Model):
    """Canonical correlation analysis between EEG and stimulus feature views.

    The algorithm whitens both views, computes the SVD of the whitened
    cross-covariance and returns the singular vectors as weights and the
    singular values as canonical correlations.

    This class implements a scikit-learn-like ``fit`` / ``score`` API.
    """

    type = "cca"

    def fit(
        self,
        eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        audio: Union[NDArray[Any], Sequence[NDArray[Any]]],
        shift: int = 0,
    ) -> "CCA":
        """Fit a CCA model to EEG and stimulus data.

        Args:
            eeg (ndarray or sequence): EEG recording or unequal-length trials.
            audio (ndarray or sequence): Corresponding stimulus recording or trials.
            shift (int): Advance EEG by this many samples after feature expansion.

        Returns:
            CCA: ``self``, with ``x_weights_``, ``y_weights_``, fitted feature
            maps, means, alignment, and ``canonical_correlations_``.
        """

        fitted = self._new()
        fitted._fit_trials(eeg, audio, shift)
        self.__dict__.clear()
        self.__dict__.update(fitted.__dict__)
        return self

    def _fit_covariances(
        self, Cxx: NDArray[Any], Cyy: NDArray[Any], Cxy: NDArray[Any],
        mx: NDArray[Any], my: NDArray[Any],
    ) -> None:
        """Fit canonical directions using the shared covariance solver."""
        self.x_mean_, self.y_mean_ = mx, my
        Wx = self._compute_whitener(Cxx, self.eeg_keep, self.rcond)
        Wy = self._compute_whitener(Cyy, self.stim_keep, self.rcond)
        U, s, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        k = self.n_components or len(s)
        self.singular_values_ = s[:k]
        self.x_weights_ = Wx @ U[:, :k]  # num data channels x n_components
        self.y_weights_ = Wy @ Vt[:k].T
        self.canonical_correlations_ = s[:k]

    def fit_transform(
        self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        audio: Union[NDArray[Any], Sequence[NDArray[Any]]], shift: int = 0,
    ) -> Tuple[Any, Any]:
        """Fit a CCA model and transform the input data.

        Args:
            eeg (array or list): EEG trials as an array or list of arrays.
            audio (array or list): Stimulus/envelope trials as an array or list.
            shift (int): EEG advance after feature expansion, in samples.

        Returns:
            tuple: Transformed arrays, or two trial lists for sequence inputs.
        """
        self.fit(eeg, audio, shift=shift)
        return self.transform(eeg, audio)

    def transform(
        self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        audio: Union[NDArray[Any], Sequence[NDArray[Any]]],
    ) -> Tuple[Any, Any]:
        """Project data onto the fitted canonical directions.

        Args:
            eeg (ndarray): EEG data, (n_samples, n_channels).
            audio (ndarray): Stimulus/envelope data, (n_samples,) or (n_samples, 1).

        Returns:
            tuple: ``(x_scores, y_scores)``, each (n_samples, n_components) -- the canonical
            component time-series for the two views.
        """
        E, S = self.prepare(eeg, audio)
        if isinstance(E, list):
            return ([(e - self.x_mean_) @ self.x_weights_ for e in E],
                    [(s - self.y_mean_) @ self.y_weights_ for s in S])
        return (E - self.x_mean_) @ self.x_weights_, (S - self.y_mean_) @ self.y_weights_

    def score(self, eeg: NDArray[Any], audio: NDArray[Any]) -> NDArray[Any]:
        """Compute per-component correlations on new data using fitted weights.

        The method projects new EEG and stimulus trials using the fitted weights
        and returns the Pearson correlation per canonical component.

        Args:
            eeg (ndarray): New EEG data.
            audio (ndarray): New stimulus data.

        Returns:
            ndarray: 1-D array of canonical correlations (one per component).
        """
        E, S = self.prepare(eeg, audio)
        return self._score_covariances(*self._compute_covariances(E, S))


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


    def fit(
        self, eeg: Union[NDArray[Any], Sequence[NDArray[Any]]],
        audio: Union[NDArray[Any], Sequence[NDArray[Any]]], shift: int = 0,
    ) -> "Regression":
        """Fit a regularized least-squares map.

        For ``type=='backward'`` the model learns to predict stimulus from EEG.
        For ``type=='forward'`` the model learns to predict EEG channels from the
        stimulus feature representation.

        Args:
            eeg (ndarray): EEG data.
            audio (ndarray): Stimulus data.
            shift (int): EEG advance after feature expansion, in samples.

        Returns:
            Regression: ``self``, with ``coef_``, fitted feature maps, means,
            and alignment.
        """

        fitted = self._new()
        fitted._fit_trials(eeg, audio, shift)
        self.__dict__.clear()
        self.__dict__.update(fitted.__dict__)
        return self

    def _fit_covariances(
        self, Cxx: NDArray[Any], Cyy: NDArray[Any], Cxy: NDArray[Any],
        mx: NDArray[Any], my: NDArray[Any],
    ) -> None:
        """Fit regression coefficients using the shared covariance solver."""
        if self.type == "forward":
            Cxx, Cxy, mx, my = Cyy, Cxy.T, my, mx
        self.x_mean_, self.y_mean_ = mx, my
        keep = self.eeg_keep if self.type == "backward" else self.stim_keep
        Wx = self._compute_whitener(Cxx, keep, self.rcond)
        self.coef_ = (Wx @ Wx.T) @ Cxy

    def score(self, eeg: NDArray[Any], audio: NDArray[Any]) -> NDArray[Any]:
        """Score predictions as per-output Pearson correlations.

        For forward models this returns correlations per EEG channel (caller may
        wish to take the maximum). For backward models it returns the
        reconstruction correlation of the stimulus.

        Args:
            eeg (ndarray): EEG data.
            audio (ndarray): Stimulus data.

        Returns:
            ndarray: 1-D array of correlation values, one per target dimension.
        """
        E, S = self.prepare(eeg, audio)
        return self._score_covariances(*self._compute_covariances(E, S))

    def predict(self, x: Union[NDArray[Any], Sequence[NDArray[Any]]]) -> Any:
        """Predict from EEG (backward) or stimulus (forward) using fitted features.

        Args:
            x (ndarray or sequence): Predictor recording or trials, with the same
                input channels and preprocessing used for fitting.

        Returns:
            ndarray or list: Predictions in the predictor's feature time frame.
            No samples are removed for the paired-view shift: with ``shift_ = s``,
            backward outputs correspond to stimulus time t - s, and forward
            outputs to EEG time t + s. ``score`` handles paired alignment.
        """
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the model before predicting.")
        trials = self._normalize_trials_as_list(x)
        side = "eeg" if self.type == "backward" else "stim"
        if trials[0].shape[1] != getattr(self, "n_" + side + "_features_in_"):
            raise ValueError("Input feature count differs from the fitted predictor.")
        trials = self._prepare_side(trials, side, False)
        if self.trial_center:
            trials = [trial - trial.mean(0) for trial in trials]
        result = [(trial - self.x_mean_) @ self.coef_ + self.y_mean_ for trial in trials]
        return result[0] if isinstance(x, np.ndarray) else result


class ForwardRegressionModel(Regression):
    def __init__(self, *args, **kwargs):
        kwargs["type"] = "forward"
        super().__init__(*args, **kwargs)

class BackwardRegressionModel(Regression):
    def __init__(self, *args, **kwargs):
        kwargs["type"] = "backward"
        super().__init__(*args, **kwargs)


def adc_model(name: str) -> Model:
    """Create one of the standard models from de Cheveigne's paper based on the 
    given name, varying the number of PCA components and the amount of
    smoothing. 
      Paper: https://doi.org/10.1016/j.neuroimage.2018.01.033.

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
