import unittest
import numpy as np

from cca_models import Model, CCA, Regression, model


class TestCCAModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_trials_and_as_2d_and_apply(self):
        a1 = np.arange(5.0)
        a2 = np.arange(6.0).reshape(3, 2)
        # _trials should convert 1-D to (n,1) and leave 2-D alone
        t1 = Model._trials(a1)
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1[0].shape, (5, 1))
        t2 = Model._trials(a2)
        self.assertEqual(t2[0].shape, (3, 2))

        # _as_2d keeps 2d and wraps 1d
        self.assertEqual(Model._as_2d(a1).shape, (5, 1))
        self.assertEqual(Model._as_2d(a2).shape, (3, 2))

        # _apply with None returns trials unchanged
        out = Model._apply(None, t2)
        self.assertIs(out, t2)

    def test_time_lag_and_moving_average_and_smoother(self):
        x = np.arange(1.0, 11.0)
        tl = Model._time_lag(x, 3)
        # for a single channel, shape should be (n_samples, n_lags)
        self.assertEqual(tl.shape, (10, 3))
        # first row should have only lag 0 filled (others zero)
        self.assertAlmostEqual(tl[0, 0], 1.0)
        self.assertAlmostEqual(tl[0, 1], 0.0)

        ma = Model._moving_average(x[:, None], 3)
        # moving average first value equals first sample
        self.assertAlmostEqual(ma[0, 0], 1.0)
        # check later value
        self.assertAlmostEqual(ma[3, 0], (2 + 3 + 4) / 3.0)

        sm = Model._smoother(x, n_bands=5, min_samples=1, max_samples=8)
        # smoother produces multiple bands
        self.assertEqual(sm.shape[0], 10)

    def test_covariances_and_whitener(self):
        # simple two-trial signals
        x = np.vstack([np.random.randn(50, 3), np.random.randn(60, 3)])
        y = np.vstack([np.random.randn(50, 2), np.random.randn(60, 2)])
        Cxx, Cyy, Cxy, mx, my = Model._covariances([x[:50], x[50:]], [y[:50], y[50:]])
        self.assertEqual(Cxx.shape, (3, 3))
        self.assertEqual(Cyy.shape, (2, 2))
        self.assertEqual(Cxy.shape, (3, 2))

        W = Model._whitener(Cxx, keep=None, rcond=1e-12)
        # whitened covariance should be approximately orthonormal when applied
        Xc = (x - mx) @ W
        cov = (Xc.T @ Xc) / Xc.shape[0]
        # diagonal elements should be near 1 (within tolerance)
        self.assertTrue(np.allclose(np.diag(cov), np.ones(cov.shape[0]), atol=1e-1))

    def test_cca_fit_score(self):
        # create a 1-d stimulus and an EEG with linear mixing + noise
        n = 400
        env = np.sin(np.linspace(0, 20, n))
        w = np.array([[1.0, 0.5, -0.3]])  # shape (1, channels)
        eeg = env[:, None] @ w + 0.01 * np.random.randn(n, 3)

        cca = CCA(type="cca", n_components=1)
        cca.fit(eeg, env)
        corr = cca.score(eeg, env)
        # correlation should be high for first canonical component
        self.assertGreater(corr[0], 0.9)

    def test_regression_backward_fit_score(self):
        # backward: reconstruct env from eeg
        n = 300
        env = np.sin(np.linspace(0, 10, n))
        w = np.array([[0.8, -0.4, 0.2]])
        eeg = env[:, None] @ w + 0.02 * np.random.randn(n, 3)

        reg = Regression(type="backward", eeg_keep=None)
        reg.fit(eeg, env)
        corr = reg.score(eeg, env)
        # single-output reconstruction correlation should be high
        self.assertGreater(corr[0], 0.9)

    def test_model_factory(self):
        m = model("cca1")
        self.assertIsInstance(m, CCA)

    def test_fit_pca(self):
        # create two trials with known variances per channel so PCA orders them
        rng = np.random.RandomState(1)
        vars = np.array([4.0, 1.0, 0.25])
        t1 = rng.randn(100, 3) * np.sqrt(vars)
        t2 = rng.randn(120, 3) * np.sqrt(vars)
        P = Model._fit_pca([t1, t2], 2)
        # shape should be (n_channels, k)
        self.assertEqual(P.shape, (3, 2))
        # columns should be orthonormal
        self.assertTrue(np.allclose(P.T @ P, np.eye(2), atol=1e-6))
        # projected variances should be descending
        X = np.vstack([t1, t2])
        proj = X @ P
        var_proj = np.var(proj, axis=0)
        self.assertGreater(var_proj[0], var_proj[1])


if __name__ == "__main__":
    unittest.main()
