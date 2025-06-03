import numpy as np
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy


class FixedSlidingWindowStrategy(TemplateStrategy):
    """
    Fixed Sliding Window (FSW):
    Uses a constant lookback window to estimate sample mean & covariance,
    then solves mean–variance with an L1 turnover penalty each day.
    """

    def __init__(self, dim, window_size: int, lam: float, kappa: float, w_max: float):
        """
        dim: number of assets
        window_size: fixed lookback length (e.g., 60)
        lam: risk-aversion parameter
        kappa: turnover penalty
        w_max: maximum weight per asset
        """
        self.dim = dim
        self.window_size = window_size
        self.optimizer = PortfolioOptimizer(
            dim=dim, lam=lam, kappa=kappa, w_min=0.0, w_max=w_max
        )
        self.buffer = []                  # list to store most recent returns
        self.w_prev = np.ones(dim) / dim
        self._jitter = 1e-6

    def _safe_cov(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] <= 1:
            return np.eye(self.dim) * self._jitter
        cov = np.cov(arr.T, bias=True)
        return cov + self._jitter * np.eye(self.dim)

    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        # Append new return and keep only last `window_size`
        self.buffer.append(x.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        arr = np.vstack(self.buffer)  # shape = (min(t, window_size), dim)
        mu_batch = arr.mean(axis=0)
        Sigma_batch = self._safe_cov(arr)

        w_new = self.optimizer.solve(mu_batch, Sigma_batch, self.w_prev)
        self.w_prev = w_new.copy()
        return w_new, False  # no change-point indicator