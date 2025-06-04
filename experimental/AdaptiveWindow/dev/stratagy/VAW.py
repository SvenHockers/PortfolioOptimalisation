import numpy as np
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy

class VolatilityAdaptiveWindowStrategy(TemplateStrategy):
    """
    Volatility-Adaptive Window (VAW):
    Adjusts lookback length L_t by ±step_size based on realized market volatility.
    Then estimates mean & covariance over the adjusted window and optimizes.
    """

    def __init__(self,
                 dim,
                 init_window: int,
                 lam: float,
                 kappa: float,
                 w_max: float,
                 lower_thr: float,
                 upper_thr: float,
                 step_size: int = 5,
                 min_window: int = 10,
                 max_window: int = 252):
        """
        dim: number of assets
        init_window: starting lookback length (e.g., 60)
        lam: risk-aversion parameter
        kappa: turnover penalty
        w_max: max weight per asset
        lower_thr, upper_thr: volatility thresholds
        step_size: days to adjust window by (default 5)
        min_window, max_window: bounds on window length
        """
        self.dim = dim
        self.L = init_window
        self.lam = lam
        self.kappa = kappa
        self.w_max = w_max
        self.lower_thr = lower_thr
        self.upper_thr = upper_thr
        self.step_size = step_size
        self.min_window = min_window
        self.max_window = max_window

        self.optimizer = PortfolioOptimizer(
            dim=dim, lam=lam, kappa=kappa, w_min=0.0, w_max=w_max
        )
        self.history = []     # list of all past returns
        self.w_prev = np.ones(dim) / dim
        self._jitter = 1e-6

    def _safe_cov(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] <= 1:
            return np.eye(self.dim) * self._jitter
        cov = np.cov(arr.T, bias=True)
        return cov + self._jitter * np.eye(self.dim)

    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        # Append return
        self.history.append(x.copy())
        t = len(self.history)

        lookback = min(self.L, t)
        arr_window = np.vstack(self.history[-lookback:])  # shape=(lookback, dim)

        # Compute realized market volatility: average cross-sectional std over window
        cs_stds = np.std(arr_window, axis=1, ddof=0)   # shape=(lookback,)
        realized_vol = cs_stds.mean()

        # Adjust window length for next step
        if realized_vol < self.lower_thr:
            self.L = min(self.L + self.step_size, self.max_window)
        elif realized_vol > self.upper_thr:
            self.L = max(self.L - self.step_size, self.min_window)

        mu_batch = arr_window.mean(axis=0)
        Sigma_batch = self._safe_cov(arr_window)

        w_new = self.optimizer.solve(mu_batch, Sigma_batch, self.w_prev)
        self.w_prev = w_new.copy()
        return w_new, False  # no change-point