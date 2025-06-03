import numpy as np 
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy

class OracleRegimeStrategy(TemplateStrategy):
    """
    Oracle Regime Model:
    Uses known regime_ids to estimate moments within each regime only.
    """

    def __init__(self, dim, regime_ids: np.ndarray, lam: float, kappa: float, w_max: float):
        """
        dim: number of assets
        regime_ids: length-T array of integer regime labels
        lam: risk-aversion parameter
        kappa: turnover penalty
        w_max: max weight per asset
        """
        self.dim = dim
        self.regime_ids = np.asarray(regime_ids)
        self.optimizer = PortfolioOptimizer(
            dim=dim, lam=lam, kappa=kappa, w_min=0.0, w_max=w_max
        )
        self.buffers = {}     # regime_id -> list of return vectors
        self.w_prev = np.ones(dim) / dim
        self.t = 0            # time index

    def _safe_cov(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] <= 1:
            return np.eye(self.dim) * 1e-6
        cov = np.cov(arr.T, bias=True)
        return cov + 1e-6 * np.eye(self.dim)

    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        rid = self.regime_ids[self.t]

        # If regime changed at this time step, reset that regime's buffer
        if rid not in self.buffers or (self.t > 0 and rid != self.regime_ids[self.t - 1]):
            self.buffers[rid] = []

        self.buffers[rid].append(x.copy())
        arr = np.vstack(self.buffers[rid])

        mu_batch = arr.mean(axis=0)
        Sigma_batch = self._safe_cov(arr)

        w_new = self.optimizer.solve(mu_batch, Sigma_batch, self.w_prev)
        self.w_prev = w_new.copy()
        self.t += 1
        return w_new, False  # no change-point