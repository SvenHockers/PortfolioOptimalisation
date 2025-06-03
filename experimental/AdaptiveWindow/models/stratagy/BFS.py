import numpy as np 
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy

class BFSOnlyStrategy(TemplateStrategy):
    """
    BFS-Only (Recursive EWMA):
    Exponential-weighted updates of mean & covariance with learning rate γ_t from BFS,
    but without BOCPD resets.
    """

    def __init__(self, dim, bocpd_params, opt_params, mu0=None, Sigma0=None):
        """
        dim: number of assets
        bocpd_params: dict with key 'p_c'
        opt_params: dict with keys 'lam', 'kappa', 'w_min', 'w_max'
        mu0, Sigma0: baseline mean & covariance for BFS
        """
        self.p_c = bocpd_params['p_c']
        self.mu0 = np.zeros(dim) if mu0 is None else mu0.copy()
        self.Sigma0 = np.eye(dim) if Sigma0 is None else Sigma0.copy()

        self.optimizer = PortfolioOptimizer(
            dim=dim,
            lam=opt_params['lam'],
            kappa=opt_params['kappa'],
            w_min=opt_params.get('w_min', 0.0),
            w_max=opt_params.get('w_max', 1.0)
        )
        self.mu_rec = self.mu0.copy()
        self.Sigma_rec = self.Sigma0.copy()
        self.w_prev = np.ones(dim) / dim
        self._jitter = 1e-6

    def _safe_cov(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] <= 1:
            return np.eye(len(self.mu_rec)) * self._jitter
        cov = np.cov(arr.T, bias=True)
        return cov + self._jitter * np.eye(cov.shape[0])

    def _pdf_with_jitter(self, x, mean, cov):
        try:
            return multivariate_normal.pdf(x, mean=mean, cov=cov)
        except np.linalg.LinAlgError:
            cov_j = cov + self._jitter * np.eye(cov.shape[0])
            return multivariate_normal.pdf(x, mean=mean, cov=cov_j)

    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        x = np.asarray(x)

        p0 = self._pdf_with_jitter(x, mean=self.mu0, cov=self.Sigma0)
        pt = self._pdf_with_jitter(x, mean=self.mu_rec, cov=self.Sigma_rec)
        bfs = p0 / (pt + 1e-12)

        m = self.p_c / (1 - self.p_c)
        gamma = (m * bfs) / (1 + m * bfs)

        delta = x - self.mu_rec
        self.mu_rec = (1 - gamma) * self.mu_rec + gamma * x
        self.Sigma_rec = ((1 - gamma) * self.Sigma_rec +
                          gamma * np.outer(delta, delta))

        w_new = self.optimizer.solve(self.mu_rec, self.Sigma_rec, self.w_prev)
        self.w_prev = w_new.copy()
        return w_new, False  # no change-point