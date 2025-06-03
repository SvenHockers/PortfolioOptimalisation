import numpy as np
from scipy.stats import multivariate_normal
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy

class BOCPDDetector:
    """
    Bayesian Online Change-Point Detector for multivariate Gaussian data
    with Normal-Inverse-Wishart prior.
    """

    def __init__(self, dim, p_c, theta_cp, max_run_length,
                 mu0=None, kappa0=1.0, psi0=None, nu0=None):
        """
        dim: data dimension
        p_c: hazard probability of a change-point
        theta_cp: threshold to declare a new regime
        max_run_length: maximum run length to track (pruning)
        mu0: prior mean (length-d array), defaults to zero
        kappa0, psi0, nu0: NIW hyperparameters
        """
        self.d = dim
        self.p_c = p_c
        self.theta = theta_cp
        self.L = max_run_length

        # Prior NIW parameters
        self.mu0 = np.zeros(dim) if mu0 is None else mu0.copy()
        self.kappa0 = kappa0
        self.nu0 = nu0 if nu0 is not None else dim + 2
        self.psi0 = np.eye(dim) if psi0 is None else psi0.copy()

        # Initialize run-length posterior
        self.run_probs = np.array([1.0])
        # Store NIW parameters for each possible run length
        self.params = [{
            'mu': self.mu0.copy(),
            'kappa': self.kappa0,
            'psi': self.psi0.copy(),
            'nu': self.nu0
        }]

    def _predictive_pdf(self, x, niw):
        """
        Student-t predictive density for x under NIW parameters.
        """
        d = self.d
        kappa, nu = niw['kappa'], niw['nu']
        mu, psi = niw['mu'], niw['psi']
        dof = nu - d + 1
        scale = (kappa + 1) / (kappa * dof) * psi
        return multivariate_normal.pdf(x, mean=mu, cov=scale)

    def _update_niw(self, niw, x):
        """
        Update NIW posterior with new observation x.
        """
        mu, kappa, psi, nu = niw['mu'], niw['kappa'], niw['psi'], niw['nu']
        kappa_new = kappa + 1
        nu_new = nu + 1
        delta = x - mu
        mu_new = (kappa * mu + x) / kappa_new
        psi_new = psi + np.outer(delta, delta) * (kappa / kappa_new)
        return {'mu': mu_new, 'kappa': kappa_new, 'psi': psi_new, 'nu': nu_new}

    def update(self, x):
        """
        Process new observation x (array of length d).
        Returns True if a change-point is declared at this time.
        """
        x = np.asarray(x)
        R_prev = self.run_probs[-self.L:]
        params_prev = self.params[-self.L:]
        max_r = len(R_prev)

        # Predictive under each run
        p_x = np.array([self._predictive_pdf(x, params_prev[r]) for r in range(max_r)])
        # Predictive under prior (new run)
        p0 = self._predictive_pdf(x, {
            'mu': self.mu0, 'kappa': self.kappa0,
            'psi': self.psi0, 'nu': self.nu0
        })

        # Allocate new run-length probabilities
        R_new = np.zeros(max_r + 1)
        # Growth (no change)
        R_new[1:] = R_prev * (1 - self.p_c) * p_x
        # Change-point probability
        R_new[0] = np.sum(R_prev * self.p_c * p0)

        R_new /= np.sum(R_new)  # normalize

        # Update NIW params
        params_new = []
        # r = 0: new run from prior
        params_new.append(self._update_niw({
            'mu': self.mu0, 'kappa': self.kappa0,
            'psi': self.psi0, 'nu': self.nu0
        }, x))
        # r >= 1: extend previous runs
        for r in range(max_r):
            params_new.append(self._update_niw(params_prev[r], x))

        # Prune to max_run_length
        if len(R_new) > self.L:
            R_new = R_new[-self.L:]
            params_new = params_new[-self.L:]

        self.run_probs = R_new
        self.params = params_new

        # Declare change if posterior mass at r_t=0 exceeds threshold
        return self.run_probs[0] > self.theta
    

""" 
And a wrapper for the BOCPD to integrate it into the simulator
"""
class PortfolioStrategy(TemplateStrategy):
    """
    Implements a BOCPD+BFS mean-variance strategy.
    Inherits from BaseStrategy. Implements .step().
    """

    def __init__(self, dim, bocpd_params, opt_params, mu0=None, Sigma0=None):
        self.bocpd = BOCPDDetector(
            dim=dim,
            p_c=bocpd_params['p_c'],
            theta_cp=bocpd_params['theta_cp'],
            max_run_length=bocpd_params['max_run_length'],
            mu0=bocpd_params.get('mu0'),
            kappa0=bocpd_params.get('kappa0', 1.0),
            psi0=bocpd_params.get('psi0'),
            nu0=bocpd_params.get('nu0')
        )
        self.optimizer = PortfolioOptimizer(
            dim=dim,
            lam=opt_params['lam'],
            kappa=opt_params['kappa'],
            w_min=opt_params.get('w_min', 0.0),
            w_max=opt_params.get('w_max', 1.0)
        )
        self.mu0 = np.zeros(dim) if mu0 is None else mu0.copy()
        self.Sigma0 = np.eye(dim) if Sigma0 is None else Sigma0.copy()

        self.mu_rec = self.mu0.copy()
        self.Sigma_rec = self.Sigma0.copy()
        self.current_run = []
        self.w_prev = np.ones(dim) / dim
        self._jitter = 1e-6

    def _safe_cov(self, arr):
        if arr.shape[0] <= 1:
            return self.Sigma_rec.copy()
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
        is_cp = self.bocpd.update(x)

        if is_cp:
            self.current_run = []

        run_arr = np.vstack(self.current_run) if self.current_run else np.empty((0, len(x)))
        if run_arr.shape[0] >= 1:
            mu_batch = run_arr.mean(axis=0)
            Sigma_batch = self._safe_cov(run_arr)
        else:
            mu_batch = self.mu_rec.copy()
            Sigma_batch = self.Sigma_rec.copy()

        p0 = self._pdf_with_jitter(x, mean=self.mu0, cov=self.Sigma0)
        pt = self._pdf_with_jitter(x, mean=mu_batch, cov=Sigma_batch)
        bfs = p0 / (pt + 1e-12)

        p_c = self.bocpd.p_c
        m = p_c / (1 - p_c)
        gamma = (m * bfs) / (1 + m * bfs)

        delta = x - self.mu_rec
        self.mu_rec = (1 - gamma) * self.mu_rec + gamma * x
        self.Sigma_rec = ((1 - gamma) * self.Sigma_rec +
                          gamma * np.outer(delta, delta))

        w_new = self.optimizer.solve(self.mu_rec, self.Sigma_rec, self.w_prev)

        self.current_run.append(x)
        self.w_prev = w_new.copy()

        return w_new, is_cp

    @property
    def initial_weights(self) -> np.ndarray:
        """
        Provide an explicit starting weight vector instead of equal-weight.
        """
        return self.w_prev.copy()