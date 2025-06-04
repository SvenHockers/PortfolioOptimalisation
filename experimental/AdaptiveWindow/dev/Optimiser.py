import numpy as np
import cvxpy as cp

class PortfolioOptimizer:
    """
    Solves mean-variance optimization with L1 turnover penalty:
        maximize w^T mu - lam * w^T Sigma w - kappa * ||w - w_prev||_1
        subject to sum(w) = 1, w >= 0, w <= w_max.
    """

    def __init__(self, dim, lam, kappa, w_min=0.0, w_max=1.0):
        """
        dim: number of assets
        lam: risk-aversion parameter
        kappa: turnover penalty
        w_min, w_max: box constraints (scalar or array-like)
        """
        self.d = dim
        self.lam = lam
        self.kappa = kappa
        self.w_min = np.zeros(dim) if np.isscalar(w_min) else np.array(w_min)
        self.w_max = np.full(dim, w_max) if np.isscalar(w_max) else np.array(w_max)

    def solve(self, mu, Sigma, w_prev):
        """
        mu: (d,) expected returns
        Sigma: (d,d) covariance matrix
        w_prev: (d,) previous weights
        Returns new weights (d,).
        """
        d = self.d
        w = cp.Variable(d)

        ret_term = mu @ w
        risk_term = cp.quad_form(w, Sigma)
        turnover_term = cp.norm1(w - w_prev)

        obj = cp.Maximize(ret_term
                          - self.lam * risk_term
                          - self.kappa * turnover_term)

        constraints = [
            cp.sum(w) == 1,
            w >= self.w_min,
            w <= self.w_max
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        return w.value