import numpy as np 
from ..Optimiser import PortfolioOptimizer
from ..utils import TemplateStrategy

class EqualWeightStrategy(TemplateStrategy):
    """
    Equal-Weight (EW):
    Maintains a 1/N allocation, rebalanced daily.
    """

    def __init__(self, dim):
        self.dim = dim
        self._w = np.ones(dim) / dim

    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        return self._w.copy(), False

    @property
    def initial_weights(self) -> np.ndarray:
        return self._w.copy()
