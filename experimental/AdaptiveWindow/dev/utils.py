from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd

class TemplateStrategy(ABC):
    """
    Abstract template class for any portfolio strategy. Requires implementing .step().
    """

    @abstractmethod
    def step(self, x: np.ndarray) -> (np.ndarray, bool):
        """
        Process a new return vector x (length-d).
        Must return (weights, info_flag).
        """
        pass

    @property
    def initial_weights(self) -> np.ndarray:
        """
        Optionally override to supply initial weight vector.
        By default, equal-weight is used.
        """
        return None
    
class NumericalTools:
    @abstractmethod
    def compute_sharpe(daily_returns: pd.Series) -> float:
        """
        Annualized Sharpe = (mean / std) * sqrt(252).
        If std == 0, return -inf so it never wins.
        """
        mean = daily_returns.mean()
        sigma = daily_returns.std(ddof=1)
        if sigma == 0:
            return -np.inf
        return (mean / sigma) * np.sqrt(252)
