from abc import ABC, abstractmethod
import numpy as np 

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