import numpy as np
import pandas as pd
from .utils import TemplateStrategy


class Simulator:
    """
    Runs any strategy over a return series, storing weights and realized returns.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        strategy: TemplateStrategy
    ):
        """
        returns: pandas.DataFrame of log-returns (index=dates, columns=assets)
        strategy: any object with a .step(return_vector) -> (weights, info) interface
        """
        # Store the returns DataFrame directly
        self.returns_df = returns.copy()
        self.dates = self.returns_df.index
        self.assets = list(self.returns_df.columns)
        self.dim = len(self.assets)

        # Assign the strategy (must implement .step())
        self.strategy = strategy

        # Storage for results
        self._weights_list = []
        self._portf_ret_list = []
        self._date_list = []

        self.weights_history = None
        self.portf_returns = None

    def run(self):
        """
        Iterate over all dates in returns_df:
          1) compute realized return using previous weights
          2) update strategy and get new weights
          3) store weights and returns
        """
        # Initialize previous weights to equal-weight, unless strategy provides initial_weights
        w_prev = np.ones(self.dim) / self.dim
        if hasattr(self.strategy, "initial_weights") and self.strategy.initial_weights is not None:
            w_prev = self.strategy.initial_weights.copy()

        for date, row in self.returns_df.iterrows():
            x_t = row.values  # this is the log-return vector for the day

            # 1) realized return from previous weights
            realized_ret = w_prev.dot(x_t)
            self._portf_ret_list.append(realized_ret)

            # 2) step strategy
            w_new, _ = self.strategy.step(x_t)

            # 3) store results
            self._weights_list.append(w_new)
            self._date_list.append(date)

            # 4) update w_prev
            w_prev = w_new.copy()

        # Build pandas structures
        self.weights_history = pd.DataFrame(
            data=np.vstack(self._weights_list),
            index=pd.DatetimeIndex(self._date_list),
            columns=self.assets
        )
        self.portf_returns = pd.Series(
            data=np.array(self._portf_ret_list),
            index=pd.DatetimeIndex(self._date_list),
            name="daily_return"
        )

    def summary_statistics(self):
        """
        Computes annualized return, volatility, and Sharpe ratio from portf_returns.
        """
        daily_ret = self.portf_returns.dropna()
        if daily_ret.empty:
            return {'ann_return': np.nan, 'ann_vol': np.nan, 'ann_sharpe': np.nan}

        ann_return = daily_ret.mean() * 252
        ann_vol = daily_ret.std(ddof=0) * np.sqrt(252)
        ann_sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

        return {
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'ann_sharpe': ann_sharpe
        }

    @property
    def cum_wealth(self):
        """
        Cumulative wealth series (start = 1) indexed by dates.
        """
        if self.portf_returns is None:
            return None
        return (1 + self.portf_returns).cumprod()
