"""
In this file I've built some custom plotting scripts for some quite complex charts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

def plot_mpt_rebalance_barchart(
    rolling_mpt,
    N: int = 15,
    figsize: tuple[float,float] = (12, 7),
    bar_width: float = 1.0,
    tol_days: int = 0,
    skip_days: int = 30,
):
    cum_wealth = (1 + rolling_mpt.portf_returns).cumprod()
    w_df   = rolling_mpt.weights_history.copy()

    w_prev = w_df.shift(1).fillna(method="bfill")
    turnover_full = (w_df - w_prev).abs().sum(axis=1)

    delta = (w_df - w_prev).abs()
    top_asset_full = delta.idxmax(axis=1)
    signed_change_full = pd.Series(0.0, index=delta.index)
    mask_rebals_full = turnover_full > 0

    for d in delta.index[mask_rebals_full]:
        tkr = top_asset_full.loc[d]
        signed_change_full.loc[d] = w_df.loc[d, tkr] - w_prev.loc[d, tkr]

    annot_df_full = pd.DataFrame({
        "turnover":       turnover_full,
        "top_asset":      top_asset_full,
        "signed_change":  signed_change_full
    }, index=w_df.index)

    if skip_days >= len(annot_df_full):
        raise ValueError(f"skip_days = {skip_days} is greater than total number of rows ({len(annot_df_full)}).")
    
    annot_df_bottom = annot_df_full.iloc[skip_days:].copy()
    dates_bottom = annot_df_bottom.index

    signed_vol_bottom = annot_df_bottom["turnover"] * np.sign(annot_df_bottom["signed_change"])

    if N > 0:
        topN_dates_bottom = annot_df_bottom["turnover"].nlargest(N).index
    else:
        topN_dates_bottom = pd.Index([])

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    cum_wealth.plot(
        ax=ax1,
        color="tab:blue",
        linewidth=1.5,
        label="Cumulative Wealth"
    )
    ax1.set_ylabel("Wealth (start = 1.0)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title("MPT Strategy: Cumulative Wealth & Turnover Bar Chart")

    bar_heights = signed_vol_bottom.values
    bar_colors = np.where(signed_vol_bottom > 0, "darkgreen", "darkred")

    ax2.bar(
        dates_bottom,
        bar_heights,
        width=bar_width,
        color=bar_colors,
        alpha=0.8,
        align="center"
    )

    # (c) Zero baseline
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax2.set_ylabel("Turnover (L₁)\n(buy / sell)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.3)

    if len(annot_df_bottom) > 0:
        max_vol_bottom = annot_df_bottom["turnover"].max()
    else:
        max_vol_bottom = 1e-3  # fallback if somehow empty

    if max_vol_bottom <= 0:
        max_vol_bottom = 1e-3

    buffer = max_vol_bottom * 1.2
    ax2.set_ylim(-buffer, buffer)

    if N > 0 and len(topN_dates_bottom) > 0:
        annotated = []  
        for d in sorted(topN_dates_bottom):
            vol = signed_vol_bottom.loc[d]
            if vol == 0:
                continue

            change = annot_df_bottom.loc[d, "signed_change"]
            ticker = annot_df_bottom.loc[d, "top_asset"]

            if change > 0:
                y0 = vol
                yoff_base = 4     
                va = "bottom"
                ha = "center"
                label = f"{ticker} +{change:.2f}"
            else:
                y0 = vol  # negative
                yoff_base = -10    # 10 points below the bar
                va = "top"
                ha = "center"
                label = f"{ticker} {change:.2f}"

            level = 0
            if tol_days > 0:
                for (prev_d, prev_level) in annotated:
                    if abs((d - prev_d).days) <= tol_days:
                        level = max(level, prev_level + 1)
            annotated.append((d, level))

            if change > 0:
                yoff = yoff_base + 10 * level
            else:
                yoff = yoff_base - 10 * level

            ax2.annotate(
                label,
                (d, y0),
                xytext=(0, yoff),
                textcoords="offset points",
                fontsize=8,
                color=("darkgreen" if change > 0 else "darkred"),
                ha=ha,
                va=va
            )

        if annotated:
            max_level = max(l for _, l in annotated)
            if max_level > 0:
                extra_pts = 14 + 10 * max_level
                bbox = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                height_inch = bbox.height
                data_span = ax2.get_ylim()[1] - ax2.get_ylim()[0]
                pts_to_data = data_span / (height_inch * 72.0)
                extra_data = extra_pts * pts_to_data
                y_min2, y_max2 = ax2.get_ylim()
                ax2.set_ylim(y_min2 - extra_data, y_max2 + extra_data * 0.05)

    ax1.legend(loc="upper left")
    plt.tight_layout()
    return fig, ax1, ax2

def plot_mpt_rebalance_timeline(
    rolling_mpt,
    N: int = 15,
    figsize: tuple[float,float] = (12, 7),
    bar_width: float = 1.5,
    tol_days: int = 0,
):
    cum_wealth = (1 + rolling_mpt.portf_returns).cumprod()

    w_df   = rolling_mpt.weights_history.copy()
    w_prev = w_df.shift(1).fillna(method="bfill")

    turnover = (w_df - w_prev).abs().sum(axis=1)

    delta = (w_df - w_prev).abs()
    top_asset = delta.idxmax(axis=1)
    top_change = delta.max(axis=1)

    annot_df = pd.DataFrame({
        "turnover":   turnover,
        "top_asset":  top_asset,
        "top_change": top_change
    }, index=w_df.index)

    topN_dates = annot_df["turnover"].nlargest(N).index

    signed_change = pd.Series(index=topN_dates, dtype=float)
    for d in topN_dates:
        ticker = annot_df.loc[d, "top_asset"]
        raw_delta = w_df.loc[d, ticker] - w_prev.loc[d, ticker]
        signed_change.loc[d] = raw_delta

    non_top_rebals = annot_df.index[
        (annot_df["turnover"] > 0) & (~annot_df.index.isin(topN_dates))
    ]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    cum_wealth.plot(
        ax=ax1,
        color="tab:blue",
        linewidth=1.5,
        label="Cumulative Wealth"
    )
    ax1.set_ylabel("Wealth (start = 1.0)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title("MPT Strategy: Cumulative Wealth and Rebalance Activity")

    if len(non_top_rebals) > 0:
        ax2.vlines(
            non_top_rebals,
            ymin=-0.005,
            ymax=0.005,
            color="gray",
            linewidth=1,
            alpha=0.5,
            label="Other Rebalances (turnover)"
        )

    buys = signed_change[signed_change > 0]
    if not buys.empty:
        ax2.bar(
            buys.index,
            buys.values,
            width=bar_width,
            color="darkgreen",
            alpha=0.7,
            label="Top‐Asset BUY (magnitude)"
        )

    sells = signed_change[signed_change < 0]
    if not sells.empty:
        ax2.bar(
            sells.index,
            sells.values,
            width=bar_width,
            color="darkred",
            alpha=0.7,
            label="Top‐Asset SELL (magnitude)"
        )

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax2.set_ylabel("Weight Δ (Top Asset)\n(+buy / −sell)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.3)

    annotated_dates: list[tuple[pd.Timestamp,int]] = []

    for d in sorted(topN_dates):
        ticker = annot_df.loc[d, "top_asset"]
        change = signed_change.loc[d]

        if change > 0:
            color = "darkgreen"
            text_prefix = f"{ticker} +{change:.2f}"
        else:
            color = "darkred"
            text_prefix = f"{ticker} {change:.2f}"

        level = 0
        if tol_days > 0:
            for prev_date, prev_level in annotated_dates:
                if abs((d - prev_date).days) <= tol_days:
                    level = max(level, prev_level + 1)

        annotated_dates.append((d, level))

        if change > 0:
            y_offset = 4 + (level * 10)
            va = "bottom"
            ha = "center"
            text_y = change
        else:
            y_offset = -10 - (level * 10)   
            va = "top"
            ha = "center"
            text_y = change

        ax2.annotate(
            text_prefix,
            (d, text_y),
            xytext=(0, y_offset),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha=ha,
            va=va
        )

    if annotated_dates:
        max_level = max(lvl for _, lvl in annotated_dates)
        extra_pts = 14 + 10 * max_level  
        bbox = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax2_height_inch = bbox.height
        data_span = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        pts_to_data = data_span / (ax2_height_inch * 72)  # how many data‐units per point
        extra_data = extra_pts * pts_to_data

        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(ymin2 - extra_data, ymax2 + extra_data * 0.05)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    return fig, ax1, ax2

def summarize_backtest(rolling_mpt, cost_per_unit: float = 0.0, weight_threshold: float = 0.0) -> pd.DataFrame:
    # Pull daily returns of the strategy
    daily_pnl = rolling_mpt.portf_returns.dropna()
    
    # Compute cumulative wealth
    cum_wealth = (1 + daily_pnl).cumprod()
    
    # Annualized return, volatility, Sharpe
    ann_return = daily_pnl.mean() * 252
    ann_vol    = daily_pnl.std() * np.sqrt(252)
    ann_sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    
    # Drawdown series and max drawdown
    running_max = cum_wealth.cummax()
    drawdown    = cum_wealth / running_max - 1.0
    max_dd      = drawdown.min()  # most negative
    
    # Calmar Ratio = annual_return / |max_drawdown|
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    
    # Sortino Ratio: use downside semideviation of daily returns
    neg_rets = daily_pnl[daily_pnl < 0]
    if len(neg_rets) > 0:
        downside_vol = np.sqrt((neg_rets**2).mean()) * np.sqrt(252)
        sortino = (ann_return - 0.0) / downside_vol if downside_vol > 0 else np.nan
    else:
        sortino = np.nan
    
    # Compute daily turnover (L1 norm) from weights_history
    w = rolling_mpt.weights_history.copy()
    w_prev = w.shift(1)
    daily_turnover = (w - w_prev).abs().sum(axis=1)
    
    avg_daily_turn     = daily_turnover.mean()
    total_turn         = daily_turnover.sum()
    
    # Average number of holdings: count weights > weight_threshold
    if weight_threshold > 0:
        num_holdings = (rolling_mpt.weights_history > weight_threshold).sum(axis=1)
    else:
        # Count any weight > 0 as a holding
        num_holdings = (rolling_mpt.weights_history > 0).sum(axis=1)
    avg_holdings = num_holdings.mean()
    
    # Transaction‐cost drag (if cost_per_unit > 0)
    if cost_per_unit > 0:
        daily_tc = daily_turnover * cost_per_unit
        total_tc_drag = daily_tc.sum() * 100  # in percentage points of AUM
    else:
        total_tc_drag = np.nan

    summary = {
        "Ann Return (%)":       ann_return * 100,
        "Ann Volatility (%)":   ann_vol * 100,
        "Ann Sharpe":           ann_sharpe,
        "Max Drawdown (%)":     max_dd * 100,
        "Calmar Ratio":         calmar,
        "Sortino Ratio":        sortino,
        "Avg Daily Turnover":   avg_daily_turn,
        "Total Turnover":       total_turn,
        "Avg # Holdings":       avg_holdings,
        "Total TC Drag (%)":    total_tc_drag
    }
    
    summary_df = pd.DataFrame(summary, index=[0])
    return summary_df

def plot_rolling_sharpe_comparison(
    rolling_mpt1,
    rolling_mpt2,
    window: int = 63,
    figsize: tuple[float, float] = (8, 5),
    labels: tuple[str, str] = ("Strategy 1", "Strategy 2"),
    # Default Tableau 10–derived colors; user can override
    colors: tuple[str, str] = ("#4C78A8", "#F58518"),
    linestyles: tuple[str, str] = ("-", "--"),
    linewidths: tuple[float, float] = (1.5, 1.5),
    title_fontsize: int = 18,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 12,
    dpi: int = 300,
):
    """
    Plot rolling annualized Sharpe for two MPT strategies with publication‐quality styling.

    Parameters
    ----------
    rolling_mpt1 : RollingPortfolio
        First backtested strategy (must have run run_backtest()). Uses .portf_returns.
    rolling_mpt2 : RollingPortfolio
        Second backtested strategy (must have run run_backtest()).
    window : int, default=63
        Rolling window length (in trading days) to compute Sharpe (63 ≈ 3 months).
    figsize : (width, height), default=(8, 5)
        Figure size in inches.
    labels : tuple[str, str]
        Legend labels for the two strategies.
    colors : tuple[str, str], default=("#4C78A8", "#F58518")
        Line colors for Strategy 1 and Strategy 2 respectively.
    linestyles : tuple[str, str], default=("-", "--")
        Linestyles for Strategy 1 and Strategy 2.
    linewidths : tuple[float, float], default=(1.5, 1.5)
        Line widths for Strategy 1 and Strategy 2.
    title_fontsize : int, default=18
        Font size for the chart title.
    label_fontsize : int, default=14
        Font size for x/y labels.
    tick_fontsize : int, default=12
        Font size for tick labels on both axes.
    legend_fontsize : int, default=12
        Font size for the legend text.
    dpi : int, default=300
        Dots per inch for saving the figure (higher for publication quality).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # 1) Extract daily returns and align indices
    pnl1 = rolling_mpt1.portf_returns.dropna()
    pnl2 = rolling_mpt2.portf_returns.dropna()
    common_idx = pnl1.index.intersection(pnl2.index)
    pnl1 = pnl1.reindex(common_idx)
    pnl2 = pnl2.reindex(common_idx)

    # 2) Compute rolling annualized Sharpe
    rolling_mean1 = pnl1.rolling(window).mean() * 252
    rolling_std1  = pnl1.rolling(window).std() * np.sqrt(252)
    rolling_sharpe1 = rolling_mean1 / rolling_std1

    rolling_mean2 = pnl2.rolling(window).mean() * 252
    rolling_std2  = pnl2.rolling(window).std() * np.sqrt(252)
    rolling_sharpe2 = rolling_mean2 / rolling_std2

    # 3) Configure rcParams for publication‐quality styling
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Palatino", "serif"],
        "axes.titlesize":    title_fontsize,
        "axes.labelsize":    label_fontsize,
        "xtick.labelsize":   tick_fontsize,
        "ytick.labelsize":   tick_fontsize,
        "legend.fontsize":   legend_fontsize,
        "figure.dpi":        dpi,
        "axes.linewidth":    1.0,
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid":         True,
        "grid.color":        "#999999",
        "grid.alpha":        0.2,
        "grid.linestyle":    "--"
    })

    # 4) Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # 5) Plot each rolling Sharpe with exposed styling
    rolling_sharpe1.plot(
        ax=ax,
        label=labels[0],
        color=colors[0],
        linestyle=linestyles[0],
        linewidth=linewidths[0],
    )
    rolling_sharpe2.plot(
        ax=ax,
        label=labels[1],
        color=colors[1],
        linestyle=linestyles[1],
        linewidth=linewidths[1],
    )

    # 6) Add a horizontal zero reference line
    ax.axhline(0, color="black", lw=0.8, linestyle="-")

    # 7) Formatting
    ax.set_title(f"Rolling {window}-Day Annualized Sharpe", pad=10)
    ax.set_xlabel("Date", fontsize=label_fontsize)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=label_fontsize)

    # Place legend slightly outside the upper-left of the plotting area
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 8) Tighten layout to accommodate legend outside plot
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])

    return fig, ax

def plot_drawdown(
    rolling_mpt,
    figsize: tuple[float, float] = (10, 4),
    title: str = "Portfolio Drawdown Over Time",
    title_fontsize: int = 18,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    dpi: int = 300,
):
    """
    Plot a publication‐quality drawdown chart using firebrick and lightcoral.

    Parameters
    ----------
    drawdown : pd.Series
        Drawdown series (values ≤ 0), indexed by date.
    figsize : (width, height), default=(10, 4)
        Figure size in inches.
    title : str, default="Portfolio Drawdown Over Time"
        Chart title.
    title_fontsize : int, default=18
        Font size for the title.
    label_fontsize : int, default=14
        Font size for axis labels.
    tick_fontsize : int, default=12
        Font size for tick labels.
    dpi : int, default=300
        Dots per inch for saving/displaying the figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    cum_wealth = (1 + rolling_mpt.portf_returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = cum_wealth / running_max - 1.0

    # 1) Configure rcParams for publication quality
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Palatino", "serif"],
        "axes.titlesize":    title_fontsize,
        "axes.labelsize":    label_fontsize,
        "xtick.labelsize":   tick_fontsize,
        "ytick.labelsize":   tick_fontsize,
        "figure.dpi":        dpi,
        "axes.linewidth":    1.0,
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid":         True,
        "grid.color":        "#999999",
        "grid.alpha":        0.2,
        "grid.linestyle":    "--"
    })

    # 2) Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # 3) Plot drawdown line in firebrick
    ax.plot(
        drawdown.index,
        drawdown.values,
        color="firebrick",
        linewidth=1.5,
        label="Drawdown"
    )

    # 4) Fill below zero with lightcoral
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        where=(drawdown.values < 0),
        color="lightcoral",
        alpha=0.5
    )

    # 5) Horizontal zero line
    ax.axhline(0, color="black", lw=0.8)

    # 6) Labels and title
    ax.set_title(title, pad=10)
    ax.set_ylabel("Drawdown (fraction)", fontsize=label_fontsize)
    ax.set_xlabel("Date", fontsize=label_fontsize)

    # 7) Grid (already enabled via rcParams)
    ax.grid(True, which="major", linestyle="--", alpha=0.2)

    # 8) Tight layout
    plt.tight_layout()

    return fig, ax

def compare_mc_qae_estimation_errors(
    all_estimates_df: pd.DataFrame,
    prices: pd.DataFrame,
    lambda_risk: float = 2.0,
    aggregator: str = "uniform",
    figsize: tuple[float, float] = (10, 6),
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 10,
    dpi: int = 300,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Compare absolute estimation errors of Monte‐Carlo (MC) vs Quantum‐AE (QAE)
    for both the mean (Xmean) and variance, using a two‐panel bar chart.

    PARAMETERS
    ----------
    all_estimates_df : pd.DataFrame
        Long‐form DataFrame containing ALL rows for both MC and QAE. Must have exactly these columns:
          - "window"    : a string of the form "YYYY-MM-DD:YYYY-MM-DD"
          - "stock"     : ticker symbol (e.g. "AAPL", "MSFT", etc.)
          - "Xmean"     : estimated mean (float)
          - "Variance"  : estimated variance (float)
          - "Estimator" : either "Monte-Carlo" or "Quantum-AE" (string)

    prices : pd.DataFrame
        Historical price data.  Must be indexed by business‐day Timestamps, 
        and columns = tickers *without* any "# " prefix (e.g. ["AAPL","MSFT",...]).
        This DataFrame is used to compute the “true” sample moments (mean & var)
        on raw daily returns for each window.

    lambda_risk : float, default=2.0
        Risk‐aversion parameter λ, used only if aggregator="hindsight_optimal".

    aggregator : str, default="uniform"
        How to aggregate per‐asset errors into a single scalar error each day:
          - "uniform"             → w_i = 1/N for all assets i
          - "hindsight_optimal"   → solve a tiny one‐period QP using true moments
                                   to get w* that maximizes w^T μ_true − λ w^T Σ_true w

    figsize : tuple[float, float], default=(10, 6)
        Figure size in inches for the final two‐panel bar chart.

    title_fontsize : int, default=16
        Font size for the two panel titles.

    label_fontsize : int, default=12
        Font size for all axis labels.

    tick_fontsize : int, default=10
        Font size for tick labels.

    legend_fontsize : int, default=10
        Font size for the legends in each panel.

    dpi : int, default=300
        DPI for the figure (publication quality).

    RETURNS
    -------
    fig : matplotlib.figure.Figure
        The figure object containing two stacked subplots.

    (ax_mu, ax_var) : tuple of matplotlib.axes.Axes
        - ax_mu  → the top panel (mean‐error Δ)
        - ax_var → the bottom panel (variance‐error Δ)
    """

    # ────────────────────────────────────────────────────────────────────────────
    # A) Sanity‐check “all_estimates_df” has exactly the five required columns
    # ────────────────────────────────────────────────────────────────────────────
    required = {"window", "stock", "Xmean", "Variance", "Estimator"}
    missing = required - set(all_estimates_df.columns)
    if missing:
        raise KeyError(f"Missing columns in all_estimates_df: {missing}")

    # ────────────────────────────────────────────────────────────────────────────
    # B) Split into MC vs QAE DataFrames
    # ────────────────────────────────────────────────────────────────────────────
    mc_df  = all_estimates_df[all_estimates_df["Estimator"] == "Monte-Carlo"].copy()
    qae_df = all_estimates_df[all_estimates_df["Estimator"] == "Quantum-AE"].copy()

    if mc_df.empty or qae_df.empty:
        raise ValueError("Ensure all_estimates_df contains both 'Monte-Carlo' and 'Quantum-AE' rows.")

    # ────────────────────────────────────────────────────────────────────────────
    # C) Pivot each long‐form subset into (window_str × stock) → estimate
    # ────────────────────────────────────────────────────────────────────────────
    est_mu_mc_long  = mc_df[["window", "stock", "Xmean"]].copy()
    est_mu_qae_long = qae_df[["window", "stock", "Xmean"]].copy()
    est_var_mc_long  = mc_df[["window", "stock", "Variance"]].copy()
    est_var_qae_long = qae_df[["window", "stock", "Variance"]].copy()

    est_mu_mc   = est_mu_mc_long.pivot(index="window", columns="stock", values="Xmean")
    est_mu_qae  = est_mu_qae_long.pivot(index="window", columns="stock", values="Xmean")
    est_var_mc  = est_var_mc_long.pivot(index="window", columns="stock", values="Variance")
    est_var_qae = est_var_qae_long.pivot(index="window", columns="stock", values="Variance")

    # ────────────────────────────────────────────────────────────────────────────
    # D) Build window_map: window_str → end_date (Timestamp), filtering to valid prices‐dates
    # ────────────────────────────────────────────────────────────────────────────
    all_window_strs = est_mu_mc.index.tolist()
    parsed_end_dates = [pd.to_datetime(w.split(":", 1)[1]) for w in all_window_strs]

    temp_map = pd.DataFrame({
        "window_str": all_window_strs,
        "end_date":   parsed_end_dates
    })
    # Keep only those whose end_date is actually in prices.index
    temp_map = temp_map[temp_map["end_date"].isin(prices.index)].copy()
    temp_map = temp_map.sort_values("end_date").reset_index(drop=True)

    window_map = pd.Series(
        data=temp_map["end_date"].values,
        index=temp_map["window_str"].values
    )

    # ────────────────────────────────────────────────────────────────────────────
    # E) Now call the “from_pivoted” subroutine (below) to do:
    #       1) Strip off “# ” from column names if present
    #       2) Compute true μ & σ² inside each window
    #       3) Aggregate errors and plot two‐panel bars
    # ────────────────────────────────────────────────────────────────────────────
    fig, (ax_mu, ax_var) = _plot_error_diff_from_pivoted(
        mu_mc_pivot  = est_mu_mc,
        mu_qae_pivot = est_mu_qae,
        var_mc_pivot = est_var_mc,
        var_qae_pivot = est_var_qae,
        prices       = prices,
        window_map   = window_map,
        lambda_risk  = lambda_risk,
        aggregator   = aggregator,
        figsize      = figsize,
        title_fontsize  = title_fontsize,
        label_fontsize  = label_fontsize,
        tick_fontsize   = tick_fontsize,
        legend_fontsize = legend_fontsize,
        dpi          = dpi,
    )

    return fig, (ax_mu, ax_var)



def _plot_error_diff_from_pivoted(
    mu_mc_pivot: pd.DataFrame,
    mu_qae_pivot: pd.DataFrame,
    var_mc_pivot: pd.DataFrame,
    var_qae_pivot: pd.DataFrame,
    prices: pd.DataFrame,
    window_map: pd.Series,
    lambda_risk: float,
    aggregator: str,
    figsize: tuple[float, float],
    title_fontsize: int,
    label_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: int,
    dpi: int,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Internal helper that assumes each of the four inputs is already pivoted in the form:
       index = window_str (e.g. "2019-01-01:2019-04-10")
       columns = stock names (possibly prefixed with "# ")
       values = the estimated quantity (mean or variance)

    It will:
      1) Strip "# " from column names so they match exactly the tickers in `prices`
      2) For each window, compute true sample mean & variance from prices
      3) Aggregate the absolute errors (MC vs QAE) under either uniform or hindsight‐optimal
      4) Plot a two‐panel bar chart of Δ|error_μ| (top) and Δ|error_σ²| (bottom)
    """

    # 1) Strip leading "# " from each column so that we get real tickers
    def strip_hash_prefix(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2.columns = [col.lstrip("# ").strip() for col in df.columns]
        return df2

    mu_mc   = strip_hash_prefix(mu_mc_pivot)
    mu_qae  = strip_hash_prefix(mu_qae_pivot)
    var_mc  = strip_hash_prefix(var_mc_pivot)
    var_qae = strip_hash_prefix(var_qae_pivot)

    # Force all four to use exactly the same asset order
    assets = sorted(mu_mc.columns.tolist())
    mu_mc   = mu_mc.reindex(columns=assets)
    mu_qae  = mu_qae.reindex(columns=assets)
    var_mc  = var_mc.reindex(columns=assets)
    var_qae = var_qae.reindex(columns=assets)

    # 2) Compute “true” sample mean & variance in each window by slicing `prices`
    #    (raw, not annualized—matches the scale of MC/QAE output)
    all_windows = [w for w in window_map.index if w in mu_mc.index]
    true_mu_df  = pd.DataFrame(index=all_windows, columns=assets, dtype=float)
    true_var_df = pd.DataFrame(index=all_windows, columns=assets, dtype=float)

    for w in all_windows:
        start_str, end_str = w.split(":", 1)
        start_date = pd.to_datetime(start_str)
        end_date   = pd.to_datetime(end_str)

        window_prices = prices.loc[start_date : end_date]
        window_rets   = window_prices.pct_change().dropna()

        true_mu_df.loc[w, :]  = window_rets.mean().reindex(assets).values
        true_var_df.loc[w, :] = window_rets.var().reindex(assets).values

    # 3) For each window, form a weight vector w_agg and compute absolute errors
    N = len(assets)
    dates_list    = []
    delta_err_mu  = []
    delta_err_var = []

    if aggregator not in ("uniform", "hindsight_optimal"):
        raise ValueError("aggregator must be 'uniform' or 'hindsight_optimal'")

    for w in all_windows:
        end_date = window_map.loc[w]
        dates_list.append(end_date)

        mu_mc_vec   = mu_mc.loc[w].values.astype(float)
        mu_qae_vec  = mu_qae.loc[w].values.astype(float)
        var_mc_vec  = var_mc.loc[w].values.astype(float)
        var_qae_vec = var_qae.loc[w].values.astype(float)

        mu_true_vec  = true_mu_df.loc[w].values.astype(float)
        var_true_vec = true_var_df.loc[w].values.astype(float)

        # 3a) Choose aggregation weights
        if aggregator == "uniform":
            w_agg = np.ones(N) / N
        else:
            # Solve one‐period hindsight QP: max { w^T μ_true  − λ w^T diag(var_true) w }
            w_var = cp.Variable(N)
            Sigma_true = np.diag(var_true_vec)
            exp_term   = mu_true_vec @ w_var
            risk_term  = cp.quad_form(w_var, Sigma_true, assume_PSD=True)
            objective  = cp.Maximize(exp_term - lambda_risk * risk_term)
            constraints = [cp.sum(w_var) == 1, w_var >= 0]

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, warm_start=True)

            if w_var.value is None:
                w_agg = np.ones(N) / N
            else:
                tmp = np.clip(w_var.value, 0, 1)
                w_agg = tmp / np.sum(tmp)

        # 3b) Compute absolute errors
        err_mu_mc   = abs(w_agg @ (mu_mc_vec   - mu_true_vec))
        err_mu_qae  = abs(w_agg @ (mu_qae_vec  - mu_true_vec))
        err_var_mc  = abs(w_agg @ (var_mc_vec   - var_true_vec))
        err_var_qae = abs(w_agg @ (var_qae_vec  - var_true_vec))

        delta_err_mu.append(err_mu_mc  - err_mu_qae)
        delta_err_var.append(err_var_mc - err_var_qae)

    delta_err_mu_series  = pd.Series(delta_err_mu,  index=dates_list).sort_index()
    delta_err_var_series = pd.Series(delta_err_var, index=dates_list).sort_index()

    # 4) Plot two‐panel bar chart with extra vertical spacing (hspace=0.3)
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Palatino", "serif"],
        "axes.titlesize":    title_fontsize,
        "axes.labelsize":    label_fontsize,
        "xtick.labelsize":   tick_fontsize,
        "ytick.labelsize":   tick_fontsize,
        "legend.fontsize":   legend_fontsize,
        "figure.dpi":        dpi,
        "axes.linewidth":    1.0,
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid":         True,
        "grid.color":        "#999999",
        "grid.alpha":        0.2,
        "grid.linestyle":    "--"
    })

    fig, (ax_mu, ax_var) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.3}
    )

    # ──── Top panel: Δ|error_mean| (MC − QAE) ────────────────────────────────
    colors_mu = np.where(delta_err_mu_series.values >= 0, "salmon", "#4477AA")
    ax_mu.bar(
        delta_err_mu_series.index,
        delta_err_mu_series.values,
        width=2,
        color=colors_mu,
        alpha=0.7,
        edgecolor="none"
    )
    ax_mu.axhline(0, color="black", lw=0.8)
    ax_mu.set_ylabel(r"$\Delta|\hat\mu - \mu^\mathrm{true}|$", fontsize=label_fontsize)
    ax_mu.set_title("Mean‐Estimation Error Δ (MC vs. QAE)", pad=6)
    ax_mu.grid(True, which="major", linestyle="--", alpha=0.2)

    import matplotlib.patches as mpatches
    qae_better_mu = mpatches.Patch(color="salmon",  label="QAE Smaller Error")
    mc_better_mu  = mpatches.Patch(color="#4477AA", label="MC Smaller Error")
    ax_mu.legend(handles=[qae_better_mu, mc_better_mu], loc="upper left", fontsize=legend_fontsize)

    # ──── Bottom panel: Δ|error_var| (MC − QAE) ─────────────────────────────
    colors_var = np.where(delta_err_var_series.values >= 0, "salmon", "#4477AA")
    ax_var.bar(
        delta_err_var_series.index,
        delta_err_var_series.values,
        width=2,
        color=colors_var,
        alpha=0.7,
        edgecolor="none"
    )
    ax_var.axhline(0, color="black", lw=0.8)
    ax_var.set_ylabel(r"$\Delta|\widehat\sigma^2 - \sigma^{2,\mathrm{true}}|$", fontsize=label_fontsize)
    ax_var.set_title("Variance‐Estimation Error Δ (MC vs. QAE)", pad=6)
    ax_var.set_xlabel("Date", fontsize=label_fontsize)
    ax_var.grid(True, which="major", linestyle="--", alpha=0.2)

    qae_better_var = mpatches.Patch(color="salmon",  label="QAE Smaller Error")
    mc_better_var  = mpatches.Patch(color="#4477AA", label="MC Smaller Error")
    ax_var.legend(handles=[qae_better_var, mc_better_var], loc="upper left", fontsize=legend_fontsize)

    plt.tight_layout()
    return fig, (ax_mu, ax_var)