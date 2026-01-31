"""
Returns analytics engine for computing fund performance metrics.
Includes alpha, beta, R², Sharpe ratio, drawdown, and capture ratios.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ReturnsAnalyzer:
    """
    Comprehensive returns analysis for bond ETFs.
    Computes regression metrics, risk metrics, and capture ratios.
    """

    ANNUALIZATION_FACTOR = 12  # Monthly to annual

    def __init__(
        self,
        returns_df: pd.DataFrame,
        benchmark_col: str,
        rf_rate: float = 0.05
    ):
        """
        Initialize the returns analyzer.

        Args:
            returns_df: DataFrame with fund returns (columns) indexed by date
            benchmark_col: Column name for benchmark returns
            rf_rate: Annual risk-free rate (default 5%)
        """
        self.returns_df = returns_df.copy()
        self.benchmark_col = benchmark_col
        self.rf_rate = rf_rate
        self.rf_monthly = rf_rate / self.ANNUALIZATION_FACTOR

        if benchmark_col not in self.returns_df.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in returns DataFrame")

        # Get fund columns (all except benchmark if different)
        self.fund_cols = [
            col for col in self.returns_df.columns
            if col != benchmark_col or col == benchmark_col
        ]

    def compute_alpha_beta_r2(self, fund_col: str) -> Dict[str, float]:
        """
        Compute alpha, beta, and R² from regression of excess fund returns on excess benchmark returns.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Dictionary with alpha (annualized), beta, r_squared, and t-stats
        """
        fund_returns = self.returns_df[fund_col].dropna()
        benchmark_returns = self.returns_df[self.benchmark_col].dropna()

        # Align the series
        aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
        fund_excess = aligned.iloc[:, 0] - self.rf_monthly
        bench_excess = aligned.iloc[:, 1] - self.rf_monthly

        # OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            bench_excess, fund_excess
        )

        # Annualize alpha (multiply monthly alpha by 12)
        alpha_annual = intercept * self.ANNUALIZATION_FACTOR

        return {
            'alpha': alpha_annual,
            'alpha_monthly': intercept,
            'beta': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err
        }

    def compute_tracking_error(self, fund_col: str) -> float:
        """
        Compute annualized tracking error (std dev of return differences).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Annualized tracking error
        """
        fund_returns = self.returns_df[fund_col]
        benchmark_returns = self.returns_df[self.benchmark_col]

        # Align and compute difference
        aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
        tracking_diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]

        # Annualize
        return tracking_diff.std() * np.sqrt(self.ANNUALIZATION_FACTOR)

    def compute_information_ratio(self, fund_col: str) -> float:
        """
        Compute information ratio (excess return over benchmark / tracking error).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Information ratio
        """
        fund_returns = self.returns_df[fund_col]
        benchmark_returns = self.returns_df[self.benchmark_col]

        # Align and compute
        aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
        tracking_diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]

        mean_excess = tracking_diff.mean() * self.ANNUALIZATION_FACTOR
        tracking_error = tracking_diff.std() * np.sqrt(self.ANNUALIZATION_FACTOR)

        if tracking_error == 0:
            return 0.0

        return mean_excess / tracking_error

    def compute_std_dev(self, fund_col: str, annualized: bool = True) -> float:
        """
        Compute standard deviation of returns.

        Args:
            fund_col: Column name for fund returns
            annualized: Whether to annualize the volatility

        Returns:
            Standard deviation (annualized if specified)
        """
        std = self.returns_df[fund_col].std()

        if annualized:
            return std * np.sqrt(self.ANNUALIZATION_FACTOR)
        return std

    def compute_sharpe_ratio(self, fund_col: str) -> float:
        """
        Compute annualized Sharpe ratio.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Annualized Sharpe ratio
        """
        returns = self.returns_df[fund_col].dropna()

        mean_return = returns.mean() * self.ANNUALIZATION_FACTOR
        std_dev = returns.std() * np.sqrt(self.ANNUALIZATION_FACTOR)

        if std_dev == 0:
            return 0.0

        return (mean_return - self.rf_rate) / std_dev

    def compute_sortino_ratio(self, fund_col: str) -> float:
        """
        Compute annualized Sortino ratio (uses downside deviation).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Annualized Sortino ratio
        """
        returns = self.returns_df[fund_col].dropna()

        mean_return = returns.mean() * self.ANNUALIZATION_FACTOR

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < self.rf_monthly]
        if len(negative_returns) == 0:
            return np.inf

        downside_dev = negative_returns.std() * np.sqrt(self.ANNUALIZATION_FACTOR)

        if downside_dev == 0:
            return np.inf

        return (mean_return - self.rf_rate) / downside_dev

    def compute_max_drawdown(self, fund_col: str) -> float:
        """
        Compute maximum drawdown (peak-to-trough decline).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Maximum drawdown as a negative decimal
        """
        returns = self.returns_df[fund_col].dropna()

        # Calculate cumulative wealth
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def compute_drawdown_series(self, fund_col: str) -> pd.Series:
        """
        Compute the full drawdown time series.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Series of drawdown values over time
        """
        returns = self.returns_df[fund_col].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max

    def compute_var(self, fund_col: str, confidence: float = 0.95) -> float:
        """
        Compute Value at Risk using historical method.

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level (default 95%)

        Returns:
            VaR as a negative decimal (potential loss)
        """
        returns = self.returns_df[fund_col].dropna()
        return returns.quantile(1 - confidence)

    def compute_expected_shortfall(self, fund_col: str, confidence: float = 0.95) -> float:
        """
        Compute Expected Shortfall (CVaR) - average loss beyond VaR.

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level (default 95%)

        Returns:
            Expected shortfall as a negative decimal
        """
        returns = self.returns_df[fund_col].dropna()
        var = self.compute_var(fund_col, confidence)
        return returns[returns <= var].mean()

    def compute_upside_capture(self, fund_col: str) -> float:
        """
        Compute upside capture ratio.
        Measures fund performance relative to benchmark in up months.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Upside capture ratio as percentage
        """
        fund_returns = self.returns_df[fund_col]
        benchmark_returns = self.returns_df[self.benchmark_col]

        # Align
        aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
        fund = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # Up months (benchmark positive)
        up_mask = bench > 0
        if up_mask.sum() == 0:
            return 0.0

        fund_up = fund[up_mask]
        bench_up = bench[up_mask]

        # Geometric mean ratio
        fund_geo = (1 + fund_up).prod() ** (1 / len(fund_up)) - 1
        bench_geo = (1 + bench_up).prod() ** (1 / len(bench_up)) - 1

        if bench_geo == 0:
            return 0.0

        return (fund_geo / bench_geo) * 100

    def compute_downside_capture(self, fund_col: str) -> float:
        """
        Compute downside capture ratio.
        Measures fund performance relative to benchmark in down months.
        Lower is better (less downside participation).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Downside capture ratio as percentage
        """
        fund_returns = self.returns_df[fund_col]
        benchmark_returns = self.returns_df[self.benchmark_col]

        # Align
        aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
        fund = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # Down months (benchmark negative)
        down_mask = bench < 0
        if down_mask.sum() == 0:
            return 0.0

        fund_down = fund[down_mask]
        bench_down = bench[down_mask]

        # Geometric mean ratio
        fund_geo = (1 + fund_down).prod() ** (1 / len(fund_down)) - 1
        bench_geo = (1 + bench_down).prod() ** (1 / len(bench_down)) - 1

        if bench_geo == 0:
            return 0.0

        return (fund_geo / bench_geo) * 100

    def compute_capture_ratio(self, fund_col: str) -> float:
        """
        Compute the capture ratio (upside / downside).
        Higher is better.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Capture ratio
        """
        upside = self.compute_upside_capture(fund_col)
        downside = self.compute_downside_capture(fund_col)

        if downside == 0:
            return np.inf

        return upside / downside

    def compute_total_return(self, fund_col: str) -> float:
        """
        Compute total cumulative return over the period.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Total return as a decimal
        """
        returns = self.returns_df[fund_col].dropna()
        return (1 + returns).prod() - 1

    def compute_annualized_return(self, fund_col: str) -> float:
        """
        Compute annualized return (CAGR).

        Args:
            fund_col: Column name for fund returns

        Returns:
            Annualized return as a decimal
        """
        returns = self.returns_df[fund_col].dropna()
        total_return = (1 + returns).prod()
        n_years = len(returns) / self.ANNUALIZATION_FACTOR

        if n_years == 0:
            return 0.0

        return total_return ** (1 / n_years) - 1

    def compute_all_metrics(self, fund_col: str) -> Dict[str, float]:
        """
        Compute all metrics for a single fund.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Dictionary with all computed metrics
        """
        regression = self.compute_alpha_beta_r2(fund_col)

        return {
            'ticker': fund_col,
            'total_return': self.compute_total_return(fund_col),
            'annualized_return': self.compute_annualized_return(fund_col),
            'volatility': self.compute_std_dev(fund_col),
            'sharpe_ratio': self.compute_sharpe_ratio(fund_col),
            'sortino_ratio': self.compute_sortino_ratio(fund_col),
            'max_drawdown': self.compute_max_drawdown(fund_col),
            'alpha': regression['alpha'],
            'beta': regression['beta'],
            'r_squared': regression['r_squared'],
            'tracking_error': self.compute_tracking_error(fund_col),
            'information_ratio': self.compute_information_ratio(fund_col),
            'upside_capture': self.compute_upside_capture(fund_col),
            'downside_capture': self.compute_downside_capture(fund_col),
            'capture_ratio': self.compute_capture_ratio(fund_col),
            'var_95': self.compute_var(fund_col, 0.95),
            'expected_shortfall_95': self.compute_expected_shortfall(fund_col, 0.95)
        }

    def generate_full_report(self, fund_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a comprehensive metrics report for all funds.

        Args:
            fund_cols: Optional list of fund columns to analyze (uses all if None)

        Returns:
            DataFrame with all metrics for all funds
        """
        if fund_cols is None:
            fund_cols = [col for col in self.returns_df.columns]

        results = []
        for col in fund_cols:
            try:
                metrics = self.compute_all_metrics(col)
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Error computing metrics for {col}: {e}")
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')

        return df

    def get_rolling_metrics(
        self,
        fund_col: str,
        window: int = 12
    ) -> pd.DataFrame:
        """
        Compute rolling metrics over time.

        Args:
            fund_col: Column name for fund returns
            window: Rolling window in months

        Returns:
            DataFrame with rolling metrics
        """
        returns = self.returns_df[fund_col].dropna()
        benchmark = self.returns_df[self.benchmark_col].dropna()

        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        fund = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        rolling_return = fund.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        rolling_vol = fund.rolling(window).std() * np.sqrt(self.ANNUALIZATION_FACTOR)
        rolling_sharpe = (
            (fund.rolling(window).mean() * self.ANNUALIZATION_FACTOR - self.rf_rate) /
            rolling_vol
        )

        return pd.DataFrame({
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe
        })
