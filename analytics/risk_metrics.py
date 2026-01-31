"""
Additional risk metrics module for bond fund analysis.
Provides VaR, Expected Shortfall, and volatility analytics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Risk metrics calculator for bond fund returns.
    Includes VaR, CVaR, volatility decomposition, and tail risk measures.
    """

    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize risk metrics calculator.

        Args:
            returns_df: DataFrame with fund returns indexed by date
        """
        self.returns_df = returns_df.copy()

    def parametric_var(
        self,
        fund_col: str,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> float:
        """
        Compute parametric VaR assuming normal distribution.

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            holding_period: Holding period in months

        Returns:
            Parametric VaR (negative value representing potential loss)
        """
        returns = self.returns_df[fund_col].dropna()
        mean = returns.mean() * holding_period
        std = returns.std() * np.sqrt(holding_period)

        z_score = stats.norm.ppf(1 - confidence)
        return mean + z_score * std

    def historical_var(
        self,
        fund_col: str,
        confidence: float = 0.95
    ) -> float:
        """
        Compute historical (empirical) VaR.

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level

        Returns:
            Historical VaR
        """
        returns = self.returns_df[fund_col].dropna()
        return returns.quantile(1 - confidence)

    def cornish_fisher_var(
        self,
        fund_col: str,
        confidence: float = 0.95
    ) -> float:
        """
        Compute VaR with Cornish-Fisher expansion for non-normal distributions.
        Adjusts for skewness and kurtosis.

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level

        Returns:
            Cornish-Fisher adjusted VaR
        """
        returns = self.returns_df[fund_col].dropna()

        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        z = stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * (kurt - 3) / 24 -
            (2*z**3 - 5*z) * skew**2 / 36
        )

        return mean + z_cf * std

    def expected_shortfall(
        self,
        fund_col: str,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Compute Expected Shortfall (Conditional VaR).

        Args:
            fund_col: Column name for fund returns
            confidence: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            Expected Shortfall
        """
        returns = self.returns_df[fund_col].dropna()

        if method == 'historical':
            var = self.historical_var(fund_col, confidence)
            return returns[returns <= var].mean()
        else:
            # Parametric ES assuming normality
            mean = returns.mean()
            std = returns.std()
            z = stats.norm.ppf(1 - confidence)
            es_z = stats.norm.pdf(z) / (1 - confidence)
            return mean - std * es_z

    def volatility_metrics(self, fund_col: str) -> Dict[str, float]:
        """
        Compute various volatility metrics.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Dictionary with volatility metrics
        """
        returns = self.returns_df[fund_col].dropna()

        # Standard volatility (annualized)
        std_vol = returns.std() * np.sqrt(12)

        # Downside volatility
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(12) if len(negative_returns) > 0 else 0

        # Upside volatility
        positive_returns = returns[returns > 0]
        upside_vol = positive_returns.std() * np.sqrt(12) if len(positive_returns) > 0 else 0

        # Semi-deviation (below mean)
        below_mean = returns[returns < returns.mean()]
        semi_dev = below_mean.std() * np.sqrt(12) if len(below_mean) > 0 else 0

        return {
            'volatility': std_vol,
            'downside_volatility': downside_vol,
            'upside_volatility': upside_vol,
            'semi_deviation': semi_dev,
            'volatility_ratio': upside_vol / downside_vol if downside_vol > 0 else np.inf
        }

    def tail_risk_metrics(self, fund_col: str) -> Dict[str, float]:
        """
        Compute tail risk statistics.

        Args:
            fund_col: Column name for fund returns

        Returns:
            Dictionary with tail risk metrics
        """
        returns = self.returns_df[fund_col].dropna()

        return {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'excess_kurtosis': returns.kurtosis() - 3,
            'min_return': returns.min(),
            'max_return': returns.max(),
            'percentile_5': returns.quantile(0.05),
            'percentile_95': returns.quantile(0.95),
            'negative_months_pct': (returns < 0).sum() / len(returns) * 100
        }

    def rolling_volatility(
        self,
        fund_col: str,
        window: int = 12
    ) -> pd.Series:
        """
        Compute rolling annualized volatility.

        Args:
            fund_col: Column name for fund returns
            window: Rolling window in months

        Returns:
            Series of rolling volatility values
        """
        returns = self.returns_df[fund_col].dropna()
        return returns.rolling(window).std() * np.sqrt(12)

    def rolling_var(
        self,
        fund_col: str,
        window: int = 12,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Compute rolling VaR.

        Args:
            fund_col: Column name for fund returns
            window: Rolling window in months
            confidence: Confidence level

        Returns:
            Series of rolling VaR values
        """
        returns = self.returns_df[fund_col].dropna()
        return returns.rolling(window).quantile(1 - confidence)

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix for all funds.

        Returns:
            Correlation matrix DataFrame
        """
        return self.returns_df.corr()

    def rolling_correlation(
        self,
        fund_col: str,
        other_col: str,
        window: int = 12
    ) -> pd.Series:
        """
        Compute rolling correlation between two funds.

        Args:
            fund_col: First fund column
            other_col: Second fund column
            window: Rolling window in months

        Returns:
            Series of rolling correlation values
        """
        return self.returns_df[fund_col].rolling(window).corr(
            self.returns_df[other_col]
        )

    def generate_risk_report(self, fund_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate comprehensive risk report for all funds.

        Args:
            fund_cols: Optional list of fund columns (uses all if None)

        Returns:
            DataFrame with risk metrics for all funds
        """
        if fund_cols is None:
            fund_cols = list(self.returns_df.columns)

        results = []
        for col in fund_cols:
            try:
                vol_metrics = self.volatility_metrics(col)
                tail_metrics = self.tail_risk_metrics(col)

                row = {
                    'ticker': col,
                    'historical_var_95': self.historical_var(col, 0.95),
                    'parametric_var_95': self.parametric_var(col, 0.95),
                    'cornish_fisher_var_95': self.cornish_fisher_var(col, 0.95),
                    'expected_shortfall_95': self.expected_shortfall(col, 0.95),
                    **vol_metrics,
                    **tail_metrics
                }
                results.append(row)
            except Exception as e:
                logger.warning(f"Error computing risk metrics for {col}: {e}")
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')

        return df
