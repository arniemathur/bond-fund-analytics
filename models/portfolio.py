"""
Portfolio container for managing multiple funds and their analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from .fund import FundFacts


@dataclass
class Portfolio:
    """
    Container for managing a portfolio/universe of bond funds.
    Handles aggregation of fund facts and returns metrics.
    """

    name: str = "Bond Fund Portfolio"
    funds: Dict[str, FundFacts] = field(default_factory=dict)
    returns_metrics: Optional[pd.DataFrame] = None
    risk_metrics: Optional[pd.DataFrame] = None
    scenario_results: Optional[pd.DataFrame] = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_fund(self, fund: FundFacts):
        """Add a fund to the portfolio."""
        self.funds[fund.ticker] = fund

    def remove_fund(self, ticker: str):
        """Remove a fund from the portfolio."""
        ticker = ticker.upper()
        if ticker in self.funds:
            del self.funds[ticker]

    def get_fund(self, ticker: str) -> Optional[FundFacts]:
        """Get a fund by ticker."""
        return self.funds.get(ticker.upper())

    @property
    def tickers(self) -> List[str]:
        """Get list of all fund tickers."""
        return list(self.funds.keys())

    @property
    def fund_count(self) -> int:
        """Get number of funds in portfolio."""
        return len(self.funds)

    def get_fund_facts_df(self) -> pd.DataFrame:
        """
        Get all fund facts as a DataFrame.

        Returns:
            DataFrame with fund facts indexed by ticker
        """
        if not self.funds:
            return pd.DataFrame()

        rows = []
        for ticker, fund in self.funds.items():
            row = {
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'aum': fund.aum,
                'expense_ratio': fund.expense_ratio,
                'yield': fund.yield_current,
                'ytm': fund.yield_to_maturity,
                'duration': fund.effective_duration,
                'maturity': fund.effective_maturity,
                'credit_rating': fund.avg_credit_rating,
                'weighted_coupon': fund.weighted_coupon,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('ticker')

        return df

    def get_combined_metrics(self) -> pd.DataFrame:
        """
        Combine fund facts with returns metrics into a single DataFrame.

        Returns:
            Combined DataFrame
        """
        facts_df = self.get_fund_facts_df()

        if self.returns_metrics is not None and not facts_df.empty:
            # Merge on index (ticker)
            combined = facts_df.join(self.returns_metrics, how='outer')
        else:
            combined = facts_df

        if self.risk_metrics is not None and not combined.empty:
            combined = combined.join(self.risk_metrics, how='outer', rsuffix='_risk')

        return combined

    def rank_funds(
        self,
        metrics: List[str],
        ascending: Optional[List[bool]] = None,
        weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Rank funds by specified metrics and compute composite score.

        Args:
            metrics: List of metric column names to rank by
            ascending: List of booleans for each metric (True = lower is better)
            weights: Optional weights for each metric in composite score

        Returns:
            DataFrame with rankings and composite score
        """
        df = self.get_combined_metrics()
        if df.empty:
            return pd.DataFrame()

        if ascending is None:
            ascending = [True] * len(metrics)

        if weights is None:
            weights = [1.0] * len(metrics)

        rankings = pd.DataFrame(index=df.index)

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=ascending[i])

        # Compute weighted composite score (lower is better)
        composite = pd.Series(0.0, index=df.index)
        for i, metric in enumerate(metrics):
            rank_col = f'{metric}_rank'
            if rank_col in rankings.columns:
                composite += rankings[rank_col] * weights[i]

        rankings['composite_score'] = composite / sum(weights)
        rankings['overall_rank'] = rankings['composite_score'].rank()

        return rankings.sort_values('overall_rank')

    def compare_funds(
        self,
        tickers: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare specific funds side by side.

        Args:
            tickers: List of fund tickers to compare
            metrics: Optional list of metrics to include

        Returns:
            DataFrame with fund comparison
        """
        tickers = [t.upper() for t in tickers]
        df = self.get_combined_metrics()

        if df.empty:
            return pd.DataFrame()

        # Filter to requested tickers
        df = df.loc[df.index.isin(tickers)]

        # Filter to requested metrics
        if metrics:
            available = [m for m in metrics if m in df.columns]
            df = df[available]

        return df.T  # Transpose for side-by-side comparison

    def get_category_summary(self) -> pd.DataFrame:
        """
        Get summary statistics by fund category.

        Returns:
            DataFrame with category-level aggregations
        """
        df = self.get_combined_metrics()
        if df.empty:
            return pd.DataFrame()

        # Get category from fund facts
        categories = {t: f.category for t, f in self.funds.items()}
        df['category'] = df.index.map(categories)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return df.groupby('category')[numeric_cols].agg(['mean', 'min', 'max', 'count'])

    def filter_by_category(self, category: str) -> 'Portfolio':
        """
        Create a new Portfolio filtered to a specific category.

        Args:
            category: The category to filter by

        Returns:
            New Portfolio containing only funds in the category
        """
        filtered = Portfolio(name=f"{self.name} - {category}")

        for ticker, fund in self.funds.items():
            if category.lower() in fund.category.lower():
                filtered.add_fund(fund)

        return filtered

    def filter_by_duration(
        self,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> 'Portfolio':
        """
        Create a new Portfolio filtered by duration range.

        Args:
            min_duration: Minimum duration (inclusive)
            max_duration: Maximum duration (inclusive)

        Returns:
            New Portfolio containing only funds in the duration range
        """
        filtered = Portfolio(name=f"{self.name} - Duration Filtered")

        for ticker, fund in self.funds.items():
            if fund.effective_duration is None:
                continue

            if min_duration and fund.effective_duration < min_duration:
                continue

            if max_duration and fund.effective_duration > max_duration:
                continue

            filtered.add_fund(fund)

        return filtered

    def get_duration_buckets(self) -> Dict[str, List[str]]:
        """
        Group funds into duration buckets.

        Returns:
            Dictionary mapping bucket names to lists of tickers
        """
        buckets = {
            'Ultra Short (0-1y)': [],
            'Short (1-3y)': [],
            'Intermediate (3-7y)': [],
            'Long (7-15y)': [],
            'Extended (15y+)': [],
        }

        for ticker, fund in self.funds.items():
            dur = fund.effective_duration
            if dur is None:
                continue

            if dur < 1:
                buckets['Ultra Short (0-1y)'].append(ticker)
            elif dur < 3:
                buckets['Short (1-3y)'].append(ticker)
            elif dur < 7:
                buckets['Intermediate (3-7y)'].append(ticker)
            elif dur < 15:
                buckets['Long (7-15y)'].append(ticker)
            else:
                buckets['Extended (15y+)'].append(ticker)

        return {k: v for k, v in buckets.items() if v}

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary."""
        return {
            'name': self.name,
            'funds': {t: f.to_dict() for t, f in self.funds.items()},
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create portfolio from dictionary."""
        portfolio = cls(name=data.get('name', 'Portfolio'))

        for ticker, fund_data in data.get('funds', {}).items():
            fund = FundFacts.from_dict(fund_data)
            portfolio.add_fund(fund)

        if 'created_at' in data:
            try:
                portfolio.created_at = datetime.fromisoformat(data['created_at'])
            except ValueError:
                pass

        return portfolio

    def __repr__(self) -> str:
        """String representation."""
        return f"Portfolio(name='{self.name}', funds={self.fund_count})"
