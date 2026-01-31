"""
Historical returns data loader using yfinance.
Fetches adjusted close prices and computes monthly total returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ReturnsLoader:
    """Loads and processes historical returns data for ETFs."""

    def __init__(self, tickers: List[str], benchmark_map: Optional[Dict[str, str]] = None):
        """
        Initialize the returns loader.

        Args:
            tickers: List of ETF ticker symbols
            benchmark_map: Optional mapping of tickers to their benchmark tickers
        """
        self.tickers = tickers
        self.benchmark_map = benchmark_map or {}
        self._prices_df: Optional[pd.DataFrame] = None
        self._returns_df: Optional[pd.DataFrame] = None

    def fetch_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 5
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for all tickers.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            lookback_years: Number of years to look back if no start_date provided

        Returns:
            DataFrame with adjusted close prices indexed by date
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start_dt = datetime.now() - timedelta(days=lookback_years * 365)
            start_date = start_dt.strftime('%Y-%m-%d')

        # Get all unique tickers including benchmarks
        all_tickers = list(set(self.tickers + list(self.benchmark_map.values())))

        logger.info(f"Fetching prices for {len(all_tickers)} tickers from {start_date} to {end_date}")

        try:
            data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,  # Use adjusted prices
                progress=False
            )

            # Handle single ticker case (yf returns Series instead of DataFrame)
            if len(all_tickers) == 1:
                self._prices_df = pd.DataFrame(data['Close'])
                self._prices_df.columns = all_tickers
            else:
                self._prices_df = data['Close']

            # Drop rows with all NaN values
            self._prices_df = self._prices_df.dropna(how='all')

            logger.info(f"Fetched {len(self._prices_df)} price observations")
            return self._prices_df

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            raise

    def compute_monthly_returns(self, prices_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute monthly total returns from daily prices.

        Args:
            prices_df: Optional prices DataFrame (uses cached if not provided)

        Returns:
            DataFrame with monthly returns indexed by month-end date
        """
        if prices_df is None:
            prices_df = self._prices_df

        if prices_df is None:
            raise ValueError("No prices data available. Call fetch_prices() first.")

        # Resample to month-end prices
        monthly_prices = prices_df.resample('ME').last()

        # Calculate monthly returns
        self._returns_df = monthly_prices.pct_change().dropna()

        logger.info(f"Computed {len(self._returns_df)} monthly return observations")
        return self._returns_df

    def compute_daily_returns(self, prices_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute daily returns from prices.

        Args:
            prices_df: Optional prices DataFrame (uses cached if not provided)

        Returns:
            DataFrame with daily returns
        """
        if prices_df is None:
            prices_df = self._prices_df

        if prices_df is None:
            raise ValueError("No prices data available. Call fetch_prices() first.")

        return prices_df.pct_change().dropna()

    def get_cumulative_returns(self, returns_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute cumulative returns from periodic returns.

        Args:
            returns_df: Optional returns DataFrame (uses cached monthly if not provided)

        Returns:
            DataFrame with cumulative returns (growth of $1)
        """
        if returns_df is None:
            returns_df = self._returns_df

        if returns_df is None:
            raise ValueError("No returns data available. Call compute_monthly_returns() first.")

        return (1 + returns_df).cumprod()

    def get_returns_with_benchmark(self, ticker: str) -> pd.DataFrame:
        """
        Get returns for a specific ticker alongside its benchmark.

        Args:
            ticker: The fund ticker

        Returns:
            DataFrame with fund and benchmark returns columns
        """
        if self._returns_df is None:
            raise ValueError("No returns data available. Call compute_monthly_returns() first.")

        benchmark = self.benchmark_map.get(ticker, ticker)

        cols_to_get = [ticker]
        if benchmark != ticker and benchmark in self._returns_df.columns:
            cols_to_get.append(benchmark)

        return self._returns_df[cols_to_get].dropna()

    @property
    def prices(self) -> Optional[pd.DataFrame]:
        """Get cached prices DataFrame."""
        return self._prices_df

    @property
    def returns(self) -> Optional[pd.DataFrame]:
        """Get cached returns DataFrame."""
        return self._returns_df

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 5
    ) -> pd.DataFrame:
        """
        Convenience method to fetch prices and compute monthly returns.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            lookback_years: Number of years to look back if no start_date

        Returns:
            DataFrame with monthly returns
        """
        self.fetch_prices(start_date, end_date, lookback_years)
        return self.compute_monthly_returns()
