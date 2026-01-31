"""
Scenario analysis module for bond funds.
Implements rate shock and credit spread shock analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fund import FundFacts

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of a scenario analysis for a single fund."""
    ticker: str
    scenario_name: str
    shock_bps: int
    price_impact_pct: float
    duration: Optional[float]
    is_applicable: bool


class ScenarioAnalyzer:
    """
    Scenario analyzer for rate and credit spread shocks.
    Uses duration-based approximation for price impact estimation.
    """

    def __init__(self, fund_facts: List[FundFacts]):
        """
        Initialize the scenario analyzer.

        Args:
            fund_facts: List of FundFacts objects for analysis
        """
        self.fund_facts = {f.ticker: f for f in fund_facts}

    def parallel_rate_shock(self, bps_change: int) -> pd.DataFrame:
        """
        Compute approximate price impact from parallel yield curve shift.

        Formula: Price Impact ≈ -Duration × ΔYield
        This is a first-order approximation (ignores convexity).

        Args:
            bps_change: Rate change in basis points (e.g., 100 = 1%)

        Returns:
            DataFrame with fund impacts
        """
        results = []

        for ticker, fund in self.fund_facts.items():
            if fund.effective_duration is None:
                results.append({
                    'ticker': ticker,
                    'name': fund.name,
                    'category': fund.category,
                    'duration': None,
                    'rate_shock_bps': bps_change,
                    'price_impact_pct': None,
                    'applicable': False
                })
                continue

            # Price impact = -Duration × ΔRate (in decimal)
            rate_change_decimal = bps_change / 10000  # Convert bps to decimal
            price_impact = -fund.effective_duration * rate_change_decimal * 100  # Convert to percent

            results.append({
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'duration': fund.effective_duration,
                'rate_shock_bps': bps_change,
                'price_impact_pct': price_impact,
                'applicable': True
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')
            # Sort by impact (most negative first for positive rate shock)
            if bps_change > 0:
                df = df.sort_values('price_impact_pct', ascending=True)
            else:
                df = df.sort_values('price_impact_pct', ascending=False)

        return df

    def credit_spread_shock(self, bps_change: int) -> pd.DataFrame:
        """
        Compute approximate price impact from credit spread changes.

        For credit-sensitive funds (corporate, high yield), spread duration
        approximates how prices react to credit spread changes.
        We use effective duration as a proxy for spread duration.

        Args:
            bps_change: Spread change in basis points

        Returns:
            DataFrame with fund impacts (only credit-sensitive funds)
        """
        results = []

        for ticker, fund in self.fund_facts.items():
            # Check if fund is credit-sensitive
            if not fund.is_credit_fund:
                results.append({
                    'ticker': ticker,
                    'name': fund.name,
                    'category': fund.category,
                    'duration': fund.effective_duration,
                    'spread_shock_bps': bps_change,
                    'price_impact_pct': None,
                    'applicable': False,
                    'reason': 'Not a credit fund'
                })
                continue

            if fund.effective_duration is None:
                results.append({
                    'ticker': ticker,
                    'name': fund.name,
                    'category': fund.category,
                    'duration': None,
                    'spread_shock_bps': bps_change,
                    'price_impact_pct': None,
                    'applicable': False,
                    'reason': 'No duration data'
                })
                continue

            # Use effective duration as proxy for spread duration
            # Apply a credit multiplier based on credit quality
            spread_duration = fund.effective_duration

            # High yield bonds typically have higher spread sensitivity
            if 'high yield' in fund.category.lower():
                spread_duration *= 1.2  # Adjust for higher spread beta

            spread_change_decimal = bps_change / 10000
            price_impact = -spread_duration * spread_change_decimal * 100

            results.append({
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'duration': fund.effective_duration,
                'spread_duration': spread_duration,
                'spread_shock_bps': bps_change,
                'price_impact_pct': price_impact,
                'applicable': True,
                'reason': None
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')
            # Sort by impact
            df = df.sort_values('price_impact_pct', ascending=True, na_position='last')

        return df

    def generate_scenario_table(
        self,
        rate_shocks: List[int] = [100, -100, 50, -50],
        spread_shocks: List[int] = [50, -50, 100]
    ) -> pd.DataFrame:
        """
        Generate comprehensive scenario analysis table.

        Args:
            rate_shocks: List of rate shocks in bps to analyze
            spread_shocks: List of spread shocks in bps to analyze

        Returns:
            DataFrame with all scenarios for all funds
        """
        results = []

        for ticker, fund in self.fund_facts.items():
            row = {
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'duration': fund.effective_duration,
                'is_credit_fund': fund.is_credit_fund,
            }

            # Add rate shock scenarios
            for shock in rate_shocks:
                col_name = f'rate_{shock:+d}bps'
                if fund.effective_duration is not None:
                    impact = -fund.effective_duration * (shock / 10000) * 100
                    row[col_name] = impact
                else:
                    row[col_name] = None

            # Add spread shock scenarios (only for credit funds)
            for shock in spread_shocks:
                col_name = f'spread_{shock:+d}bps'
                if fund.is_credit_fund and fund.effective_duration is not None:
                    spread_dur = fund.effective_duration
                    if 'high yield' in fund.category.lower():
                        spread_dur *= 1.2
                    impact = -spread_dur * (shock / 10000) * 100
                    row[col_name] = impact
                else:
                    row[col_name] = None

            results.append(row)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')

            # Add ranking columns based on first rate shock
            rate_cols = [c for c in df.columns if c.startswith('rate_')]
            if rate_cols and df[rate_cols[0]].notna().any():
                df['rate_risk_rank'] = pd.to_numeric(df[rate_cols[0]], errors='coerce').abs().rank(ascending=False, na_option='bottom')
                df = df.sort_values('rate_risk_rank')

            spread_cols = [c for c in df.columns if c.startswith('spread_')]
            if spread_cols and df[spread_cols[0]].notna().any():
                df['spread_risk_rank'] = pd.to_numeric(df[spread_cols[0]], errors='coerce').abs().rank(ascending=False, na_option='bottom')

        return df

    def get_rate_sensitivity_report(self) -> pd.DataFrame:
        """
        Generate a rate sensitivity report with rankings.

        Returns:
            DataFrame with rate sensitivity metrics
        """
        data = []

        for ticker, fund in self.fund_facts.items():
            if fund.effective_duration is None:
                continue

            # Calculate impacts for standard scenarios
            impact_plus100 = -fund.effective_duration * 0.01 * 100
            impact_minus100 = -impact_plus100

            data.append({
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'duration': fund.effective_duration,
                'impact_+100bps': impact_plus100,
                'impact_-100bps': impact_minus100,
                'sensitivity_score': fund.get_rate_sensitivity_score(),
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.set_index('ticker')
            df['rate_risk_rank'] = df['duration'].rank(ascending=False)
            df = df.sort_values('rate_risk_rank')

        return df

    def get_credit_sensitivity_report(self) -> pd.DataFrame:
        """
        Generate a credit spread sensitivity report.

        Returns:
            DataFrame with credit sensitivity metrics for credit funds only
        """
        data = []

        for ticker, fund in self.fund_facts.items():
            if not fund.is_credit_fund:
                continue

            if fund.effective_duration is None:
                continue

            spread_dur = fund.effective_duration
            if 'high yield' in fund.category.lower():
                spread_dur *= 1.2

            impact_plus50 = -spread_dur * 0.005 * 100
            impact_plus100 = -spread_dur * 0.01 * 100

            data.append({
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'duration': fund.effective_duration,
                'spread_duration': spread_dur,
                'impact_+50bps_spread': impact_plus50,
                'impact_+100bps_spread': impact_plus100,
                'sensitivity_score': fund.get_credit_sensitivity_score(),
                'ig_pct': fund.investment_grade_pct,
                'hy_pct': fund.high_yield_pct,
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.set_index('ticker')
            df['credit_risk_rank'] = df['spread_duration'].rank(ascending=False)
            df = df.sort_values('credit_risk_rank')

        return df

    def stress_test(
        self,
        scenario_name: str,
        rate_shock_bps: int = 0,
        spread_shock_bps: int = 0
    ) -> pd.DataFrame:
        """
        Run a combined stress test scenario.

        Args:
            scenario_name: Name for the scenario
            rate_shock_bps: Rate shock in basis points
            spread_shock_bps: Credit spread shock in basis points

        Returns:
            DataFrame with combined impact
        """
        results = []

        for ticker, fund in self.fund_facts.items():
            if fund.effective_duration is None:
                continue

            # Rate impact
            rate_impact = -fund.effective_duration * (rate_shock_bps / 10000) * 100

            # Spread impact (only for credit funds)
            spread_impact = 0
            if fund.is_credit_fund:
                spread_dur = fund.effective_duration
                if 'high yield' in fund.category.lower():
                    spread_dur *= 1.2
                spread_impact = -spread_dur * (spread_shock_bps / 10000) * 100

            total_impact = rate_impact + spread_impact

            results.append({
                'ticker': ticker,
                'name': fund.name,
                'category': fund.category,
                'scenario': scenario_name,
                'rate_shock_bps': rate_shock_bps,
                'spread_shock_bps': spread_shock_bps,
                'rate_impact_pct': rate_impact,
                'spread_impact_pct': spread_impact if fund.is_credit_fund else None,
                'total_impact_pct': total_impact,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('ticker')
            df = df.sort_values('total_impact_pct')

        return df

    def run_historical_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Run historical stress scenarios based on past market events.

        Returns:
            Dictionary mapping scenario names to result DataFrames
        """
        scenarios = {
            '2013 Taper Tantrum': {'rate_shock_bps': 100, 'spread_shock_bps': 50},
            '2020 COVID Crisis': {'rate_shock_bps': -150, 'spread_shock_bps': 300},
            '2022 Rate Hikes': {'rate_shock_bps': 200, 'spread_shock_bps': 75},
            'Flight to Quality': {'rate_shock_bps': -50, 'spread_shock_bps': 100},
            'Risk-On Rally': {'rate_shock_bps': 50, 'spread_shock_bps': -50},
        }

        results = {}
        for name, params in scenarios.items():
            results[name] = self.stress_test(name, **params)

        return results
