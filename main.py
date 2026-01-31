#!/usr/bin/env python3
"""
CLI entry point for Fixed Income Fund Analytics.
Provides command-line interface for running analyses and generating reports.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

from data.loaders.returns import ReturnsLoader
from data.scrapers.morningstar import MorningstarScraper
from data.scrapers.blackrock import BlackRockScraper
from analytics.returns_engine import ReturnsAnalyzer
from analytics.risk_metrics import RiskMetrics
from analytics.scenarios import ScenarioAnalyzer
from models.fund import FundFacts
from models.portfolio import Portfolio
from outputs.excel_report import ExcelReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/funds.yaml') -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_analysis(
    tickers: list,
    lookback_years: int = 5,
    rf_rate: float = 0.05,
    output_file: str = None,
    scrape_facts: bool = True
) -> Portfolio:
    """
    Run complete bond fund analysis.

    Args:
        tickers: List of ETF tickers to analyze
        lookback_years: Years of historical data
        rf_rate: Risk-free rate for Sharpe calculations
        output_file: Optional Excel output file path
        scrape_facts: Whether to scrape fund facts

    Returns:
        Portfolio object with analysis results
    """
    config = load_config()
    fund_config = config.get('funds', {})

    # Build benchmark mapping
    benchmark_map = {}
    for ticker in tickers:
        if ticker in fund_config:
            benchmark_map[ticker] = fund_config[ticker].get('benchmark', 'AGG')
        else:
            benchmark_map[ticker] = 'AGG'

    logger.info(f"Analyzing {len(tickers)} funds: {', '.join(tickers)}")

    # Load returns data
    logger.info("Loading historical returns data...")
    loader = ReturnsLoader(tickers, benchmark_map)
    returns_df = loader.load_data(lookback_years=lookback_years)
    logger.info(f"Loaded {len(returns_df)} months of returns data")

    # Compute returns metrics
    logger.info("Computing returns metrics...")
    primary_benchmark = 'AGG' if 'AGG' in returns_df.columns else tickers[0]
    returns_analyzer = ReturnsAnalyzer(returns_df, primary_benchmark, rf_rate)
    metrics_df = returns_analyzer.generate_full_report(tickers)

    # Compute risk metrics
    logger.info("Computing risk metrics...")
    risk_analyzer = RiskMetrics(returns_df)
    risk_df = risk_analyzer.generate_risk_report(tickers)

    # Scrape fund facts
    fund_facts_list = []
    if scrape_facts:
        logger.info("Scraping fund facts...")
        for ticker in tickers:
            issuer = fund_config.get(ticker, {}).get('issuer', 'morningstar')

            try:
                if issuer == 'blackrock':
                    scraper = BlackRockScraper()
                else:
                    scraper = MorningstarScraper()

                facts_data = scraper.scrape(ticker)

                if facts_data:
                    facts_data['ticker'] = ticker
                    facts_data['category'] = fund_config.get(ticker, {}).get('category', 'Unknown')
                    facts_data['benchmark'] = fund_config.get(ticker, {}).get('benchmark', 'AGG')

                    fund_facts = FundFacts.from_dict(facts_data)
                    fund_facts_list.append(fund_facts)
                    logger.info(f"  {ticker}: Scraped successfully")
                else:
                    logger.warning(f"  {ticker}: No data returned from scraper")
            except Exception as e:
                logger.warning(f"  {ticker}: Scraping failed - {e}")

    # Create portfolio
    portfolio = Portfolio(name="Bond Fund Analysis")
    for ff in fund_facts_list:
        portfolio.add_fund(ff)
    portfolio.returns_metrics = metrics_df
    portfolio.risk_metrics = risk_df

    # Run scenario analysis
    if fund_facts_list:
        logger.info("Running scenario analysis...")
        scenario_analyzer = ScenarioAnalyzer(fund_facts_list)
        portfolio.scenario_results = scenario_analyzer.generate_scenario_table()

    # Generate Excel report
    if output_file:
        logger.info(f"Generating Excel report: {output_file}")
        report_gen = ExcelReportGenerator()
        report_gen.generate_report(
            metrics_df=metrics_df,
            fund_facts_df=portfolio.get_fund_facts_df(),
            returns_df=returns_df[tickers],
            scenario_df=portfolio.scenario_results,
            output_path=output_file
        )
        logger.info(f"Report saved to {output_file}")

    return portfolio


def print_summary(portfolio: Portfolio):
    """Print analysis summary to console."""
    print("\n" + "=" * 60)
    print("BOND FUND ANALYSIS SUMMARY")
    print("=" * 60)

    if portfolio.returns_metrics is not None and not portfolio.returns_metrics.empty:
        print("\nPERFORMANCE METRICS")
        print("-" * 40)

        metrics = portfolio.returns_metrics
        summary_cols = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha']
        available = [c for c in summary_cols if c in metrics.columns]

        for ticker in metrics.index:
            print(f"\n{ticker}:")
            for col in available:
                val = metrics.loc[ticker, col]
                if 'ratio' not in col.lower():
                    print(f"  {col.replace('_', ' ').title()}: {val:.2%}")
                else:
                    print(f"  {col.replace('_', ' ').title()}: {val:.2f}")

    if portfolio.scenario_results is not None and not portfolio.scenario_results.empty:
        print("\n\nSCENARIO ANALYSIS (Rate +100bps)")
        print("-" * 40)

        scenarios = portfolio.scenario_results
        if 'rate_+100bps' in scenarios.columns:
            sorted_df = scenarios.sort_values('rate_+100bps')
            for ticker in sorted_df.index:
                impact = sorted_df.loc[ticker, 'rate_+100bps']
                if not pd.isna(impact):
                    print(f"  {ticker}: {impact:.2f}%")

    print("\n" + "=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fixed Income Fund Analytics CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default fund universe
  python main.py

  # Analyze specific funds
  python main.py -t AGG BND TLT

  # Generate Excel report
  python main.py -t AGG HYG LQD -o report.xlsx

  # Run with custom parameters
  python main.py -t AGG TLT --years 3 --rf-rate 4.5

  # Launch Streamlit dashboard
  python main.py --dashboard
        """
    )

    parser.add_argument(
        '-t', '--tickers',
        nargs='+',
        help='ETF tickers to analyze (default: all configured funds)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output Excel file path'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )

    parser.add_argument(
        '--rf-rate',
        type=float,
        default=5.0,
        help='Risk-free rate in percent (default: 5.0)'
    )

    parser.add_argument(
        '--no-scrape',
        action='store_true',
        help='Skip web scraping of fund facts'
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch Streamlit dashboard'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Launch dashboard if requested
    if args.dashboard:
        import subprocess
        subprocess.run(['streamlit', 'run', 'app.py'])
        return 0

    # Get tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        config = load_config()
        tickers = list(config.get('funds', {}).keys())
        if not tickers:
            tickers = ['AGG', 'BND', 'VCIT', 'LQD', 'HYG', 'TLT', 'SHY', 'TIP']

    # Set output file
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'bond_fund_report_{timestamp}.xlsx'

    # Run analysis
    try:
        portfolio = run_analysis(
            tickers=tickers,
            lookback_years=args.years,
            rf_rate=args.rf_rate / 100,
            output_file=output_file,
            scrape_facts=not args.no_scrape
        )

        # Print summary
        print_summary(portfolio)

        print(f"\nDone. Report saved to: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
