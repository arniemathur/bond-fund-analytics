"""
Streamlit dashboard for Fixed Income Fund Analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging

from data.loaders.returns import ReturnsLoader
from analytics.returns_engine import ReturnsAnalyzer
from analytics.risk_metrics import RiskMetrics
from analytics.scenarios import ScenarioAnalyzer
from models.fund import FundFacts
from models.portfolio import Portfolio
from outputs.excel_report import ExcelReportGenerator
from outputs.visualizations import ChartGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Bond Fund Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_config():
    config_path = Path(__file__).parent / 'config' / 'funds.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {'funds': {}, 'analysis': {'risk_free_rate': 0.05, 'default_lookback_years': 5}}


@st.cache_data(ttl=3600)
def load_returns_data(tickers: tuple, benchmark_map: dict, lookback_years: int):
    loader = ReturnsLoader(list(tickers), benchmark_map)
    returns = loader.load_data(lookback_years=lookback_years)
    return returns, loader.prices


def show_landing_page():
    """Display the landing/home page with app overview."""

    st.title("Bond Fund Analytics")

    st.markdown("""
    ### What is this?

    This is a tool I built to analyze bond ETFs. If you're trying to figure out which bond funds
    to invest in, or you just want to understand how different funds perform and react to market
    changes, this should help.

    I got tired of jumping between different websites to compare funds, so I made something that
    pulls everything together in one place.
    """)

    st.warning("""
    **Disclaimer:** This is just a tool for looking at data. It's not investment advice and I'm not
    a financial advisor. Do your own research before making any investment decisions.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### What you can do here

        **Performance Analysis**
        - See how funds have performed over time
        - Compare returns across multiple ETFs
        - Check out key stats like Sharpe ratio, volatility, max drawdown
        - View cumulative growth charts

        **Risk Metrics**
        - Sharpe ratio vs drawdown scatter plots (find the efficient funds)
        - Downside capture analysis (how much pain do you feel in down markets?)
        - Correlation matrix (see which funds move together)
        - Drawdown charts over time

        **Fund Profiles**
        - Duration and yield data for each fund
        - Expense ratios and credit ratings
        - Category breakdowns
        - Duration bucket classification
        """)

    with col2:
        st.markdown("""
        ### More features

        **Scenario Analysis**
        - What happens if rates go up 100bps? 200bps?
        - Credit spread shock modeling
        - Historical scenario simulations (2020 COVID crash, 2022 rate hikes, etc.)
        - See which funds get hit hardest

        **Fund Comparison**
        - Radar charts to compare multiple funds at once
        - Side by side metrics tables
        - Pick up to 4 funds and see how they stack up

        **Export Options**
        - Download full Excel reports with all the data
        - CSV export for the metrics
        - Use the data however you want
        """)

    st.divider()

    st.markdown("""
    ### Funds Currently Supported
    """)

    config = load_config()
    fund_universe = config.get('funds', {})

    fund_data = []
    for ticker, info in fund_universe.items():
        fund_data.append({
            'Ticker': ticker,
            'Name': info.get('name', ''),
            'Category': info.get('category', ''),
            'Duration': info.get('duration', ''),
            'Yield': f"{info.get('yield', 0)*100:.1f}%" if info.get('yield') else ''
        })

    if fund_data:
        st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("""
    ### How to use this

    1. **Select your funds** in the sidebar on the left
    2. **Pick your timeframe** (how many years of data to analyze)
    3. **Set the risk free rate** (defaults to 5%, but change it if you want)
    4. **Click through the tabs** to explore different analyses
    5. **Export your data** when you're done

    The data comes from Yahoo Finance for returns and I've got the fund characteristics
    (duration, yield, etc.) stored locally. Everything updates when you change your selections.
    """)

    st.divider()

    st.markdown("""
    ### Quick notes

    - All returns are calculated using adjusted close prices (accounts for dividends)
    - Monthly returns are used for most calculations
    - Scenario analysis uses duration based approximations (first order, ignores convexity)
    - The benchmark for most calculations is AGG unless specified otherwise

    **Ready to get started?** Select some funds in the sidebar and switch to the Performance tab.
    """)


def main():
    config = load_config()
    fund_universe = config.get('funds', {})
    analysis_config = config.get('analysis', {})

    with st.sidebar:
        st.header("Settings")

        available_funds = list(fund_universe.keys())
        if not available_funds:
            available_funds = ['AGG', 'BND', 'VCIT', 'LQD', 'HYG', 'TLT', 'SHY', 'TIP']

        selected_funds = st.multiselect(
            "Funds",
            options=available_funds,
            default=available_funds[:4] if len(available_funds) >= 4 else available_funds
        )

        lookback_years = st.slider(
            "Lookback (Years)",
            min_value=1,
            max_value=10,
            value=analysis_config.get('default_lookback_years', 5)
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        st.caption(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        rf_rate = st.number_input(
            "Risk Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=analysis_config.get('risk_free_rate', 0.05) * 100,
            step=0.25
        ) / 100

        st.divider()

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Create tabs
    tab_home, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Home",
        "Performance",
        "Risk",
        "Fund Profile",
        "Scenarios",
        "Comparison"
    ])

    with tab_home:
        show_landing_page()

    # Check if funds are selected for other tabs
    if not selected_funds:
        with tab1:
            st.warning("Select at least one fund from the sidebar to see performance data.")
        with tab2:
            st.warning("Select at least one fund from the sidebar to see risk analysis.")
        with tab3:
            st.warning("Select at least one fund from the sidebar to see fund profiles.")
        with tab4:
            st.warning("Select at least one fund from the sidebar to run scenarios.")
        with tab5:
            st.warning("Select at least one fund from the sidebar to compare funds.")
        return

    # Build benchmark mapping
    benchmark_map = {}
    for ticker in selected_funds:
        if ticker in fund_universe:
            benchmark_map[ticker] = fund_universe[ticker].get('benchmark', 'AGG')
        else:
            benchmark_map[ticker] = 'AGG'

    # Load returns data
    try:
        returns_df, prices_df = load_returns_data(
            tuple(selected_funds),
            benchmark_map,
            lookback_years
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Initialize analyzers
    primary_benchmark = 'AGG' if 'AGG' in returns_df.columns else selected_funds[0]
    returns_analyzer = ReturnsAnalyzer(returns_df, primary_benchmark, rf_rate)
    risk_analyzer = RiskMetrics(returns_df)
    chart_gen = ChartGenerator()

    # Compute metrics
    metrics_df = returns_analyzer.generate_full_report(selected_funds)
    risk_df = risk_analyzer.generate_risk_report(selected_funds)

    # Build fund facts from config
    fund_facts_list = []
    for ticker in selected_funds:
        fund_config = fund_universe.get(ticker, {})
        fund_facts = FundFacts(
            ticker=ticker,
            name=fund_config.get('name', ticker),
            category=fund_config.get('category', 'Unknown'),
            benchmark=fund_config.get('benchmark', 'AGG'),
            issuer=fund_config.get('issuer', ''),
            effective_duration=fund_config.get('duration'),
            yield_current=fund_config.get('yield'),
            expense_ratio=fund_config.get('expense_ratio'),
            avg_credit_rating=fund_config.get('credit_rating')
        )
        fund_facts_list.append(fund_facts)

    portfolio = Portfolio(name="Analysis Portfolio")
    for ff in fund_facts_list:
        portfolio.add_fund(ff)

    # Tab 1: Performance
    with tab1:
        st.header("Performance")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = chart_gen.cumulative_returns_chart(
                returns_df[selected_funds],
                "Cumulative Returns"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Summary")
            summary_cols = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
            available_cols = [c for c in summary_cols if c in metrics_df.columns]
            if available_cols:
                display_df = metrics_df[available_cols].copy()
                display_df.columns = ['Return', 'Vol', 'Sharpe', 'Max DD']
                st.dataframe(
                    display_df.style.format({
                        'Return': '{:.2%}',
                        'Vol': '{:.2%}',
                        'Sharpe': '{:.2f}',
                        'Max DD': '{:.2%}'
                    }),
                    use_container_width=True
                )

        st.subheader("All Metrics")
        st.dataframe(
            metrics_df.style.format({col: '{:.2%}' for col in metrics_df.columns
                                     if 'return' in col.lower() or 'capture' in col.lower()
                                     or col in ['volatility', 'alpha', 'max_drawdown', 'var_95', 'expected_shortfall_95']}),
            use_container_width=True
        )

    # Tab 2: Risk
    with tab2:
        st.header("Risk")

        col1, col2 = st.columns(2)

        with col1:
            fig = chart_gen.sharpe_vs_drawdown_scatter(metrics_df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = chart_gen.downside_capture_scatter(metrics_df)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlations")
        fig = chart_gen.correlation_heatmap(returns_df[selected_funds])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdowns")
        selected_for_dd = st.selectbox("Fund", selected_funds, key="dd_select")
        fig = chart_gen.drawdown_chart(returns_df, selected_for_dd)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Fund Profile
    with tab3:
        st.header("Fund Profile")

        fund_facts_df = portfolio.get_fund_facts_df()

        if not fund_facts_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                if 'duration' in fund_facts_df.columns and 'yield' in fund_facts_df.columns:
                    fig = chart_gen.yield_vs_duration_scatter(fund_facts_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Duration/yield data not available")

            with col2:
                st.subheader("Characteristics")
                display_cols = ['name', 'category', 'expense_ratio', 'duration', 'yield', 'credit_rating']
                available_display = [c for c in display_cols if c in fund_facts_df.columns]
                if available_display:
                    st.dataframe(fund_facts_df[available_display], use_container_width=True)

            st.subheader("Duration Buckets")
            buckets = portfolio.get_duration_buckets()
            if buckets:
                for bucket, tickers in buckets.items():
                    st.write(f"**{bucket}**: {', '.join(tickers)}")
            else:
                st.info("Duration data not available")
        else:
            st.warning("Fund data not available")

    # Tab 4: Scenarios
    with tab4:
        st.header("Scenarios")

        st.markdown("""
        This shows estimated price impact based on duration. When rates go up, bond prices go down
        (and vice versa). The longer the duration, the bigger the move.
        """)

        if fund_facts_list:
            scenario_analyzer = ScenarioAnalyzer(fund_facts_list)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Rate Shock")
                rate_shock = st.slider(
                    "Rate Change (bps)",
                    min_value=-200,
                    max_value=200,
                    value=100,
                    step=25
                )

            with col2:
                st.subheader("Spread Shock")
                spread_shock = st.slider(
                    "Spread Change (bps)",
                    min_value=-100,
                    max_value=200,
                    value=50,
                    step=25
                )

            scenario_df = scenario_analyzer.generate_scenario_table(
                rate_shocks=[rate_shock, -rate_shock],
                spread_shocks=[spread_shock]
            )

            if not scenario_df.empty:
                impact_col = f'rate_{rate_shock:+d}bps'
                if impact_col in scenario_df.columns:
                    fig = chart_gen.scenario_impact_bar(
                        scenario_df,
                        impact_col,
                        f"Rate Shock: {rate_shock:+d}bps"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Impact Table")
                display_scenario = scenario_df.drop(columns=['name', 'is_credit_fund'], errors='ignore')
                format_cols = [col for col in display_scenario.columns if 'rate_' in col or 'spread_' in col]
                formatter = {col: lambda x: f'{x:.2f}%' if pd.notna(x) else '' for col in format_cols}
                st.dataframe(
                    display_scenario.style.format(formatter, na_rep=''),
                    use_container_width=True
                )

                st.subheader("Historical Scenarios")
                st.markdown("See how funds would have reacted to past market events:")

                historical = scenario_analyzer.run_historical_scenarios()
                selected_scenario = st.selectbox(
                    "Scenario",
                    options=list(historical.keys()),
                    key="hist_scenario"
                )

                if selected_scenario in historical:
                    hist_df = historical[selected_scenario]
                    if not hist_df.empty and 'total_impact_pct' in hist_df.columns:
                        fig = chart_gen.scenario_impact_bar(
                            hist_df,
                            'total_impact_pct',
                            selected_scenario
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Fund data required for scenario analysis")

    # Tab 5: Comparison
    with tab5:
        st.header("Comparison")

        st.markdown("Pick funds to compare side by side. The radar chart normalizes everything to a 0 to 100 scale so you can see relative strengths.")

        compare_funds = st.multiselect(
            "Select funds (max 4)",
            options=selected_funds,
            default=selected_funds[:min(2, len(selected_funds))],
            max_selections=4,
            key="compare_select"
        )

        if len(compare_funds) >= 2:
            fig = chart_gen.metrics_comparison_radar(
                metrics_df,
                compare_funds,
                metrics=['sharpe_ratio', 'alpha', 'upside_capture', 'information_ratio'],
                title="Metrics Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Side by Side")
            comparison_df = metrics_df.loc[compare_funds].T
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Select at least 2 funds to compare")

    # Export section
    st.divider()
    st.header("Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Excel Report"):
            with st.spinner("Generating..."):
                try:
                    report_gen = ExcelReportGenerator()
                    fund_facts_df = portfolio.get_fund_facts_df()
                    scenario_df_export = scenario_df if 'scenario_df' in dir() else pd.DataFrame()

                    output_path = 'bond_fund_report.xlsx'
                    report_gen.generate_report(
                        metrics_df=metrics_df,
                        fund_facts_df=fund_facts_df,
                        returns_df=returns_df[selected_funds],
                        scenario_df=scenario_df_export,
                        output_path=output_path
                    )

                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download Excel",
                            data=f,
                            file_name=f"bond_fund_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        csv = metrics_df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"fund_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
