# Fixed Income Bond Fund Analytics

A comprehensive Python analytics model for bond ETFs with web-scraped data, risk metrics, scenario analysis, and dual output (Streamlit dashboard + Excel reports).

## Features

- **Returns Analytics**: Alpha, beta, R², Sharpe ratio, Sortino ratio, max drawdown, capture ratios
- **Risk Metrics**: VaR, Expected Shortfall, volatility decomposition, correlation analysis
- **Scenario Analysis**: Interest rate shocks, credit spread shocks, historical stress tests
- **Web Scraping**: Automated fund facts from Morningstar and BlackRock iShares
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **Excel Reports**: Professional multi-sheet workbooks with charts and formatting

## Fund Universe

| Ticker | Name | Category |
|--------|------|----------|
| AGG | iShares Core U.S. Aggregate Bond | Intermediate Core Bond |
| BND | Vanguard Total Bond Market | Intermediate Core Bond |
| VCIT | Vanguard Intermediate-Term Corporate | Corporate Bond |
| LQD | iShares iBoxx $ Investment Grade Corporate | Corporate Bond |
| HYG | iShares iBoxx $ High Yield Corporate | High Yield Bond |
| TLT | iShares 20+ Year Treasury Bond | Long Government |
| SHY | iShares 1-3 Year Treasury Bond | Short Government |
| TIP | iShares TIPS Bond | Inflation-Protected |

## Installation

```bash
# Clone or navigate to the project directory
cd bond_fund_analytics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Run analysis on default fund universe
python main.py

# Analyze specific funds
python main.py -t AGG BND TLT HYG

# Generate Excel report with custom parameters
python main.py -t AGG LQD HYG --years 3 --rf-rate 4.5 -o my_report.xlsx

# Skip web scraping (faster, returns-only analysis)
python main.py -t AGG TLT --no-scrape
```

### Streamlit Dashboard

```bash
# Launch interactive dashboard
python main.py --dashboard

# Or directly with Streamlit
streamlit run app.py
```

## Project Structure

```
bond_fund_analytics/
├── config/
│   └── funds.yaml              # Fund universe and settings
├── data/
│   ├── scrapers/
│   │   ├── base.py             # Base scraper with retry/caching
│   │   ├── morningstar.py      # Morningstar scraper
│   │   └── blackrock.py        # iShares scraper
│   ├── loaders/
│   │   └── returns.py          # yfinance data loader
│   └── cache/                  # Scraped data cache
├── analytics/
│   ├── returns_engine.py       # Performance metrics
│   ├── risk_metrics.py         # Risk analytics
│   └── scenarios.py            # Scenario analysis
├── models/
│   ├── fund.py                 # FundFacts dataclass
│   └── portfolio.py            # Portfolio container
├── outputs/
│   ├── excel_report.py         # Excel generator
│   └── visualizations.py       # Plotly charts
├── app.py                      # Streamlit dashboard
├── main.py                     # CLI entry point
└── requirements.txt
```

## Usage Examples

### Python API

```python
from data.loaders.returns import ReturnsLoader
from analytics.returns_engine import ReturnsAnalyzer

# Load returns data
loader = ReturnsLoader(['AGG', 'BND', 'TLT'], {'AGG': 'AGG', 'BND': 'AGG', 'TLT': 'AGG'})
returns = loader.load_data(lookback_years=5)

# Compute metrics
analyzer = ReturnsAnalyzer(returns, benchmark_col='AGG', rf_rate=0.05)
metrics = analyzer.generate_full_report()
print(metrics)
```

### Scenario Analysis

```python
from analytics.scenarios import ScenarioAnalyzer
from models.fund import FundFacts

# Create fund facts (typically from scraping)
funds = [
    FundFacts(ticker='AGG', effective_duration=6.2, category='Intermediate Core Bond'),
    FundFacts(ticker='TLT', effective_duration=17.5, category='Long Government'),
    FundFacts(ticker='HYG', effective_duration=3.8, category='High Yield Bond'),
]

# Run scenarios
analyzer = ScenarioAnalyzer(funds)

# +100bps rate shock
rate_impact = analyzer.parallel_rate_shock(100)
print(rate_impact)

# Historical stress tests
historical = analyzer.run_historical_scenarios()
print(historical['2022 Rate Hikes'])
```

## Metrics Calculated

### Performance Metrics
- **Alpha**: Annualized excess return vs benchmark
- **Beta**: Sensitivity to benchmark returns
- **R²**: Explained variance from benchmark
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Information Ratio**: Active return per unit tracking error

### Risk Metrics
- **Volatility**: Annualized standard deviation
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%)**: Value at Risk at 95% confidence
- **Expected Shortfall**: Average loss beyond VaR
- **Upside/Downside Capture**: Performance in up/down markets

### Scenario Analysis
- **Rate Shocks**: Price impact from yield curve shifts
- **Spread Shocks**: Price impact from credit spread changes
- **Historical Scenarios**: 2013 Taper Tantrum, 2020 COVID, 2022 Rate Hikes

## Configuration

Edit `config/funds.yaml` to customize:

```yaml
funds:
  AGG:
    name: "iShares Core U.S. Aggregate Bond"
    category: "Intermediate Core Bond"
    benchmark: "AGG"
    issuer: "blackrock"

analysis:
  default_lookback_years: 5
  risk_free_rate: 0.05
  var_confidence: 0.95

scraping:
  rate_limit_seconds: 2.0
  cache_ttl_hours: 24
```

## Dashboard Features

The Streamlit dashboard provides:

1. **Performance Tab**: Cumulative returns chart, key metrics table
2. **Risk Analysis Tab**: Sharpe vs drawdown scatter, correlation heatmap, drawdown charts
3. **Fixed Income Profile Tab**: Yield vs duration scatter, fund characteristics
4. **Scenario Analysis Tab**: Interactive rate/spread shock sliders, impact charts
5. **Comparison Tab**: Multi-fund radar chart, side-by-side metrics
6. **Export**: Download Excel reports and CSV data

## Notes

- Web scraping may be blocked by some sites; cached data is used when available
- Duration-based scenario analysis is a first-order approximation (ignores convexity)
- Returns are calculated using adjusted close prices to capture distributions
- All percentage metrics are stored as decimals (0.05 = 5%)

## License

MIT License
