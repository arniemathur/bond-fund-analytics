# Bond Fund Analytics

A tool for analyzing bond ETFs. Built this because I got tired of jumping between different websites to compare funds.

## What it does

- **Performance tracking**: See how funds have done over time, compare returns, check Sharpe ratios and other key stats
- **Risk analysis**: Drawdown charts, correlation matrices, downside capture ratios
- **Fund profiles**: Duration, yield, expense ratios, credit ratings all in one place
- **Scenario modeling**: What happens if rates go up 100bps? Run stress tests on your funds
- **Comparison tools**: Radar charts and side by side tables to compare multiple funds

## Funds included

| Ticker | Name | Category | Duration |
|--------|------|----------|----------|
| AGG | iShares Core U.S. Aggregate Bond | Intermediate Core Bond | 6.1 |
| BND | Vanguard Total Bond Market | Intermediate Core Bond | 6.0 |
| VCIT | Vanguard Intermediate Term Corporate | Corporate Bond | 6.2 |
| LQD | iShares iBoxx $ Investment Grade Corporate | Corporate Bond | 8.4 |
| HYG | iShares iBoxx $ High Yield Corporate | High Yield Bond | 3.8 |
| TLT | iShares 20+ Year Treasury Bond | Long Government | 16.8 |
| SHY | iShares 1 3 Year Treasury Bond | Short Government | 1.8 |
| TIP | iShares TIPS Bond | Inflation Protected | 6.7 |

## Running it locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Command line option

If you just want to generate a report without the dashboard:

```bash
python main.py -t AGG TLT HYG -o report.xlsx
```

## How it works

- Returns data comes from Yahoo Finance (using yfinance)
- Fund characteristics (duration, yield, etc.) are stored in config/funds.yaml
- Scenario analysis uses duration based approximations
- Everything is calculated with monthly returns

## Tech stack

- Python
- Streamlit for the dashboard
- Plotly for charts
- Pandas for data stuff
- yfinance for market data

## Notes

- All returns use adjusted close prices (includes dividends)
- Scenario analysis ignores convexity (first order approximation only)
- The default benchmark is AGG for most calculations
