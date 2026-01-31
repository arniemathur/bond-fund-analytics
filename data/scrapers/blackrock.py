"""
BlackRock iShares product page scraper.
Extracts detailed fund facts, holdings, duration, and credit data.
"""

import re
import json
from typing import Optional, Dict, Any, List
import logging
from .base import BaseScraper

logger = logging.getLogger(__name__)


# Known iShares fund IDs for common bond ETFs
ISHARES_FUND_IDS = {
    'AGG': '239458',
    'LQD': '239566',
    'HYG': '239565',
    'TLT': '239454',
    'SHY': '239452',
    'TIP': '239467',
    'IEF': '239456',
    'IEI': '239455',
    'IGSB': '239451',
    'IGIB': '239463',
    'GOVT': '239468',
    'MUB': '239766',
    'EMB': '239572',
    'IAGG': '279626',
    'USIG': '239460',
    'SHYG': '258100',
}


class BlackRockScraper(BaseScraper):
    """
    Scraper for BlackRock iShares product pages.
    URL pattern: https://www.ishares.com/us/products/{fund-id}/
    """

    BASE_URL = "https://www.ishares.com/us/products"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session.headers.update({
            'Referer': 'https://www.ishares.com/',
        })

    def _get_fund_id(self, ticker: str) -> Optional[str]:
        """Get the iShares fund ID for a ticker."""
        return ISHARES_FUND_IDS.get(ticker.upper())

    def _build_url(self, fund_id: str) -> str:
        """Build the iShares product URL."""
        return f"{self.BASE_URL}/{fund_id}/"

    def _build_api_url(self, fund_id: str, endpoint: str = 'key-facts') -> str:
        """Build API-style URL for fund data."""
        return f"{self.BASE_URL}/{fund_id}/{endpoint}"

    def scrape(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Scrape fund data from iShares.

        Args:
            ticker: The ETF ticker symbol

        Returns:
            Dictionary with scraped fund data
        """
        fund_id = self._get_fund_id(ticker)
        if not fund_id:
            logger.warning(f"Unknown iShares fund ID for {ticker}")
            return None

        url = self._build_url(fund_id)
        logger.info(f"Scraping iShares data for {ticker} (ID: {fund_id})")

        html = self.fetch_page(url)
        if html is None:
            logger.error(f"Failed to fetch iShares page for {ticker}")
            return None

        data = self._parse_product_page(html, ticker)
        data['fund_id'] = fund_id
        data['source'] = 'blackrock'

        return data

    def _parse_product_page(self, html: str, ticker: str) -> Dict[str, Any]:
        """Parse the iShares product page."""
        soup = self.parse_html(html)
        data = {'ticker': ticker.upper()}

        # Extract fund name
        name_elem = soup.find('h1', class_=re.compile('product', re.IGNORECASE))
        if name_elem:
            data['name'] = self.clean_text(name_elem.get_text())
        else:
            title = soup.find('title')
            if title:
                data['name'] = self.clean_text(title.get_text().split('|')[0])

        # Look for key facts section
        self._parse_key_facts(soup, data)

        # Look for fixed income characteristics
        self._parse_fi_characteristics(soup, data)

        # Look for sector/credit breakdown
        self._parse_allocations(soup, data)

        # Try to extract from embedded JSON/JS data
        self._extract_embedded_data(soup, data)

        return data

    def _parse_key_facts(self, soup, data: Dict[str, Any]):
        """Parse key facts section."""
        # Common patterns for iShares pages

        # Net Assets / AUM
        aum = self._find_labeled_value(soup, ['Net Assets', 'Total Net Assets', 'AUM'])
        if aum:
            data['aum'] = self.parse_number(aum)

        # Expense Ratio
        expense = self._find_labeled_value(soup, ['Expense Ratio', 'Net Expense Ratio', 'Management Fee'])
        if expense:
            data['expense_ratio'] = self.parse_percentage(expense)

        # Yield
        yield_val = self._find_labeled_value(soup, [
            'Distribution Yield', '30-Day SEC Yield', '12-Month Trailing Yield',
            'SEC Yield', 'Yield'
        ])
        if yield_val:
            data['yield'] = self.parse_percentage(yield_val)

        # Inception Date
        inception = self._find_labeled_value(soup, ['Inception Date', 'Fund Inception'])
        if inception:
            data['inception_date'] = self.clean_text(inception)

        # Benchmark
        benchmark = self._find_labeled_value(soup, ['Benchmark', 'Index', 'Underlying Index'])
        if benchmark:
            data['benchmark_name'] = self.clean_text(benchmark)

    def _parse_fi_characteristics(self, soup, data: Dict[str, Any]):
        """Parse fixed income characteristics section."""

        # Effective Duration
        duration = self._find_labeled_value(soup, [
            'Effective Duration', 'Modified Duration', 'Duration', 'Avg Duration'
        ])
        if duration:
            # Remove 'yrs' or 'years' suffix
            duration_clean = re.sub(r'\s*(yrs?|years?)\s*', '', duration, flags=re.IGNORECASE)
            data['effective_duration'] = self.parse_number(duration_clean)

        # Yield to Maturity
        ytm = self._find_labeled_value(soup, [
            'Yield to Maturity', 'YTM', 'Weighted Avg YTM'
        ])
        if ytm:
            data['yield_to_maturity'] = self.parse_percentage(ytm)

        # Weighted Average Maturity
        maturity = self._find_labeled_value(soup, [
            'Weighted Avg Maturity', 'Average Maturity', 'Effective Maturity'
        ])
        if maturity:
            # Remove 'yrs' suffix
            maturity_clean = re.sub(r'\s*(yrs?|years?)\s*', '', maturity, flags=re.IGNORECASE)
            data['effective_maturity'] = self.parse_number(maturity_clean)

        # Weighted Average Coupon
        coupon = self._find_labeled_value(soup, [
            'Weighted Avg Coupon', 'Average Coupon', 'Coupon'
        ])
        if coupon:
            data['weighted_coupon'] = self.parse_percentage(coupon)

        # Weighted Average Price
        price = self._find_labeled_value(soup, [
            'Weighted Avg Price', 'Average Price'
        ])
        if price:
            data['weighted_price'] = self.parse_number(price)

        # Credit Quality
        credit = self._find_labeled_value(soup, [
            'Average Credit Quality', 'Credit Quality', 'Credit Rating'
        ])
        if credit:
            data['avg_credit_rating'] = self.clean_text(credit)

    def _parse_allocations(self, soup, data: Dict[str, Any]):
        """Parse sector and credit allocations."""

        # Look for allocation tables
        tables = soup.find_all('table')

        for table in tables:
            header = table.find_previous(['h2', 'h3', 'h4', 'div'])
            if header:
                header_text = header.get_text().lower()

                if 'sector' in header_text or 'allocation' in header_text:
                    data['sector_weights'] = self._parse_table_to_dict(table)
                elif 'credit' in header_text or 'quality' in header_text:
                    data['quality_breakdown'] = self._parse_table_to_dict(table)

    def _parse_table_to_dict(self, table) -> Dict[str, float]:
        """Parse a two-column table into a dictionary."""
        result = {}
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                label = self.clean_text(cells[0].get_text())
                value = self.parse_percentage(cells[1].get_text())
                if label and value is not None:
                    result[label] = value

        return result

    def _find_labeled_value(self, soup, labels: List[str]) -> Optional[str]:
        """Find a value by searching for label patterns."""
        for label in labels:
            # Try various common HTML patterns

            # Pattern 1: dt/dd pairs
            for dt in soup.find_all('dt'):
                if label.lower() in dt.get_text().lower():
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        return dd.get_text(strip=True)

            # Pattern 2: th/td pairs
            for th in soup.find_all('th'):
                if label.lower() in th.get_text().lower():
                    td = th.find_next_sibling('td')
                    if td:
                        return td.get_text(strip=True)

            # Pattern 3: label/span pairs
            for elem in soup.find_all(string=re.compile(label, re.IGNORECASE)):
                parent = elem.parent
                if parent:
                    # Look for value in next sibling
                    sibling = parent.find_next_sibling()
                    if sibling:
                        text = sibling.get_text(strip=True)
                        if text and text not in ['-', 'N/A', '--']:
                            return text

            # Pattern 4: div with data attributes
            for div in soup.find_all('div', attrs={'data-label': re.compile(label, re.IGNORECASE)}):
                value = div.get('data-value') or div.get_text(strip=True)
                if value:
                    return value

        return None

    def _extract_embedded_data(self, soup, data: Dict[str, Any]):
        """Extract data from embedded JSON in script tags."""

        # Look for window.__INITIAL_STATE__ or similar patterns
        for script in soup.find_all('script'):
            if script.string:
                # Look for JSON objects in script content
                json_matches = re.findall(r'(\{[^{}]*"[^"]+"\s*:\s*[^{}]+\})', script.string)

                for match in json_matches:
                    try:
                        json_data = json.loads(match)
                        self._extract_from_json(json_data, data)
                    except json.JSONDecodeError:
                        continue

                # Look for specific data patterns
                patterns = [
                    (r'"effectiveDuration"\s*:\s*([\d.]+)', 'effective_duration'),
                    (r'"yieldToMaturity"\s*:\s*([\d.]+)', 'yield_to_maturity'),
                    (r'"expenseRatio"\s*:\s*([\d.]+)', 'expense_ratio'),
                    (r'"netAssets"\s*:\s*([\d.]+)', 'aum'),
                    (r'"secYield"\s*:\s*([\d.]+)', 'yield'),
                ]

                for pattern, field in patterns:
                    match = re.search(pattern, script.string)
                    if match and field not in data:
                        value = float(match.group(1))
                        # Normalize percentages
                        if field in ['yield_to_maturity', 'expense_ratio', 'yield']:
                            if value > 1:
                                value = value / 100
                        data[field] = value

    def _extract_from_json(self, json_data: Any, data: Dict[str, Any]):
        """Extract relevant fields from JSON data."""
        if not isinstance(json_data, dict):
            return

        field_mappings = {
            'effectiveDuration': 'effective_duration',
            'modifiedDuration': 'modified_duration',
            'yieldToMaturity': 'yield_to_maturity',
            'weightedAvgMaturity': 'effective_maturity',
            'weightedAvgCoupon': 'weighted_coupon',
            'weightedAvgPrice': 'weighted_price',
            'expenseRatio': 'expense_ratio',
            'netAssets': 'aum',
            'secYield': 'yield',
            'distributionYield': 'yield',
            'avgCreditQuality': 'avg_credit_rating',
        }

        for json_key, data_key in field_mappings.items():
            if json_key in json_data and data_key not in data:
                value = json_data[json_key]
                if isinstance(value, (int, float)):
                    # Normalize percentages
                    if data_key in ['yield_to_maturity', 'expense_ratio', 'yield', 'weighted_coupon']:
                        if value > 1:
                            value = value / 100
                    data[data_key] = value
                elif isinstance(value, str):
                    data[data_key] = self.clean_text(value)


def get_available_isshares_tickers() -> List[str]:
    """Return list of tickers with known iShares fund IDs."""
    return list(ISHARES_FUND_IDS.keys())
