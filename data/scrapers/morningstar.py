"""
Morningstar ETF page scraper.
Extracts fund facts, duration, yield, credit quality, and sector weights.
"""

import re
from typing import Optional, Dict, Any, List
import logging
from .base import BaseScraper

logger = logging.getLogger(__name__)


class MorningstarScraper(BaseScraper):
    """
    Scraper for Morningstar ETF pages.
    URL pattern: https://www.morningstar.com/etfs/{exchange}/{ticker}/quote
    """

    BASE_URL = "https://www.morningstar.com/etfs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Morningstar-specific headers
        self.session.headers.update({
            'Referer': 'https://www.morningstar.com/',
        })

    def _build_url(self, ticker: str, exchange: str = 'arcx') -> str:
        """Build the Morningstar ETF URL."""
        return f"{self.BASE_URL}/{exchange}/{ticker.lower()}/quote"

    def _build_portfolio_url(self, ticker: str, exchange: str = 'arcx') -> str:
        """Build the Morningstar ETF portfolio URL."""
        return f"{self.BASE_URL}/{exchange}/{ticker.lower()}/portfolio"

    def scrape(self, ticker: str, exchange: str = 'arcx') -> Optional[Dict[str, Any]]:
        """
        Scrape fund data from Morningstar.

        Args:
            ticker: The ETF ticker symbol
            exchange: The exchange code (default: arcx for NYSE Arca)

        Returns:
            Dictionary with scraped fund data
        """
        quote_url = self._build_url(ticker, exchange)
        portfolio_url = self._build_portfolio_url(ticker, exchange)

        logger.info(f"Scraping Morningstar data for {ticker}")

        # Fetch quote page
        quote_html = self.fetch_page(quote_url)
        if quote_html is None:
            logger.error(f"Failed to fetch Morningstar quote page for {ticker}")
            return None

        # Parse quote page
        data = self._parse_quote_page(quote_html, ticker)

        # Fetch and parse portfolio page for additional details
        portfolio_html = self.fetch_page(portfolio_url)
        if portfolio_html:
            portfolio_data = self._parse_portfolio_page(portfolio_html)
            data.update(portfolio_data)

        return data

    def _parse_quote_page(self, html: str, ticker: str) -> Dict[str, Any]:
        """Parse the main quote page for basic fund facts."""
        soup = self.parse_html(html)
        data = {'ticker': ticker.upper(), 'source': 'morningstar'}

        # Try to extract fund name from title or header
        title_elem = soup.find('title')
        if title_elem:
            title_text = title_elem.get_text()
            # Extract name before the pipe or ticker
            name_match = re.search(r'^([^|]+)', title_text)
            if name_match:
                data['name'] = self.clean_text(name_match.group(1))

        # Look for key-value pairs in common Morningstar patterns
        # These selectors may need adjustment based on current page structure

        # Try to find expense ratio
        expense_ratio = self._find_metric(soup, ['expense ratio', 'net expense ratio'])
        if expense_ratio:
            data['expense_ratio'] = self.parse_percentage(expense_ratio)

        # Try to find yield
        yield_value = self._find_metric(soup, ['yield', 'ttm yield', 'sec yield', '30-day sec yield'])
        if yield_value:
            data['yield'] = self.parse_percentage(yield_value)

        # Try to find AUM/Net Assets
        aum = self._find_metric(soup, ['net assets', 'total assets', 'aum', 'fund size'])
        if aum:
            data['aum'] = self.parse_number(aum)

        # Try to find category
        category = self._find_metric(soup, ['category', 'morningstar category'])
        if category:
            data['category'] = self.clean_text(category)

        # Extract from data attributes or JSON in page
        self._extract_embedded_data(soup, data)

        return data

    def _parse_portfolio_page(self, html: str) -> Dict[str, Any]:
        """Parse the portfolio page for duration, credit quality, etc."""
        soup = self.parse_html(html)
        data = {}

        # Duration metrics
        duration = self._find_metric(soup, ['effective duration', 'modified duration', 'duration'])
        if duration:
            data['effective_duration'] = self.parse_number(duration)

        mod_duration = self._find_metric(soup, ['modified duration'])
        if mod_duration:
            data['modified_duration'] = self.parse_number(mod_duration)

        # Maturity
        maturity = self._find_metric(soup, ['effective maturity', 'average maturity', 'weighted avg maturity'])
        if maturity:
            data['effective_maturity'] = self.parse_number(maturity)

        # Credit quality
        credit = self._find_metric(soup, ['average credit quality', 'credit quality', 'avg credit rating'])
        if credit:
            data['avg_credit_rating'] = self.clean_text(credit)

        # Coupon
        coupon = self._find_metric(soup, ['weighted avg coupon', 'average coupon', 'coupon'])
        if coupon:
            data['weighted_coupon'] = self.parse_percentage(coupon)

        # YTM
        ytm = self._find_metric(soup, ['yield to maturity', 'ytm'])
        if ytm:
            data['yield_to_maturity'] = self.parse_percentage(ytm)

        # Sector weights
        sector_weights = self._parse_sector_weights(soup)
        if sector_weights:
            data['sector_weights'] = sector_weights

        # Credit quality breakdown
        quality_breakdown = self._parse_credit_breakdown(soup)
        if quality_breakdown:
            data['quality_breakdown'] = quality_breakdown

        return data

    def _find_metric(self, soup, labels: List[str]) -> Optional[str]:
        """
        Find a metric value by searching for label patterns.

        Args:
            soup: BeautifulSoup object
            labels: List of possible label strings to search for

        Returns:
            The metric value text or None
        """
        for label in labels:
            # Try finding by text content
            label_lower = label.lower()

            # Search for elements containing the label
            for elem in soup.find_all(string=re.compile(label, re.IGNORECASE)):
                parent = elem.parent
                if parent:
                    # Look for value in sibling or parent's sibling
                    next_sibling = parent.find_next_sibling()
                    if next_sibling:
                        value = next_sibling.get_text(strip=True)
                        if value and value != '-':
                            return value

                    # Try parent's next sibling
                    parent_sibling = parent.parent.find_next_sibling() if parent.parent else None
                    if parent_sibling:
                        value = parent_sibling.get_text(strip=True)
                        if value and value != '-':
                            return value

            # Try finding by class or id patterns
            for elem in soup.find_all(class_=re.compile(label_lower.replace(' ', ''), re.IGNORECASE)):
                value = elem.get_text(strip=True)
                if value and value != '-':
                    return value

        return None

    def _extract_embedded_data(self, soup, data: Dict[str, Any]):
        """Extract data from embedded JSON or data attributes."""
        # Look for script tags with JSON data
        for script in soup.find_all('script', type='application/json'):
            try:
                import json
                json_data = json.loads(script.string)
                self._extract_from_json(json_data, data)
            except (json.JSONDecodeError, TypeError):
                continue

        # Look for data attributes
        for elem in soup.find_all(attrs={'data-value': True}):
            data_label = elem.get('data-label', '').lower()
            data_value = elem.get('data-value', '')

            if 'expense' in data_label:
                data['expense_ratio'] = self.parse_percentage(data_value)
            elif 'yield' in data_label:
                data['yield'] = self.parse_percentage(data_value)
            elif 'duration' in data_label:
                data['effective_duration'] = self.parse_number(data_value)

    def _extract_from_json(self, json_data: Any, data: Dict[str, Any]):
        """Recursively extract relevant data from JSON structure."""
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                key_lower = key.lower()
                if 'expenseratio' in key_lower or 'expense_ratio' in key_lower:
                    if isinstance(value, (int, float)):
                        data['expense_ratio'] = value / 100 if value > 1 else value
                elif 'yield' in key_lower and 'to' not in key_lower:
                    if isinstance(value, (int, float)):
                        data['yield'] = value / 100 if value > 1 else value
                elif 'duration' in key_lower:
                    if isinstance(value, (int, float)):
                        data['effective_duration'] = value
                elif 'netassets' in key_lower or 'aum' in key_lower:
                    if isinstance(value, (int, float)):
                        data['aum'] = value
                elif isinstance(value, (dict, list)):
                    self._extract_from_json(value, data)
        elif isinstance(json_data, list):
            for item in json_data:
                self._extract_from_json(item, data)

    def _parse_sector_weights(self, soup) -> Optional[Dict[str, float]]:
        """Parse sector allocation from portfolio page."""
        sectors = {}
        sector_labels = ['government', 'corporate', 'securitized', 'municipal', 'cash', 'other']

        for label in sector_labels:
            value = self._find_metric(soup, [label])
            if value:
                parsed = self.parse_percentage(value)
                if parsed is not None:
                    sectors[label] = parsed

        return sectors if sectors else None

    def _parse_credit_breakdown(self, soup) -> Optional[Dict[str, float]]:
        """Parse credit quality breakdown from portfolio page."""
        quality = {}
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Below B', 'Not Rated']

        for rating in ratings:
            value = self._find_metric(soup, [rating])
            if value:
                parsed = self.parse_percentage(value)
                if parsed is not None:
                    quality[rating] = parsed

        return quality if quality else None
