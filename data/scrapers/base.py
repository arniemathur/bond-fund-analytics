"""
Base scraper class with retry logic, rate limiting, and caching.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.
    Provides rate limiting, retries, caching, and common utilities.
    """

    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    def __init__(
        self,
        cache_dir: str = 'data/cache',
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize the base scraper.

        Args:
            cache_dir: Directory for caching scraped data
            rate_limit_seconds: Minimum seconds between requests
            max_retries: Maximum number of retry attempts
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.cache_ttl_hours = cache_ttl_hours
        self._last_request_time: Optional[float] = None
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_seconds:
                sleep_time = self.rate_limit_seconds - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False

        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = modified_time + timedelta(hours=self.cache_ttl_hours)
        return datetime.now() < expiry_time

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Read data from cache if valid."""
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Cache hit for {cache_key}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache: {e}")
                return None

        return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any]):
        """Write data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Cached data for {cache_key}")
        except IOError as e:
            logger.warning(f"Error writing cache: {e}")

    def fetch_page(self, url: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch a web page with retries and rate limiting.

        Args:
            url: The URL to fetch
            use_cache: Whether to use caching

        Returns:
            The page HTML content or None if failed
        """
        cache_key = self._get_cache_key(url)

        # Check cache first
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached and 'html' in cached:
                return cached['html']

        # Fetch with retries
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                html = response.text

                # Cache the result
                if use_cache:
                    self._write_cache(cache_key, {
                        'url': url,
                        'html': html,
                        'fetched_at': datetime.now().isoformat()
                    })

                return html

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limit_seconds
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None

        return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content into BeautifulSoup object."""
        return BeautifulSoup(html, 'html.parser')

    def clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text content."""
        if text is None:
            return None
        return ' '.join(text.strip().split())

    def parse_number(self, text: Optional[str]) -> Optional[float]:
        """Parse a number from text, handling common formats."""
        if text is None:
            return None

        # Clean the text
        text = self.clean_text(text)
        if not text:
            return None

        # Remove common suffixes and prefixes
        text = text.replace('$', '').replace('%', '').replace(',', '')
        text = text.replace('B', 'e9').replace('M', 'e6').replace('K', 'e3')

        try:
            return float(text)
        except ValueError:
            return None

    def parse_percentage(self, text: Optional[str]) -> Optional[float]:
        """Parse a percentage value (returns as decimal)."""
        value = self.parse_number(text)
        if value is not None:
            # If text contained %, value is already divided by removing %
            # But we need to convert to decimal
            if '%' in (text or ''):
                return value / 100
            return value
        return None

    @abstractmethod
    def scrape(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Scrape fund data for a given ticker.

        Args:
            ticker: The fund ticker symbol

        Returns:
            Dictionary of scraped data or None if failed
        """
        pass

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.

        Args:
            ticker: If provided, clear only this ticker's cache.
                   If None, clear all cache.
        """
        if ticker:
            # Clear specific ticker cache files
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        if ticker.upper() in data.get('url', '').upper():
                            cache_file.unlink()
                            logger.info(f"Cleared cache for {ticker}")
                except Exception:
                    continue
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()
            logger.info("Cleared all cache")
