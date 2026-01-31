# Web scrapers for fund data
from .base import BaseScraper
from .morningstar import MorningstarScraper
from .blackrock import BlackRockScraper

__all__ = ['BaseScraper', 'MorningstarScraper', 'BlackRockScraper']
