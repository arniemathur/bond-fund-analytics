"""
Fund data models for bond fund analytics.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class FundFacts:
    """
    Comprehensive fund facts data model.
    Stores static fund characteristics scraped from data sources.
    """

    # Identification
    ticker: str
    name: str = ""
    category: str = ""
    benchmark: str = ""
    issuer: str = ""

    # Size and cost
    aum: Optional[float] = None  # Assets under management in dollars
    expense_ratio: Optional[float] = None  # As decimal (0.01 = 1%)

    # Yield metrics
    yield_current: Optional[float] = None  # Current/distribution yield
    yield_to_maturity: Optional[float] = None  # YTM as decimal
    weighted_coupon: Optional[float] = None  # Weighted average coupon
    weighted_price: Optional[float] = None  # Weighted average price

    # Duration and maturity
    effective_duration: Optional[float] = None  # In years
    modified_duration: Optional[float] = None  # In years
    effective_maturity: Optional[float] = None  # In years

    # Credit quality
    avg_credit_rating: Optional[str] = None  # e.g., "AA", "BBB"

    # Sector allocation (as decimals)
    sector_weights: Dict[str, float] = field(default_factory=dict)
    # Example: {"government": 0.40, "corporate": 0.35, "securitized": 0.20, "cash": 0.05}

    # Credit quality breakdown (as decimals)
    quality_breakdown: Dict[str, float] = field(default_factory=dict)
    # Example: {"AAA": 0.30, "AA": 0.25, "A": 0.25, "BBB": 0.15, "BB": 0.05}

    # Metadata
    source: str = ""  # Data source (morningstar, blackrock)
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        """Normalize data after initialization."""
        self.ticker = self.ticker.upper()
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FundFacts':
        """Create FundFacts from a dictionary."""
        # Map common field variations
        field_mappings = {
            'yield': 'yield_current',
            'distribution_yield': 'yield_current',
            'sec_yield': 'yield_current',
            'ytm': 'yield_to_maturity',
            'duration': 'effective_duration',
            'maturity': 'effective_maturity',
            'credit_rating': 'avg_credit_rating',
            'credit_quality': 'avg_credit_rating',
        }

        # Create normalized dict
        normalized = {}
        for key, value in data.items():
            # Apply field mappings
            target_key = field_mappings.get(key.lower(), key)

            # Convert camelCase to snake_case
            import re
            target_key = re.sub(r'([a-z])([A-Z])', r'\1_\2', target_key).lower()

            normalized[target_key] = value

        # Filter to valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in normalized.items() if k in valid_fields}

        # Handle last_updated
        if 'last_updated' in filtered and isinstance(filtered['last_updated'], str):
            try:
                filtered['last_updated'] = datetime.fromisoformat(filtered['last_updated'])
            except ValueError:
                filtered['last_updated'] = None

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if data['last_updated']:
            data['last_updated'] = data['last_updated'].isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'FundFacts':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @property
    def has_duration(self) -> bool:
        """Check if duration data is available."""
        return self.effective_duration is not None

    @property
    def has_credit_data(self) -> bool:
        """Check if credit quality data is available."""
        return bool(self.quality_breakdown) or self.avg_credit_rating is not None

    @property
    def is_credit_fund(self) -> bool:
        """Determine if this is a credit-sensitive fund based on category."""
        credit_keywords = ['corporate', 'credit', 'high yield', 'investment grade']
        return any(kw in self.category.lower() for kw in credit_keywords)

    @property
    def is_government_fund(self) -> bool:
        """Determine if this is a government bond fund."""
        govt_keywords = ['treasury', 'government', 'tips', 'inflation']
        return any(kw in self.category.lower() for kw in govt_keywords)

    @property
    def investment_grade_pct(self) -> Optional[float]:
        """Calculate percentage invested in investment grade bonds."""
        if not self.quality_breakdown:
            return None

        ig_ratings = ['AAA', 'AA', 'A', 'BBB']
        total = sum(
            self.quality_breakdown.get(rating, 0)
            for rating in ig_ratings
        )
        return total

    @property
    def high_yield_pct(self) -> Optional[float]:
        """Calculate percentage invested in high yield bonds."""
        if not self.quality_breakdown:
            return None

        hy_ratings = ['BB', 'B', 'CCC', 'CC', 'C', 'D', 'Below B']
        total = sum(
            self.quality_breakdown.get(rating, 0)
            for rating in hy_ratings
        )
        return total

    def get_rate_sensitivity_score(self) -> Optional[float]:
        """
        Get a normalized rate sensitivity score (0-100).
        Higher = more sensitive to rate changes.
        """
        if self.effective_duration is None:
            return None

        # Normalize duration to 0-100 scale (assuming max duration of 25 years)
        max_duration = 25.0
        return min(100, (self.effective_duration / max_duration) * 100)

    def get_credit_sensitivity_score(self) -> Optional[float]:
        """
        Get a normalized credit sensitivity score (0-100).
        Higher = more sensitive to credit spread changes.
        """
        if not self.is_credit_fund:
            return 0.0

        if self.effective_duration is None:
            return None

        # For credit funds, use duration as proxy for spread sensitivity
        # Weight more heavily for lower-rated funds
        base_score = min(100, (self.effective_duration / 15.0) * 100)

        # Adjust for credit quality
        hy_pct = self.high_yield_pct
        if hy_pct is not None:
            # High yield increases spread sensitivity
            credit_multiplier = 1 + (hy_pct * 0.5)
            base_score = min(100, base_score * credit_multiplier)

        return base_score

    def merge(self, other: 'FundFacts') -> 'FundFacts':
        """
        Merge another FundFacts object, filling in missing values.
        Self takes precedence for non-None values.
        """
        merged_data = self.to_dict()

        for key, value in other.to_dict().items():
            if merged_data.get(key) is None and value is not None:
                merged_data[key] = value
            elif key in ['sector_weights', 'quality_breakdown']:
                # Merge dictionaries
                if isinstance(value, dict) and value:
                    if not merged_data.get(key):
                        merged_data[key] = value
                    else:
                        merged_data[key] = {**value, **merged_data[key]}

        return FundFacts.from_dict(merged_data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FundFacts(ticker='{self.ticker}', name='{self.name}', "
            f"duration={self.effective_duration}, expense_ratio={self.expense_ratio})"
        )
