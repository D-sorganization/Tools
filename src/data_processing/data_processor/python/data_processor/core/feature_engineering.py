"""Feature Engineering Automation Module.

Provides automated feature engineering capabilities for time series
and tabular data, including feature generation, selection, and transformation.

This facade re-exports all classes from the decomposed submodules for
backward compatibility. New code should import directly from:
- feature_types: Enums, config, and result dataclasses
- feature_extractor: FeatureExtractor and extract_features
- feature_selector: FeatureSelector and select_features
- feature_transformer: FeatureTransformer
"""

# Re-export all public symbols for backward compatibility
from .feature_extractor import FeatureExtractor, extract_features  # noqa: F401
from .feature_selector import FeatureSelector, select_features  # noqa: F401
from .feature_transformer import FeatureTransformer  # noqa: F401
from .feature_types import (  # noqa: F401
    FeatureCategory,
    FeatureConfig,
    FeatureResult,
    SelectionMethod,
    SelectionResult,
    TransformationType,
)

__all__ = [
    "FeatureCategory",
    "SelectionMethod",
    "TransformationType",
    "FeatureConfig",
    "FeatureResult",
    "SelectionResult",
    "FeatureExtractor",
    "FeatureSelector",
    "FeatureTransformer",
    "extract_features",
    "select_features",
]
