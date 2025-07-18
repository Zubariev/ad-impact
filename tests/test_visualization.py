"""
Unit tests for visualization module.
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.visualization import (
    create_interpretation_hints,
    create_vif_table,
    display_descriptive_stats,
    MODEL_METRIC_DISPATCH,
)


class TestVisualization:
    """Test class for visualization functions."""

    def test_create_vif_table(self):
        """Test VIF table creation."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
            "feature3": [1, 3, 2, 4, 3]   # Less correlated
        })
        
        vif_df = create_vif_table(df, ["feature1", "feature2", "feature3"])
        
        assert "Variable" in vif_df.columns
        assert "VIF" in vif_df.columns
        assert len(vif_df) == 3
        assert all(vif_df["VIF"] > 0)

    def test_create_interpretation_hints(self):
        """Test interpretation hints creation."""
        hints = create_interpretation_hints("MLR")
        
        assert isinstance(hints, list)
        assert len(hints) > 0
        assert all(isinstance(hint, str) for hint in hints)

    def test_create_interpretation_hints_unknown_model(self):
        """Test interpretation hints for unknown model."""
        hints = create_interpretation_hints("UnknownModel")
        
        assert isinstance(hints, list)
        assert len(hints) == 1
        assert "No specific hints available" in hints[0]

    def test_model_metric_dispatch_coverage(self):
        """Test that all expected models are in the dispatch."""
        expected_models = [
            "MLR", "Distributed Lag", "ML + SHAP", "DiD", 
            "VAR", "Synthetic Control", "CausalImpact", "PSM"
        ]
        
        for model_name in expected_models:
            assert model_name in MODEL_METRIC_DISPATCH
            assert callable(MODEL_METRIC_DISPATCH[model_name])

    @patch("streamlit.expander")
    def test_display_descriptive_stats(self, mock_expander):
        """Test descriptive statistics display."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [1, 2, 3, 4, 5],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature2": [1.1, 1.2, 1.3, 1.4, 1.5]
        })
        
        # Should not raise any exceptions
        display_descriptive_stats(df, "date", "target", ["feature1", "feature2"])
        
        # Verify expander was called
        mock_expander.assert_called_once()

    def test_display_descriptive_stats_with_missing_data(self):
        """Test descriptive statistics with missing data."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [1, 2, None, 4, 5],
            "feature1": [0.1, 0.2, 0.3, None, 0.5],
            "feature2": [1.1, 1.2, 1.3, 1.4, 1.5]
        })
        
        # Should handle missing data gracefully
        with patch("streamlit.expander"):
            display_descriptive_stats(df, "date", "target", ["feature1", "feature2"])

    def test_display_descriptive_stats_empty_features(self):
        """Test descriptive statistics with empty feature list."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [1, 2, 3, 4, 5]
        })
        
        # Should handle empty feature list gracefully
        with patch("streamlit.expander"):
            display_descriptive_stats(df, "date", "target", [])

    def test_interpretation_hints_all_models(self):
        """Test that all models have interpretation hints."""
        expected_models = [
            "MLR", "Distributed Lag", "ML + SHAP", "DiD", 
            "VAR", "Synthetic Control", "CausalImpact", "PSM"
        ]
        
        for model_name in expected_models:
            hints = create_interpretation_hints(model_name)
            assert isinstance(hints, list)
            assert len(hints) > 0
            assert all(isinstance(hint, str) for hint in hints)
            assert all(len(hint) > 0 for hint in hints) 