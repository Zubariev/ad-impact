"""
Unit tests for model training module.
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.model_training import (
    save_model_and_predictions,
    train_mlr,
    TRAIN_FUNCTIONS,
)


class TestModelTraining:
    """Test class for model training functions."""

    def test_train_mlr_basic(self):
        """Test basic MLR training."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "feature2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        })
        
        model, predictions, fig = train_mlr(df, "date", "target", ["feature1", "feature2"])
        
        assert model is not None
        assert predictions is not None
        assert fig is not None
        assert len(predictions) == len(df)
        assert "prediction" in predictions.columns

    def test_train_mlr_single_feature(self):
        """Test MLR training with single feature."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [1, 2, 3, 4, 5],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        model, predictions, fig = train_mlr(df, "date", "target", ["feature1"])
        
        assert model is not None
        assert predictions is not None
        assert fig is not None
        assert len(predictions) == len(df)

    def test_train_functions_dispatcher(self):
        """Test that all model training functions are available."""
        expected_models = [
            "MLR", "Distributed Lag", "ML + SHAP", "DiD", 
            "VAR", "Synthetic Control", "CausalImpact", "PSM"
        ]
        
        for model_name in expected_models:
            assert model_name in TRAIN_FUNCTIONS
            assert callable(TRAIN_FUNCTIONS[model_name])

    @patch("src.model_training.joblib.dump")
    @patch("src.model_training.pd.DataFrame.to_csv")
    def test_save_model_and_predictions(self, mock_to_csv, mock_dump):
        """Test model and predictions saving."""
        model = Mock()
        predictions = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        model_name = "TestModel"
        
        model_path, pred_path = save_model_and_predictions(model, predictions, model_name)
        
        assert model_path.endswith(f"{model_name}.pkl")
        assert pred_path.endswith(f"{model_name}_predictions.csv")
        mock_dump.assert_called_once()
        mock_to_csv.assert_called_once()

    def test_train_mlr_with_missing_data(self):
        """Test MLR training with missing data."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "target": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
            "feature1": [0.1, 0.2, 0.3, None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "feature2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        })
        
        # Should handle missing data gracefully
        with pytest.raises(Exception):
            train_mlr(df, "date", "target", ["feature1", "feature2"])

    def test_train_mlr_empty_dataframe(self):
        """Test MLR training with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(Exception):
            train_mlr(df, "date", "target", ["feature1", "feature2"])

    def test_train_mlr_missing_columns(self):
        """Test MLR training with missing columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "target": [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(Exception):
            train_mlr(df, "date", "target", ["missing_feature"]) 