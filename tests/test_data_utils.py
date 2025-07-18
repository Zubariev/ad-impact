"""
Unit tests for data utilities module.
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.data_utils import (
    detect_column_type,
    enforce_feature_limits,
    handle_missing_data,
    standardize_columns,
    validate_data_for_training,
)


class TestDataUtils:
    """Test class for data utilities."""

    def test_standardize_columns(self):
        """Test column standardization."""
        df = pd.DataFrame({
            "Column Name": [1, 2, 3],
            "Another Column": [4, 5, 6],
            "UPPER_CASE": [7, 8, 9]
        })
        
        result = standardize_columns(df)
        
        expected_columns = ["column_name", "another_column", "upper_case"]
        assert list(result.columns) == expected_columns

    def test_detect_column_type_datetime(self):
        """Test datetime column detection."""
        df = pd.DataFrame({
            "date_column": pd.date_range("2023-01-01", periods=5),
            "other_column": [1, 2, 3, 4, 5]
        })
        
        result = detect_column_type(df, "date_column")
        assert result == "datetime"

    def test_detect_column_type_numeric(self):
        """Test numeric column detection."""
        df = pd.DataFrame({
            "numeric_column": [1.1, 2.2, 3.3, 4.4, 5.5],
            "other_column": ["a", "b", "c", "d", "e"]
        })
        
        result = detect_column_type(df, "numeric_column")
        assert result == "numeric"

    def test_detect_column_type_categorical(self):
        """Test categorical column detection."""
        df = pd.DataFrame({
            "categorical_column": ["a", "b", "c", "d", "e"],
            "other_column": [1, 2, 3, 4, 5]
        })
        
        result = detect_column_type(df, "categorical_column")
        assert result == "categorical"

    def test_enforce_feature_limits_valid(self):
        """Test feature limit enforcement with valid input."""
        with patch("src.data_utils.MODEL_CONFIGS", {"TestModel": {"variables_allowed": (2, 10)}}):
            result = enforce_feature_limits("TestModel", ["feature1", "feature2", "feature3"])
            assert result is True

    def test_enforce_feature_limits_too_few(self):
        """Test feature limit enforcement with too few features."""
        with patch("src.data_utils.MODEL_CONFIGS", {"TestModel": {"variables_allowed": (5, 10)}}):
            result = enforce_feature_limits("TestModel", ["feature1", "feature2"])
            assert result is False

    def test_enforce_feature_limits_too_many(self):
        """Test feature limit enforcement with too many features."""
        with patch("src.data_utils.MODEL_CONFIGS", {"TestModel": {"variables_allowed": (1, 3)}}):
            result = enforce_feature_limits("TestModel", ["feature1", "feature2", "feature3", "feature4"])
            assert result is False

    def test_handle_missing_data_remove_rows(self):
        """Test missing data handling with row removal."""
        df = pd.DataFrame({
            "target": [1, 2, None, 4, 5],
            "feature1": [1, 2, 3, None, 5],
            "feature2": [1, 2, 3, 4, 5]
        })
        
        result = handle_missing_data(df, "Remove rows with missing data (Listwise deletion)", ["target", "feature1"])
        
        assert len(result) == 3  # Should remove 2 rows with missing data
        assert result["target"].isna().sum() == 0
        assert result["feature1"].isna().sum() == 0

    def test_handle_missing_data_fill_zeros(self):
        """Test missing data handling with zero filling."""
        df = pd.DataFrame({
            "target": [1, 2, None, 4, 5],
            "feature1": [1, 2, 3, None, 5],
            "feature2": [1, 2, 3, 4, 5]
        })
        
        result = handle_missing_data(df, "Fill with zeros", ["target", "feature1"])
        
        assert len(result) == 5  # Should keep all rows
        assert result["target"].isna().sum() == 0
        assert result["feature1"].isna().sum() == 0
        assert result.loc[2, "target"] == 0
        assert result.loc[3, "feature1"] == 0

    def test_validate_data_for_training_valid(self):
        """Test data validation with valid input."""
        df = pd.DataFrame({
            "target": [1, 2, 3, 4, 5],
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [1, 2, 3, 4, 5]
        })
        
        result = validate_data_for_training(df, "target", ["feature1", "feature2"])
        assert result is True

    def test_validate_data_for_training_insufficient_data(self):
        """Test data validation with insufficient data."""
        df = pd.DataFrame({
            "target": [1, 2, 3],
            "feature1": [1, 2, 3],
            "feature2": [1, 2, 3]
        })
        
        result = validate_data_for_training(df, "target", ["feature1", "feature2"])
        assert result is False

    def test_validate_data_for_training_missing_columns(self):
        """Test data validation with missing columns."""
        df = pd.DataFrame({
            "target": [1, 2, 3, 4, 5],
            "feature1": [1, 2, 3, 4, 5]
        })
        
        result = validate_data_for_training(df, "target", ["feature1", "missing_feature"])
        assert result is False 