"""
Data utilities for the Ad Impact Modeling Dashboard.
Handles file ingestion, data cleaning, and preprocessing operations.
"""

import io
import logging
from typing import List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from config import (
    DATE_KEYWORDS,
    MISSING_DATA_METHODS,
    MIN_OBSERVATIONS_FOR_TRAINING,
    OUTLIER_THRESHOLD,
    SUPPORTED_FILE_TYPES,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lower-cases and strips column names for consistency.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_data
def load_uploaded_files(files: List[io.BytesIO]) -> pd.DataFrame:
    """
    Load multiple uploaded files (csv/xls/xlsx) into a single DataFrame.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        Unified DataFrame with standardized column names
        
    Raises:
        ValueError: If no valid files are provided
    """
    frames = []
    for f in files:
        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(f)
            elif name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(f)
            else:
                st.warning(f"Unsupported file type for {f.name}; skipping.")
                continue
            frames.append(standardize_columns(df))
            logger.info(f"Successfully loaded {f.name} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {f.name}: {str(e)}")
            st.warning(f"Error loading {f.name}: {str(e)}")
            continue
    
    if not frames:
        error_msg = "No valid data found in uploaded files."
        logger.error(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
    
    unified = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined {len(frames)} files into {len(unified)} total rows")
    return unified


def enforce_feature_limits(model_name: str, selected_features: List[str]) -> bool:
    """
    Check if selected features meet model requirements.
    
    Args:
        model_name: Name of the model
        selected_features: List of selected feature names
        
    Returns:
        True if limits are satisfied, False otherwise
    """
    from config import MODEL_CONFIGS
    
    min_vars, max_vars = MODEL_CONFIGS[model_name]["variables_allowed"]
    n = len(selected_features)
    
    if n < min_vars or n > max_vars:
        error_msg = (
            f"{model_name} requires between {min_vars} and {max_vars} features. "
            f"Currently selected: {n}."
        )
        logger.warning(error_msg)
        st.error(error_msg)
        return False
    return True


def detect_column_type(df: pd.DataFrame, column: str) -> str:
    """
    Detect if a column is datetime, numeric, or categorical.
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        
    Returns:
        Column type: 'datetime', 'numeric', or 'categorical'
    """
    col_data = df[column].dropna()
    
    # Check for datetime
    if any(keyword in column.lower() for keyword in DATE_KEYWORDS):
        try:
            col_data_converted = pd.to_datetime(col_data, errors='coerce')
            valid_dates = col_data_converted.notna().sum()
            
            if valid_dates > len(col_data) * 0.7:
                unique_years = col_data_converted.dropna().dt.year.nunique()
                if unique_years > 1 or col_data_converted.dropna().dt.year.iloc[0] > 1990:
                    return 'datetime'
        except:
            pass
    
    # Check for numeric
    try:
        col_numeric = pd.to_numeric(col_data, errors='coerce').dropna()
        if len(col_numeric) > len(col_data) * 0.7:
            return 'numeric'
    except:
        pass
    
    return 'categorical'


def handle_missing_data(
    df: pd.DataFrame, 
    method: str, 
    required_cols: List[str]
) -> pd.DataFrame:
    """
    Handle missing data according to specified method.
    
    Args:
        df: Input DataFrame
        method: Method to handle missing data
        required_cols: Columns that need to be processed
        
    Returns:
        DataFrame with missing data handled
    """
    df_processed = df.copy()
    original_count = len(df_processed)
    
    if method == "Remove rows with missing data (Listwise deletion)":
        df_processed = df_processed.dropna(subset=required_cols)
        logger.info(f"Removed {original_count - len(df_processed)} rows with missing data")
        
    elif method == "Fill with zeros":
        df_processed[required_cols] = df_processed[required_cols].fillna(0)
        logger.info("Filled missing values with zeros")
        
    elif method == "Fill with column mean":
        for col in required_cols:
            if df_processed[col].dtype in ['float64', 'int64']:
                mean_val = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(mean_val)
                logger.info(f"Filled {col} missing values with mean: {mean_val:.2f}")
            else:
                mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                df_processed[col] = df_processed[col].fillna(mode_val)
                
    elif method == "Fill with column median":
        for col in required_cols:
            if df_processed[col].dtype in ['float64', 'int64']:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val:.2f}")
            else:
                mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                df_processed[col] = df_processed[col].fillna(mode_val)
                
    elif method == "Forward fill (carry last observation forward)":
        df_processed[required_cols] = df_processed[required_cols].ffill()
        remaining_na = df_processed[required_cols].isnull().sum().sum()
        if remaining_na > 0:
            df_processed[required_cols] = df_processed[required_cols].fillna(0)
        logger.info("Applied forward fill")
        
    elif method == "Backward fill (carry next observation backward)":
        df_processed[required_cols] = df_processed[required_cols].bfill()
        remaining_na = df_processed[required_cols].isnull().sum().sum()
        if remaining_na > 0:
            df_processed[required_cols] = df_processed[required_cols].fillna(0)
        logger.info("Applied backward fill")
        
    elif method == "Linear interpolation":
        for col in required_cols:
            if df_processed[col].dtype in ['float64', 'int64']:
                df_processed[col] = df_processed[col].interpolate(method='linear')
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            else:
                mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                df_processed[col] = df_processed[col].fillna(mode_val)
        logger.info("Applied linear interpolation")
        
    elif method == "Fill with mode (most frequent value)":
        for col in required_cols:
            mode_val = df_processed[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else 0
            df_processed[col] = df_processed[col].fillna(fill_val)
            logger.info(f"Filled {col} missing values with mode: {fill_val}")
    
    return df_processed


def validate_data_for_training(df: pd.DataFrame, target: str, features: List[str]) -> bool:
    """
    Validate that data is suitable for model training.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        
    Returns:
        True if data is valid for training, False otherwise
    """
    if len(df) < MIN_OBSERVATIONS_FOR_TRAINING:
        error_msg = f"Insufficient data for training. Need at least {MIN_OBSERVATIONS_FOR_TRAINING} observations."
        logger.error(error_msg)
        st.error(error_msg)
        return False
    
    required_cols = [target] + features
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        st.error(error_msg)
        return False
    
    return True


def get_data_summary_stats(df: pd.DataFrame, target: str, features: List[str]) -> dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Dictionary containing summary statistics
    """
    numeric_cols = [c for c in [target] + features if pd.api.types.is_numeric_dtype(df[c])]
    
    summary = {
        "total_observations": len(df),
        "numeric_columns": numeric_cols,
        "missing_data": df[[target] + features].isna().sum().to_dict(),
        "outliers": {}
    }
    
    if numeric_cols:
        summary["descriptive_stats"] = df[numeric_cols].agg(["mean", "std", "min", "max"]).T.to_dict()
        
        # Count outliers (top 1%)
        for col in numeric_cols:
            threshold = df[col].quantile(OUTLIER_THRESHOLD)
            summary["outliers"][col] = int((df[col] > threshold).sum())
    
    return summary 