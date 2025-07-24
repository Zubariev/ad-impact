"""
Data utilities for the Ad Impact Modeling Dashboard.
Handles file ingestion, data cleaning, and preprocessing operations.
"""

import io
import logging
from typing import List, Optional, Tuple, Union, Dict

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
    unified.insert(0, "id", unified.index)
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
        Column type: 'datetime', 'numeric', 'integer', or 'categorical'
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
            if (col_numeric % 1 == 0).all():
                return 'integer'
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


def create_treated_column_by_location(
    df: pd.DataFrame, 
    location_col: str, 
    treated_locations: List[str]
) -> pd.DataFrame:
    """
    Create a 'treated' column based on specific locations.
    
    Args:
        df: Input DataFrame
        location_col: Name of the location/city column
        treated_locations: List of location names to mark as treated
        
    Returns:
        DataFrame with 'treated' column added
    """
    df_new = df.copy()
    df_new['treated'] = (df_new[location_col].isin(treated_locations)).astype(int)
    
    treated_count = df_new['treated'].sum()
    control_count = (df_new['treated'] == 0).sum()
    
    logger.info(f"Created 'treated' column: {treated_count} treated, {control_count} control observations")
    return df_new


def create_post_column_by_date(
    df: pd.DataFrame, 
    date_col: str, 
    cutoff_date: Union[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Create a 'post' column based on a date cutoff.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        cutoff_date: Date cutoff (observations after this are marked as post=1)
        
    Returns:
        DataFrame with 'post' column added
    """
    df_new = df.copy()
    
    # Convert cutoff_date to pandas datetime if it's a string
    if isinstance(cutoff_date, str):
        cutoff_date = pd.to_datetime(cutoff_date)
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_new[date_col]):
        df_new[date_col] = pd.to_datetime(df_new[date_col])
    
    df_new['post'] = (df_new[date_col] > cutoff_date).astype(int)
    
    pre_count = (df_new['post'] == 0).sum()
    post_count = df_new['post'].sum()
    
    logger.info(f"Created 'post' column with cutoff {cutoff_date}: {pre_count} pre, {post_count} post observations")
    return df_new


def create_post_column_by_period(
    df: pd.DataFrame, 
    period_col: str, 
    cutoff_value: Union[int, float, str]
) -> pd.DataFrame:
    """
    Create a 'post' column based on a period cutoff (e.g., week, month, year).
    
    Args:
        df: Input DataFrame
        period_col: Name of the period column (e.g., 'Week', 'Month', 'Year')
        cutoff_value: Period cutoff (observations after this are marked as post=1)
        
    Returns:
        DataFrame with 'post' column added
    """
    df_new = df.copy()
    df_new['post'] = (df_new[period_col] > cutoff_value).astype(int)
    
    pre_count = (df_new['post'] == 0).sum()
    post_count = df_new['post'].sum()
    
    logger.info(f"Created 'post' column with cutoff {cutoff_value}: {pre_count} pre, {post_count} post observations")
    return df_new


def create_treated_column_by_quantile(
    df: pd.DataFrame, 
    metric_col: str, 
    quantile: float = 0.5,
    above_threshold: bool = True
) -> pd.DataFrame:
    """
    Create a 'treated' column based on a quantile threshold of some metric.
    
    Args:
        df: Input DataFrame
        metric_col: Column to use for threshold calculation
        quantile: Quantile threshold (0.0 to 1.0)
        above_threshold: If True, values above threshold are treated; if False, below
        
    Returns:
        DataFrame with 'treated' column added
    """
    df_new = df.copy()
    threshold = df_new[metric_col].quantile(quantile)
    
    if above_threshold:
        df_new['treated'] = (df_new[metric_col] > threshold).astype(int)
    else:
        df_new['treated'] = (df_new[metric_col] <= threshold).astype(int)
    
    treated_count = df_new['treated'].sum()
    control_count = (df_new['treated'] == 0).sum()
    
    direction = "above" if above_threshold else "below"
    logger.info(f"Created 'treated' column ({direction} {quantile} quantile = {threshold:.2f}): {treated_count} treated, {control_count} control")
    return df_new


def prepare_did_data(
    df: pd.DataFrame,
    location_col: str,
    treated_locations: List[str],
    date_col: str,
    cutoff_date: Union[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Prepare data for Difference-in-Differences analysis.
    
    Args:
        df: Input DataFrame
        location_col: Name of the location/city column
        treated_locations: List of location names to mark as treated
        date_col: Name of the date column
        cutoff_date: Date cutoff for pre/post periods
        
    Returns:
        DataFrame ready for DiD analysis with 'treated' and 'post' columns
    """
    logger.info("Preparing data for DiD analysis...")
    
    # Create treated column
    df_prepared = create_treated_column_by_location(df, location_col, treated_locations)
    
    # Create post column
    df_prepared = create_post_column_by_date(df_prepared, date_col, cutoff_date)
    
    # Validate the 2x2 design
    crosstab = pd.crosstab(df_prepared['treated'], df_prepared['post'], margins=True)
    logger.info(f"DiD 2x2 design:\n{crosstab}")
    
    # Check for sufficient observations in each cell
    min_obs_per_cell = 5
    for treated in [0, 1]:
        for post in [0, 1]:
            cell_count = ((df_prepared['treated'] == treated) & (df_prepared['post'] == post)).sum()
            if cell_count < min_obs_per_cell:
                logger.warning(f"Low observations in treated={treated}, post={post} cell: {cell_count}")
    
    return df_prepared


def prepare_synthetic_control_data(
    df: pd.DataFrame,
    location_col: str,
    treated_location: str
) -> pd.DataFrame:
    """
    Prepare data for Synthetic Control analysis.
    
    Args:
        df: Input DataFrame
        location_col: Name of the location/city column
        treated_location: Single location name to mark as treated
        
    Returns:
        DataFrame ready for Synthetic Control with 'treated' column
    """
    logger.info("Preparing data for Synthetic Control analysis...")
    
    df_prepared = create_treated_column_by_location(df, location_col, [treated_location])
    
    treated_count = df_prepared['treated'].sum()
    control_count = (df_prepared['treated'] == 0).sum()
    
    if treated_count == 0:
        raise ValueError(f"No observations found for treated location: {treated_location}")
    if control_count < 10:
        logger.warning(f"Few control observations ({control_count}) for synthetic control")
    
    logger.info(f"Synthetic Control setup: 1 treated location ({treated_location}), {len(df_prepared[location_col].unique())-1} control locations")
    
    return df_prepared


def suggest_model_data_requirements(df: pd.DataFrame) -> Dict[str, str]:
    """
    Analyze dataset and suggest how to prepare it for different models.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with suggestions for each model type
    """
    suggestions = {}
    cols = list(df.columns)
    
    # Look for location/geographic columns
    location_cols = [col for col in cols if any(keyword in col.lower() 
                    for keyword in ['city', 'location', 'region', 'state', 'country', 'store', 'unit'])]
    
    # Look for time-related columns
    time_cols = [col for col in cols if any(keyword in col.lower() 
                for keyword in ['date', 'time', 'week', 'month', 'year', 'period'])]
    
    # Base models that work with any numeric data
    suggestions["MLR"] = "Ready - just select numeric features"
    suggestions["Distributed Lag"] = "Ready - works with time series features"
    suggestions["ML + SHAP"] = "Ready - handles both numeric and categorical features"
    suggestions["VAR"] = "Ready - good for time series with multiple variables"
    suggestions["CausalImpact"] = "Ready - works with time series data"
    
    # Advanced models requiring special columns
    if location_cols:
        suggestions["DiD"] = f"Add 'treated' (based on {location_cols}) and 'post' (based on {time_cols if time_cols else 'time cutoff'}) columns"
        suggestions["Synthetic Control"] = f"Add 'treated' column (pick 1 treated from {location_cols})"
        suggestions["PSM"] = f"Add 'treated' column (based on {location_cols} or intervention logic)"
    else:
        suggestions["DiD"] = "Need location/unit identifier to create treatment groups"
        suggestions["Synthetic Control"] = "Need location/unit identifier to define treated vs control"
        suggestions["PSM"] = "Need intervention/treatment indicator column"
    
    return suggestions


# Update the existing validate_data_for_training function
def validate_data_for_training(df: pd.DataFrame, target: str, features: List[str], model_name: str = None) -> bool:
    """
    Validate that data is suitable for model training.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        model_name: Name of the model (optional, for model-specific validation)
        
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
    
    # Model-specific validation
    if model_name in ["DiD"]:
        if "treated" not in df.columns or "post" not in df.columns:
            st.error("DiD model requires 'treated' and 'post' columns. Use data preparation utilities to create them.")
            st.info("💡 Go to 'Data Preparation for Advanced Models' section above to create these columns.")
            return False
    elif model_name in ["Synthetic Control", "PSM"]:
        if "treated" not in df.columns:
            st.error(f" {model_name} model requires 'treated' column. Use data preparation utilities to create it.")
            st.info("💡 Go to 'Data Preparation for Advanced Models' section above to create this column.")
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