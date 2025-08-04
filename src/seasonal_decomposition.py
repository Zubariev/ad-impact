"""
Seasonal decomposition module for time series analysis.
Provides functionality to decompose target variables into trend, seasonal, and residual components.
"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional, Any
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.x13 import x13_arima_analysis
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeasonalDecomposer:
    """
    A class to handle seasonal decomposition of time series data.
    Supports multiple decomposition methods and provides visualization.
    """
    
    def __init__(self, method: str = 'STL', period: Optional[int] = None):
        """
        Initialize the seasonal decomposer.
        
        Args:
            method: Decomposition method ('STL', 'seasonal_decompose', 'x13')
            period: Seasonal period (if None, will be auto-detected)
        """
        self.method = method
        self.period = period
        self.decomposition_result = None
        self.original_data = None
        self.decomposed_data = None
        
    def auto_detect_period(self, data: pd.Series) -> int:
        """
        Auto-detect seasonal period from the data.
        
        Args:
            data: Time series data
            
        Returns:
            Detected seasonal period
        """
        if len(data) < 12:
            # For very small datasets, use a conservative period
            return min(4, len(data) // 2) if len(data) >= 4 else 2
            
        # Try to detect period based on data frequency
        if hasattr(data.index, 'freq'):
            freq = data.index.freq
            if freq:
                if freq.name in ['D', 'day']:
                    return 7  # Weekly seasonality
                elif freq.name in ['W', 'week']:
                    return 52  # Yearly seasonality
                elif freq.name in ['M', 'month']:
                    return 12  # Yearly seasonality
                elif freq.name in ['Q', 'quarter']:
                    return 4  # Quarterly seasonality
                    
        # If no frequency info, try to detect from data length
        if len(data) >= 24:
            # Try common periods
            for period in [12, 4, 7, 52]:
                if len(data) % period == 0:
                    return period
                    
        # Default to 12 for monthly-like data
        return 12
    
    def validate_decomposition_requirements(self, data: pd.Series, period: int) -> bool:
        """
        Validate that data meets requirements for decomposition.
        
        Args:
            data: Time series data
            period: Seasonal period
            
        Returns:
            True if requirements are met, False otherwise
        """
        logger.info(f"Validating data: length={len(data)}, period={period}")
        
        # Check minimum data requirements (relaxed for small datasets)
        min_required = max(period + 1, 10)  # At least period+1 or 10 points
        if len(data) < min_required:
            logger.warning(f"Data length ({len(data)}) is less than minimum required ({min_required}). Decomposition may be unreliable.")
            return False
            
        # Check for sufficient non-null values
        null_count = data.isnull().sum()
        null_percentage = (null_count / len(data)) * 100
        logger.info(f"Null values: {null_count} ({null_percentage:.1f}%)")
        if null_count > len(data) * 0.5:
            logger.warning("More than 50% of data is null. Decomposition may fail.")
            return False
            
        # Check for constant data
        data_std = data.std()
        logger.info(f"Data standard deviation: {data_std}")
        if data_std == 0:
            logger.warning("Data has zero variance. Decomposition not meaningful.")
            return False
            
        logger.info("Data validation passed")
        return True
    
    def decompose(self, data: pd.Series, date_index: pd.DatetimeIndex = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data: Time series data to decompose
            date_index: Optional datetime index for the data
            
        Returns:
            Dictionary containing trend, seasonal, residual, and non-seasonal components
        """
        if date_index is not None:
            data = pd.Series(data.values, index=date_index)
            
        self.original_data = data.copy()
        
        # Auto-detect period if not provided
        if self.period is None:
            self.period = self.auto_detect_period(data)
            logger.info(f"Auto-detected period: {self.period}")
        else:
            logger.info(f"Using provided period: {self.period}")
            
        # Validate decomposition requirements
        logger.info(f"Validating decomposition requirements...")
        if not self.validate_decomposition_requirements(data, self.period):
            raise ValueError("Data does not meet decomposition requirements")
            
        logger.info(f"Decomposing data with method: {self.method}, period: {self.period}")
        
        try:
            if self.method == 'STL':
                # STL decomposition (more robust)
                try:
                    decomposition = STL(data, period=self.period, robust=True).fit()
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid
                except Exception as stl_error:
                    logger.warning(f"STL decomposition failed: {stl_error}. Trying with non-robust method.")
                    try:
                        decomposition = STL(data, period=self.period, robust=False).fit()
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid
                    except Exception as stl_error2:
                        logger.warning(f"STL non-robust also failed: {stl_error2}. Falling back to seasonal_decompose.")
                        decomposition = seasonal_decompose(data, period=self.period, extrapolate_trend='freq')
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid
                
            elif self.method == 'seasonal_decompose':
                # Classical decomposition
                decomposition = seasonal_decompose(
                    data, 
                    period=self.period, 
                    extrapolate_trend='freq'
                )
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
            elif self.method == 'x13':
                # X-13ARIMA-SEATS decomposition (requires X-13ARIMA-SEATS)
                try:
                    decomposition = x13_arima_analysis(data, period=self.period)
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid
                except Exception as e:
                    logger.warning(f"X-13 decomposition failed: {e}. Falling back to STL.")
                    decomposition = STL(data, period=self.period, robust=True).fit()
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid
            else:
                raise ValueError(f"Unknown decomposition method: {self.method}")
                
            # Calculate non-seasonal component (trend + residual)
            non_seasonal = trend + residual
            
            # Handle NaN values in decomposition components
            # Fill NaN values in trend and residual before calculating non_seasonal
            if trend.isnull().any():
                logger.warning(f"Found {trend.isnull().sum()} NaN values in trend component. Filling with forward/backward fill.")
                trend = trend.ffill().bfill()
            
            if residual.isnull().any():
                logger.warning(f"Found {residual.isnull().sum()} NaN values in residual component. Filling with forward/backward fill.")
                residual = residual.ffill().bfill()
            
            # Recalculate non-seasonal component with cleaned data
            non_seasonal = trend + residual
            
            # Final check: ensure non_seasonal has no NaN values
            if non_seasonal.isnull().any():
                logger.warning(f"Found {non_seasonal.isnull().sum()} NaN values in non-seasonal component. Filling with forward/backward fill.")
                non_seasonal = non_seasonal.ffill().bfill()
            
            # Also clean seasonal component for consistency
            if seasonal.isnull().any():
                logger.warning(f"Found {seasonal.isnull().sum()} NaN values in seasonal component. Filling with forward/backward fill.")
                seasonal = seasonal.ffill().bfill()
            
            self.decomposition_result = {
                'original': data,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'non_seasonal': non_seasonal
            }
            
            # Create decomposed DataFrame
            self.decomposed_data = pd.DataFrame({
                'original': data,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'non_seasonal': non_seasonal
            })
            
            logger.info("Decomposition completed successfully")
            return self.decomposition_result
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Return original data as non-seasonal if decomposition fails
            return {
                'original': data,
                'trend': data,
                'seasonal': pd.Series(0, index=data.index),
                'residual': pd.Series(0, index=data.index),
                'non_seasonal': data
            }
    
    def get_non_seasonal_component(self) -> pd.Series:
        """
        Get the non-seasonal component for modeling.
        
        Returns:
            Non-seasonal component (trend + residual)
        """
        if self.decomposition_result is None:
            raise ValueError("Decomposition must be performed first")
        return self.decomposition_result['non_seasonal']
    
    def create_decomposition_plot(self) -> go.Figure:
        """
        Create a comprehensive decomposition visualization.
        
        Returns:
            Plotly figure showing all decomposition components
        """
        if self.decomposition_result is None:
            raise ValueError("Decomposition must be performed first")
            
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original Data', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['original'].index,
                y=self.decomposition_result['original'].values,
                mode='lines',
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['trend'].index,
                y=self.decomposition_result['trend'].values,
                mode='lines',
                name='Trend',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['seasonal'].index,
                y=self.decomposition_result['seasonal'].values,
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['residual'].index,
                y=self.decomposition_result['residual'].values,
                mode='lines',
                name='Residual',
                line=dict(color='orange')
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Seasonal Decomposition",
            showlegend=False
        )
        
        return fig
    
    def create_comparison_plot(self) -> go.Figure:
        """
        Create a comparison plot showing original vs non-seasonal components.
        
        Returns:
            Plotly figure comparing original and non-seasonal data
        """
        if self.decomposition_result is None:
            raise ValueError("Decomposition must be performed first")
            
        fig = go.Figure()
        
        # Original data
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['original'].index,
                y=self.decomposition_result['original'].values,
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=2)
            )
        )
        
        # Non-seasonal component
        fig.add_trace(
            go.Scatter(
                x=self.decomposition_result['non_seasonal'].index,
                y=self.decomposition_result['non_seasonal'].values,
                mode='lines',
                name='Non-Seasonal Component',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title_text="Original vs Non-Seasonal Component",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500
        )
        
        return fig


@st.cache_resource
def apply_seasonal_decomposition(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    method: str = 'STL',
    period: Optional[int] = None
) -> Tuple[pd.DataFrame, SeasonalDecomposer, go.Figure]:
    """
    Apply seasonal decomposition to the target variable and return modified DataFrame.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        date_col: Date column name
        method: Decomposition method
        period: Seasonal period (if None, auto-detected)
        
    Returns:
        Tuple of (modified DataFrame, decomposer object, decomposition plot)
    """
    try:
        # Ensure date column is datetime
        df_copy = df.copy()
        logger.info(f"Original DataFrame shape: {df_copy.shape}")
        logger.info(f"Target column '{target}' data type: {df_copy[target].dtype}")
        logger.info(f"Date column '{date_col}' data type: {df_copy[date_col].dtype}")
        
        # Handle year-week format (e.g., "2023_0")
        if df_copy[date_col].dtype == 'object' and df_copy[date_col].str.contains('_').any():
            from data_utils import convert_year_week_to_datetime
            df_copy[date_col] = convert_year_week_to_datetime(df_copy[date_col])
            logger.info(f"Converted year-week format to datetime")
        else:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Sort by date
        df_copy = df_copy.sort_values(date_col).reset_index(drop=True)
        logger.info(f"DataFrame shape after sorting: {df_copy.shape}")
        
        # Handle missing values in target variable
        missing_count = df_copy[target].isnull().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in target variable. Filling with forward fill.")
            df_copy[target] = df_copy[target].ffill().bfill()
        
        # Remove rows with missing target values
        df_copy = df_copy.dropna(subset=[target])
        logger.info(f"DataFrame shape after removing missing target values: {df_copy.shape}")
        
        if len(df_copy) < 10:
            raise ValueError(f"Insufficient data after preprocessing: {len(df_copy)} rows (minimum 10 required)")
        
        # Log target variable statistics
        logger.info(f"Target variable statistics:")
        logger.info(f"  - Min: {df_copy[target].min()}")
        logger.info(f"  - Max: {df_copy[target].max()}")
        logger.info(f"  - Mean: {df_copy[target].mean()}")
        logger.info(f"  - Std: {df_copy[target].std()}")
        logger.info(f"  - Null count: {df_copy[target].isnull().sum()}")
        
        # Create decomposer
        decomposer = SeasonalDecomposer(method=method, period=period)
        
        # Perform decomposition
        target_series = df_copy[target]
        logger.info(f"Starting decomposition with method: {method}, period: {period}")
        logger.info(f"Target series length: {len(target_series)}")
        logger.info(f"Target series range: {target_series.index.min()} to {target_series.index.max()}")
        
        decomposition_result = decomposer.decompose(target_series, df_copy[date_col])
        
        # Replace target with non-seasonal component
        df_copy[f'{target}_original'] = df_copy[target]
        
        # Fix index alignment issue
        non_seasonal_series = decomposition_result['non_seasonal']
        if len(non_seasonal_series) != len(df_copy):
            logger.warning(f"Length mismatch: non_seasonal ({len(non_seasonal_series)}) vs DataFrame ({len(df_copy)})")
            # Try to align by taking the first len(df_copy) values
            if len(non_seasonal_series) > len(df_copy):
                non_seasonal_series = non_seasonal_series.iloc[:len(df_copy)]
            else:
                # Pad with the last value if shorter
                last_val = non_seasonal_series.iloc[-1]
                padding = pd.Series([last_val] * (len(df_copy) - len(non_seasonal_series)), 
                                  index=range(len(non_seasonal_series), len(df_copy)))
                non_seasonal_series = pd.concat([non_seasonal_series, padding])
        
        # Reset index to match DataFrame index
        non_seasonal_series = non_seasonal_series.reset_index(drop=True)
        
        df_copy[target] = non_seasonal_series.ffill().bfill()
        
        # Add decomposition components as features
        df_copy[f'{target}_trend'] = decomposition_result['trend']
        df_copy[f'{target}_seasonal'] = decomposition_result['seasonal']
        df_copy[f'{target}_residual'] = decomposition_result['residual']
        
        # Final validation: ensure target variable has no NaN values
        if df_copy[target].isnull().any():
            logger.warning(f"Found {df_copy[target].isnull().sum()} NaN values in target after decomposition. Filling with forward/backward fill.")
            df_copy[target] = df_copy[target].ffill().bfill()
        
        # Additional validation: check for infinite values
        if np.isinf(df_copy[target]).any():
            logger.warning(f"Found infinite values in target after decomposition. Replacing with NaN and filling.")
            df_copy[target] = df_copy[target].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Create decomposition plot
        decomposition_plot = decomposer.create_decomposition_plot()
        
        logger.info(f"Seasonal decomposition applied successfully using {method} method")
        
        return df_copy, decomposer, decomposition_plot
        
    except Exception as e:
        logger.error(f"Seasonal decomposition failed: {e}")
        # Log specific error details for debugging
        if "Data does not meet decomposition requirements" in str(e):
            logger.error("Data validation failed - insufficient data or invalid characteristics")
        elif "period" in str(e).lower():
            logger.error("Period-related error - check data length and period settings")
        elif "stl" in str(e).lower():
            logger.error("STL decomposition error - check data quality and period")
        else:
            logger.error(f"Unexpected decomposition error: {type(e).__name__}")
        
        # Return original data if decomposition fails
        return df, None, None


def create_decomposition_ui(model_name: str = "default") -> Tuple[str, Optional[int]]:
    """
    Create UI for seasonal decomposition settings.
    
    Args:
        model_name: Model name for unique key generation
        
    Returns:
        Tuple of (decomposition method, seasonal period)
    """
    st.subheader("Seasonal Decomposition Settings")
    
    # Decomposition method selection
    method = st.selectbox(
        "Decomposition Method",
        options=['STL', 'seasonal_decompose', 'x13'],
        index=0,
        key=f"decomposition_method_{model_name}",
        help="STL: Robust decomposition, seasonal_decompose: Classical decomposition, x13: X-13ARIMA-SEATS"
    )
    
    # Seasonal period
    period = st.number_input(
        "Seasonal Period (leave empty for auto-detection)",
        min_value=2,
        max_value=365,
        value=None,
        key=f"decomposition_period_{model_name}",
        help="Common values: 7 (weekly), 12 (monthly), 52 (yearly), 4 (quarterly)"
    )
    
    if period is not None and period <= 0:
        period = None
    
    return method, period


def display_decomposition_info(decomposer: SeasonalDecomposer) -> None:
    """
    Display information about the decomposition results.
    
    Args:
        decomposer: SeasonalDecomposer object with results
    """
    if decomposer is None or decomposer.decomposition_result is None:
        return
        
    st.subheader("Decomposition Information")
    
    # Calculate decomposition statistics
    original = decomposer.decomposition_result['original']
    trend = decomposer.decomposition_result['trend']
    seasonal = decomposer.decomposition_result['seasonal']
    residual = decomposer.decomposition_result['residual']
    
    # Calculate variance explained
    total_var = original.var()
    seasonal_var = seasonal.var()
    trend_var = trend.var()
    residual_var = residual.var()
    
    seasonal_explained = (seasonal_var / total_var) * 100 if total_var > 0 else 0
    trend_explained = (trend_var / total_var) * 100 if total_var > 0 else 0
    residual_explained = (residual_var / total_var) * 100 if total_var > 0 else 0
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Seasonal Variance Explained", f"{seasonal_explained:.1f}%")
        
    with col2:
        st.metric("Trend Variance Explained", f"{trend_explained:.1f}%")
        
    with col3:
        st.metric("Residual Variance Explained", f"{residual_explained:.1f}%")
    
    # Display decomposition method and period
    st.info(f"Decomposition Method: {decomposer.method}, Period: {decomposer.period}")
    
    # Show comparison plot
    comparison_plot = decomposer.create_comparison_plot()
    st.plotly_chart(comparison_plot, use_container_width=True) 