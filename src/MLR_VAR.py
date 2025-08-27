"""
Multiple Linear Regression (MLR) and Vector Autoregression (VAR) models
with advanced time series diagnostics for ad impact modeling.

This script can be run independently or imported as a module.
"""

import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core analysis utilities
try:
    from time_series_utils import (
        check_stationarity,
        apply_differencing,
        check_autocorrelation,
        multicollinearity_check_with_vif,
        perform_complete_time_series_analysis,
        fit_optimal_model
    )
except ImportError:
    logger.warning("Could not import time_series_utils. Run this script from the root directory.")
    
    # Try with relative import
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from src.time_series_utils import (
            check_stationarity,
            apply_differencing,
            check_autocorrelation,
            multicollinearity_check_with_vif,
            perform_complete_time_series_analysis,
            fit_optimal_model
        )
    except ImportError:
        logger.error("Failed to import time_series_utils. Time series functionality will not be available.")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Excel file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with data
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            logger.info(f"Loaded CSV file: {file_path}, shape: {data.shape}")
        elif file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
            logger.info(f"Loaded Excel file: {file_path}, shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def convert_year_week_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert Year_Week format (e.g., '2023_01') to datetime.
    
    Args:
        series: Series with Year_Week format
        
    Returns:
        Series with datetime values
    """
    def parse_year_week(value):
        try:
            if pd.isna(value):
                return pd.NaT
                
            value = str(value).strip()
            parts = value.split('_')
            
            if len(parts) == 2:
                year, week = int(parts[0]), int(parts[1])
                return pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w')
            else:
                return pd.NaT
        except Exception:
            return pd.NaT
    
    return series.apply(parse_year_week)


def preprocess_data(data: pd.DataFrame, date_col: str = 'Y&W', 
                   target: str = 'visits_dynamics') -> pd.DataFrame:
    """
    Preprocess data for time series analysis.
    
    Args:
        data: Input DataFrame
        date_col: Date column name
        target: Target variable name
        
    Returns:
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Convert date column if it's in Year_Week format
    if date_col in df.columns and df[date_col].dtype == 'object':
        if df[date_col].astype(str).str.contains('_').any():
            logger.info(f"Converting {date_col} from Year_Week format to datetime")
            df[date_col] = convert_year_week_to_datetime(df[date_col])
    
    # Sort by date
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values detected: {missing_values[missing_values > 0].to_dict()}")
        
        # Handle missing values
        for col in df.columns:
            if missing_values[col] > 0:
                if col == date_col:
                    # Drop rows with missing dates
                    df = df.dropna(subset=[date_col])
                    logger.info(f"Dropped {missing_values[col]} rows with missing dates")
                else:
                    # Forward/backward fill for other columns
                    df[col] = df[col].ffill().bfill()
                    logger.info(f"Filled {missing_values[col]} missing values in {col}")
    
    return df


def run_complete_analysis(data: pd.DataFrame, target: str, features: List[str],
                        date_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete analysis including stationarity, multicollinearity, and autocorrelation.
    
    Args:
        data: DataFrame with target and features
        target: Target column name
        features: Feature column names
        date_col: Date column name (optional)
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info(f"Starting complete analysis for target: {target}, features: {features}")
    
    # Run time series analysis
    results = perform_complete_time_series_analysis(data, target, features)
    
    # Fit optimal model
    logger.info("Finding optimal model configuration")
    model_results = fit_optimal_model(data, target, features, date_col)
    
    # Combine results
    results['optimal_model'] = model_results
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    logger.info("Analysis completed successfully")
    return results


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path
    """
    try:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def main():
    """Main entry point when running as script."""
    logger.info("Starting MLR and VAR analysis")
    
    # Default parameters
    file_path = os.path.join('data', 'main.csv')
    date_col = 'Y&W'
    target = 'visits_dynamics'
    features = ['metro_max_click', 'metro_display_ins', 'metro_tv+dv_ins_all_25-55_50k+_(inc_guests)_sov']
    output_file = 'mlr_var_analysis_results.json'
    
    # Load data
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        # Try with different path format
        file_path = '/Users/upqrade/econ/data/main.csv'
        logger.info(f"Trying alternative path: {file_path}")
        data = load_data(file_path)
    
    # Preprocess data
    data = preprocess_data(data, date_col, target)
    
    # Run complete analysis
    results = run_complete_analysis(data, target, features, date_col)
    
    # Save results
    save_results(results, output_file)
    
    logger.info("Analysis completed")
    
    # Print summary of results
    print("\nAnalysis Summary:")
    print("-" * 50)
    
    # Stationarity summary
    stationary_series = results['summary'].get('stationary_series', [])
    non_stationary_series = results['summary'].get('non_stationary_series', [])
    
    print(f"Stationarity: {len(stationary_series)} stationary, {len(non_stationary_series)} non-stationary")
    if non_stationary_series:
        print(f"Non-stationary series: {', '.join(non_stationary_series)}")
    
    # Multicollinearity summary
    has_multicollinearity = results['multicollinearity'].get('high_multicollinearity', False)
    print(f"Multicollinearity: {'Present' if has_multicollinearity else 'Not present'}")
    
    if has_multicollinearity:
        problematic_vars = results['multicollinearity'].get('problematic_variables', [])
        print(f"Problematic variables: {', '.join(problematic_vars)}")
    
    # Autocorrelation summary
    has_autocorrelation = results.get('autocorrelation', {}).get('positive_autocorr', False)
    print(f"Autocorrelation: {'Present' if has_autocorrelation else 'Not present'}")
    
    if has_autocorrelation:
        dw_stat = results.get('autocorrelation', {}).get('durbin_watson')
        if dw_stat:
            print(f"Durbin-Watson statistic: {dw_stat:.3f}")
    
    # Optimal model summary
    suggested_model = results.get('optimal_model', {}).get('suggested_model')
    if suggested_model:
        print(f"Suggested model: {suggested_model}")
    
    best_model = results.get('optimal_model', {}).get('best_model')
    if best_model:
        model_info = results.get('optimal_model', {}).get('models', {}).get(best_model, {})
        r2 = model_info.get('r_squared')
        if r2 is not None:
            print(f"Best model ({best_model}) R²: {r2:.3f}")
    
    print("\nDetailed results saved to:", output_file)


if __name__ == "__main__":
    main()
