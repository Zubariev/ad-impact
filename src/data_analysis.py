"""
Comprehensive data analysis module for the Ad Impact Modeling Dashboard.
Generates detailed statistical summaries and data quality reports.
Integrated version of data_description.py for JSON output.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Union
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """
    Comprehensive data analysis tool for DataFrames.
    Generates detailed statistical summaries and data quality reports in JSON format.
    """
    
    def __init__(self):
        pass
        
    def analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric columns with comprehensive statistics."""
        analysis = {}
        
        # Basic statistics
        analysis['data_type'] = str(series.dtype)
        analysis['count'] = int(len(series))
        analysis['non_null_count'] = int(series.count())
        analysis['null_count'] = int(series.isnull().sum())
        analysis['null_percentage'] = float((series.isnull().sum() / len(series)) * 100)
        
        if series.count() > 0:  # Only if we have non-null values
            analysis['mean'] = float(series.mean()) if pd.notna(series.mean()) else None
            analysis['median'] = float(series.median()) if pd.notna(series.median()) else None
            analysis['std'] = float(series.std()) if pd.notna(series.std()) else None
            analysis['variance'] = float(series.var()) if pd.notna(series.var()) else None
            analysis['min'] = float(series.min()) if pd.notna(series.min()) else None
            analysis['max'] = float(series.max()) if pd.notna(series.max()) else None
            
            if analysis['min'] is not None and analysis['max'] is not None:
                analysis['range'] = float(analysis['max'] - analysis['min'])
            
            # Quartiles
            try:
                analysis['q25'] = float(series.quantile(0.25)) if pd.notna(series.quantile(0.25)) else None
                analysis['q75'] = float(series.quantile(0.75)) if pd.notna(series.quantile(0.75)) else None
                if analysis['q25'] is not None and analysis['q75'] is not None:
                    analysis['iqr'] = float(analysis['q75'] - analysis['q25'])
            except:
                analysis['q25'] = None
                analysis['q75'] = None
                analysis['iqr'] = None
            
            # Distribution characteristics
            try:
                clean_series = series.dropna()
                if len(clean_series) > 1:
                    analysis['skewness'] = float(stats.skew(clean_series))
                    analysis['kurtosis'] = float(stats.kurtosis(clean_series))
                else:
                    analysis['skewness'] = None
                    analysis['kurtosis'] = None
            except:
                analysis['skewness'] = None
                analysis['kurtosis'] = None
            
            # Value characteristics
            analysis['unique_count'] = int(series.nunique())
            analysis['zero_count'] = int((series == 0).sum())
            analysis['negative_count'] = int((series < 0).sum())
            analysis['positive_count'] = int((series > 0).sum())
            
            # Outlier detection (IQR method)
            try:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                if pd.notna(q1) and pd.notna(q3):
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())
                    analysis['outliers_count'] = outliers_count
                    analysis['outliers_percentage'] = float((outliers_count / len(series)) * 100)
                else:
                    analysis['outliers_count'] = 0
                    analysis['outliers_percentage'] = 0.0
            except:
                analysis['outliers_count'] = 0
                analysis['outliers_percentage'] = 0.0
                
        return analysis
    
    def analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical/string columns."""
        analysis = {}
        
        analysis['data_type'] = str(series.dtype)
        analysis['count'] = int(len(series))
        analysis['non_null_count'] = int(series.count())
        analysis['null_count'] = int(series.isnull().sum())
        analysis['null_percentage'] = float((series.isnull().sum() / len(series)) * 100)
        analysis['unique_count'] = int(series.nunique())
        
        if series.count() > 0:
            # Most frequent values
            try:
                value_counts = series.value_counts()
                analysis['most_frequent'] = {str(k): int(v) for k, v in value_counts.head(10).items()}
                analysis['mode'] = str(series.mode().iloc[0]) if not series.mode().empty else None
            except:
                analysis['most_frequent'] = {}
                analysis['mode'] = None
            
            # Unique values (limit to first 20 for readability)
            try:
                unique_values = series.dropna().unique()
                if len(unique_values) <= 20:
                    analysis['unique_values'] = [str(val) for val in unique_values]
                else:
                    analysis['unique_values_sample'] = [str(val) for val in unique_values[:20]]
                    analysis['unique_values_note'] = f"Too many unique values ({len(unique_values)}). Showing first 20."
            except:
                analysis['unique_values'] = []
            
            # String length statistics if applicable
            if series.dtype == 'object':
                try:
                    str_lengths = series.dropna().astype(str).str.len()
                    if len(str_lengths) > 0:
                        analysis['avg_string_length'] = float(str_lengths.mean())
                        analysis['min_string_length'] = int(str_lengths.min())
                        analysis['max_string_length'] = int(str_lengths.max())
                        analysis['empty_strings'] = int((series == '').sum())
                except:
                    pass
            
        return analysis
    
    def analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime columns."""
        analysis = {}
        
        analysis['data_type'] = str(series.dtype)
        analysis['count'] = int(len(series))
        analysis['non_null_count'] = int(series.count())
        analysis['null_count'] = int(series.isnull().sum())
        analysis['null_percentage'] = float((series.isnull().sum() / len(series)) * 100)
        
        if series.count() > 0:
            try:
                min_date = series.min()
                max_date = series.max()
                analysis['min_date'] = min_date.isoformat() if pd.notna(min_date) else None
                analysis['max_date'] = max_date.isoformat() if pd.notna(max_date) else None
                
                if pd.notna(min_date) and pd.notna(max_date):
                    analysis['date_range_days'] = int((max_date - min_date).days)
                analysis['unique_dates'] = int(series.nunique())
            except:
                analysis['min_date'] = None
                analysis['max_date'] = None
                analysis['date_range_days'] = None
                analysis['unique_dates'] = int(series.nunique())
            
        return analysis
    
    def analyze_dataframe_comprehensive(self, df: pd.DataFrame, filename: str = "dataset") -> Dict[str, Any]:
        """Perform comprehensive analysis of a DataFrame."""
        analysis = {
            'filename': filename,
            'shape': {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1])
            },
            'total_cells': int(df.shape[0] * df.shape[1]),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'column_count': int(len(df.columns)),
            'row_count': int(len(df)),
            'duplicate_rows': int(df.duplicated().sum()),
            'columns_analysis': {},
            'data_quality': {},
            'column_types_summary': {}
        }
        
        # Overall data quality
        total_missing = df.isnull().sum().sum()
        analysis['total_missing_values'] = int(total_missing)
        analysis['overall_completeness_percentage'] = float(((analysis['total_cells'] - total_missing) / analysis['total_cells']) * 100)
        
        # Column type summary
        numeric_cols = 0
        categorical_cols = 0
        datetime_cols = 0
        
        # Column-wise analysis
        for column in df.columns:
            series = df[column]
            
            try:
                if pd.api.types.is_numeric_dtype(series):
                    analysis['columns_analysis'][column] = self.analyze_numeric_column(series)
                    analysis['columns_analysis'][column]['column_type'] = 'numeric'
                    numeric_cols += 1
                elif pd.api.types.is_datetime64_any_dtype(series):
                    analysis['columns_analysis'][column] = self.analyze_datetime_column(series)
                    analysis['columns_analysis'][column]['column_type'] = 'datetime'
                    datetime_cols += 1
                else:
                    analysis['columns_analysis'][column] = self.analyze_categorical_column(series)
                    analysis['columns_analysis'][column]['column_type'] = 'categorical'
                    categorical_cols += 1
            except Exception as e:
                # Fallback analysis for problematic columns
                analysis['columns_analysis'][column] = {
                    'column_type': 'unknown',
                    'data_type': str(series.dtype),
                    'count': int(len(series)),
                    'non_null_count': int(series.count()),
                    'null_count': int(series.isnull().sum()),
                    'null_percentage': float((series.isnull().sum() / len(series)) * 100),
                    'unique_count': int(series.nunique()),
                    'analysis_error': str(e)
                }
        
        # Column types summary
        analysis['column_types_summary'] = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'total_columns': numeric_cols + categorical_cols + datetime_cols
        }
        
        # Data quality summary
        columns_with_missing = sum(1 for col_analysis in analysis['columns_analysis'].values() 
                                 if col_analysis.get('null_count', 0) > 0)
        
        analysis['data_quality'] = {
            'completeness_percentage': analysis['overall_completeness_percentage'],
            'columns_with_missing_data': columns_with_missing,
            'columns_with_missing_percentage': float((columns_with_missing / len(df.columns)) * 100),
            'duplicate_rows_percentage': float((analysis['duplicate_rows'] / analysis['row_count']) * 100),
            'memory_efficiency_mb_per_1k_rows': float(analysis['memory_usage_mb'] / (analysis['row_count'] / 1000)) if analysis['row_count'] > 0 else 0
        }
        
        return analysis


def analyze_dataset_for_report(df: pd.DataFrame, filename: str = "dataset") -> Dict[str, Any]:
    """
    Convenience function to analyze a dataset and return JSON-compatible results.
    
    Args:
        df: Input DataFrame to analyze
        filename: Name of the dataset for reporting
        
    Returns:
        Dictionary containing comprehensive dataset analysis
    """
    analyzer = DataAnalyzer()
    return analyzer.analyze_dataframe_comprehensive(df, filename) 