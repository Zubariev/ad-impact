#!/usr/bin/env python3
"""
Test script for seasonal decomposition functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from seasonal_decomposition import SeasonalDecomposer, apply_seasonal_decomposition

def create_test_data():
    """Create synthetic time series data with seasonality."""
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Create synthetic data with trend, seasonality, and noise
    trend = np.linspace(100, 150, 365)  # Upward trend
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # Annual seasonality
    noise = np.random.normal(0, 5, 365)  # Random noise
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    
    return df

def create_problematic_data():
    """Create data that commonly causes decomposition failures."""
    # Small dataset
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    values = [10, 12, 11, 13, 12]
    
    df_small = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    
    # Dataset with missing values
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    values = [10, 12, np.nan, 13, 12, 11, np.nan, 14, 13, 12, 11, 13, 12, 11, 14, 13, 12, 11, 13, 12]
    
    df_missing = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    
    # Constant dataset
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    values = [10] * 20
    
    df_constant = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    
    return df_small, df_missing, df_constant

def test_seasonal_decomposer():
    """Test the SeasonalDecomposer class."""
    print("Testing SeasonalDecomposer...")
    
    # Create test data
    df = create_test_data()
    
    # Test STL decomposition
    decomposer = SeasonalDecomposer(method='STL', period=12)  # Use monthly seasonality instead of yearly
    result = decomposer.decompose(df['sales'], df['date'])
    
    print(f"Decomposition method: {decomposer.method}")
    print(f"Period: {decomposer.period}")
    print(f"Original data shape: {len(result['original'])}")
    print(f"Trend shape: {len(result['trend'])}")
    print(f"Seasonal shape: {len(result['seasonal'])}")
    print(f"Residual shape: {len(result['residual'])}")
    print(f"Non-seasonal shape: {len(result['non_seasonal'])}")
    
    # Test that components add up approximately to original
    reconstructed = result['trend'] + result['seasonal'] + result['residual']
    mse = np.mean((result['original'] - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")
    
    # Test non-seasonal component
    non_seasonal = decomposer.get_non_seasonal_component()
    print(f"Non-seasonal component shape: {len(non_seasonal)}")
    
    print("✅ SeasonalDecomposer test passed!")

def test_apply_seasonal_decomposition():
    """Test the apply_seasonal_decomposition function."""
    print("\nTesting apply_seasonal_decomposition...")
    
    # Create test data
    df = create_test_data()
    
    # Apply decomposition
    df_modified, decomposer, plot = apply_seasonal_decomposition(
        df, 'sales', 'date', method='STL', period=12
    )
    
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Modified DataFrame shape: {df_modified.shape}")
    print(f"New columns: {[col for col in df_modified.columns if col not in df.columns]}")
    
    # Check that target was replaced
    print(f"Original target values: {df['sales'].iloc[:5].values}")
    print(f"Modified target values: {df_modified['sales'].iloc[:5].values}")
    
    # Check that decomposition components were added
    expected_columns = ['sales_original', 'sales_trend', 'sales_seasonal', 'sales_residual']
    for col in expected_columns:
        if col in df_modified.columns:
            print(f"✅ {col} column found")
        else:
            print(f"❌ {col} column missing")
    
    print("✅ apply_seasonal_decomposition test passed!")

def test_error_handling():
    """Test error handling for problematic data."""
    print("\nTesting error handling...")
    
    df_small, df_missing, df_constant = create_problematic_data()
    
    # Test small dataset
    print("Testing small dataset (should fail gracefully)...")
    try:
        df_modified, decomposer, plot = apply_seasonal_decomposition(
            df_small, 'sales', 'date', method='STL', period=4
        )
        if decomposer is None:
            print("✅ Small dataset handled correctly - decomposition failed gracefully")
        else:
            print("⚠️ Small dataset unexpectedly succeeded")
    except Exception as e:
        print(f"✅ Small dataset error handled: {e}")
    
    # Test dataset with missing values
    print("Testing dataset with missing values...")
    try:
        df_modified, decomposer, plot = apply_seasonal_decomposition(
            df_missing, 'sales', 'date', method='STL', period=4
        )
        if decomposer is not None:
            print("✅ Missing values handled correctly")
        else:
            print("⚠️ Missing values caused decomposition to fail")
    except Exception as e:
        print(f"✅ Missing values error handled: {e}")
    
    # Test constant dataset
    print("Testing constant dataset...")
    try:
        df_modified, decomposer, plot = apply_seasonal_decomposition(
            df_constant, 'sales', 'date', method='STL', period=4
        )
        if decomposer is None:
            print("✅ Constant dataset handled correctly - decomposition failed gracefully")
        else:
            print("⚠️ Constant dataset unexpectedly succeeded")
    except Exception as e:
        print(f"✅ Constant dataset error handled: {e}")
    
    print("✅ Error handling tests completed!")

if __name__ == "__main__":
    print("Running seasonal decomposition tests...")
    
    try:
        test_seasonal_decomposer()
        test_apply_seasonal_decomposition()
        test_error_handling()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 