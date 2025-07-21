"""
Debug utilities for multicollinearity analysis integration
"""

import streamlit as st
import os
import sys
from pathlib import Path

def debug_integration():
    """Debug the multicollinearity analysis integration."""
    
    st.markdown("## 🔧 Multicollinearity Integration Debug")
    
    # Check file existence
    st.markdown("### 📁 File Checks")
    
    files_to_check = [
        'src/multicollinearity_analysis.py',
        'src/dashboard_integration.py', 
        'src/multicollinearity_streamlit.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            st.success(f" {file_path} exists ({file_size:.1f} KB)")
        else:
            st.error(f" {file_path} missing")
    
    # Check imports
    st.markdown("### 📦 Import Checks")
    
    try:
        from multicollinearity_analysis import MulticollinearityAnalyzer
        st.success(" MulticollinearityAnalyzer imported successfully")
    except ImportError as e:
        st.error(f" Failed to import MulticollinearityAnalyzer: {e}")
    
    try:
        from dashboard_integration import DashboardAnalysisIntegrator
        st.success(" DashboardAnalysisIntegrator imported successfully")
    except ImportError as e:
        st.error(f" Failed to import DashboardAnalysisIntegrator: {e}")
    
    try:
        from multicollinearity_streamlit import add_multicollinearity_analysis
        st.success(" Streamlit integration imported successfully")
    except ImportError as e:
        st.error(f" Failed to import Streamlit integration: {e}")
    
    # Check current working directory
    st.markdown("### 📂 Working Directory")
    st.write(f"Current directory: {os.getcwd()}")
    
    # Check Python path
    st.markdown("### 🐍 Python Path")
    st.write("Current Python path:")
    for i, path in enumerate(sys.path[:10]):  # Show first 10 paths
        st.write(f"{i+1}. {path}")
    
    # Check package dependencies
    st.markdown("### 📦 Required Packages")
    
    required_packages = [
        'pandas', 'numpy', 'plotly', 'streamlit', 
        'scikit-learn', 'statsmodels', 'xgboost'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            st.success(f" {package} available")
        except ImportError:
            st.error(f" {package} missing")
    
    # Test basic functionality
    st.markdown("### 🧪 Quick Functionality Test")
    
    if st.button(" Run Quick Test"):
        try:
            import pandas as pd
            import numpy as np
            
            # Create sample data
            np.random.seed(42)
            test_data = pd.DataFrame({
                'target': np.random.randn(100),
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100) * 0.8,  # correlated
                'feature3': np.random.randn(100)
            })
            
            # Make feature2 correlated with feature1
            test_data['feature2'] = test_data['feature1'] * 0.9 + test_data['feature2'] * 0.1
            
            st.write("Sample data created:")
            st.dataframe(test_data.head())
            
            # Test correlation calculation
            correlation = test_data.corr()
            st.write("Correlation matrix:")
            st.dataframe(correlation.round(3))
            
            st.success(" Basic functionality test passed!")
            
        except Exception as e:
            st.error(f" Quick test failed: {e}")

if __name__ == "__main__":
    debug_integration()
