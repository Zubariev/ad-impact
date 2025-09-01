"""
MLR Enhancements for Ad Impact Modeling Dashboard.

Integrates advanced analytical processes for MLR models with Streamlit UI components.
Adds time series functionality, stationarity checks, and multicollinearity analysis.
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
import time
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import os
import uuid

# Import background process utilities
from src.background_processes import (
    run_background_process, 
    get_process_status, 
    get_process_result,
    display_process_ui,
    stationarity_analysis_process,
    multicollinearity_analysis_process,
    autocorrelation_analysis_process,
    optimal_model_selection_process
)

# Import time series utilities
from src.time_series_utils import (
    check_stationarity,
    apply_differencing,
    check_autocorrelation,
    multicollinearity_check_with_vif,
    perform_complete_time_series_analysis,
    fit_optimal_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mlr_enhancement_ui(df: pd.DataFrame, target: str, features: List[str], 
                             model_name: str = "MLR") -> None:
    """
    Create enhanced MLR UI section with advanced analytics.
    
    Args:
        df: DataFrame with target and features
        target: Target column name
        features: Feature column names
        model_name: Model name for UI
    """
    st.markdown("---")
    st.header("🧪 Advanced MLR Analytics")
    st.caption("Background processes for enhanced model analysis")
    
    # Initialize session state for process tracking
    if f"{model_name}_processes" not in st.session_state:
        st.session_state[f"{model_name}_processes"] = {}
    
    process_state = st.session_state[f"{model_name}_processes"]
    
    # Create tabs for different analysis categories
    tabs = st.tabs([
        "📊 Stationarity & TS Analysis", 
        "🔄 Multicollinearity", 
        "📈 Autocorrelation", 
        "🚀 Optimal Model"
    ])
    
    # Tab 1: Stationarity & Time Series Analysis
    with tabs[0]:
        create_stationarity_tab(df, target, features, model_name, process_state)
    
    # Tab 2: Multicollinearity Analysis
    with tabs[1]:
        create_multicollinearity_tab(df, target, features, model_name, process_state)
    
    # Tab 3: Autocorrelation Analysis
    with tabs[2]:
        create_autocorrelation_tab(df, target, features, model_name, process_state)
    
    # Tab 4: Optimal Model Selection
    with tabs[3]:
        create_optimal_model_tab(df, target, features, model_name, process_state)


def create_stationarity_tab(df: pd.DataFrame, target: str, features: List[str], 
                          model_name: str, process_state: Dict) -> None:
    """Create stationarity analysis tab."""
    st.subheader("Stationarity Analysis")
    st.info("""
    Stationarity is crucial for time series regression. Non-stationary data can lead to 
    spurious correlations and invalid statistical tests.
    """)
    
    # Run stationarity analysis button
    run_key = f"{model_name}_stationarity_run"
    result_key = f"{model_name}_stationarity_result"
    
    if 'stationarity_process' not in process_state:
        st.info("Stationarity analysis will start automatically after model training.")
        return

    process_id = process_state['stationarity_process']
    display_process_ui(
        process_id=process_id,
        title="Stationarity Analysis Status",
        on_complete_callback=display_stationarity_results,
        result_key=result_key,
        refresh_interval=1
    )
    
    # Display existing results if available
    if result_key in st.session_state:
        display_stationarity_results(st.session_state[result_key])


def display_stationarity_results(result: Dict[str, Any]) -> None:
    """Display stationarity analysis results."""
    if not result:
        return
    
    st.success("✅ Stationarity Analysis Completed")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stationary Series", result.get('stationary_series', 0))
    with col2:
        st.metric("Non-stationary Series", result.get('non_stationary_series', 0))
    
    # Display detailed results
    if 'stationarity_results' in result:
        st.markdown("### Detailed Series Analysis")
        
        # Create tabs for each series
        series_names = list(result['stationarity_results'].keys())
        if series_names:
            series_tabs = st.tabs(series_names)
            
            for i, series_name in enumerate(series_names):
                series_result = result['stationarity_results'][series_name]
                
                with series_tabs[i]:
                    # Display stationarity status with emoji
                    status_emoji = "✅" if series_result['is_stationary'] else "❌"
                    st.markdown(f"**Status: {status_emoji} {'' if series_result['is_stationary'] else 'Not'} Stationary**")
                    
                    # Create columns for ADF and KPSS results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### ADF Test")
                        st.write(f"p-value: {series_result['adf_test']['p_value']:.4f}")
                        st.write(f"Test statistic: {series_result['adf_test']['test_statistic']:.4f}")
                        st.caption(series_result['adf_test']['message'])
                    
                    with col2:
                        st.markdown("##### KPSS Test")
                        st.write(f"p-value: {series_result['kpss_test']['p_value']:.4f}")
                        st.write(f"Test statistic: {series_result['kpss_test']['test_statistic']:.4f}")
                        st.caption(series_result['kpss_test']['message'])
                    
                    # Suggested transformation
                    if not series_result['is_stationary']:
                        st.info(f"**Suggested transformation:** {series_result.get('suggested_transformation', 'differencing')}")


def create_multicollinearity_tab(df: pd.DataFrame, target: str, features: List[str], 
                               model_name: str, process_state: Dict) -> None:
    """Create multicollinearity analysis tab."""
    st.subheader("Multicollinearity Analysis")
    st.info("""
    Multicollinearity can destabilize model coefficients and inflate standard errors. 
    This analysis helps identify and address collinearity issues between predictors.
    """)
    
    # Run multicollinearity analysis button
    run_key = f"{model_name}_multicol_run"
    result_key = f"{model_name}_multicol_result"
    
    if 'multicol_process' not in process_state:
        st.info("Multicollinearity analysis will start automatically after model training.")
        return

    process_id = process_state['multicol_process']
    display_process_ui(
        process_id=process_id,
        title="Multicollinearity Analysis Status",
        on_complete_callback=display_multicollinearity_results,
        result_key=result_key,
        refresh_interval=1
    )
    
    # Display existing results if available
    if result_key in st.session_state:
        display_multicollinearity_results(st.session_state[result_key])


def display_multicollinearity_results(result: Dict[str, Any]) -> None:
    """Display multicollinearity analysis results."""
    if not result:
        return
    
    st.success("✅ Multicollinearity Analysis Completed")
    
    # Display VIF results
    if 'vif_analysis' in result and 'vif_values' in result['vif_analysis']:
        st.markdown("### VIF Analysis")
        
        # Convert VIF values to DataFrame for display
        vif_values = result['vif_analysis']['vif_values']
        vif_df = pd.DataFrame({
            'Variable': list(vif_values.keys()),
            'VIF': list(vif_values.values())
        })
        
        # Sort by VIF value
        vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)
        
        # Display as styled DataFrame
        st.dataframe(
            vif_df.style.apply(
                lambda x: ['background-color: #ffcccc' if v > 10 else 
                          'background-color: #ffffcc' if v > 5 else '' 
                          for v in x], 
                subset=['VIF']
            ),
            use_container_width=True
        )
        
        # Display problematic variables
        problematic_vars = result['vif_analysis'].get('problematic_variables', [])
        if problematic_vars:
            st.warning(f"⚠️ Found {len(problematic_vars)} variables with high VIF (>10): {', '.join(problematic_vars)}")
    
    # Display correlation pairs
    if 'high_correlation_pairs' in result:
        st.markdown("### High Correlation Pairs")
        
        high_corr_pairs = result['high_correlation_pairs']
        if high_corr_pairs:
            # Convert to DataFrame
            corr_df = pd.DataFrame(high_corr_pairs)
            
            # Sort by correlation strength
            corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)
            
            # Display as styled DataFrame
            st.dataframe(
                corr_df.style.apply(
                    lambda x: ['background-color: #ffcccc' if abs(v) > 0.9 else 
                              'background-color: #ffffcc' if abs(v) > 0.8 else '' 
                              for v in x], 
                    subset=['correlation']
                ),
                use_container_width=True
            )
        else:
            st.success("✅ No high correlation pairs found (threshold > 0.7)")
    
    # Display suggested actions
    if 'vif_analysis' in result and 'suggested_actions' in result['vif_analysis']:
        suggested_actions = result['vif_analysis']['suggested_actions']
        if suggested_actions:
            st.markdown("### Suggested Actions")
            for i, action in enumerate(suggested_actions, 1):
                st.markdown(f"{i}. {action}")


def create_autocorrelation_tab(df: pd.DataFrame, target: str, features: List[str], 
                             model_name: str, process_state: Dict) -> None:
    """Create autocorrelation analysis tab."""
    st.subheader("Autocorrelation Analysis")
    st.info("""
    Autocorrelation in residuals indicates that time-dependent patterns have not been captured
    by the model, which can lead to inefficient estimates and invalid inference.
    """)
    
    # Run autocorrelation analysis button
    run_key = f"{model_name}_autocorr_run"
    result_key = f"{model_name}_autocorr_result"
    
    if 'autocorr_process' not in process_state:
        st.info("Autocorrelation analysis will start automatically after model training.")
        return

    process_id = process_state['autocorr_process']
    display_process_ui(
        process_id=process_id,
        title="Autocorrelation Analysis Status",
        on_complete_callback=display_autocorrelation_results,
        result_key=result_key,
        refresh_interval=1
    )
    
    # Display existing results if available
    if result_key in st.session_state:
        display_autocorrelation_results(st.session_state[result_key])


def display_autocorrelation_results(result: Dict[str, Any]) -> None:
    """Display autocorrelation analysis results."""
    if not result:
        return
    
    st.success("✅ Autocorrelation Analysis Completed")
    
    # Check if there was an error
    if 'error' in result:
        st.error(f"Analysis Error: {result['error']}")
        return
    
    # Display Durbin-Watson statistic
    if 'autocorrelation' in result and 'durbin_watson' in result['autocorrelation']:
        dw_stat = result['autocorrelation']['durbin_watson']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display DW with color coding
            if dw_stat < 1.5:
                st.metric("Durbin-Watson Statistic", f"{dw_stat:.3f}", delta="Positive autocorrelation", delta_color="off")
            elif dw_stat > 2.5:
                st.metric("Durbin-Watson Statistic", f"{dw_stat:.3f}", delta="Negative autocorrelation", delta_color="off")
            else:
                st.metric("Durbin-Watson Statistic", f"{dw_stat:.3f}", delta="No autocorrelation", delta_color="normal")
        
        with col2:
            # Display Ljung-Box test result if available
            if 'ljung_box' in result['autocorrelation']:
                lb_pvalue = result['autocorrelation']['ljung_box'].get('p_value')
                if lb_pvalue is not None:
                    lb_status = "Autocorrelation present" if lb_pvalue < 0.05 else "No autocorrelation"
                    st.metric("Ljung-Box p-value", f"{lb_pvalue:.3f}", delta=lb_status, delta_color="off")
        
        # Display message
        st.info(result['autocorrelation'].get('message', ''))
    
    # Display model diagnostics
    if 'model_diagnostics' in result:
        st.markdown("### Model Diagnostics")
        
        diagnostics = result['model_diagnostics']
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R²", f"{diagnostics.get('r_squared', 0):.3f}")
        with col2:
            st.metric("Adjusted R²", f"{diagnostics.get('adj_r_squared', 0):.3f}")
        with col3:
            st.metric("AIC", f"{diagnostics.get('aic', 0):.3f}")
    
    # Display suggested action
    if result.get('has_autocorrelation') and 'suggested_action' in result['autocorrelation']:
        st.markdown("### Suggested Action")
        st.info(result['autocorrelation']['suggested_action'])


def create_optimal_model_tab(df: pd.DataFrame, target: str, features: List[str], 
                           model_name: str, process_state: Dict) -> None:
    """Create optimal model selection tab."""
    st.subheader("Optimal Model Selection")
    st.info("""
    This process automatically evaluates various model configurations to find the optimal
    approach based on the data characteristics, including stationarity, multicollinearity,
    and autocorrelation.
    """)
    
    # Run optimal model selection button
    run_key = f"{model_name}_optimal_run"
    result_key = f"{model_name}_optimal_result"
    
    if 'optimal_process' not in process_state:
        st.info("Optimal model selection will start automatically after model training.")
        return

    process_id = process_state['optimal_process']
    display_process_ui(
        process_id=process_id,
        title="Optimal Model Selection Status",
        on_complete_callback=display_optimal_model_results,
        result_key=result_key,
        refresh_interval=1
    )
    
    # Display existing results if available
    if result_key in st.session_state:
        display_optimal_model_results(st.session_state[result_key])


def display_optimal_model_results(result: Dict[str, Any]) -> None:
    """Display optimal model selection results."""
    if not result:
        return
    
    st.success("✅ Optimal Model Selection Completed")
    
    # Display suggested model
    if 'suggested_model' in result:
        suggested_model = result['suggested_model']
        st.markdown(f"### Suggested Model: **{suggested_model}**")
        
        # Model description
        model_descriptions = {
            'base_ols': "Standard OLS model - no issues detected that require special handling",
            'lagged_dependent': "OLS with lagged dependent variable - addresses autocorrelation",
            'newey_west': "OLS with Newey-West robust standard errors - addresses autocorrelation",
            'ridge': "Ridge Regression - addresses multicollinearity by penalizing large coefficients",
            'pca_regression': "Principal Component Regression - addresses severe multicollinearity through dimensionality reduction"
        }
        
        st.info(model_descriptions.get(suggested_model, "Custom model configuration"))

        # Add a button to apply the optimal model
        if st.button(f"🚀 Apply {suggested_model} Model", key=f"apply_optimal_model_button_{model_name}"):
            # Store the optimal model suggestions in session state
            st.session_state['optimal_model_applied'] = True
            st.session_state['applied_optimal_model_name'] = suggested_model
            
            # Retrieve optimized feature set from multicollinearity analysis if available
            multicol_result_key = f"{model_name}_multicol_result"
            if multicol_result_key in st.session_state:
                multicol_results = st.session_state[multicol_result_key]
                if 'vif_analysis' in multicol_results and 'optimized_feature_set' in multicol_results['vif_analysis']:
                    st.session_state['applied_optimal_features'] = multicol_results['vif_analysis']['optimized_feature_set']
                else:
                    st.session_state['applied_optimal_features'] = features # Fallback to original if no optimization
            else:
                st.session_state['applied_optimal_features'] = features # Fallback to original if no multicol analysis yet
            
            # Store optimal model result for comparison
            if result.get('best_model') and result.get('models', {}).get(result['best_model']):
                best_model_data = result['models'][result['best_model']]
                st.session_state['optimal_model_data'] = best_model_data
                st.session_state['optimal_model_features'] = features  # Store the features used by the optimal model
                st.info(f"Optimal model data stored for comparison: {result['best_model']}")
            
            # Trigger a rerun to apply the new model
            st.experimental_rerun()

    # Display model comparison
    if 'model_comparison' in result:
        st.markdown("### Model Comparison")
        
        comparison = result['model_comparison']
        if comparison:
            # Convert to DataFrame
            comparison_df = pd.DataFrame(comparison)
            
            # Handle cases with missing metrics
            for col in ['r_squared', 'adj_r_squared', 'aic']:
                if col not in comparison_df.columns:
                    comparison_df[col] = np.nan
            
            # Format the DataFrame
            st.dataframe(
                comparison_df[['model_name', 'model_type', 'r_squared', 'adj_r_squared', 'aic']],
                use_container_width=True
            )
    
    # Display feature importance
    if 'feature_importance' in result and result['feature_importance']:
        st.markdown("### Feature Importance")
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(result['feature_importance'].keys()),
            'Importance': list(result['feature_importance'].values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Display as bar chart
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Display best model details
    if 'best_model' in result and result['best_model'] in result.get('models', {}):
        best_model_key = result['best_model']
        best_model = result['models'][best_model_key]
        
        st.markdown("### Best Model Details")
        
        # Display coefficients if available
        if 'coefficients' in best_model:
            st.markdown("#### Coefficients")
            
            # Convert to DataFrame
            coef_df = pd.DataFrame({
                'Variable': list(best_model['coefficients'].keys()),
                'Coefficient': list(best_model['coefficients'].values())
            })
            
            # Add p-values if available
            if 'p_values' in best_model:
                p_values = best_model['p_values']
                coef_df['p-value'] = [p_values.get(var, np.nan) for var in coef_df['Variable']]
                coef_df['Significance'] = coef_df['p-value'].apply(
                    lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                )
            
            # Sort by absolute coefficient value
            coef_df['abs_coef'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns=['abs_coef'])
            
            st.dataframe(coef_df, use_container_width=True)


def add_mlr_enhancements_to_ui(tab_container, df: pd.DataFrame, target: str, features: List[str], model_name: str = "MLR"):
    """
    Add MLR enhancements to existing UI tab.
    
    Args:
        tab_container: Streamlit tab container
        df: DataFrame with target and features
        target: Target column name
        features: Feature column names
        model_name: Model name for UI
    """
    with tab_container:
        create_mlr_enhancement_ui(df, target, features, model_name)
