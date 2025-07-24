"""
Main Streamlit application for the Ad Impact Modeling Dashboard.
Contains the UI logic and orchestrates the modular components.
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Any, Dict
import tempfile
import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import numpy as np

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import (
    MODEL_TABLE,
    MODEL_CONFIGS,
    MISSING_DATA_METHODS,
    SUPPORTED_FILE_TYPES,
    MIN_OBSERVATIONS_FOR_TRAINING,
)
from data_utils import (
    detect_column_type,
    enforce_feature_limits,
    handle_missing_data,
    load_uploaded_files,
    validate_data_for_training,
    suggest_model_data_requirements
)
from model_training import TRAIN_FUNCTIONS, save_model_and_predictions
from visualization import display_descriptive_stats, display_model_metrics, display_interpretation_hints, collect_model_report_data
from combine_reports import combine_uploaded_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_range_selector(df: pd.DataFrame, range_col: str, model_name: str) -> pd.DataFrame:
    """
    Create range selector UI and return filtered DataFrame.
    
    Args:
        df: Input DataFrame
        range_col: Range column name
        model_name: Model name for unique keys
        
    Returns:
        Filtered DataFrame
    """
    col_data = df[range_col].dropna()
    df_filtered = df.copy()
    
    # Detect column type
    col_type = detect_column_type(df, range_col)
    
    if col_type == 'datetime':
        # Date range selector
        df[range_col] = pd.to_datetime(df[range_col], errors='coerce')
        min_val, max_val = df[range_col].min(), df[range_col].max()
        
        start_val, end_val = st.date_input(
            f"Date Range ({range_col})",
            value=(min_val.date(), max_val.date()),
            min_value=min_val.date(),
            max_value=max_val.date(),
            key=f"range_picker_{model_name}",
        )
        
        start_val = pd.to_datetime(start_val)
        end_val = pd.to_datetime(end_val)
        df_filtered = df[(df[range_col] >= start_val) & (df[range_col] <= end_val)]
        st.info(f"Date range selected: {start_val.date()} to {end_val.date()}")
        
    elif col_type in ['numeric', 'integer']:
        # Numeric range selector
        if col_type == 'integer':
            min_val, max_val = int(col_data.min()), int(col_data.max())
            step = 1
        else:
            min_val, max_val = float(col_data.min()), float(col_data.max())
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
        
        st.info(f"Column '{range_col}' range: {min_val} to {max_val}")
        
        start_val, end_val = st.slider(
            f"Select Range ({range_col})",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step,
            key=f"range_picker_{model_name}",
        )
        
        df[range_col] = pd.to_numeric(df[range_col], errors='coerce')
        df_filtered = df[(df[range_col] >= start_val) & (df[range_col] <= end_val)]
        st.info(f"Numeric range selected: {start_val} to {end_val}")
        
    else:
        # Categorical selector
        unique_vals = sorted([str(x) for x in col_data.unique() if pd.notna(x)])
        st.info(f"Column '{range_col}' has {len(unique_vals)} unique values")
        
        selection_method = st.radio(
            "How would you like to select data?",
            options=["Select specific values", "Select by row range"],
            key=f"selection_method_{model_name}",
            help="Choose between selecting specific categorical values or using row indices"
        )
        
        if selection_method == "Select specific values":
            selected_vals = st.multiselect(
                f"Select values ({range_col})",
                options=unique_vals,
                default=unique_vals[:min(10, len(unique_vals))],
                key=f"range_picker_vals_{model_name}",
            )
            if selected_vals:
                df_filtered = df[df[range_col].astype(str).isin(selected_vals)]
                st.info(f"Selected {len(selected_vals)} values: {selected_vals[:3]}{'...' if len(selected_vals) > 3 else ''}")
            else:
                df_filtered = pd.DataFrame()
        else:
            # Row-based selection
            max_rows = len(df) - 1
            st.info(f"Dataset has {len(df)} rows (indices 0 to {max_rows})")
            
            start_row, end_row = st.slider(
                "Select row range (by index)",
                min_value=0,
                max_value=max_rows,
                value=(0, min(1000, max_rows)),
                key=f"range_picker_rows_{model_name}",
                help=f"Select which rows to include by their position (0 to {max_rows})"
            )
            df_filtered = df.iloc[start_row:end_row+1].copy()
            st.info(f"Selected rows {start_row} to {end_row} ({len(df_filtered)} observations)")
    
    return df_filtered


def handle_missing_data_ui(df: pd.DataFrame, target_var: str, feature_vars: List[str], model_name: str) -> pd.DataFrame:
    """
    Handle missing data UI and processing.
    
    Args:
        df: Input DataFrame
        target_var: Target variable name
        feature_vars: Feature variable names
        model_name: Model name for unique keys
        
    Returns:
        Processed DataFrame
    """
    required_cols = [target_var] + feature_vars
    missing_data = df[required_cols].isnull().sum()
    
    if missing_data.sum() > 0:
        st.warning(f" Missing data found: {missing_data[missing_data > 0].to_dict()}")
        
        missing_method = st.selectbox(
            "How to handle missing data?",
            options=MISSING_DATA_METHODS,
            key=f"missing_method_{model_name}",
            help="Choose how to handle missing values in your dataset"
        )
        
        df_processed = handle_missing_data(df, missing_method, required_cols)
        
        # Final validation
        final_missing = df_processed[required_cols].isnull().sum().sum()
        if final_missing > 0:
            st.warning(f"⚠️ {final_missing} missing values still remain. These will be removed during training.")
        
        st.session_state[f"df_processed_{model_name}"] = df_processed
        st.success(f" Data prepared: {len(df_processed)} observations ready for {model_name} training")
        
        if len(df_processed) < 5:
            st.error(" Insufficient data for training. Need at least 5 observations after processing.")
        else:
            st.info(f" Dataset ready with {len(feature_vars)} features and {len(df_processed)} observations")
    else:
        st.success(" No missing data found in selected columns!")
        st.session_state[f"df_processed_{model_name}"] = df.copy()
    
    return st.session_state[f"df_processed_{model_name}"]


def create_data_preparation_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a data preparation section for advanced models.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Modified DataFrame with new columns if created
    """
    
    st.subheader("Data Preparation for Advanced Models")
    
    # Show suggestions
    suggestions = suggest_model_data_requirements(df)
    
    # Check which models need preparation
    needs_prep = any("Need" in sugg for sugg in suggestions.values())
    ready_models = [model for model, sugg in suggestions.items() if "Ready" in sugg]
    prep_models = [model for model, sugg in suggestions.items() if "Need" in sugg]
    
    if not needs_prep:
        st.success(" **Great! Your dataset is ready for all models!**")
        st.info("You can proceed directly to model training in any tab.")
        return df
    
    # Show status overview
    col1, col2 = st.columns(2)
    with col1:
        st.success(f" **Ready Models** ({len(ready_models)}):")
        for model in ready_models:
            st.write(f"• {model}")
    
    with col2:
        st.warning(f"**Need Preparation** ({len(prep_models)}):")
        for model in prep_models:
            st.write(f"• {model}")
    
    # Detailed requirements
    with st.expander("View Detailed Model Requirements", expanded=False):
        for model, suggestion in suggestions.items():
            st.markdown(f"**{model}**: {suggestion}")
    
    st.markdown("---")
    
    df_modified = df.copy()
    st.info("Data preparation completed")
    return df_modified


def integrate_comprehensive_multicollinearity_analysis(
    df: pd.DataFrame, 
    target: str, 
    features: List[str], 
    model_name: str,
    model: Any,
    predictions: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run comprehensive multicollinearity analysis and return enhanced metrics for integration.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        model_name: Model name
        model: Trained model object
        predictions: Model predictions
        
    Returns:
        Dictionary containing comprehensive multicollinearity analysis results
    """
    try:
        import tempfile
        import os
        from multicollinearity_analysis import MulticollinearityAnalyzer
        
        logger.info(f"Running comprehensive multicollinearity analysis for {model_name}")
        
        # Create temporary data file for analysis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Include target and features in the analysis dataset
            analysis_columns = [target] + features
            analysis_df = df[analysis_columns].copy()
            analysis_df.to_csv(temp_file.name, index=False)
            temp_data_path = temp_file.name
        
        # Initialize multicollinearity analyzer
        analyzer = MulticollinearityAnalyzer(
            target_column=target,
            correlation_threshold=0.8,
            vif_threshold=10.0,
            test_size=0.2,
            random_state=42
        )
        
        # Run comprehensive analysis
        analyzer.load_data(temp_data_path)
        analyzer.calculate_correlation_matrix()
        analyzer.identify_high_correlations()
        analyzer.calculate_vif()
        reduction_suggestions = analyzer.generate_reduction_suggestions()
        
        # Run baseline models for comparison
        baseline_results = analyzer.run_baseline_models(use_reduced_features=True)
        
        # Detect overfitting
        overfitting_alerts = analyzer.detect_overfitting()
        
        # Generate comprehensive report
        comprehensive_report = analyzer.generate_comprehensive_report()
        
        # Create integration results combining main model with multicollinearity analysis
        integration_results = {
            "multicollinearity_analysis_complete": True,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "current_model_analysis": {
                "model_name": model_name,
                "model_object_type": str(type(model).__name__),
                "features_used": features,
                "target_variable": target
            },
            "correlation_analysis_detailed": comprehensive_report.get('correlation_analysis', {}),
            "vif_analysis_detailed": comprehensive_report.get('vif_analysis', {}),
            "variable_reduction_recommendations": comprehensive_report.get('variable_reduction', {}),
            "baseline_model_comparison": {
                "baseline_results": baseline_results,
                "comparison_with_current_model": _compare_current_model_with_baselines(
                    model, df, target, features, model_name, baseline_results
                ),
                "performance_summary": comprehensive_report.get('baseline_models', {}).get('performance_summary', {})
            },
            "overfitting_analysis_detailed": {
                "overfitting_alerts": overfitting_alerts,
                "overall_risk": comprehensive_report.get('overfitting_analysis', {}).get('overall_overfitting_risk', 'unknown'),
                "risk_factors": _extract_risk_factors(overfitting_alerts)
            },
            "multicollinearity_severity": {
                "high_correlation_pairs": len(analyzer.high_corr_pairs),
                "high_vif_variables": int(analyzer.vif_results['High_Multicollinearity'].sum()) if analyzer.vif_results is not None else 0,
                "reduction_needed": len(analyzer.suggested_removals),
                "severity_level": _assess_severity_level(analyzer.high_corr_pairs, analyzer.vif_results, analyzer.suggested_removals)
            },
            "actionable_recommendations": comprehensive_report.get('recommendations', []),
            "feature_optimization": {
                "original_features": len(features),
                "recommended_features": len(analyzer.reduced_features),
                "features_to_remove": analyzer.suggested_removals,
                "optimized_feature_set": analyzer.reduced_features,
                "reduction_impact": {
                    "percentage_reduction": round((len(analyzer.suggested_removals) / len(features)) * 100, 1),
                    "expected_performance_impact": "Likely improvement in model stability and generalization"
                }
            },
            "comprehensive_model_metrics": _extract_comprehensive_metrics(baseline_results, model, df, target, features, model_name)
        }
        
        # Cleanup temporary file
        try:
            os.unlink(temp_data_path)
        except:
            pass
        
        # Ensure all numpy types are converted to Python types
        def clean_integration_results(obj):
            """Clean integration results of numpy types."""
            if isinstance(obj, dict):
                return {k: clean_integration_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_integration_results(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        integration_results = clean_integration_results(integration_results)
            
        logger.info("Comprehensive multicollinearity analysis completed successfully")
        return integration_results
        
    except Exception as e:
        logger.error(f"Error in comprehensive multicollinearity analysis: {str(e)}")
        return {
            "multicollinearity_analysis_complete": False,
            "error": str(e),
            "fallback_note": "Basic multicollinearity analysis available in multicollinearity_integration section"
        }


def _compare_current_model_with_baselines(model, df, target, features, model_name, baseline_results):
    """Compare current model performance with baseline models."""
    try:
        current_model_metrics = {}
        
        # Try to get metrics for current model
        X = df[features].dropna()
        y = df.loc[X.index, target]
        
        if hasattr(model, 'predict'):
            try:
                predictions = model.predict(X)
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                current_r2 = r2_score(y, predictions)
                current_mae = mean_absolute_error(y, predictions)
                current_rmse = np.sqrt(mean_squared_error(y, predictions))
                
                current_model_metrics = {
                    "current_model_r2": float(current_r2),
                    "current_model_mae": float(current_mae),
                    "current_model_rmse": float(current_rmse)
                }
            except:
                pass
        
        # Compare with baselines
        comparison = {
            "current_model_metrics": current_model_metrics,
            "baseline_comparison": {},
            "performance_ranking": "Unable to determine"
        }
        
        if baseline_results and current_model_metrics:
            valid_baselines = {name: results for name, results in baseline_results.items() 
                             if 'error' not in results}
            
            if valid_baselines:
                current_r2 = current_model_metrics["current_model_r2"]
                baseline_r2s = {name: res.get('test_r2', 0) for name, res in valid_baselines.items()}
                
                better_than = sum(1 for r2 in baseline_r2s.values() if current_r2 > r2)
                total_baselines = len(baseline_r2s)
                
                comparison["baseline_comparison"] = {
                    "better_than_count": better_than,
                    "total_baselines": total_baselines,
                    "better_than_percentage": round((better_than / total_baselines) * 100, 1),
                    "baseline_r2_values": baseline_r2s
                }
                
                if better_than == total_baselines:
                    comparison["performance_ranking"] = "Best performing model"
                elif better_than >= total_baselines * 0.7:
                    comparison["performance_ranking"] = "Above average performance"
                elif better_than >= total_baselines * 0.3:
                    comparison["performance_ranking"] = "Average performance"
                else:
                    comparison["performance_ranking"] = "Below average performance"
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}


def _extract_risk_factors(overfitting_alerts):
    """Extract risk factors from overfitting alerts."""
    risk_factors = []
    for alert in overfitting_alerts:
        for sub_alert in alert.get('alerts', []):
            risk_factors.append({
                "model": alert.get('model', 'unknown'),
                "risk_type": sub_alert.get('type', 'unknown'),
                "severity": sub_alert.get('severity', 'unknown'),
                "message": sub_alert.get('message', '')
            })
    return risk_factors


def _assess_severity_level(high_corr_pairs, vif_results, suggested_removals):
    """Assess overall multicollinearity severity level."""
    high_corr_count = len(high_corr_pairs)
    # Safely handle VIF results and convert numpy types
    if vif_results is not None and isinstance(vif_results, pd.DataFrame) and 'High_Multicollinearity' in vif_results.columns:
        high_vif_count = int(vif_results['High_Multicollinearity'].sum())
    else:
        high_vif_count = 0
    removal_count = len(suggested_removals)
    
    if removal_count > 10 or high_corr_count > 50 or high_vif_count > 10:
        return "critical"
    elif removal_count > 5 or high_corr_count > 20 or high_vif_count > 5:
        return "high"
    elif removal_count > 2 or high_corr_count > 10 or high_vif_count > 2:
        return "medium"
    else:
        return "low"


def _extract_comprehensive_metrics(baseline_results, model, df, target, features, model_name):
    """Extract comprehensive performance metrics including baselines."""
    try:
        comprehensive_metrics = {
            "baseline_models_performance": baseline_results,
            "current_model_type": model_name,
            "analysis_scope": "comprehensive_multicollinearity_and_performance"
        }
        
        # Add current model metrics if available
        try:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if hasattr(model, 'predict'):
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                predictions = model.predict(X)
                
                comprehensive_metrics["current_model_detailed"] = {
                    "r_squared": float(r2_score(y, predictions)),
                    "mean_absolute_error": float(mean_absolute_error(y, predictions)),
                    "root_mean_squared_error": float(np.sqrt(mean_squared_error(y, predictions))),
                    "observations_used": len(y),
                    "features_used": len(features)
                }
        except Exception as metric_error:
            comprehensive_metrics["current_model_detailed"] = {"error": str(metric_error)}
        
        # Performance summary
        if baseline_results:
            valid_baselines = {name: res for name, res in baseline_results.items() if 'error' not in res}
            if valid_baselines:
                test_r2_values = [res.get('test_r2', 0) for res in valid_baselines.values()]
                mae_values = [res.get('test_mae', 0) for res in valid_baselines.values()]
                
                comprehensive_metrics["performance_benchmarks"] = {
                    "best_baseline_r2": float(max(test_r2_values)),
                    "worst_baseline_r2": float(min(test_r2_values)),
                    "average_baseline_r2": float(np.mean(test_r2_values)),
                    "best_baseline_mae": float(min(mae_values)),
                    "worst_baseline_mae": float(max(mae_values)),
                    "average_baseline_mae": float(np.mean(mae_values)),
                    "model_count": len(valid_baselines)
                }
        
        return comprehensive_metrics
        
    except Exception as e:
        return {"error": str(e)}


def train_model_safely(model_name: str, df: pd.DataFrame, date_col: str, target: str, features: List[str]):
    """
    Safely train a model with error handling and user feedback.
    
    Args:
        model_name: Name of the model to train
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
    """
    try:
        with st.spinner(f"Training {model_name} model, please wait..."):
            model_func = TRAIN_FUNCTIONS[model_name]
            model, predictions, fig = model_func(
                df.copy(),
                date_col=date_col,
                target=target,
                features=features,
            )
            
            # Save model and predictions
            model_path, pred_path = save_model_and_predictions(model, predictions, model_name)
            
            st.success(f" {model_name} training complete!")
            st.plotly_chart(fig, use_container_width=True)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(model_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Trained Model (.pkl)",
                        data=f,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream",
                        key=f"download_model_{model_name}",
                    )
            
            with col2:
                with open(pred_path, "rb") as f:
                    st.download_button(
                        label=" Download Predictions (.csv)",
                        data=f,
                        file_name=f"{model_name}_predictions.csv",
                        mime="text/csv",
                        key=f"download_pred_{model_name}",
                    )
            
            with col3:
                # Generate comprehensive JSON report
                try:
                    import json
                    
                    # Collect range selection info if available
                    range_selection = {}
                    if f"range_picker_{model_name}" in st.session_state:
                        range_selection = {
                            "range_column": date_col,
                            "selected_range": str(st.session_state[f"range_picker_{model_name}"])
                        }
                    
                    report_data = collect_model_report_data(
                        model_name=model_name,
                        df=df,
                        date_col=date_col,
                        target=target,
                        features=features,
                        model=model,
                        predictions=predictions,
                        range_selection=range_selection
                    )
                    
                    # Integrate multicollinearity analysis
                    multicollinearity_integration = integrate_comprehensive_multicollinearity_analysis(
                        df=df,
                        target=target,
                        features=features,
                        model_name=model_name,
                        model=model,
                        predictions=predictions
                    )
                    
                    # Merge multicollinearity integration into the main report
                    report_data.update(multicollinearity_integration)
                    
                    # Convert numpy types for JSON serialization
                    def convert_numpy_types_for_json(obj):
                        """Convert numpy types to JSON-serializable Python types."""
                        if isinstance(obj, dict):
                            return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types_for_json(v) for v in obj]
                        elif isinstance(obj, tuple):
                            return [convert_numpy_types_for_json(v) for v in obj]
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, pd.Timestamp):
                            return obj.isoformat()
                        elif hasattr(obj, 'isoformat'):  # Handle other datetime objects
                            return obj.isoformat()
                        elif pd.isna(obj):  # Handle NaN values
                            return None
                        elif obj is None:
                            return None
                        elif isinstance(obj, pd.Series):
                            return obj.to_dict()
                        elif isinstance(obj, pd.DataFrame):
                            return obj.to_dict('records')
                        else:
                            return obj
                    
                    # Apply type conversion to ensure JSON compatibility
                    report_data_serializable = convert_numpy_types_for_json(report_data)
                    
                    try:
                        json_str = json.dumps(report_data_serializable, indent=2, ensure_ascii=False)
                    except TypeError as json_error:
                        # If JSON serialization still fails, provide detailed error info
                        logger.error(f"JSON serialization error: {json_error}")
                        # Create a simplified report with error information
                        simplified_report = {
                            "error": "JSON serialization failed",
                            "error_details": str(json_error),
                            "model_name": model_name,
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "basic_info": {
                                "features_count": len(features),
                                "observations_count": len(df),
                                "target_variable": target
                            },
                            "note": "Complete analysis available in UI, but JSON export encountered type conversion issues"
                        }
                        json_str = json.dumps(simplified_report, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="Download JSON Report",
                        data=json_str,
                        file_name=f"{model_name}_comprehensive_report.json",
                        mime="application/json",
                        key=f"download_report_{model_name}",
                        help="Download comprehensive analysis report with all model diagnostics, dataset overview, and results"
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating JSON report: {str(e)}")
                    st.error(f"Could not generate JSON report: {str(e)}")
                    
                    # Provide debugging information
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.write("**Error details:**")
                        st.code(str(e))
                        st.write("**Possible solutions:**")
                        st.write("1. Try training a simpler model with fewer features")
                        st.write("2. Check for any non-standard data types in your dataset")
                        st.write("3. The interactive analysis above should still work normally")
                        
                        # Try to create a minimal report
                        try:
                            minimal_report = {
                                "model_name": model_name,
                                "error": str(e),
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "features": features,
                                "target": target,
                                "dataset_shape": list(df.shape)
                            }
                            minimal_json = json.dumps(minimal_report, indent=2, ensure_ascii=False)
                            
                            st.download_button(
                                label="📋 Download Minimal Report (JSON)",
                                data=minimal_json,
                                file_name=f"{model_name}_minimal_report.json",
                                mime="application/json",
                                key=f"download_minimal_report_{model_name}",
                                help="Download basic model information despite the error"
                            )
                        except:
                            st.info("📝 Report generation failed, but all analysis is visible in the UI above.")
            
            # Display model metrics and interpretation hints
            display_model_metrics(
                model_name,
                df=df,
                target=target,
                features=features,
                model=model,
                predictions=predictions,
            )
            
            # 🔍 MULTICOLLINEARITY ANALYSIS (now integrated into JSON report above)
            # Note: Comprehensive multicollinearity analysis is now integrated directly into the JSON report
            # The analysis includes: correlation analysis, VIF analysis, baseline model comparison, 
            # overfitting detection, variable importance, and actionable recommendations
            
            # Optional: Add Streamlit UI analysis for interactive exploration
            try:
                from multicollinearity_streamlit import add_multicollinearity_analysis
                
                # Create model results for analysis
                model_results = {}
                if hasattr(model, 'score') and predictions is not None:
                    try:
                        # Try to calculate R²
                        X_test = df[features].dropna()
                        y_test = df.loc[X_test.index, target]
                        if len(X_test) > 0:
                            test_score = model.score(X_test, y_test)
                            model_results['test_r2'] = test_score
                    except Exception as e:
                        logger.warning(f"Could not calculate model score: {e}")
                
                # Run interactive multicollinearity analysis for UI exploration
                add_multicollinearity_analysis(
                    df=df,
                    target_column=target,
                    feature_columns=features,
                    model_name=model_name,
                    model_results=model_results
                )
                
            except ImportError as e:
                st.info(f"Interactive multicollinearity analysis: {e}")
                st.info("📋 **Complete multicollinearity analysis has been integrated into your JSON report above.**")
            except Exception as e:
                st.warning(f"⚠️ Interactive analysis issue: {e}")
                st.success("📋 **Complete multicollinearity analysis is included in your comprehensive JSON report.**")
            
            display_interpretation_hints(model_name)
            
            # Information about the JSON report
            with st.expander("About the Enhanced Comprehensive JSON Report", expanded=False):
                st.markdown("""
                **The enhanced comprehensive JSON report now includes:**
                
                ### Dataset Analysis
                - **Dataset Overview**: Total observations, date coverage, missing data summary, descriptive statistics, outlier analysis
                - **Comprehensive Dataset Analysis**: Detailed column-by-column analysis including:
                  - Numeric columns: mean, median, std, skewness, kurtosis, quartiles, outliers, value distributions
                  - Categorical columns: unique values, most frequent values, string length statistics
                  - Datetime columns: date ranges, unique dates, temporal coverage
                  - Data quality metrics: completeness, missing data patterns, duplicate detection
                
                ### Model Performance & Comparison
                - **Model Performance Metrics**: R², Adjusted R², MAE, RMSE, MAPE for your trained model
                - **Cross-Validation Results**: K-fold CV scores, model stability analysis, performance consistency
                - **Baseline Model Comparison**: Performance vs Linear Regression, Random Forest, XGBoost
                - **Performance Rankings**: How your model compares to baseline alternatives
                - **Overfitting Analysis**: Risk assessment, performance gaps, generalization metrics
                
                ### Variable Importance & Feature Analysis
                - **Variable Importance**: Model-specific importance scores, rankings, statistical significance
                - **SHAP Analysis**: Feature contributions, positive/negative impacts, importance distributions
                - **Feature Optimization**: Recommended feature sets, removal candidates, impact predictions
                
                ### Advanced Model Diagnostics
                - **Model-Specific Diagnostics**: R², F-statistics, coefficients, p-values, confidence intervals
                - **Statistical Assumptions**: Normality tests, autocorrelation, homoscedasticity checks
                - **Residual Analysis**: Standardized residuals, outlier detection, fitted vs residual plots
                
                ### Multicollinearity Analysis
                - **Correlation Analysis**: High correlation pairs, correlation matrix summary, severity assessment
                - **VIF Analysis**: Variance inflation factors, multicollinearity levels, removal recommendations
                - **Variable Reduction**: Optimized feature sets, reduction impact, performance implications
                
                ### Enhanced Channel Contributions
                - **Coefficient Analysis**: Individual feature impacts, elasticity measures, time series contributions
                - **Feature Statistics**: Descriptive stats for each feature, contribution ranges
                - **Economic Interpretation**: Positive/negative drivers, coefficient significance
                
                ### Complete Model Predictions
                - **Prediction Data**: Full prediction dataset with timestamps and values
                - **Prediction Quality**: Coverage, accuracy metrics, prediction ranges
                
                ### Actionable Insights
                - **Multicollinearity Recommendations**: Specific variables to remove, expected improvements
                - **Model Optimization**: Feature selection guidance, complexity reduction suggestions
                - **Performance Enhancement**: Data quality improvements, modeling technique recommendations
                - **Risk Mitigation**: Overfitting warnings, stability concerns, validation recommendations
                
                ### Executive Summary
                - **Performance Benchmarks**: Best/worst/average baseline performance
                - **Severity Levels**: Critical, high, medium, low risk classifications
                - **Priority Actions**: Immediate, medium-term, and long-term recommendations
                
                ** This enterprise-grade analysis is suitable for:**
                - Research publications and academic work
                - Regulatory compliance and audit requirements  
                - Stakeholder presentations and business reporting
                - Model validation and risk assessment
                - Feature engineering and data science workflow optimization
                - Comparative model analysis and selection
                """)
            
            
            logger.info(f"{model_name} model trained and displayed successfully")
            
    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("Please check your data and feature selection.")


def create_combine_reports_tab():
    """Create the Combine Reports tab functionality."""
    st.header("📊 Combine Reports & Predictions")
    st.markdown("Upload your model reports and prediction files to create a comprehensive combined JSON report.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Upload Analysis Reports")
        st.markdown("Upload JSON files containing model analysis reports:")
        
        json_files = st.file_uploader(
            "Select JSON Report Files",
            type=['json'],
            accept_multiple_files=True,
            key="json_reports_uploader",
            help="Upload multicollinearity analysis files and other model reports"
        )
        
        if json_files:
            st.success(f"✅ {len(json_files)} JSON files uploaded:")
            for f in json_files:
                st.write(f"• {f.name}")
    
    with col2:
        st.subheader("📈 Upload Prediction Files")
        st.markdown("Upload CSV files containing model predictions:")
        
        prediction_files = st.file_uploader(
            "Select Prediction CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            key="predictions_uploader",
            help="Upload prediction CSV files from saved_predictions folder"
        )
        
        if prediction_files:
            st.success(f"✅ {len(prediction_files)} prediction files uploaded:")
            for f in prediction_files:
                st.write(f"• {f.name}")
    
    # Combine and process files
    if json_files or prediction_files:
        st.markdown("---")
        
        if st.button("🔄 Combine Reports", type="primary", use_container_width=True):
            with st.spinner("Combining reports and predictions..."):
                try:
                    combined_report = combine_uploaded_files(json_files or [], prediction_files or [])
                    
                    if combined_report:
                        st.success("✅ Reports combined successfully!")
                        
                        # Display summary
                        st.subheader("📊 Combined Report Summary")
                        
                        metadata = combined_report.get("combined_report_metadata", {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Models", metadata.get("total_models", 0))
                        with col2:
                            st.metric("Prediction Sets", metadata.get("total_prediction_sets", 0))
                        with col3:
                            st.metric("Report Sections", len(combined_report.get("models", {})))
                        with col4:
                            file_size_mb = len(json.dumps(combined_report).encode('utf-8')) / (1024 * 1024)
                            st.metric("File Size (MB)", f"{file_size_mb:.2f}")
                        
                        # Show included models
                        if metadata.get("combined_from"):
                            st.write("**📋 Analysis Reports Included:**")
                            for model in metadata["combined_from"]:
                                st.write(f"• {model}")
                        
                        if metadata.get("prediction_files"):
                            st.write("**📈 Prediction Files Included:**")
                            for pred in metadata["prediction_files"]:
                                st.write(f"• {pred}")
                        
                        # Store in session state for download
                        st.session_state['combined_report'] = combined_report
                        st.session_state['combined_report_generated'] = True
                        
                    else:
                        st.error("❌ Failed to combine reports. Please check your files and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error combining reports: {str(e)}")
    
    # Download section
    if st.session_state.get('combined_report_generated', False):
        st.markdown("---")
        st.subheader("📥 Download Combined Report")
        
        combined_report = st.session_state.get('combined_report', {})
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"combined_comprehensive_report_{timestamp}.json"
        
        # Create download button
        json_str = json.dumps(combined_report, indent=2, ensure_ascii=False)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Download Combined JSON",
                data=json_str.encode('utf-8'),
                file_name=filename,
                mime="application/json",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.info(f"File: {filename} | Size: {len(json_str.encode('utf-8')) / 1024:.1f} KB")
        
        # Preview section
        with st.expander("🔍 Preview Combined Report Structure", expanded=False):
            st.json({
                "combined_report_metadata": combined_report.get("combined_report_metadata", {}),
                "dataset_overview": "..." if "dataset_overview" in combined_report else "Not included",
                "models": list(combined_report.get("models", {}).keys()),
                "predictions": list(combined_report.get("predictions", {}).keys())
            })
    
    # Instructions section
    with st.expander("ℹ️ How to Use This Tab", expanded=False):
        st.markdown("""
        ### Instructions:
        
        1. **Upload Analysis Reports (JSON)**:
           - Upload multicollinearity analysis files (e.g., `multicollinearity_analysis_MLR_20250724_1540.json`)
           - Upload other model comprehensive reports in JSON format
           
        2. **Upload Prediction Files (CSV)**:
           - Upload prediction CSV files from your `saved_predictions` folder
           - Files like `MLR_predictions.csv`, `ML + SHAP_predictions.csv`, etc.
           
        3. **Combine Reports**:
           - Click "Combine Reports" to merge all uploaded files
           - Common sections (like dataset overview) appear only once
           - Model-specific sections are preserved separately
           
        4. **Download**:
           - Download the combined JSON file containing all your analysis
           - The file includes metadata, analysis results, and predictions
        
        ### What Gets Combined:
        - **Analysis Reports**: Multicollinearity analysis, model diagnostics, performance metrics
        - **Predictions**: All prediction datasets with model identification
        - **Metadata**: Information about when and how the combination was created
        """)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Impact Modeling Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Impact Modeling Dashboard")
    st.markdown("**Analyze impact using multiple econometric and ML models**")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

    # File upload section
    st.subheader("📁 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload your data files",
        type=[t.lstrip('.') for t in SUPPORTED_FILE_TYPES],
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(SUPPORTED_FILE_TYPES)}"
    )

    if uploaded_files:
        st.session_state['df'] = load_uploaded_files(uploaded_files)

    # Data preparation section (new)
    if not st.session_state['df'].empty:
        with st.expander("Data Preparation for Advanced Models", expanded=False):
            st.session_state['df'] = create_data_preparation_section(st.session_state['df'])

    # Create tabs for each model + combine reports tab
    model_names = [m["name"] for m in MODEL_TABLE]
    all_tab_names = model_names + ["📊 Combine Reports"]
    tabs = st.tabs(all_tab_names)

    # Handle model tabs
    for tab, model_name in zip(tabs[:-1], model_names):
        with tab:
            st.header(f"{model_name} Model")
            
            if not st.session_state['df'].empty:
                df = st.session_state['df']
                try:
                    # Column selection
                    range_col = st.selectbox(
                        "Select Range Column (for filtering observations)",
                        options=df.columns,
                        help="Select a column to filter the range of observations for training",
                        key=f"range_{model_name}",
                    )
                    
                    target_var = st.selectbox(
                        "Target Variable",
                        options=[c for c in df.columns if c != range_col],
                        key=f"target_{model_name}",
                    )
                    
                    feature_vars = st.multiselect(
                        "Feature Variables",
                        options=[c for c in df.columns if c not in {range_col, target_var}],
                        key=f"features_{model_name}",
                    )

                    # Feature limit validation - now non-blocking
                    feature_limits_ok = enforce_feature_limits(model_name, feature_vars)
                    
                    if feature_limits_ok:
                        # Range selection
                        try:
                            df_filtered = create_range_selector(df, range_col, model_name)
                            
                            # Validate filtered data
                            if df_filtered is not None and not df_filtered.empty:
                                st.success(f"Selected {len(df_filtered)} observations out of {len(df)} total.")

                                # Show descriptive statistics
                                if feature_vars:
                                    display_descriptive_stats(df_filtered, range_col, target_var, feature_vars)

                                    # Handle missing data
                                    df_for_training = handle_missing_data_ui(df_filtered, target_var, feature_vars, model_name)
                                    
                                    # Validate data for training (now with model-specific checks)
                                    training_data_ok = validate_data_for_training(df_for_training, target_var, feature_vars, model_name)
                                    
                                    if training_data_ok:
                                        # Show Apply button only if data is ready
                                        if len(df_for_training) >= 5:
                                            if st.button("🚀 Apply & Train Model", key=f"apply_{model_name}", type="primary"):
                                                train_model_safely(model_name, df_for_training, range_col, target_var, feature_vars)
                                        else:
                                            st.warning("Need at least 5 observations for training after data processing.")
                                    else:
                                        st.info("Please resolve data validation issues above before training.")
                                else:
                                    st.info("Please select feature variables to proceed with training.")
                            else:
                                st.error("No data found in the selected range. Please adjust your selection.")
                        except Exception as range_error:
                            st.error(f"Range selection error: {str(range_error)}")
                            st.info("Try selecting a different range column or check your data format.")
                    else:
                        st.info("Please adjust your feature selection to meet the model requirements above.")
                            
                except Exception as e:
                    logger.error(f"Error in {model_name} tab: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Try refreshing the page or checking your data format.")
            else:
                st.info("📁 Please upload data files using the uploader at the top of the page.")

    # Handle combine reports tab
    with tabs[-1]:
        create_combine_reports_tab()


if __name__ == "__main__":
    main() 