"""
Main Streamlit application for the Ad Impact Modeling Dashboard.
Contains the UI logic and orchestrates the modular components.
"""

import logging
import sys
import os
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

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
    suggest_model_data_requirements,
    create_treated_column_by_location,
    create_post_column_by_date,
    create_post_column_by_period,
    prepare_did_data,
    prepare_synthetic_control_data
)
from model_training import TRAIN_FUNCTIONS, save_model_and_predictions
from visualization import display_descriptive_stats, display_model_metrics, display_interpretation_hints, collect_model_report_data

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
        st.warning(f"⚠️ Missing data found: {missing_data[missing_data > 0].to_dict()}")
        
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
            st.info(f"📈 Dataset ready with {len(feature_vars)} features and {len(df_processed)} observations")
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
    st.info("**Model Requirements:**")
    for model, suggestion in suggestions.items():
        st.markdown(f"• **{model}**: {suggestion}")
    
    # Check if any models need preparation
    needs_prep = any("📋" in sugg or "" in sugg for sugg in suggestions.values())
    
    if not needs_prep:
        st.success("✅ Your dataset is ready for all models!")
        return df
    
    st.markdown("---")
    
    # Data preparation options
    prep_option = st.selectbox(
        "What would you like to prepare?",
        [
            "None",
            "Create 'treated' column (for DiD, Synthetic Control, PSM)",
            "Create 'post' column (for DiD)",
            "Prepare complete DiD dataset",
            "Prepare Synthetic Control dataset"
        ],
        help="Select what type of data preparation you need"
    )
    
    df_modified = df.copy()
    
    if prep_option != "None":
        st.markdown("### Data Preparation Settings")
        
        if prep_option == "Create 'treated' column (for DiD, Synthetic Control, PSM)":
            # Location-based treated column
            location_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['city', 'location', 'region', 'state', 'country', 'store', 'unit'])]
            
            if location_cols:
                location_col = st.selectbox("Select location/unit column:", location_cols)
                unique_locations = sorted(df[location_col].unique())
                treated_locations = st.multiselect(
                    "Select treated locations/units:",
                    unique_locations,
                    help="Choose which locations should be marked as treated (1)"
                )
                
                if treated_locations and st.button("Create 'treated' column"):
                    df_modified = create_treated_column_by_location(df_modified, location_col, treated_locations)
                    st.success(f"✅ Created 'treated' column based on {len(treated_locations)} treated locations")
                    
                    # Show preview
                    preview = df_modified.groupby([location_col, 'treated']).size().reset_index(name='count')
                    st.dataframe(preview)
            else:
                st.warning("No location/unit columns detected. You may need to upload data with geographic identifiers.")
        
        elif prep_option == "Create 'post' column (for DiD)":
            # Time-based post column
            time_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'time', 'week', 'month', 'year', 'period'])]
            
            if time_cols:
                time_col = st.selectbox("Select time column:", time_cols)
                
                # Check if it's a date column or numeric
                if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    # Date-based cutoff
                    min_date = df[time_col].min()
                    max_date = df[time_col].max()
                    cutoff_date = st.date_input(
                        "Select cutoff date (observations after this will be post=1):",
                        value=min_date + (max_date - min_date) / 2,
                        min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                        max_value=max_date.date() if hasattr(max_date, 'date') else max_date
                    )
                    
                    if st.button("Create 'post' column"):
                        df_modified = create_post_column_by_date(df_modified, time_col, cutoff_date)
                        st.success(f"✅ Created 'post' column with cutoff {cutoff_date}")
                else:
                    # Numeric period cutoff
                    min_val = df[time_col].min()
                    max_val = df[time_col].max()
                    cutoff_val = st.number_input(
                        f"Select cutoff value for {time_col} (observations after this will be post=1):",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(min_val + (max_val - min_val) / 2)
                    )
                    
                    if st.button("Create 'post' column"):
                        df_modified = create_post_column_by_period(df_modified, time_col, cutoff_val)
                        st.success(f"✅ Created 'post' column with cutoff {cutoff_val}")
            else:
                st.warning("No time columns detected. You may need a date, week, month, or year column.")
        
        elif prep_option == "Prepare complete DiD dataset":
            st.info("This will create both 'treated' and 'post' columns for Difference-in-Differences analysis.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Treatment Assignment:**")
                location_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['city', 'location', 'region', 'state', 'country', 'store', 'unit'])]
                
                if location_cols:
                    location_col = st.selectbox("Location column:", location_cols, key="did_location")
                    unique_locations = sorted(df[location_col].unique())
                    treated_locations = st.multiselect(
                        "Treated locations:",
                        unique_locations,
                        key="did_treated"
                    )
                else:
                    st.error("Need a location/unit column")
                    return df
            
            with col2:
                st.markdown("**Time Cutoff:**")
                time_cols = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['date', 'time', 'week', 'month', 'year', 'period'])]
                
                if time_cols:
                    time_col = st.selectbox("Time column:", time_cols, key="did_time")
                    
                    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        min_date = df[time_col].min()
                        max_date = df[time_col].max()
                        cutoff_date = st.date_input(
                            "Cutoff date:",
                            value=min_date + (max_date - min_date) / 2,
                            key="did_cutoff"
                        )
                    else:
                        min_val = df[time_col].min()
                        max_val = df[time_col].max()
                        cutoff_date = st.number_input(
                            "Cutoff value:",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val + (max_val - min_val) / 2),
                            key="did_cutoff_num"
                        )
                else:
                    st.error("Need a time column")
                    return df
            
            if treated_locations and st.button("Prepare DiD Dataset", type="primary"):
                df_modified = prepare_did_data(df_modified, location_col, treated_locations, time_col, cutoff_date)
                st.success("✅ DiD dataset prepared successfully!")
                
                # Show 2x2 table
                crosstab = pd.crosstab(df_modified['treated'], df_modified['post'], margins=True)
                st.markdown("**DiD 2x2 Design:**")
                st.dataframe(crosstab)
        
        elif prep_option == "Prepare Synthetic Control dataset":
            st.info("This will create a 'treated' column for Synthetic Control analysis (1 treated unit, rest as controls).")
            
            location_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['city', 'location', 'region', 'state', 'country', 'store', 'unit'])]
            
            if location_cols:
                location_col = st.selectbox("Location column:", location_cols, key="sc_location")
                unique_locations = sorted(df[location_col].unique())
                treated_location = st.selectbox(
                    "Select the treated unit (only one):",
                    unique_locations,
                    key="sc_treated"
                )
                
                if st.button("Prepare Synthetic Control Dataset", type="primary"):
                    df_modified = prepare_synthetic_control_data(df_modified, location_col, treated_location)
                    st.success("✅ Synthetic Control dataset prepared successfully!")
                    
                    # Show summary
                    summary = df_modified.groupby([location_col, 'treated']).size().reset_index(name='observations')
                    st.dataframe(summary)
            else:
                st.error("Need a location/unit column for Synthetic Control")
    
    return df_modified


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
                    
                    json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📋 Download JSON Report",
                        data=json_str,
                        file_name=f"{model_name}_comprehensive_report.json",
                        mime="application/json",
                        key=f"download_report_{model_name}",
                        help="Download comprehensive analysis report with all model diagnostics, dataset overview, and results"
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating JSON report: {str(e)}")
                    st.error(f"Could not generate JSON report: {str(e)}")
            
            # Display model metrics and interpretation hints
            display_model_metrics(
                model_name,
                df=df,
                target=target,
                features=features,
                model=model,
                predictions=predictions,
            )
            
            # 🔍 ADD MULTICOLLINEARITY ANALYSIS AFTER MODEL TRAINING
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
                
                # Run multicollinearity analysis
                add_multicollinearity_analysis(
                    df=df,
                    target_column=target,
                    feature_columns=features,
                    model_name=model_name,
                    model_results=model_results
                )
                
            except ImportError as e:
                st.warning(f"⚠️ Multicollinearity analysis not available: {e}")
                st.info("To enable correlation and overfitting analysis, ensure all analysis modules are properly installed.")
            except Exception as e:
                st.error(f" Multicollinearity analysis failed: {e}")
                st.info("Debug this issue by running the analysis manually or checking the debug panel.")
            
            display_interpretation_hints(model_name)
            
            # Information about the JSON report
            with st.expander("ℹ️ About the JSON Report", expanded=False):
                st.markdown("""
                **The comprehensive JSON report includes:**
                
                 **Dataset Overview**: Total observations, date coverage, missing data summary, descriptive statistics, outlier analysis
                
                🔍 **Comprehensive Dataset Analysis**: Detailed column-by-column analysis including:
                - Numeric columns: mean, median, std, skewness, kurtosis, quartiles, outliers, value distributions
                - Categorical columns: unique values, most frequent values, string length statistics
                - Datetime columns: date ranges, unique dates, temporal coverage
                - Data quality metrics: completeness, missing data patterns, duplicate detection
                
                🎯 **Model Configuration**: Selected columns (range, target, features), model parameters, data filtering details
                
                📈 **Model Diagnostics**: R², F-statistics, coefficients, p-values, confidence intervals, model-specific metrics
                
                🔍 **Variance Inflation Factor (VIF)**: Multicollinearity detection for each feature
                
                📉 **Residual Diagnostics**: Residual statistics, fitted vs residual values for model validation
                
                🏷️ **Channel Contributions**: Individual feature contributions, time series data, coefficient impacts
                
                📋 **Model Predictions**: Full prediction dataset with timestamps and values
                
                💡 **Interpretation Hints**: Guidelines for understanding and interpreting the model results
                
                This comprehensive report provides enterprise-grade analysis suitable for research, compliance, sharing with stakeholders, or documentation purposes.
                """)
            
            
            logger.info(f"{model_name} model trained and displayed successfully")
            
    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("Please check your data and feature selection.")


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
        with st.expander("🔧 Data Preparation for Advanced Models", expanded=False):
            st.session_state['df'] = create_data_preparation_section(st.session_state['df'])

    # Create tabs for each model
    model_names = [m["name"] for m in MODEL_TABLE]
    tabs = st.tabs(model_names)

    for tab, model_name in zip(tabs, model_names):
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

                    # Feature limit validation
                    if not enforce_feature_limits(model_name, feature_vars):
                        st.stop()

                    # Range selection
                    df_filtered = create_range_selector(df, range_col, model_name)
                    
                    # Validate filtered data
                    if df_filtered is None or df_filtered.empty:
                        st.error("No data found in the selected range. Please adjust your selection.")
                        st.stop()
                    
                    st.success(f" Selected {len(df_filtered)} observations out of {len(df)} total.")

                    # Show descriptive statistics
                    if feature_vars:
                        display_descriptive_stats(df_filtered, range_col, target_var, feature_vars)

                    # Handle missing data
                    if len(feature_vars) > 0:
                        df_for_training = handle_missing_data_ui(df_filtered, target_var, feature_vars, model_name)
                        
                        # Validate data for training (now with model-specific checks)
                        if not validate_data_for_training(df_for_training, target_var, feature_vars, model_name):
                            st.stop()
                        
                        # Show Apply button only if data is ready
                        if len(df_for_training) >= 5:
                            if st.button(" Apply & Train Model", key=f"apply_{model_name}", type="primary"):
                                train_model_safely(model_name, df_for_training, range_col, target_var, feature_vars)
                        else:
                            st.info(" Please select feature variables to proceed with training.")
                            
                except Exception as e:
                    logger.error(f"Error in {model_name} tab: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.info(" Please upload data files using the uploader at the top of the page.")


if __name__ == "__main__":
    main() 