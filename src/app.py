"""
Main Streamlit application for the Ad Impact Modeling Dashboard.
Contains the UI logic and orchestrates the modular components.
"""

import logging
from typing import List

import pandas as pd
import streamlit as st

from config import MODEL_CONFIGS, MODEL_TABLE, MISSING_DATA_METHODS
from data_utils import (
    detect_column_type,
    enforce_feature_limits,
    handle_missing_data,
    load_uploaded_files,
    validate_data_for_training,
)
from model_training import TRAIN_FUNCTIONS, save_model_and_predictions
from visualization import display_descriptive_stats, display_model_metrics, display_interpretation_hints

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
        st.success(f"✅ Data prepared: {len(df_processed)} observations ready for {model_name} training")
        
        if len(df_processed) < 5:
            st.error("❌ Insufficient data for training. Need at least 5 observations after processing.")
        else:
            st.info(f"📈 Dataset ready with {len(feature_vars)} features and {len(df_processed)} observations")
    else:
        st.success("✅ No missing data found in selected columns!")
        st.session_state[f"df_processed_{model_name}"] = df.copy()
    
    return st.session_state[f"df_processed_{model_name}"]


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
            
            st.success(f"✅ {model_name} training complete!")
            st.plotly_chart(fig, use_container_width=True)
            
            # Download buttons
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model (.pkl)",
                    data=f,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream",
                    key=f"download_model_{model_name}",
                )
            
            with open(pred_path, "rb") as f:
                st.download_button(
                    label="📊 Download Predictions (.csv)",
                    data=f,
                    file_name=f"{model_name}_predictions.csv",
                    mime="text/csv",
                    key=f"download_pred_{model_name}",
                )
            
            # Display model metrics and interpretation hints
            display_model_metrics(
                model_name,
                df=df,
                target=target,
                features=features,
                model=model,
                predictions=predictions,
            )
            
            display_interpretation_hints(model_name)
            
            logger.info(f"{model_name} model trained and displayed successfully")
            
    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("Please check your data and feature selection.")


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Ad Impact Modeling Dashboard", layout="wide")
    st.title("Ad Impact Modeling Dashboard")
    st.markdown("Upload your supermarket foot-traffic datasets and build attribution models.")

    # Centralized file uploader
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

    uploaded_files = st.file_uploader(
        "Upload Data Files (CSV, XLS, XLSX)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Drag and drop multiple files or click to select. Data will be available across all tabs.",
        key="global_uploader",
    )

    if uploaded_files:
        st.session_state['df'] = load_uploaded_files(uploaded_files)

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
                    
                    st.success(f"✅ Selected {len(df_filtered)} observations out of {len(df)} total.")

                    # Show descriptive statistics
                    if feature_vars:
                        display_descriptive_stats(df_filtered, range_col, target_var, feature_vars)

                    # Handle missing data
                    if len(feature_vars) > 0:
                        df_for_training = handle_missing_data_ui(df_filtered, target_var, feature_vars, model_name)
                        
                        # Validate data for training
                        if not validate_data_for_training(df_for_training, target_var, feature_vars):
                            st.stop()
                        
                        # Show Apply button only if data is ready
                        if len(df_for_training) >= 5:
                            if st.button("🚀 Apply & Train Model", key=f"apply_{model_name}", type="primary"):
                                train_model_safely(model_name, df_for_training, range_col, target_var, feature_vars)
                        else:
                            st.info("👆 Please select feature variables to proceed with training.")
                            
                except Exception as e:
                    logger.error(f"Error in {model_name} tab: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.info("👆 Please upload data files using the uploader at the top of the page.")


if __name__ == "__main__":
    main() 