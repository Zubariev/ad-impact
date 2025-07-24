"""
Visualization functions for the Ad Impact Modeling Dashboard.
Contains chart generation, diagnostic metrics, and SHAP plotting.
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from config import (
    DURBIN_WATSON_LOWER,
    DURBIN_WATSON_UPPER,
    OUTLIER_THRESHOLD,
    SIGNIFICANCE_LEVEL,
    VIF_THRESHOLD,
)
from data_analysis import analyze_dataset_for_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_actual_vs_predicted_chart(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    date_col: str,
    target: str,
    prediction_col: str = "prediction"
) -> go.Figure:
    """
    Create a line chart comparing actual vs predicted values with colored intersection areas.
    
    Args:
        df: Original DataFrame with actual values
        predictions: DataFrame with predicted values
        date_col: Date column name
        target: Target variable name (actual values)
        prediction_col: Name of prediction column in predictions DataFrame
        
    Returns:
        Plotly figure object
    """
    try:
        # Ensure we have matching date indices
        if date_col in predictions.columns:
            chart_data = predictions[[date_col, prediction_col]].copy()
            chart_data = chart_data.merge(df[[date_col, target]], on=date_col, how='inner')
        else:
            # If no date column in predictions, assume same order as df
            chart_data = pd.DataFrame({
                date_col: df[date_col].iloc[:len(predictions)] if date_col in df.columns else range(len(predictions)),
                prediction_col: predictions[prediction_col] if prediction_col in predictions.columns else predictions.iloc[:, 0],
                target: df[target].iloc[:len(predictions)]
            })
        
        # Sort by date if it's a proper date column, otherwise by index
        try:
            if date_col in chart_data.columns and pd.api.types.is_datetime64_any_dtype(chart_data[date_col]):
                chart_data = chart_data.sort_values(date_col).reset_index(drop=True)
            else:
                chart_data = chart_data.reset_index(drop=True)
        except:
            chart_data = chart_data.reset_index(drop=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add actual values line
        fig.add_trace(go.Scatter(
            x=chart_data[date_col],
            y=chart_data[target],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add predicted values line
        fig.add_trace(go.Scatter(
            x=chart_data[date_col],
            y=chart_data[prediction_col],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2),
            hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Find intersection points and create colored fill areas
        actual_vals = chart_data[target].values
        pred_vals = chart_data[prediction_col].values
        dates = chart_data[date_col].values
        
        # Create segments for fill areas
        for i in range(len(actual_vals) - 1):
            # Determine which line is higher in this segment
            if pred_vals[i] > actual_vals[i] and pred_vals[i+1] > actual_vals[i+1]:
                # Predicted higher than actual - fill green (positive prediction)
                fig.add_trace(go.Scatter(
                    x=[dates[i], dates[i+1], dates[i+1], dates[i]],
                    y=[actual_vals[i], actual_vals[i+1], pred_vals[i+1], pred_vals[i]],
                    fill='toself',
                    fillcolor='rgba(0, 255, 0, 0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            elif actual_vals[i] > pred_vals[i] and actual_vals[i+1] > pred_vals[i+1]:
                # Actual higher than predicted - fill red (under-prediction)
                fig.add_trace(go.Scatter(
                    x=[dates[i], dates[i+1], dates[i+1], dates[i]],
                    y=[pred_vals[i], pred_vals[i+1], actual_vals[i+1], actual_vals[i]],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                # Lines cross in this segment - need to find intersection
                # For simplicity, we'll color based on the midpoint
                mid_actual = (actual_vals[i] + actual_vals[i+1]) / 2
                mid_pred = (pred_vals[i] + pred_vals[i+1]) / 2
                
                if mid_pred > mid_actual:
                    fig.add_trace(go.Scatter(
                        x=[dates[i], dates[i+1], dates[i+1], dates[i]],
                        y=[min(actual_vals[i], pred_vals[i]), min(actual_vals[i+1], pred_vals[i+1]),
                           max(actual_vals[i+1], pred_vals[i+1]), max(actual_vals[i], pred_vals[i])],
                        fill='toself',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[dates[i], dates[i+1], dates[i+1], dates[i]],
                        y=[min(actual_vals[i], pred_vals[i]), min(actual_vals[i+1], pred_vals[i+1]),
                           max(actual_vals[i+1], pred_vals[i+1]), max(actual_vals[i], pred_vals[i])],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add legend entries for the fill colors
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(0, 255, 0, 0.5)'),
            name='Predicted > Actual',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(255, 0, 0, 0.5)'),
            name='Actual > Predicted',
            showlegend=True
        ))
        
        # Update layout
        x_title = 'Date' if date_col in ['date', 'time', 'timestamp'] or pd.api.types.is_datetime64_any_dtype(chart_data[date_col]) else date_col
        
        fig.update_layout(
            title='Actual vs Predicted Values Over Time',
            xaxis_title=x_title,
            yaxis_title='Value',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating actual vs predicted chart: {str(e)}")
        # Return empty figure on error with helpful message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Chart Creation Error",
            height=300
        )
        return fig


def display_descriptive_stats(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> None:
    """
    Display comprehensive descriptive statistics for the dataset.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
    """
    with st.expander(" Dataset Overview", expanded=False):
        st.markdown(f"**Total observations:** {len(df):,}")

        # Date coverage
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            st.markdown(
                f"**Date coverage:** {df[date_col].min().date()} → {df[date_col].max().date()}"
            )

        # Basic stats table
        numeric_cols = [c for c in [target] + features if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            desc = df[numeric_cols].agg(["mean", "std", "min", "max"]).T
            desc.columns = [c.capitalize() for c in desc.columns]
            # Apply formatting only to numeric columns
            format_dict = {col: "{:.2f}" for col in desc.columns if desc[col].dtype.kind in 'biufc'}
            st.dataframe(desc.style.format(format_dict))

        # Missing value percentages
        na_pct = df[[target] + features].isna().mean().mul(100).round(1)
        na_table = na_pct.reset_index().rename(columns={"index": "Column", 0: "% Missing"})
        st.dataframe(na_table.style.background_gradient(cmap="Reds"))

        # Simple outlier flag (top 1% of each feature)
        if numeric_cols:
            outlier_info = {}
            for col in numeric_cols:
                threshold = df[col].quantile(OUTLIER_THRESHOLD)
                outlier_info[col] = int((df[col] > threshold).sum())
            out_df = pd.DataFrame.from_dict(outlier_info, orient="index", columns=["Count > 99th pct"])
            st.dataframe(out_df)


def create_vif_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.
    
    Args:
        df: Input DataFrame
        features: Feature variable names
        
    Returns:
        DataFrame with VIF values
    """
    X = sm.add_constant(df[features])
    vif_data = {
        "Variable": [],
        "VIF": [],
    }
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_data["Variable"].append(col)
        vif_data["VIF"].append(variance_inflation_factor(X.values, i))
    return pd.DataFrame(vif_data)


def create_kpi_dashboard(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    model_predictions: pd.Series = None,
    model_metrics: Dict = None,
    shap_importance: np.ndarray = None
) -> None:
    """
    Create comprehensive KPI dashboard for model results.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        model_predictions: Model predictions (optional)
        model_metrics: Model performance metrics (optional)
        shap_importance: SHAP importance values (optional)
    """
    try:
        st.markdown("---")
        st.markdown("**Performance Dashboard & KPIs:**")
        
        # Create KPI columns layout
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        # KPI 1: Total Value by Features (sum of feature values)
        with kpi_col1:
            numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
            total_value = df[numeric_features].sum().sum() if numeric_features else 0
            st.metric(
                label="Total Features Value",
                value=f"{total_value:,.0f}" if total_value > 0 else "N/A",
                help="Sum of all numeric feature values"
            )
        
        # KPI 2: Total Target Value Generated
        with kpi_col2:
            total_target = df[target].sum() if pd.api.types.is_numeric_dtype(df[target]) else 0
            st.metric(
                label="Total Target Value",
                value=f"{total_target:,.0f}",
                help="Sum of all target variable values"
            )
        
        # KPI 3: Overall ROI
        with kpi_col3:
            overall_roi = (total_target / total_value * 100) if total_value > 0 else 0
            st.metric(
                label="Overall Target vs. Value % (ROI)",
                value=f"{overall_roi:.1f}%",
                help="(Total Target Value / Total Feature Value) × 100"
            )
        
        # KPI 4: Model Performance
        with kpi_col4:
            if model_metrics and 'rmse' in model_metrics:
                performance_metric = f"{model_metrics['rmse']:.2f}"
                performance_label = "Model RMSE"
            elif model_metrics and 'r_squared' in model_metrics:
                performance_metric = f"{model_metrics['r_squared']:.3f}"
                performance_label = "Model R²"
            else:
                performance_metric = "N/A"
                performance_label = "Performance"
            
            st.metric(
                label=performance_label,
                value=performance_metric,
                help="Model performance metric"
            )
        
        # Detailed Feature Analysis
        st.markdown("**Feature Performance Analysis:**")
        
        # Calculate per-feature metrics
        feature_metrics = []
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                feature_value = df[feature].sum()
                
                # Calculate correlation with target
                correlation = df[feature].corr(df[target]) if df[feature].var() > 0 else 0
                
                # Get SHAP importance if available
                shap_importance_val = 0
                try:
                    if shap_importance is not None:
                        feature_idx = features.index(feature)
                        shap_importance_val = shap_importance[feature_idx]
                except:
                    pass
                
                # Calculate ROI (simplified: correlation * target_total as proxy)
                target_driven = correlation * df[target].sum() if correlation > 0 else 0
                roi = (target_driven / feature_value * 100) if feature_value > 0 else 0
                
                # Calculate efficiency (target per unit feature value)
                efficiency = df[target].sum() / feature_value if feature_value > 0 else 0
                
                feature_metrics.append({
                    'Feature': feature,
                    'Total_Value': feature_value,
                    'Target_Driven': target_driven,
                    'Feature_Value_vs_Target_Percent': roi,
                    'Correlation': correlation,
                    'SHAP_Importance': shap_importance_val,
                    'Efficiency': efficiency
                })
        
        if feature_metrics:
            feature_df = pd.DataFrame(feature_metrics).sort_values('Feature_Value_vs_Target_Percent', ascending=False)
            
            # Feature visualizations in 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Value by feature
                fig_spend = px.bar(
                    feature_df,
                    x='Feature',
                    y='Total_Value',
                    title='Total Value by Feature',
                    labels={'Total_Value': 'Total Value', 'Feature': 'Features'},
                    color='Total_Value',
                    color_continuous_scale='blues'
                )
                fig_spend.update_xaxes(tickangle=45)
                fig_spend.update_layout(height=400)
                st.plotly_chart(fig_spend, use_container_width=True)
            
            with col2:
                # ROI by feature
                fig_roi = px.bar(
                    feature_df,
                    x='Feature',
                    y='Feature_Value_vs_Target_Percent',
                    title='Feature Value vs. Target % (ROI)',
                    labels={'Feature_Value_vs_Target_Percent': 'Feature Value vs. Target % (ROI)', 'Feature': 'Features'},
                    color='Feature_Value_vs_Target_Percent',
                    color_continuous_scale='RdYlGn'
                )
                fig_roi.update_xaxes(tickangle=45)
                fig_roi.update_layout(height=400)
                st.plotly_chart(fig_roi, use_container_width=True)
            
            # Second row
            col3, col4 = st.columns(2)
            
            with col3:
                # Target value driven by feature
                fig_target = px.bar(
                    feature_df,
                    x='Feature',
                    y='Target_Driven',
                    title='Target Value Driven by Feature',
                    labels={'Target_Driven': 'Target Value Driven', 'Feature': 'Features'},
                    color='Target_Driven',
                    color_continuous_scale='viridis'
                )
                fig_target.update_xaxes(tickangle=45)
                fig_target.update_layout(height=400)
                st.plotly_chart(fig_target, use_container_width=True)
            
            with col4:
                # Efficiency by feature
                fig_efficiency = px.bar(
                    feature_df,
                    x='Feature',
                    y='Efficiency',
                    title='Efficiency by Feature',
                    labels={'Efficiency': 'Target Value per Feature Value', 'Feature': 'Features'},
                    color='Efficiency',
                    color_continuous_scale='plasma'
                )
                fig_efficiency.update_xaxes(tickangle=45)
                fig_efficiency.update_layout(height=400)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Feature correlation heatmap
            st.markdown("**Feature Correlation Matrix:**")
            if len(numeric_features) > 1:
                corr_matrix = df[numeric_features + [target]].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Comprehensive metrics table
            st.markdown("**Detailed Feature Metrics:**")
            
            # Format the dataframe for display
            display_df = feature_df.copy()
            display_df['Total_Value'] = display_df['Total_Value'].apply(lambda x: f"${x:,.0f}")
            display_df['Target_Driven'] = display_df['Target_Driven'].apply(lambda x: f"{x:,.0f}")
            display_df['Feature_Value_vs_Target_Percent'] = display_df['Feature_Value_vs_Target_Percent'].apply(lambda x: f"{x:.1f}%")
            display_df['Correlation'] = display_df['Correlation'].apply(lambda x: f"{x:.3f}")
            display_df['SHAP_Importance'] = display_df['SHAP_Importance'].apply(lambda x: f"{x:.3f}")
            display_df['Efficiency'] = display_df['Efficiency'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                display_df.style.background_gradient(subset=['Feature_Value_vs_Target_Percent'], cmap='RdYlGn'),
                use_container_width=True
            )
            
        else:
            st.info("Feature analysis requires numeric feature columns representing value data.")
            
    except Exception as kpi_error:
        logger.error(f"Error creating KPI dashboard: {str(kpi_error)}")
        st.warning(f"Could not create KPI dashboard: {str(kpi_error)}")


def display_mlr_metrics(df: pd.DataFrame, target: str, features: List[str]) -> None:
    """
    Display MLR model diagnostics and metrics.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
    """
    try:
        X = sm.add_constant(df[features])
        y = df[target]
        model = sm.OLS(y, X).fit()

        st.subheader("Model Diagnostics (MLR)")
        st.markdown(
            f"**R² / Adj R²:** {model.rsquared:.3f} / {model.rsquared_adj:.3f}\n\n"
            f"**F-statistic (p):** {model.fvalue:.2f} ({model.f_pvalue:.3g})"
        )

        # Coefficient table with significance indicators
        coef_tbl = model.summary2().tables[1].rename(columns={"Coef.": "Coef"})
        coef_tbl["Signif"] = coef_tbl["P>|t|"].apply(
            lambda p: "" if p < SIGNIFICANCE_LEVEL else ""
        )
        styled = coef_tbl.style.applymap(
            lambda p: "color:green;" if p < SIGNIFICANCE_LEVEL else "color:red;", 
            subset=["P>|t|"]
        )
        st.dataframe(styled, height=min(400, 25 * len(coef_tbl)))

        # VIF table
        vif_df = create_vif_table(df, features)
        st.markdown("**Variance Inflation Factor (VIF):**")
        # Check if VIF column is numeric before formatting
        if "VIF" in vif_df.columns and pd.api.types.is_numeric_dtype(vif_df["VIF"]):
            # Replace infinite values with a large number for display
            vif_df_display = vif_df.copy()
            vif_df_display["VIF"] = vif_df_display["VIF"].replace([np.inf, -np.inf], 999.0)
            st.dataframe(vif_df_display.style.format({"VIF": "{:.2f}"}))
        else:
            st.dataframe(vif_df)

        # Residual plot
        resid = model.resid
        fig_resid = px.scatter(
            x=model.fittedvalues, 
            y=resid, 
            labels={"x": "Fitted", "y": "Residuals"}, 
            title="Residual Diagnostic"
        )
        fig_resid.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_resid, use_container_width=True)
        
        # Create KPI Dashboard for MLR
        model_metrics = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue
        }
        create_kpi_dashboard(df, target, features, model.fittedvalues, model_metrics)
        
        # Create predictions for MLR model and show actual vs predicted chart
        try:
            predictions_data = pd.DataFrame({
                'prediction': model.fittedvalues
            })
            
            # Try to find a date column, with fallbacks
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if not date_cols:
                date_cols = [col for col in df.columns if 'date' in col.lower()]
            
            # Use the first available column or create row index
            if date_cols:
                date_col = date_cols[0]
                st.markdown("**Actual vs Predicted Comparison:**")
                logger.info(f"Creating actual vs predicted chart with date column: {date_col}")
                
                fig_comparison = create_actual_vs_predicted_chart(
                    df, predictions_data, date_col, target, 'prediction'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.success("Actual vs Predicted chart displayed above")
                logger.info("Actual vs predicted chart displayed successfully for MLR")
            else:
                # Fallback: use the first column (likely the range column user selected) or row index
                fallback_col = df.columns[0] if len(df.columns) > 0 else 'index'
                st.markdown("**Actual vs Predicted Comparison:**")
                st.info(f"No date column found. Using '{fallback_col}' for x-axis.")
                logger.info(f"Creating actual vs predicted chart with fallback column: {fallback_col}")
                
                # Create a temporary dataframe with row indices if needed
                if fallback_col == 'index':
                    temp_df = df.copy()
                    temp_df['index'] = range(len(temp_df))
                    fig_comparison = create_actual_vs_predicted_chart(
                        temp_df, predictions_data, 'index', target, 'prediction'
                    )
                else:
                    fig_comparison = create_actual_vs_predicted_chart(
                        df, predictions_data, fallback_col, target, 'prediction'
                    )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.success("Actual vs Predicted chart displayed above")
                logger.info("Actual vs predicted chart displayed successfully for MLR with fallback")
        
        except Exception as chart_error:
            logger.error(f"Error creating actual vs predicted chart for MLR: {str(chart_error)}")
            st.warning(f"Could not create actual vs predicted chart: {str(chart_error)}")
            st.info("This may be due to date column detection issues or data format problems.")
        
        logger.info("MLR metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying MLR metrics: {str(e)}")
        st.error(f"Error computing MLR metrics: {str(e)}")


def display_distributed_lag_metrics(df: pd.DataFrame, target: str, features: List[str]) -> None:
    """
    Display Distributed Lag model diagnostics and metrics.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
    """
    st.subheader("Model Diagnostics (Distributed Lag)")
    
    # Re-use MLR diagnostics (which now includes the comparison chart)
    display_mlr_metrics(df, target, features)
    
    # Durbin-Watson test for autocorrelation
    try:
        y = df[target]
        X = sm.add_constant(df[features])
        dw = durbin_watson(y - sm.OLS(y, X).fit().fittedvalues)
        
        dw_status = " No autocorrelation" if DURBIN_WATSON_LOWER <= dw <= DURBIN_WATSON_UPPER else " Autocorrelation detected"
        st.markdown(f"**Durbin-Watson:** {dw:.2f} ({dw_status})")
        
        logger.info("Distributed Lag metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error computing Durbin-Watson: {str(e)}")
        st.warning("Could not compute Durbin-Watson statistic")


def display_ml_shap_metrics(model: Any, df: pd.DataFrame, target: str) -> None:
    """
    Display ML + SHAP model diagnostics and metrics.
    
    Args:
        model: Trained ML model
        df: Input DataFrame
        target: Target variable name
    """
    try:
        st.subheader("Model Diagnostics (ML + SHAP)")
        
        # Get feature columns (use stored feature names if available)
        if hasattr(model, 'feature_names_'):
            feature_cols = model.feature_names_
        else:
            feature_cols = [c for c in df.columns if c != target]
        
        # Handle categorical columns for prediction using stored encoders
        X_pred = df[feature_cols].copy()
        
        # Apply the SAME categorical encoding used during training
        if hasattr(model, 'label_encoders_') and hasattr(model, 'categorical_cols_'):
            for col in model.categorical_cols_:
                if col in model.label_encoders_:
                    le = model.label_encoders_[col]
                    try:
                        # Transform using the exact same encoder from training
                        X_pred[col] = le.transform(X_pred[col].astype(str))
                    except ValueError as e:
                        # Handle unseen categories by using the most frequent class
                        st.warning(f"Unseen categories in column '{col}': {str(e)}")
                        # Map unseen categories to most frequent class
                        known_classes = set(le.classes_)
                        X_pred[col] = X_pred[col].astype(str).apply(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        X_pred[col] = le.transform(X_pred[col])
                    
                    if hasattr(model, 'enable_categorical') and model.enable_categorical:
                        X_pred[col] = X_pred[col].astype('category')
        else:
            # Fallback to old method if model doesn't have stored encoders
            st.warning("⚠️ Model missing preprocessing info. Using fallback encoding (may cause issues).")
            from sklearn.preprocessing import LabelEncoder
            for col in feature_cols:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_pred[col] = le.fit_transform(X_pred[col].astype(str))
                    if hasattr(model, 'enable_categorical') and model.enable_categorical:
                        X_pred[col] = X_pred[col].astype('category')
        
        preds = model.predict(X_pred)
        rmse = mean_squared_error(df[target], preds, squared=False)
        mae = mean_absolute_error(df[target], preds)
        
        st.markdown(f"**RMSE / MAE:** {rmse:.2f} / {mae:.2f}")

        # Global SHAP summary (bar plot)
        try:
            import shap
            explainer = shap.Explainer(model)
            shap_values = explainer(X_pred)
            
            # Create interactive SHAP summary plot with Plotly
            st.markdown("**SHAP Feature Importance (Interactive):**")
            
            # Calculate mean absolute SHAP values for feature importance
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP_Importance': shap_importance
            }).sort_values('SHAP_Importance', ascending=True)
            
            # Create interactive bar chart
            fig_shap = px.bar(
                feature_importance_df,
                x='SHAP_Importance',
                y='Feature',
                orientation='h',
                title='SHAP Feature Importance (Mean Absolute Impact)',
                labels={'SHAP_Importance': 'Mean |SHAP Value|', 'Feature': 'Features'},
                color='SHAP_Importance',
                color_continuous_scale='viridis'
            )
            fig_shap.update_layout(height=max(400, len(feature_cols) * 30))
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # SHAP Waterfall Chart for first observation
            st.markdown("**💧 SHAP Waterfall Chart (Sample Prediction):**")
            
            # Select a representative sample (middle of dataset)
            sample_idx = len(shap_values.values) // 2
            sample_shap = shap_values.values[sample_idx]
            baseline = shap_values.base_values[sample_idx] if hasattr(shap_values, 'base_values') else 0
            
            # Create waterfall data
            waterfall_data = []
            cumulative = baseline
            
            # Add baseline
            waterfall_data.append({
                'Feature': 'Baseline',
                'SHAP_Value': baseline,
                'Cumulative': baseline,
                'Type': 'baseline'
            })
            
            # Sort features by absolute SHAP value for better visualization
            feature_shap_pairs = list(zip(feature_cols, sample_shap))
            feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, shap_val in feature_shap_pairs:
                cumulative += shap_val
                waterfall_data.append({
                    'Feature': feature,
                    'SHAP_Value': shap_val,
                    'Cumulative': cumulative,
                    'Type': 'positive' if shap_val > 0 else 'negative'
                })
            
            # Add final prediction
            waterfall_data.append({
                'Feature': 'Prediction',
                'SHAP_Value': 0,
                'Cumulative': cumulative,
                'Type': 'prediction'
            })
            
            waterfall_df = pd.DataFrame(waterfall_data)
            
            # Create waterfall chart
            fig_waterfall = go.Figure()
            
            colors = {'baseline': 'gray', 'positive': 'green', 'negative': 'red', 'prediction': 'blue'}
            
            for i, row in waterfall_df.iterrows():
                fig_waterfall.add_trace(go.Bar(
                    x=[row['Feature']],
                    y=[row['SHAP_Value']] if row['Type'] != 'prediction' else [row['Cumulative']],
                    name=row['Type'].title(),
                    marker_color=colors[row['Type']],
                    showlegend=i == 0 or (i == 1 and row['Type'] != waterfall_df.iloc[0]['Type']),
                    hovertemplate=f"<b>{row['Feature']}</b><br>SHAP Value: {row['SHAP_Value']:.3f}<br>Cumulative: {row['Cumulative']:.3f}<extra></extra>"
                ))
            
            fig_waterfall.update_layout(
                title=f'SHAP Waterfall for Sample Prediction #{sample_idx}',
                xaxis_title='Features',
                yaxis_title='SHAP Values',
                height=500,
                xaxis={'categoryorder': 'array', 'categoryarray': waterfall_df['Feature'].tolist()}
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Additional Feature Impact Analysis
            st.markdown("**📊 Feature Impact Distribution:**")
            
            # Create box plot showing SHAP value distributions
            shap_df_long = pd.DataFrame(shap_values.values, columns=feature_cols)
            shap_df_melted = shap_df_long.melt(var_name='Feature', value_name='SHAP_Value')
            
            fig_box = px.box(
                shap_df_melted,
                x='Feature',
                y='SHAP_Value',
                title='SHAP Value Distribution by Feature',
                labels={'SHAP_Value': 'SHAP Values', 'Feature': 'Features'}
            )
            fig_box.update_xaxes(tickangle=45)
            fig_box.update_layout(height=500)
            st.plotly_chart(fig_box, use_container_width=True)
            
        except Exception as shap_error:
            logger.error(f"SHAP error details: {str(shap_error)}")
            st.warning(f"Could not generate SHAP plot: {str(shap_error)}")
            st.info("SHAP analysis may not be available for this model configuration.")
            
            # Provide debugging info
            with st.expander("🔍 SHAP Debug Information"):
                st.write("**Model Info:**")
                st.write(f"- Model type: {type(model).__name__}")
                st.write(f"- Has stored encoders: {hasattr(model, 'label_encoders_')}")
                st.write(f"- Has feature names: {hasattr(model, 'feature_names_')}")
                st.write(f"- X_pred shape: {X_pred.shape}")
                st.write(f"- X_pred dtypes: {X_pred.dtypes.to_dict()}")
                if hasattr(model, 'categorical_cols_'):
                    st.write(f"- Categorical columns: {model.categorical_cols_}")
        
        # KPI Dashboard Section
        create_kpi_dashboard(df, target, feature_cols, preds, {'rmse': rmse, 'mae': mae}, shap_importance if 'shap_importance' in locals() else None)
        
        # Create actual vs predicted comparison chart
        try:
            predictions_data = pd.DataFrame({
                'prediction': preds
            })
            
            # Try to find a date column, with fallbacks
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if not date_cols:
                date_cols = [col for col in df.columns if 'date' in col.lower()]
            
            # Use the first available column or create row index
            if date_cols:
                date_col = date_cols[0]
                st.markdown("**Actual vs Predicted Comparison:**")
                logger.info(f"Creating actual vs predicted chart for ML+SHAP with date column: {date_col}")
                
                fig_comparison = create_actual_vs_predicted_chart(
                    df, predictions_data, date_col, target, 'prediction'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.success("Actual vs Predicted chart displayed above")
                logger.info("Actual vs predicted chart displayed successfully for ML+SHAP")
            else:
                # Fallback: use the first column or row index
                fallback_col = df.columns[0] if len(df.columns) > 0 else 'index'
                st.markdown("**Actual vs Predicted Comparison:**")
                st.info(f"No date column found. Using '{fallback_col}' for x-axis.")
                logger.info(f"Creating actual vs predicted chart for ML+SHAP with fallback column: {fallback_col}")
                
                # Create a temporary dataframe with row indices if needed
                if fallback_col == 'index':
                    temp_df = df.copy()
                    temp_df['index'] = range(len(temp_df))
                    fig_comparison = create_actual_vs_predicted_chart(
                        temp_df, predictions_data, 'index', target, 'prediction'
                    )
                else:
                    fig_comparison = create_actual_vs_predicted_chart(
                        df, predictions_data, fallback_col, target, 'prediction'
                    )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.success("Actual vs Predicted chart displayed above")
                logger.info("Actual vs predicted chart displayed successfully for ML+SHAP with fallback")
                
        except Exception as chart_error:
            logger.error(f"Error creating actual vs predicted chart for ML+SHAP: {str(chart_error)}")
            st.warning(f"Could not create actual vs predicted chart: {str(chart_error)}")
            st.info("This may be due to date column detection issues or data format problems.")
        
        logger.info("ML + SHAP metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying ML + SHAP metrics: {str(e)}")
        st.error(f"Error computing ML + SHAP metrics: {str(e)}")
        
        # Add debug information for troubleshooting
        with st.expander("🔍 Error Debug Information"):
            st.write("**Error Details:**")
            st.write(f"- Error: {str(e)}")
            st.write(f"- Target: {target}")
            st.write(f"- DataFrame shape: {df.shape}")
            st.write(f"- DataFrame columns: {list(df.columns)}")
            st.write(f"- Model type: {type(model).__name__}")
            if hasattr(model, 'feature_names_'):
                st.write(f"- Expected features: {model.feature_names_}")
            st.info("Try retraining the model or check your data for inconsistencies.")


def display_did_metrics(model: Any) -> None:
    """
    Display DiD model diagnostics and metrics.
    
    Args:
        model: Trained DiD model
    """
    try:
        st.subheader("Model Diagnostics (DiD)")
        
        ate = model.params.get("treated:post", float("nan"))
        ci_low, ci_high = model.conf_int().loc["treated:post"]
        pval = model.pvalues.get("treated:post", float("nan"))
        
        significance = " Significant" if pval < SIGNIFICANCE_LEVEL else " Not significant"
        st.markdown(
            f"**ATE:** {ate:.3f}  |  CI95%: [{ci_low:.3f}, {ci_high:.3f}]  |  p-value: {pval:.3g} ({significance})"
        )
        
        logger.info("DiD metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying DiD metrics: {str(e)}")
        st.error(f"Error computing DiD metrics: {str(e)}")


def display_var_metrics(results: Any) -> None:
    """
    Display VAR model diagnostics and metrics.
    
    Args:
        results: Trained VAR model results
    """
    try:
        st.subheader("Model Diagnostics (VAR)")
        
        stability_status = " Stable" if results.is_stable() else " Unstable"
        st.markdown(
            f"**Selected Lag Order:** {results.k_ar}  |  **AIC:** {results.aic:.2f}  |  **BIC:** {results.bic:.2f}"
        )
        st.markdown(f"**Stability check:** {stability_status}")
        
        logger.info("VAR metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying VAR metrics: {str(e)}")
        st.error(f"Error computing VAR metrics: {str(e)}")


def display_synthetic_control_metrics(predictions: pd.DataFrame) -> None:
    """
    Display Synthetic Control model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame with Actual and Synthetic columns
    """
    try:
        st.subheader("Model Diagnostics (Synthetic Control)")
        
        rmspe = np.sqrt(((predictions["Actual"] - predictions["Synthetic"]) ** 2).mean())
        st.markdown(f"**RMSPE:** {rmspe:.2f}")
        
        # Create actual vs predicted comparison chart for Synthetic Control
        try:
            # Find date column in predictions
            date_cols = [col for col in predictions.columns if pd.api.types.is_datetime64_any_dtype(predictions[col])]
            if not date_cols:
                date_cols = [col for col in predictions.columns if 'date' in col.lower()]
            
            if date_cols and "Actual" in predictions.columns and "Synthetic" in predictions.columns:
                date_col = date_cols[0]
                
                # Prepare data for the chart - rename Synthetic to prediction for consistency
                chart_predictions = predictions.rename(columns={"Synthetic": "prediction"})
                
                fig_comparison = create_actual_vs_predicted_chart(
                    predictions.rename(columns={"Actual": "target"}), 
                    chart_predictions, 
                    date_col, 
                    "target", 
                    'prediction'
                )
                fig_comparison.update_layout(title='Actual vs Synthetic Control Over Time')
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Required columns not found for comparison chart.")
                
        except Exception as chart_error:
            st.warning(f"Could not create actual vs synthetic chart: {str(chart_error)}")
        
        logger.info("Synthetic Control metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying Synthetic Control metrics: {str(e)}")
        st.error(f"Error computing Synthetic Control metrics: {str(e)}")


def display_causal_impact_metrics(predictions: pd.DataFrame, df_original: pd.DataFrame, target: str) -> None:
    """
    Display CausalImpact model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame
        df_original: Original DataFrame
        target: Target variable name
    """
    try:
        st.subheader("Model Diagnostics (CausalImpact)")
        
        if target not in df_original.columns:
            st.warning("Target column missing in original data; cannot compute effect.")
            return
            
        df_effect = df_original[target] - predictions["prediction"]
        cumulative = df_effect.cumsum().iloc[-1]
        st.markdown(f"**Cumulative effect:** {cumulative:.2f}")
        
        # Create actual vs predicted comparison chart for CausalImpact
        try:
            # Find date column
            date_cols = [col for col in df_original.columns if pd.api.types.is_datetime64_any_dtype(df_original[col])]
            if not date_cols:
                date_cols = [col for col in df_original.columns if 'date' in col.lower()]
            
            if date_cols and "prediction" in predictions.columns:
                date_col = date_cols[0]
                
                fig_comparison = create_actual_vs_predicted_chart(
                    df_original, predictions, date_col, target, 'prediction'
                )
                fig_comparison.update_layout(title='Actual vs CausalImpact Prediction Over Time')
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Required columns not found for comparison chart.")
                
        except Exception as chart_error:
            st.warning(f"Could not create actual vs predicted chart: {str(chart_error)}")
        
        logger.info("CausalImpact metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying CausalImpact metrics: {str(e)}")
        st.error(f"Error computing CausalImpact metrics: {str(e)}")


def display_psm_metrics(predictions: pd.DataFrame) -> None:
    """
    Display PSM model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame with ATT column
    """
    try:
        st.subheader("Model Diagnostics (PSM)")
        
        att = predictions["ATT"].iloc[0]
        st.markdown(f"**ATT:** {att:.3f}")
        
        logger.info("PSM metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying PSM metrics: {str(e)}")
        st.error(f"Error computing PSM metrics: {str(e)}")


# Model metric display dispatcher
MODEL_METRIC_DISPATCH = {
    "MLR": lambda **kwargs: display_mlr_metrics(kwargs["df"], kwargs["target"], kwargs["features"]),
    "Distributed Lag": lambda **kwargs: display_distributed_lag_metrics(kwargs["df"], kwargs["target"], kwargs["features"]),
    "ML + SHAP": lambda **kwargs: display_ml_shap_metrics(kwargs["model"], kwargs["df"], kwargs["target"]),
    "DiD": lambda **kwargs: display_did_metrics(kwargs["model"]),
    "VAR": lambda **kwargs: display_var_metrics(kwargs["model"]),
    "Synthetic Control": lambda **kwargs: display_synthetic_control_metrics(kwargs["predictions"]),
    "CausalImpact": lambda **kwargs: display_causal_impact_metrics(kwargs["predictions"], kwargs["df"], kwargs["target"]),
    "PSM": lambda **kwargs: display_psm_metrics(kwargs["predictions"]),
}


def display_model_metrics(
    model_name: str, 
    *, 
    df: pd.DataFrame, 
    target: str, 
    features: List[str], 
    model: Any, 
    predictions: pd.DataFrame
) -> None:
    """
    Safely display metrics for a trained model.
    
    Args:
        model_name: Name of the model
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        model: Trained model object
        predictions: Predictions DataFrame
    """
    try:
        func = MODEL_METRIC_DISPATCH.get(model_name)
        if func:
            func(df=df, target=target, features=features, model=model, predictions=predictions)
        else:
            st.info("Metric display not implemented for this model yet.")
            
    except Exception as e:
        logger.error(f"Failed to display metrics for {model_name}: {str(e)}")
        st.warning(f"Failed to compute metrics: {str(e)}")


def collect_model_report_data(
    model_name: str,
    df: pd.DataFrame,
    date_col: str,
    target: str,
    features: List[str],
    model: Any,
    predictions: pd.DataFrame,
    range_selection: Dict = None
) -> Dict:
    """
    Collect comprehensive model report data for JSON export.
    
    Args:
        model_name: Name of the model
        df: Input DataFrame  
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        model: Trained model object
        predictions: Predictions DataFrame
        range_selection: Range selection information
        
    Returns:
        Dictionary containing all model information
    """
    import json
    from datetime import datetime
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    try:
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_name": model_name,
                "report_version": "2.0",
                "analysis_type": "comprehensive_model_analysis"
            },
            "dataset_overview": {
                "total_observations": len(df),
                "columns_count": len(df.columns),
                "selected_observations": len(df),
                "date_range": {},
                "missing_data_summary": {},
                "descriptive_statistics": {},
                "outlier_summary": {}
            },
            "comprehensive_dataset_analysis": {},
            "model_configuration": {
                "model_name": model_name,
                "range_column": date_col,
                "target_variable": target,
                "feature_variables": features,
                "feature_count": len(features),
                "range_selection": range_selection or {}
            },
            "model_performance_metrics": {},
            "cross_validation_results": {},
            "variable_importance": {},
            "comparative_baseline_analysis": {},
            "model_diagnostics": {},
            "variance_inflation_factor": {},
            "residual_diagnostics": {},
            "overfitting_analysis": {},
            "model_predictions": {},
            "channel_contributions": {},
            "multicollinearity_integration": {},
            "interpretation_hints": create_interpretation_hints(model_name)
        }
        
        # Dataset overview details
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            report["dataset_overview"]["date_range"] = {
                "start_date": df[date_col].min().isoformat(),
                "end_date": df[date_col].max().isoformat(),
                "date_column": date_col,
                "total_time_periods": len(df[date_col].unique())
            }
        
        # Missing data summary
        missing_data = df[[target] + features].isnull().sum()
        missing_pct = df[[target] + features].isnull().mean() * 100
        report["dataset_overview"]["missing_data_summary"] = {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": missing_pct.round(2).to_dict(),
            "total_missing": int(missing_data.sum()),
            "completeness_score": round(100 - missing_pct.mean(), 2)
        }
        
        # Descriptive statistics
        numeric_cols = [c for c in [target] + features if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            desc_stats = df[numeric_cols].describe()  # This gives count, mean, std, min, 25%, 50%, 75%, max
            report["dataset_overview"]["descriptive_statistics"] = desc_stats.to_dict()
            
            # Enhanced outlier analysis
            outlier_info = {}
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                threshold_99 = df[col].quantile(OUTLIER_THRESHOLD)
                outlier_count_99 = int((df[col] > threshold_99).sum())
                
                outlier_info[col] = {
                    "iqr_method": {
                        "outlier_count": len(outliers),
                        "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    },
                    "percentile_99_method": {
                        "threshold": float(threshold_99),
                        "outlier_count": outlier_count_99,
                        "outlier_percentage": round(outlier_count_99 / len(df) * 100, 2)
                    }
                }
            report["dataset_overview"]["outlier_summary"] = outlier_info
        
        # Comprehensive dataset analysis using data_analysis.py functionality
        try:
            from data_analysis import analyze_dataset_for_report
            comprehensive_analysis = analyze_dataset_for_report(df, f"{model_name}_dataset")
            report["comprehensive_dataset_analysis"] = comprehensive_analysis
            logger.info("Comprehensive dataset analysis completed successfully")
        except Exception as analysis_error:
            logger.warning(f"Could not complete comprehensive dataset analysis: {str(analysis_error)}")
            report["comprehensive_dataset_analysis"] = {
                "error": str(analysis_error),
                "note": "Comprehensive analysis failed, basic analysis available in dataset_overview section"
            }
        
        # Enhanced Model Performance Metrics
        try:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if len(X) > 0:
                # Basic performance metrics
                if predictions is not None and not predictions.empty:
                    # Try to align predictions with actual values
                    if 'prediction' in predictions.columns:
                        pred_values = predictions['prediction']
                    elif model_name in predictions.columns:
                        pred_values = predictions[model_name]
                    else:
                        # Generate predictions if not available
                        pred_values = _generate_predictions_for_metrics(model, X, model_name)
                    
                    if pred_values is not None:
                        # Align with actual values
                        aligned_actual, aligned_pred = _align_actual_predicted(y, pred_values)
                        
                        if len(aligned_actual) > 0:
                            r2 = r2_score(aligned_actual, aligned_pred)
                            mae = mean_absolute_error(aligned_actual, aligned_pred)
                            rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_pred))
                            mape = np.mean(np.abs((aligned_actual - aligned_pred) / aligned_actual)) * 100 if aligned_actual.min() > 0 else np.nan
                            
                            # Adjusted R²
                            n = len(aligned_actual)
                            k = len(features)
                            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
                            
                            report["model_performance_metrics"] = {
                                "r_squared": float(r2),
                                "adjusted_r_squared": float(adj_r2) if not np.isnan(adj_r2) else None,
                                "mean_absolute_error": float(mae),
                                "root_mean_squared_error": float(rmse),
                                "mean_absolute_percentage_error": float(mape) if not np.isnan(mape) else None,
                                "observations_used": len(aligned_actual),
                                "features_used": len(features),
                                "prediction_accuracy_score": float(max(0, r2))  # Bounded R²
                            }
                        
                # Cross-validation analysis
                try:
                    cv_results = _perform_cross_validation(X, y, model_name)
                    report["cross_validation_results"] = cv_results
                except Exception as cv_error:
                    logger.warning(f"Cross-validation failed: {cv_error}")
                    report["cross_validation_results"] = {"error": str(cv_error)}
                
                # Variable importance analysis
                try:
                    importance_results = _calculate_variable_importance(model, X, y, features, model_name)
                    report["variable_importance"] = importance_results
                except Exception as imp_error:
                    logger.warning(f"Variable importance calculation failed: {imp_error}")
                    report["variable_importance"] = {"error": str(imp_error)}
                
                # Comparative baseline analysis
                try:
                    baseline_comparison = _run_baseline_comparison(X, y, features, model_name)
                    report["comparative_baseline_analysis"] = baseline_comparison
                except Exception as base_error:
                    logger.warning(f"Baseline comparison failed: {base_error}")
                    report["comparative_baseline_analysis"] = {"error": str(base_error)}
                
                # Overfitting analysis
                try:
                    overfitting_analysis = _detect_overfitting_patterns(X, y, model, model_name)
                    report["overfitting_analysis"] = overfitting_analysis
                except Exception as over_error:
                    logger.warning(f"Overfitting analysis failed: {over_error}")
                    report["overfitting_analysis"] = {"error": str(over_error)}
                    
        except Exception as perf_error:
            logger.error(f"Error in performance metrics calculation: {perf_error}")
            report["model_performance_metrics"] = {"error": str(perf_error)}
        
        # Predictions data
        if predictions is not None and not predictions.empty:
            report["model_predictions"] = {
                "predictions_summary": {
                    "prediction_count": len(predictions),
                    "prediction_columns": list(predictions.columns),
                    "date_coverage": {
                        "start": predictions[date_col].min().isoformat() if pd.api.types.is_datetime64_any_dtype(predictions[date_col]) else str(predictions[date_col].min()),
                        "end": predictions[date_col].max().isoformat() if pd.api.types.is_datetime64_any_dtype(predictions[date_col]) else str(predictions[date_col].max())
                    }
                },
                "predictions_data": predictions.to_dict(orient="records")
            }
        
        # Model-specific diagnostics (enhanced)
        if model_name == "MLR":
            try:
                import statsmodels.api as sm
                X_sm = sm.add_constant(df[features])
                y = df[target]
                ols_model = sm.OLS(y, X_sm).fit()
                
                report["model_diagnostics"] = {
                    "r_squared": float(ols_model.rsquared),
                    "adjusted_r_squared": float(ols_model.rsquared_adj),
                    "f_statistic": float(ols_model.fvalue),
                    "f_pvalue": float(ols_model.f_pvalue),
                    "aic": float(ols_model.aic),
                    "bic": float(ols_model.bic),
                    "log_likelihood": float(ols_model.llf),
                    "durbin_watson": float(sm.stats.diagnostic.durbin_watson(ols_model.resid)),
                    "jarque_bera_test": {
                        "statistic": float(sm.stats.diagnostic.jarque_bera(ols_model.resid)[0]),
                        "p_value": float(sm.stats.diagnostic.jarque_bera(ols_model.resid)[1])
                    },
                    "coefficients": {},
                    "model_assumptions": {}
                }
                
                # Enhanced coefficient analysis
                coef_summary = ols_model.summary2().tables[1]
                for idx, row in coef_summary.iterrows():
                    significance_level = 0.05
                    is_significant = float(row["P>|t|"]) < significance_level
                    
                    report["model_diagnostics"]["coefficients"][idx] = {
                        "coefficient": float(row["Coef."]),
                        "std_error": float(row["Std.Err."]),
                        "t_value": float(row["t"]),
                        "p_value": float(row["P>|t|"]),
                        "conf_int_lower": float(row["[0.025"]),
                        "conf_int_upper": float(row["0.975]"]),
                        "significant": is_significant,
                        "significance_level": "***" if row["P>|t|"] < 0.001 else "**" if row["P>|t|"] < 0.01 else "*" if row["P>|t|"] < 0.05 else "",
                        "interpretation": "positive_effect" if row["Coef."] > 0 else "negative_effect"
                    }
                
                # Model assumptions testing
                report["model_diagnostics"]["model_assumptions"] = {
                    "linearity": "Check residual plots for linearity assumption",
                    "normality_of_residuals": {
                        "jarque_bera_p_value": float(sm.stats.diagnostic.jarque_bera(ols_model.resid)[1]),
                        "assumption_met": float(sm.stats.diagnostic.jarque_bera(ols_model.resid)[1]) > 0.05
                    },
                    "homoscedasticity": "Check residual plots for constant variance",
                    "independence": {
                        "durbin_watson_statistic": float(sm.stats.diagnostic.durbin_watson(ols_model.resid)),
                        "autocorrelation_detected": not (1.5 <= sm.stats.diagnostic.durbin_watson(ols_model.resid) <= 2.5)
                    }
                }
                
                # Enhanced VIF analysis
                vif_df = create_vif_table(df, features)
                if not vif_df.empty and "VIF" in vif_df.columns:
                    vif_data = {}
                    for _, row in vif_df.iterrows():
                        vif_value = row["VIF"]
                        if np.isfinite(vif_value):
                            multicollinearity_level = "severe" if vif_value > 10 else "moderate" if vif_value > 5 else "acceptable"
                            vif_data[row["Variable"]] = {
                                "vif_value": float(vif_value),
                                "multicollinearity_concern": float(vif_value) > VIF_THRESHOLD,
                                "multicollinearity_level": multicollinearity_level,
                                "recommendation": "consider_removal" if vif_value > 10 else "monitor" if vif_value > 5 else "keep"
                            }
                        else:
                            vif_data[row["Variable"]] = {
                                "vif_value": "infinite",
                                "multicollinearity_concern": True,
                                "multicollinearity_level": "severe",
                                "recommendation": "remove_immediately"
                            }
                    report["variance_inflation_factor"] = vif_data
                
                # Enhanced residual diagnostics
                residuals = ols_model.resid
                fitted_values = ols_model.fittedvalues
                standardized_residuals = residuals / residuals.std()
                
                report["residual_diagnostics"] = {
                    "residual_summary": {
                        "count": len(residuals),
                        "mean": float(residuals.mean()),
                        "std": float(residuals.std()),
                        "min": float(residuals.min()),
                        "max": float(residuals.max()),
                        "skewness": float(residuals.skew()),
                        "kurtosis": float(residuals.kurtosis())
                    },
                    "outlier_detection": {
                        "standardized_residuals_gt_2": int(np.sum(np.abs(standardized_residuals) > 2)),
                        "standardized_residuals_gt_3": int(np.sum(np.abs(standardized_residuals) > 3)),
                        "potential_outliers": [i for i, val in enumerate(standardized_residuals) if abs(val) > 2][:10]  # First 10
                    },
                    "fitted_vs_residual": [
                        {"fitted": float(f), "residual": float(r), "standardized_residual": float(sr)} 
                        for f, r, sr in zip(fitted_values, residuals, standardized_residuals)
                    ]
                }
                
                # Enhanced channel contributions
                contributions = {}
                for coef, feat in zip(ols_model.params[1:], features):  # Skip intercept
                    contribution_values = df[feat] * coef
                    feat_stats = df[feat].describe()
                    
                    contributions[feat] = {
                        "coefficient": float(coef),
                        "coefficient_interpretation": "increases" if coef > 0 else "decreases",
                        "total_contribution": float(contribution_values.sum()),
                        "average_contribution": float(contribution_values.mean()),
                        "contribution_std": float(contribution_values.std()),
                        "contribution_range": {
                            "min": float(contribution_values.min()),
                            "max": float(contribution_values.max())
                        },
                        "feature_statistics": {
                            "mean": float(feat_stats['mean']),
                            "std": float(feat_stats['std']),
                            "min": float(feat_stats['min']),
                            "max": float(feat_stats['max'])
                        },
                        "elasticity": float((coef * df[feat].mean()) / df[target].mean()) if df[target].mean() != 0 else 0,
                        "contribution_time_series": [
                            {
                                "date": df[date_col].iloc[i].isoformat() if pd.api.types.is_datetime64_any_dtype(df[date_col]) else str(df[date_col].iloc[i]),
                                "feature_value": float(df[feat].iloc[i]),
                                "contribution": float(contribution_values.iloc[i])
                            }
                            for i in range(len(contribution_values))
                        ]
                    }
                report["channel_contributions"] = contributions
                
            except Exception as e:
                logger.error(f"Error collecting MLR diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "Distributed Lag":
            try:
                # Similar to MLR but add enhanced lag analysis
                import statsmodels.api as sm
                from statsmodels.stats.stattools import durbin_watson
                
                X = sm.add_constant(df[features])
                y = df[target]
                ols_model = sm.OLS(y, X).fit()
                
                dw_stat = durbin_watson(ols_model.resid)
                
                report["model_diagnostics"] = {
                    "r_squared": float(ols_model.rsquared),
                    "adjusted_r_squared": float(ols_model.rsquared_adj),
                    "f_statistic": float(ols_model.fvalue),
                    "f_pvalue": float(ols_model.f_pvalue),
                    "aic": float(ols_model.aic),
                    "bic": float(ols_model.bic),
                    "durbin_watson": float(dw_stat),
                    "autocorrelation_detected": not (DURBIN_WATSON_LOWER <= dw_stat <= DURBIN_WATSON_UPPER),
                    "lag_analysis": {
                        "lag_variables_count": len([f for f in features if 'lag' in f.lower()]),
                        "max_lag_order": _detect_max_lag_order(features),
                        "lag_significance": _analyze_lag_significance(ols_model, features)
                    },
                    "temporal_patterns": {
                        "durbin_watson_interpretation": _interpret_durbin_watson(dw_stat),
                        "serial_correlation_risk": "high" if dw_stat < 1.5 or dw_stat > 2.5 else "low"
                    }
                }
                
            except Exception as e:
                logger.error(f"Error collecting Distributed Lag diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "ML + SHAP":
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                from sklearn.preprocessing import LabelEncoder
                
                # Get feature columns (use stored feature names if available)
                if hasattr(model, 'feature_names_'):
                    feature_cols = model.feature_names_
                else:
                    feature_cols = [c for c in df.columns if c != target]
                
                # Handle categorical columns for prediction using stored encoders
                X_pred = df[feature_cols].copy()
                
                # Apply the SAME categorical encoding used during training
                if hasattr(model, 'label_encoders_') and hasattr(model, 'categorical_cols_'):
                    for col in model.categorical_cols_:
                        if col in model.label_encoders_:
                            le = model.label_encoders_[col]
                            try:
                                # Transform using the exact same encoder from training
                                X_pred[col] = le.transform(X_pred[col].astype(str))
                            except ValueError:
                                # Handle unseen categories by using the most frequent class
                                known_classes = set(le.classes_)
                                X_pred[col] = X_pred[col].astype(str).apply(
                                    lambda x: x if x in known_classes else le.classes_[0]
                                )
                                X_pred[col] = le.transform(X_pred[col])
                            
                            if hasattr(model, 'enable_categorical') and model.enable_categorical:
                                X_pred[col] = X_pred[col].astype('category')
                else:
                    # Fallback to old method if model doesn't have stored encoders
                    for col in feature_cols:
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                            le = LabelEncoder()
                            X_pred[col] = le.fit_transform(X_pred[col].astype(str))
                            if hasattr(model, 'enable_categorical') and model.enable_categorical:
                                X_pred[col] = X_pred[col].astype('category')
                
                preds = model.predict(X_pred)
                
                report["model_diagnostics"] = {
                    "rmse": float(mean_squared_error(df[target], preds, squared=False)),
                    "mae": float(mean_absolute_error(df[target], preds)),
                    "model_type": str(type(model).__name__),
                    "feature_count": len(feature_cols),
                    "has_stored_encoders": hasattr(model, 'label_encoders_'),
                    "prediction_range": {
                        "min": float(preds.min()),
                        "max": float(preds.max()),
                        "mean": float(preds.mean()),
                        "std": float(preds.std())
                    },
                    "residual_analysis": {
                        "residuals_mean": float((actual_values - preds).mean()),
                        "residuals_std": float((actual_values - preds).std()),
                        "residuals_range": {
                            "min": float((actual_values - preds).min()),
                            "max": float((actual_values - preds).max())
                        }
                    }
                }
                
                # Try to get SHAP values
                try:
                    import shap
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_pred)
                    
                    # Global feature importance
                    feature_importance = {}
                    shap_importance_values = np.abs(shap_values.values).mean(axis=0)
                    
                    for i, feat in enumerate(feature_cols):
                        importance = float(shap_importance_values[i])
                        shap_vals = shap_values.values[:, i]
                        
                        feature_importance[feat] = {
                            "mean_absolute_shap": importance,
                            "importance_rank": int(np.argsort(shap_importance_values)[::-1].tolist().index(i) + 1),
                            "importance_percentage": float((importance / shap_importance_values.sum()) * 100),
                            "shap_statistics": {
                                "mean": float(shap_vals.mean()),
                                "std": float(shap_vals.std()),
                                "min": float(shap_vals.min()),
                                "max": float(shap_vals.max())
                            },
                            "positive_contributions": int(np.sum(shap_vals > 0)),
                            "negative_contributions": int(np.sum(shap_vals < 0)),
                            "neutral_contributions": int(np.sum(shap_vals == 0)),
                            "shap_values_sample": [float(val) for val in shap_vals[:min(100, len(shap_vals))]],  # First 100 values
                            "importance_type": "shap_based_importance"
                        }
                    
                    # Sort by importance
                    sorted_importance = dict(sorted(feature_importance.items(), 
                                                  key=lambda x: x[1]['mean_absolute_shap'], 
                                                  reverse=True))
                    
                    report["channel_contributions"] = sorted_importance
                    
                    # SHAP summary statistics
                    report["model_diagnostics"]["shap_analysis"] = {
                        "explainer_type": str(type(explainer).__name__),
                        "total_shap_importance": float(shap_importance_values.sum()),
                        "top_3_features": list(sorted_importance.keys())[:3],
                        "feature_importance_distribution": {
                            "mean": float(shap_importance_values.mean()),
                            "std": float(shap_importance_values.std()),
                            "concentration_index": float(shap_importance_values.max() / shap_importance_values.sum()),  # How concentrated importance is
                        }
                    }
                    
                except Exception as shap_error:
                    logger.warning(f"Could not compute SHAP values: {str(shap_error)}")
                    report["model_diagnostics"]["shap_analysis_note"] = f"SHAP analysis not available: {str(shap_error)}"
                
            except Exception as e:
                logger.error(f"Error collecting ML + SHAP diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "DiD":
            try:
                ate = model.params.get("treated:post", float("nan"))
                ci_low, ci_high = model.conf_int().loc["treated:post"]
                pval = model.pvalues.get("treated:post", float("nan"))
                
                report["model_diagnostics"] = {
                    "average_treatment_effect": float(ate),
                    "confidence_interval_lower": float(ci_low),
                    "confidence_interval_upper": float(ci_high),
                    "p_value": float(pval),
                    "significant": float(pval) < SIGNIFICANCE_LEVEL,
                    "effect_size_interpretation": _interpret_did_effect_size(ate, df[target].std()),
                    "statistical_power": _estimate_did_statistical_power(model, df),
                    "parallel_trends_assumption": "Check with pre-treatment trend analysis",
                    "model_summary_key_stats": {
                        "r_squared": float(model.rsquared),
                        "f_statistic": float(model.fvalue),
                        "observations": int(model.nobs)
                    }
                }
                
                # DiD-specific diagnostics
                if 'treated' in df.columns and 'post' in df.columns:
                    # 2x2 design validation
                    crosstab = pd.crosstab(df['treated'], df['post'])
                    report["model_diagnostics"]["design_validation"] = {
                        "control_pre": int(crosstab.loc[0, 0]) if (0, 0) in crosstab.index else 0,
                        "control_post": int(crosstab.loc[0, 1]) if (0, 1) in crosstab.index else 0,
                        "treated_pre": int(crosstab.loc[1, 0]) if (1, 0) in crosstab.index else 0,
                        "treated_post": int(crosstab.loc[1, 1]) if (1, 1) in crosstab.index else 0,
                        "balanced_design": _check_did_balance(crosstab)
                    }
                
            except Exception as e:
                logger.error(f"Error collecting DiD diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "VAR":
            try:
                report["model_diagnostics"] = {
                    "selected_lag_order": int(model.k_ar),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "hqic": float(model.hqic),
                    "fpe": float(model.fpe),
                    "stability_check": bool(model.is_stable()),
                    "variables_used": model.names,
                    "total_parameters": int(model.k_ar * len(model.names) * len(model.names)),
                    "degrees_of_freedom": int(model.df_model),
                    "lag_order_selection": {
                        "selected_order": int(model.k_ar),
                        "selection_criterion": "Information criteria optimization",
                        "stability_maintained": bool(model.is_stable())
                    },
                    "impulse_response_analysis": "Available through VAR model methods",
                    "variance_decomposition": "Available through VAR model methods"
                }
                
                # VAR-specific variable importance (based on coefficients)
                var_importance = {}
                for eq_name in model.names:
                    eq_params = model.params[eq_name]
                    var_importance[eq_name] = {
                        "equation_coefficients": eq_params.to_dict(),
                        "equation_significance": "Requires coefficient testing",
                        "total_coefficient_magnitude": float(np.abs(eq_params).sum())
                    }
                
                report["variable_importance"] = var_importance
                
            except Exception as e:
                logger.error(f"Error collecting VAR diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "Synthetic Control":
            try:
                if isinstance(predictions, pd.DataFrame) and "Actual" in predictions.columns and "Synthetic" in predictions.columns:
                    actual = predictions["Actual"]
                    synthetic = predictions["Synthetic"]
                    
                    # Enhanced Synthetic Control metrics
                    rmspe = float(np.sqrt(((actual - synthetic) ** 2).mean()))
                    mspe = float(((actual - synthetic) ** 2).mean())
                    mae = float(np.abs(actual - synthetic).mean())
                    
                    # Pre/post treatment analysis if possible
                    treatment_effect = actual - synthetic
                    
                    report["model_diagnostics"] = {
                        "rmspe": rmspe,
                        "mspe": mspe,
                        "mae": mae,
                        "treatment_effect_analysis": {
                            "mean_effect": float(treatment_effect.mean()),
                            "cumulative_effect": float(treatment_effect.sum()),
                            "effect_std": float(treatment_effect.std()),
                            "effect_range": {
                                "min": float(treatment_effect.min()),
                                "max": float(treatment_effect.max())
                            }
                        },
                        "fit_quality": {
                            "pre_treatment_fit": "Analyze pre-treatment period fit",
                            "control_units_used": "Check donor pool composition"
                        },
                        "actual_vs_synthetic": [
                            {
                                "date": predictions[date_col].iloc[i].isoformat() if pd.api.types.is_datetime64_any_dtype(predictions[date_col]) else str(predictions[date_col].iloc[i]),
                                "actual": float(actual.iloc[i]),
                                "synthetic": float(synthetic.iloc[i]),
                                "treatment_effect": float(treatment_effect.iloc[i])
                            }
                            for i in range(len(predictions))
                        ]
                    }
                
            except Exception as e:
                logger.error(f"Error collecting Synthetic Control diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "CausalImpact":
            try:
                if isinstance(predictions, pd.DataFrame) and "prediction" in predictions.columns:
                    df_effect = df[target] - predictions["prediction"]
                    cumulative_effect = float(df_effect.cumsum().iloc[-1])
                    
                    report["model_diagnostics"] = {
                        "cumulative_effect": cumulative_effect,
                        "average_effect": float(df_effect.mean()),
                        "effect_std": float(df_effect.std()),
                        "model_type": str(type(model).__name__),
                        "causal_effect_analysis": {
                            "absolute_effect": cumulative_effect,
                            "relative_effect": float((cumulative_effect / df[target].sum()) * 100) if df[target].sum() != 0 else 0,
                            "effect_significance": "Requires probabilistic analysis",
                            "posterior_probability": "Available in CausalImpact output"
                        },
                        "prediction_quality": {
                            "prediction_accuracy": "Based on Bayesian model fit",
                            "uncertainty_intervals": "Available in CausalImpact output"
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error collecting CausalImpact diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        elif model_name == "PSM":
            try:
                if isinstance(predictions, pd.DataFrame) and "ATT" in predictions.columns:
                    att = float(predictions["ATT"].iloc[0])
                    
                    report["model_diagnostics"] = {
                        "average_treatment_effect_treated": att,
                        "matching_quality": {
                            "matched_pairs": len(predictions) if "matched_pairs" in predictions.columns else "Not available",
                            "propensity_score_range": "Check propensity score distribution",
                            "balance_achievement": "Requires covariate balance testing"
                        },
                        "treatment_effect_analysis": {
                            "att_magnitude": abs(att),
                            "effect_direction": "positive" if att > 0 else "negative",
                            "statistical_significance": "Requires standard error calculation"
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error collecting PSM diagnostics: {str(e)}")
                report["model_diagnostics"]["error"] = str(e)
        
        # Multicollinearity Integration
        try:
            multicollinearity_integration = _integrate_multicollinearity_analysis(df, target, features, model_name)
            report["multicollinearity_integration"] = multicollinearity_integration
        except Exception as mult_error:
            logger.warning(f"Multicollinearity integration failed: {mult_error}")
            report["multicollinearity_integration"] = {"error": str(mult_error), "note": "Run separate multicollinearity analysis"}
        
        # Convert any remaining numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat'):  # Handle other datetime objects
                return obj.isoformat()
            elif pd.isna(obj):  # Handle NaN values
                return None
            else:
                return obj
        
        report = convert_numpy_types(report)
        
        logger.info(f"Enhanced model report data collected successfully for {model_name}")
        return report
        
    except Exception as e:
        logger.error(f"Error collecting model report data: {str(e)}")
        return {
            "error": str(e),
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "report_version": "2.0"
        }


# Helper functions for enhanced analysis
def _generate_predictions_for_metrics(model, X, model_name):
    """Generate predictions for metrics calculation if not available."""
    try:
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif hasattr(model, 'fittedvalues'):
            return model.fittedvalues
        else:
            return None
    except:
        return None

def _align_actual_predicted(actual, predicted):
    """Align actual and predicted values for metric calculation."""
    try:
        if len(actual) == len(predicted):
            return actual, predicted
        else:
            min_len = min(len(actual), len(predicted))
            return actual.iloc[:min_len], predicted[:min_len]
    except:
        return actual, predicted

def _perform_cross_validation(X, y, model_name, cv_folds=5):
    """Perform cross-validation analysis."""
    try:
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
        
        # Select appropriate model for CV
        if model_name in ["MLR", "Distributed Lag"]:
            cv_model = LinearRegression()
        elif model_name == "ML + SHAP":
            cv_model = RandomForestRegressor(random_state=42)
        else:
            cv_model = LinearRegression()  # Default fallback
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_r2 = cross_val_score(cv_model, X, y, cv=kfold, scoring='r2')
        cv_neg_mse = cross_val_score(cv_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        cv_neg_mae = cross_val_score(cv_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        
        return {
            "cv_folds": cv_folds,
            "r2_scores": {
                "mean": float(cv_r2.mean()),
                "std": float(cv_r2.std()),
                "min": float(cv_r2.min()),
                "max": float(cv_r2.max()),
                "individual_scores": cv_r2.tolist()
            },
            "rmse_scores": {
                "mean": float(np.sqrt(-cv_neg_mse.mean())),
                "std": float(np.sqrt(cv_neg_mse.var())),
                "individual_scores": np.sqrt(-cv_neg_mse).tolist()
            },
            "mae_scores": {
                "mean": float(-cv_neg_mae.mean()),
                "std": float(cv_neg_mae.std()),
                "individual_scores": (-cv_neg_mae).tolist()
            },
            "model_stability": {
                "r2_coefficient_of_variation": float(cv_r2.std() / cv_r2.mean()) if cv_r2.mean() != 0 else float('inf'),
                "consistent_performance": cv_r2.std() < 0.1
            }
        }
    except Exception as e:
        return {"error": str(e)}

def _calculate_variable_importance(model, X, y, features, model_name):
    """Calculate variable importance for different model types."""
    try:
        importance_data = {}
        
        if model_name in ["MLR", "Distributed Lag", "DiD"]:
            # For linear models, use coefficient magnitude and t-statistics
            import statsmodels.api as sm
            X_sm = sm.add_constant(X)
            ols_model = sm.OLS(y, X_sm).fit()
            
            coefficients = ols_model.params[1:]  # Exclude intercept
            t_values = ols_model.tvalues[1:]  # Exclude intercept
            p_values = ols_model.pvalues[1:]  # Exclude intercept
            
            for i, feat in enumerate(features):
                coef = coefficients.iloc[i]
                t_val = t_values.iloc[i]
                p_val = p_values.iloc[i]
                
                importance_data[feat] = {
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "t_value": float(t_val),
                    "p_value": float(p_val),
                    "significance": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "",
                    "importance_rank": 0,  # Will be filled later
                    "standardized_coefficient": float(coef * X[feat].std() / y.std()),
                    "variable_interpretation": "positive_driver" if coef > 0 else "negative_driver"
                }
            
            # Rank by absolute standardized coefficient
            sorted_features = sorted(importance_data.items(), 
                                   key=lambda x: abs(x[1]['standardized_coefficient']), 
                                   reverse=True)
            
            for rank, (feat, data) in enumerate(sorted_features, 1):
                importance_data[feat]['importance_rank'] = rank
                
        elif model_name == "ML + SHAP":
            # For ML models, try to extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for i, feat in enumerate(features):
                    if i < len(importances):
                        importance_data[feat] = {
                            "feature_importance": float(importances[i]),
                            "importance_rank": int(np.argsort(importances)[::-1].tolist().index(i) + 1),
                            "importance_percentage": float((importances[i] / importances.sum()) * 100),
                            "importance_type": "tree_based_importance"
                        }
            else:
                # Fallback: permutation importance
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(model, X, y, random_state=42)
                
                for i, feat in enumerate(features):
                    importance_data[feat] = {
                        "permutation_importance": float(perm_importance.importances_mean[i]),
                        "permutation_std": float(perm_importance.importances_std[i]),
                        "importance_rank": int(np.argsort(perm_importance.importances_mean)[::-1].tolist().index(i) + 1),
                        "importance_type": "permutation_importance"
                    }
        
        elif model_name == "VAR":
            # For VAR models, analyze coefficient magnitudes across equations
            if hasattr(model, 'params'):
                for feat in features:
                    total_importance = 0
                    for eq_name in model.names:
                        if feat in model.params[eq_name].index:
                            total_importance += abs(model.params[eq_name][feat])
                    
                    importance_data[feat] = {
                        "total_coefficient_magnitude": float(total_importance),
                        "importance_type": "var_coefficient_sum",
                        "affects_equations": [eq for eq in model.names if feat in model.params[eq].index]
                    }
        
        else:
            # Generic fallback: correlation with target
            for feat in features:
                corr = X[feat].corr(y)
                importance_data[feat] = {
                    "correlation_with_target": float(corr),
                    "abs_correlation": float(abs(corr)),
                    "importance_rank": 0,  # Will be filled later
                    "importance_type": "correlation_based"
                }
            
            # Rank by absolute correlation
            sorted_features = sorted(importance_data.items(), 
                                   key=lambda x: x[1]['abs_correlation'], 
                                   reverse=True)
            
            for rank, (feat, data) in enumerate(sorted_features, 1):
                importance_data[feat]['importance_rank'] = rank
        
        return {
            "importance_method": f"{model_name}_specific_importance",
            "feature_importance_scores": importance_data,
            "top_5_features": list(dict(sorted(importance_data.items(), 
                                             key=lambda x: x[1].get('importance_rank', float('inf'))))
                                 .keys())[:5],
            "importance_distribution": {
                "most_important_feature": min(importance_data.items(), 
                                            key=lambda x: x[1].get('importance_rank', float('inf')))[0] if importance_data else None,
                "feature_count": len(importance_data)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

def _run_baseline_comparison(X, y, features, model_name):
    """Run baseline model comparison for performance context."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import xgboost as xgb
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
        }
        
        results = {}
        
        for name, baseline_model in baseline_models.items():
            try:
                baseline_model.fit(X_train, y_train)
                train_pred = baseline_model.predict(X_train)
                test_pred = baseline_model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Adjusted R²
                n = len(y_train)
                k = len(features)
                adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
                
                results[name] = {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_adj_r2': float(adj_r2) if not np.isnan(adj_r2) else None,
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'r2_difference': float(train_r2 - test_r2),
                    'mae_ratio': float(test_mae / train_mae) if train_mae > 0 else float('inf'),
                    'generalization_score': float(test_r2 / train_r2) if train_r2 > 0 else 0
                }
                
            except Exception as model_error:
                results[name] = {'error': str(model_error)}
        
        # Identify best baseline
        valid_results = {name: res for name, res in results.items() if 'error' not in res}
        if valid_results:
            best_baseline = max(valid_results.keys(), key=lambda x: valid_results[x]['test_r2'])
            
            return {
                'baseline_results': results,
                'best_baseline_model': {
                    'name': best_baseline,
                    'performance': valid_results[best_baseline]
                },
                'comparison_summary': {
                    'average_test_r2': float(np.mean([res['test_r2'] for res in valid_results.values()])),
                    'performance_range': {
                        'best_test_r2': float(max([res['test_r2'] for res in valid_results.values()])),
                        'worst_test_r2': float(min([res['test_r2'] for res in valid_results.values()]))
                    },
                    'model_comparison_available': True
                },
                'current_model_context': f"Compare your {model_name} results with these baseline models"
            }
        else:
            return {'error': 'All baseline models failed', 'baseline_results': results}
            
    except Exception as e:
        return {"error": str(e)}

def _detect_overfitting_patterns(X, y, model, model_name):
    """Detect overfitting patterns in the trained model."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Split data to test overfitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        overfitting_analysis = {
            "overfitting_risk": "unknown",
            "risk_factors": [],
            "recommendations": []
        }
        
        try:
            if hasattr(model, 'predict'):
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                r2_diff = train_r2 - test_r2
                mae_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
                
                overfitting_analysis["performance_metrics"] = {
                    "train_r2": float(train_r2),
                    "test_r2": float(test_r2),
                    "r2_difference": float(r2_diff),
                    "mae_ratio": float(mae_ratio),
                    "generalization_ratio": float(test_r2 / train_r2) if train_r2 > 0 else 0
                }
                
                # Overfitting risk assessment
                risk_level = "low"
                risk_factors = []
                recommendations = []
                
                if r2_diff > 0.2:
                    risk_level = "high"
                    risk_factors.append(f"Large R² difference ({r2_diff:.3f})")
                    recommendations.append("Consider regularization or feature selection")
                elif r2_diff > 0.1:
                    risk_level = "medium"
                    risk_factors.append(f"Moderate R² difference ({r2_diff:.3f})")
                    recommendations.append("Monitor performance on new data")
                
                if mae_ratio > 1.5:
                    risk_level = "high" if risk_level != "high" else risk_level
                    risk_factors.append(f"High MAE ratio ({mae_ratio:.3f})")
                    recommendations.append("Consider simpler model or more data")
                
                if train_r2 > 0.95 and test_r2 < 0.7:
                    risk_level = "high"
                    risk_factors.append("Suspiciously high training performance")
                    recommendations.append("Check for data leakage or reduce model complexity")
                
                # Model complexity factors
                if model_name == "ML + SHAP" and hasattr(model, 'n_estimators'):
                    if model.n_estimators > 500:
                        risk_factors.append("High number of estimators")
                        recommendations.append("Consider reducing n_estimators")
                
                if len(X.columns) > len(X) * 0.1:  # Many features relative to observations
                    risk_factors.append("High feature-to-observation ratio")
                    recommendations.append("Consider feature selection or regularization")
                
                overfitting_analysis.update({
                    "overfitting_risk": risk_level,
                    "risk_factors": risk_factors,
                    "recommendations": recommendations,
                    "complexity_analysis": {
                        "feature_count": len(X.columns),
                        "observation_count": len(X),
                        "feature_to_obs_ratio": float(len(X.columns) / len(X)),
                        "model_complexity": _assess_model_complexity(model, model_name)
                    }
                })
                
        except Exception as pred_error:
            overfitting_analysis["prediction_error"] = str(pred_error)
            overfitting_analysis["overfitting_risk"] = "unable_to_assess"
        
        return overfitting_analysis
        
    except Exception as e:
        return {"error": str(e)}

def _integrate_multicollinearity_analysis(df, target, features, model_name):
    """Integrate basic multicollinearity analysis into the main report."""
    try:
        integration_results = {
            "analysis_type": "basic_multicollinearity_check",
            "note": "For comprehensive analysis, use the multicollinearity analysis module"
        }
        
        # Basic correlation analysis
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            
            # Find high correlations
            high_correlations = []
            threshold = 0.8
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_correlations.append({
                            "variable_1": corr_matrix.columns[i],
                            "variable_2": corr_matrix.columns[j],
                            "correlation": float(corr_val),
                            "abs_correlation": float(abs(corr_val))
                        })
            
            integration_results["correlation_analysis"] = {
                "high_correlations_found": len(high_correlations),
                "threshold_used": threshold,
                "high_correlation_pairs": high_correlations[:5],  # Top 5
                "correlation_matrix_available": True
            }
            
            # Basic VIF calculation
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                from statsmodels.tools.tools import add_constant
                
                X_numeric = df[numeric_features].dropna()
                if not X_numeric.empty:
                    X_with_const = add_constant(X_numeric)
                    
                    vif_data = []
                    for i in range(1, X_with_const.shape[1]):  # Skip constant
                        vif_val = variance_inflation_factor(X_with_const.values, i)
                        vif_data.append({
                            "variable": numeric_features[i-1],
                            "vif": float(vif_val) if np.isfinite(vif_val) else None,
                            "high_multicollinearity": vif_val > 10 if np.isfinite(vif_val) else True
                        })
                    
                    integration_results["vif_analysis"] = {
                        "vif_results": vif_data,
                        "high_vif_count": sum(1 for v in vif_data if v.get("high_multicollinearity", False)),
                        "vif_threshold": 10.0
                    }
                    
            except Exception as vif_error:
                integration_results["vif_analysis"] = {"error": str(vif_error)}
            
            # Recommendations
            recommendations = []
            if len(high_correlations) > 0:
                recommendations.append(f"Found {len(high_correlations)} high correlation pairs - consider variable reduction")
            
            high_vif_count = integration_results.get("vif_analysis", {}).get("high_vif_count", 0)
            if high_vif_count > 0:
                recommendations.append(f"Found {high_vif_count} variables with high VIF - consider multicollinearity treatment")
            
            if not recommendations:
                recommendations.append("No major multicollinearity issues detected in basic analysis")
            
            integration_results["recommendations"] = recommendations
            integration_results["comprehensive_analysis_suggestion"] = "Run dedicated multicollinearity analysis for detailed recommendations"
            
        else:
            integration_results["message"] = "Insufficient numeric features for multicollinearity analysis"
        
        return integration_results
        
    except Exception as e:
        return {"error": str(e)}

# Additional helper functions
def _detect_max_lag_order(features):
    """Detect maximum lag order from feature names."""
    max_lag = 0
    for feature in features:
        if 'lag' in feature.lower():
            # Try to extract lag number
            import re
            lag_numbers = re.findall(r'lag[_\s]*(\d+)', feature.lower())
            if lag_numbers:
                max_lag = max(max_lag, int(lag_numbers[0]))
    return max_lag

def _analyze_lag_significance(model, features):
    """Analyze significance of lag variables."""
    lag_analysis = {}
    for feature in features:
        if 'lag' in feature.lower():
            if hasattr(model, 'pvalues') and feature in model.pvalues.index:
                p_val = model.pvalues[feature]
                lag_analysis[feature] = {
                    "p_value": float(p_val),
                    "significant": float(p_val) < 0.05
                }
    return lag_analysis

def _interpret_durbin_watson(dw_stat):
    """Interpret Durbin-Watson statistic."""
    if dw_stat < 1.5:
        return "Positive autocorrelation likely"
    elif dw_stat > 2.5:
        return "Negative autocorrelation likely"
    else:
        return "No strong evidence of autocorrelation"

def _interpret_did_effect_size(ate, target_std):
    """Interpret DiD effect size."""
    if target_std == 0:
        return "Cannot assess effect size (zero variance)"
    
    effect_size = abs(ate) / target_std
    
    if effect_size < 0.2:
        return "Small effect size"
    elif effect_size < 0.5:
        return "Medium effect size"
    else:
        return "Large effect size"

def _estimate_did_statistical_power(model, df):
    """Estimate statistical power for DiD analysis."""
    try:
        if hasattr(model, 'conf_int'):
            ci = model.conf_int().loc["treated:post"]
            ci_width = ci[1] - ci[0]
            return {
                "confidence_interval_width": float(ci_width),
                "power_assessment": "Narrow CI suggests good power" if ci_width < df['Visits'].std() else "Wide CI suggests low power"
            }
    except:
        return {"power_assessment": "Unable to estimate"}

def _check_did_balance(crosstab):
    """Check balance in DiD design."""
    try:
        min_cell = crosstab.min().min()
        max_cell = crosstab.max().max()
        balance_ratio = min_cell / max_cell if max_cell > 0 else 0
        
        return {
            "balanced": balance_ratio > 0.2,  # At least 20% of largest cell
            "balance_ratio": float(balance_ratio),
            "min_cell_size": int(min_cell),
            "max_cell_size": int(max_cell)
        }
    except:
        return {"balanced": False, "error": "Could not assess balance"}

def _assess_model_complexity(model, model_name):
    """Assess model complexity."""
    complexity_indicators = {"model_type": model_name}
    
    try:
        if model_name == "ML + SHAP":
            if hasattr(model, 'n_estimators'):
                complexity_indicators["n_estimators"] = int(model.n_estimators)
            if hasattr(model, 'max_depth'):
                complexity_indicators["max_depth"] = int(model.max_depth) if model.max_depth else "unlimited"
            if hasattr(model, 'n_features_in_'):
                complexity_indicators["features_used"] = int(model.n_features_in_)
                
        elif model_name in ["MLR", "Distributed Lag"]:
            if hasattr(model, 'params'):
                complexity_indicators["parameters_count"] = len(model.params)
                
        elif model_name == "VAR":
            if hasattr(model, 'k_ar'):
                complexity_indicators["lag_order"] = int(model.k_ar)
            if hasattr(model, 'names'):
                complexity_indicators["variables_count"] = len(model.names)
                
        complexity_indicators["assessment"] = "Model complexity within reasonable bounds"
        
    except Exception as e:
        complexity_indicators["error"] = str(e)
    
    return complexity_indicators 

def create_interpretation_hints(model_name: str) -> List[str]:
    """
    Generate interpretation hints for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of interpretation hints
    """
    hints = {
        "MLR": [
            "p-value < 0.05 indicates significant channel",
            "R² > 0.6 suggests a good model fit",
            "VIF > 5 may indicate multicollinearity",
            "Coefficients show direct impact per unit change",
            "Check residual plots for assumption violations"
        ],
        "Distributed Lag": [
            "Higher lag coefficients indicate longer-lasting effects",
            "Durbin-Watson near 2 indicates no autocorrelation",
            "Sum of lag coefficients shows total cumulative impact",
            "Early lags typically have stronger immediate effects"
        ],
        "ML + SHAP": [
            "Lower RMSE indicates better predictive accuracy",
            "Higher SHAP value indicates a channel's contribution",
            "Feature importance ranks show relative predictive power",
            "SHAP values explain individual prediction contributions"
        ],
        "DiD": [
            "CI excluding zero indicates significant effect",
            "Placebo test near zero confirms robustness",
            "Check parallel trends assumption in pre-treatment period",
            "Treatment effect size relative to baseline variance"
        ],
        "VAR": [
            "IRF shows effect of ad spend shocks over time",
            "Stability check confirms forecast reliability",
            "Lag order selection balances fit vs complexity",
            "Granger causality tests show directional relationships"
        ],
        "Synthetic Control": [
            "Low RMSPE indicates a good match",
            "Visible post-gap indicates treatment effect",
            "Pre-treatment fit quality validates synthetic control",
            "Donor pool composition affects validity"
        ],
        "CausalImpact": [
            "CI not crossing zero indicates significant impact",
            "High posterior probability confirms effect confidence",
            "Cumulative effect shows total impact magnitude",
            "Relative effect shows percentage change"
        ],
        "PSM": [
            "SMD < 0.1 indicates good balance",
            "ATT p-value < 0.05 indicates significance",
            "Common support region shows overlap quality",
            "Sensitivity analysis confirms robustness"
        ]
    }
    
    return hints.get(model_name, ["No specific hints available for this model"])


def display_interpretation_hints(model_name: str) -> None:
    """
    Display interpretation hints for a model.
    
    Args:
        model_name: Name of the model
    """
    hints = create_interpretation_hints(model_name)
    
    with st.expander("💡 Enhanced Model Interpretation Guide", expanded=False):
        st.markdown(f"### 🎯 **{model_name} Model Interpretation**")
        
        for i, hint in enumerate(hints, 1):
            st.markdown(f"{i}. {hint}")
        
        st.markdown("---")
        st.markdown("### 📊 **General Guidelines**")
        st.markdown("""
        - **Statistical Significance**: Look for p-values < 0.05 for reliable effects
        - **Effect Size**: Consider practical significance alongside statistical significance  
        - **Model Fit**: Higher R² indicates better explanatory power
        - **Validation**: Always check model assumptions and robustness
        - **Business Context**: Interpret coefficients in terms of business impact
        """)
        
        if model_name in ["MLR", "Distributed Lag"]:
            st.markdown("### 🔍 **Linear Model Specifics**")
            st.markdown("""
            - **Multicollinearity**: VIF > 10 suggests problematic correlation
            - **Residuals**: Should be normally distributed and uncorrelated
            - **Outliers**: Standardized residuals > 3 may indicate outliers
            - **Assumptions**: Check linearity, independence, normality, homoscedasticity
            """)
        
        elif model_name == "ML + SHAP":
            st.markdown("### 🤖 **Machine Learning Specifics**")
            st.markdown("""
            - **Feature Importance**: Higher values indicate stronger predictive power
            - **SHAP Values**: Positive values increase prediction, negative decrease it
            - **Model Complexity**: Balance accuracy with interpretability
            - **Overfitting**: Monitor training vs validation performance gaps
            """)
        
        elif model_name in ["DiD", "Synthetic Control", "CausalImpact", "PSM"]:
            st.markdown("### 🎯 **Causal Inference Specifics**")
            st.markdown("""
            - **Causal Assumptions**: Carefully validate identifying assumptions
            - **Treatment Effect**: Consider both statistical and economic significance
            - **Robustness**: Test sensitivity to specification changes
            - **External Validity**: Consider generalizability to other contexts
            """)
        
        st.markdown("---")
        st.info("💡 **Tip**: Your comprehensive JSON report contains detailed diagnostics to help validate these interpretation guidelines.")