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
from statsmodels.tsa.seasonal import STL

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


def display_mlr_metrics(df: pd.DataFrame, target: str, features: List[str], model: Any = None, client_specific_vars: List[str] = None, predictions: pd.DataFrame = None) -> None:
    """
    Display MLR model diagnostics and metrics with client-specific effect analysis.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        features: Feature variable names
        model: Trained model object (for client-specific analysis)
        client_specific_vars: List of client-specific variables
    """
    try:
        # Check if this is a client-specific analysis
        if model and hasattr(model, 'summary') and client_specific_vars:
            # Client-specific analysis
            st.subheader(" Client-Specific Advertising Effects Analysis")
            
            # Display model summary
            st.markdown("**Model Summary:**")
            st.text(str(model.summary()))
            
            # Client-specific effects section
            st.subheader("Client-Specific Channel Effects")
            
            from data_analysis import analyze_client_specific_effects
            client_analysis = analyze_client_specific_effects(model, df, target, client_specific_vars)
            
            if "client_specific_effects" in client_analysis:
                # Display client-specific effects
                effects_data = []
                for term, effect in client_analysis["client_specific_effects"].items():
                    # Handle None values safely
                    coef = effect.get('coefficient', None)
                    p_val = effect.get('p_value', None)
                    ci_lower = effect.get('confidence_interval_lower', None)
                    ci_upper = effect.get('confidence_interval_upper', None)
                    
                    effects_data.append({
                        "Channel": effect.get("original_variable", term),
                        "Coefficient": f"{coef:.4f}" if coef is not None else "N/A",
                        "P-Value": f"{p_val:.4f}" if p_val is not None else "N/A",
                        "Significance": effect.get("significance", "Unknown"),
                        "Effect Direction": effect.get("effect_direction", "Unknown"),
                        "CI Lower": f"{ci_lower:.4f}" if ci_lower is not None else "N/A",
                        "CI Upper": f"{ci_upper:.4f}" if ci_upper is not None else "N/A"
                    })
                
                effects_df = pd.DataFrame(effects_data)
                st.dataframe(effects_df, use_container_width=True)
                
                # Business insights
                if "business_insights" in client_analysis:
                    st.subheader("💼 Business Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        effective_count = len(client_analysis["business_insights"]["effective_channels"])
                        st.metric("Effective Channels", effective_count)
                    with col2:
                        ineffective_count = len(client_analysis["business_insights"]["ineffective_channels"])
                        st.metric("Ineffective Channels", ineffective_count)
                    with col3:
                        uncertain_count = len(client_analysis["business_insights"]["uncertain_channels"])
                        st.metric("Uncertain Channels", uncertain_count)
                    
                    # Recommendations
                    if client_analysis["business_insights"]["recommendations"]:
                        st.subheader("Recommendations")
                        for rec in client_analysis["business_insights"]["recommendations"]:
                            st.info(rec)
                
                # Model performance metrics
                try:
                    r_squared = model.rsquared if hasattr(model, 'rsquared') else None
                    adjusted_r_squared = model.rsquared_adj if hasattr(model, 'rsquared_adj') else None
                    rmse = np.sqrt(model.mse_resid) if hasattr(model, 'mse_resid') else None
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R-squared", f"{r_squared:.4f}" if r_squared is not None else "N/A")
                    with col2:
                        st.metric("Adjusted R-squared", f"{adjusted_r_squared:.4f}" if adjusted_r_squared is not None else "N/A")
                    with col3:
                        st.metric("RMSE", f"{rmse:.4f}" if rmse is not None else "N/A")
                except Exception as metrics_error:
                    logger.error(f"Error computing model metrics: {metrics_error}")
                    st.warning("Could not compute model performance metrics")
                
                # Residual plot
                try:
                    if hasattr(model, 'resid') and hasattr(model, 'fittedvalues'):
                        resid = model.resid
                        fitted = model.fittedvalues
                        fig_resid = px.scatter(
                            x=fitted, 
                            y=resid, 
                            labels={"x": "Fitted", "y": "Residuals"}, 
                            title="Residual Diagnostic"
                        )
                        fig_resid.add_hline(y=0, line_dash="dash")
                        st.plotly_chart(fig_resid, use_container_width=True)
                    else:
                        st.warning("Could not create residual plot - model missing required attributes")
                except Exception as plot_error:
                    logger.error(f"Error creating residual plot: {plot_error}")
                    st.warning("Could not create residual plot")
                
                # Create actual vs predicted chart for client-specific analysis
                try:
                    # Use passed predictions if available, otherwise create from model
                    if predictions is not None and not predictions.empty:
                        predictions_data = predictions.copy()
                        if 'prediction' not in predictions_data.columns and len(predictions_data.columns) > 0:
                            # Use the first column as prediction if no 'prediction' column exists
                            predictions_data['prediction'] = predictions_data.iloc[:, 0]
                    else:
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
                        logger.info("Actual vs predicted chart displayed successfully for MLR client-specific analysis")
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
                        logger.info("Actual vs predicted chart displayed successfully for MLR client-specific analysis with fallback")
                
                except Exception as chart_error:
                    logger.error(f"Error creating actual vs predicted chart for MLR client-specific analysis: {str(chart_error)}")
                    st.warning(f"Could not create actual vs predicted chart: {str(chart_error)}")
                    st.info("This may be due to date column detection issues or data format problems.")
                
            else:
                st.warning("No client-specific effects found in the model.")
        
        else:
            # Standard MLR analysis
            X = sm.add_constant(df[features])
            y = df[target]
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})

            st.subheader("Standard MLR Model Diagnostics")
            st.markdown(
                f"**R² / Adj R²:** {model.rsquared:.3f} / {model.rsquared_adj:.3f}\n\n"
                f"**F-statistic (p):** {model.fvalue:.2f} ({model.f_pvalue:.3g})"
            )

            # Coefficient table with significance indicators
            coef_tbl = model.summary2().tables[1].rename(columns={"Coef.": "Coef"})
            coef_tbl["Signif"] = coef_tbl["P>|t|"].apply(
                lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
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
                # Use passed predictions if available, otherwise create from model
                if predictions is not None and not predictions.empty:
                    predictions_data = predictions.copy()
                    if 'prediction' not in predictions_data.columns and len(predictions_data.columns) > 0:
                        # Use the first column as prediction if no 'prediction' column exists
                        predictions_data['prediction'] = predictions_data.iloc[:, 0]
                else:
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
        dw = durbin_watson(y - sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1}).fittedvalues)
        
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
        st.subheader("✨ Machine Learning + SHAP Analysis")
        
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
        
        # Model Performance Metrics
        st.markdown("### 📊 Model Performance Metrics")
        
        # Calculate additional metrics
        from sklearn.metrics import r2_score
        r2 = r2_score(df[target], preds)
        
        # Calculate relative metrics
        baseline_mean = df[target].mean()
        relative_rmse = rmse / baseline_mean if baseline_mean != 0 else None
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
        with col4:
            st.metric("Relative RMSE", f"{relative_rmse:.3f}" if relative_rmse is not None else "N/A")
        
        # Model quality assessment
        if r2 > 0.9:
            quality = "Excellent"
            quality_color = "🟢"
        elif r2 > 0.8:
            quality = "Good"
            quality_color = "🟡"
        elif r2 > 0.6:
            quality = "Fair"
            quality_color = "🟠"
        else:
            quality = "Poor"
            quality_color = "🔴"
        
        st.markdown(f"**{quality_color} Model Quality: {quality}** (R² = {r2:.4f})")

        # Global SHAP summary (bar plot)
        try:
            import shap
            explainer = shap.Explainer(model)
            shap_values = explainer(X_pred)
            
            # Create interactive SHAP summary plot with Plotly
            st.markdown("### 🔍 SHAP Feature Importance Analysis")
            st.info("📊 SHAP values explain individual predictions by quantifying each feature's contribution.")
            
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
            st.markdown("### 💧 SHAP Waterfall Analysis")
            st.info("🔍 Shows how each feature contributes to a specific prediction, starting from the baseline.")
            
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
            st.markdown("### 📈 Feature Impact Distribution")
            st.info("📆 Distribution of SHAP values across all observations shows feature behavior patterns.")
            
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
            with st.expander("SHAP Debug Information"):
                st.write("**Model Info:**")
                st.write(f"- Model type: {type(model).__name__}")
                st.write(f"- Has stored encoders: {hasattr(model, 'label_encoders_')}")
                st.write(f"- Has feature names: {hasattr(model, 'feature_names_')}")
                st.write(f"- X_pred shape: {X_pred.shape}")
                st.write(f"- X_pred dtypes: {X_pred.dtypes.to_dict()}")
                if hasattr(model, 'categorical_cols_'):
                    st.write(f"- Categorical columns: {model.categorical_cols_}")
        
        # Add business interpretation section before KPI dashboard
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_ml_shap_interpretation(
            r2, rmse, mae, quality,
            shap_importance if 'shap_importance' in locals() else None,
            feature_cols
        )
        
        # Display interpretation with appropriate styling
        if quality in ["Excellent", "Good"]:
            st.success(interpretation["summary"])
        elif quality == "Fair":
            st.warning(interpretation["summary"])
        else:
            st.error(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        # KPI Dashboard Section
        st.markdown("### 📊 Model Performance Dashboard")
        create_kpi_dashboard(df, target, feature_cols, preds, {'rmse': rmse, 'mae': mae, 'r2': r2}, shap_importance if 'shap_importance' in locals() else None)
        
        # Create actual vs predicted comparison chart
        st.markdown("### 📈 Actual vs Predicted Comparison")
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
        
        logger.info("Enhanced ML + SHAP metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying ML + SHAP metrics: {str(e)}")
        st.error(f"Error computing ML + SHAP metrics: {str(e)}")
        
        # Add debug information for troubleshooting
        with st.expander("Error Debug Information"):
            st.write("**Error Details:**")
            st.write(f"- Error: {str(e)}")
            st.write(f"- Target: {target}")
            st.write(f"- DataFrame shape: {df.shape}")
            st.write(f"- DataFrame columns: {list(df.columns)}")
            st.write(f"- Model type: {type(model).__name__}")
            if hasattr(model, 'feature_names_'):
                st.write(f"- Expected features: {model.feature_names_}")
            st.info("Try retraining the model or check your data for inconsistencies.")


def create_ml_shap_interpretation(r2: float, rmse: float, mae: float, quality: str,
                                shap_importance: np.ndarray = None, feature_names: List[str] = None) -> Dict[str, any]:
    """
    Create business interpretation for ML + SHAP results.
    
    Args:
        r2: R-squared score
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        quality: Model quality assessment (Excellent/Good/Fair/Poor)
        shap_importance: SHAP feature importance values
        feature_names: List of feature names
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Create summary based on model quality
    if quality == "Excellent":
        summary = f"Machine Learning model demonstrates excellent predictive performance (R² = {r2:.4f}). The model accurately captures complex patterns with low prediction errors (RMSE: {rmse:.4f}, MAE: {mae:.4f})."
    elif quality == "Good":
        summary = f"Machine Learning model shows good predictive performance (R² = {r2:.4f}). The model captures most patterns effectively with reasonable prediction accuracy (RMSE: {rmse:.4f}, MAE: {mae:.4f})."
    elif quality == "Fair":
        summary = f"Machine Learning model provides fair predictive performance (R² = {r2:.4f}). There is room for improvement in capturing data patterns (RMSE: {rmse:.4f}, MAE: {mae:.4f})."
    else:
        summary = f"Machine Learning model shows poor predictive performance (R² = {r2:.4f}). Significant improvements needed for reliable predictions (RMSE: {rmse:.4f}, MAE: {mae:.4f})."
    
    # Create detailed explanation
    details = f"""
    **Machine Learning + SHAP Analysis:**
    • Advanced ML model (likely Gradient Boosting) with explainable AI capabilities
    • R² of {r2:.4f} indicates the model explains {r2*100:.1f}% of the variance in the target variable
    • SHAP (SHapley Additive exPlanations) provides local and global feature importance
    • Each prediction is decomposed into feature contributions, enabling transparency
    
    **Model Performance:**
    • RMSE: {rmse:.4f} (average prediction error magnitude)
    • MAE: {mae:.4f} (average absolute prediction error)
    • Model complexity allows capturing non-linear relationships and feature interactions
    • {'Excellent' if quality == 'Excellent' else 'Good' if quality == 'Good' else 'Moderate' if quality == 'Fair' else 'Poor'} balance between model accuracy and explainability
    
    **Feature Importance Insights:**
    • SHAP values provide both magnitude and direction of feature impacts
    • {'Feature importance analysis available' if shap_importance is not None else 'Feature importance analysis not available'}
    • {'Waterfall charts show individual prediction breakdowns' if shap_importance is not None else 'Consider running SHAP analysis for detailed insights'}
    • Feature impact distributions reveal consistency across different observations
    """
    
    # Create recommendations
    recommendations = []
    
    if quality in ["Excellent", "Good"]:
        recommendations.extend([
            "Model performance is strong - suitable for production deployment",
            "Use SHAP insights to understand key business drivers and feature relationships",
            "Leverage individual prediction explanations for stakeholder communication",
            "Monitor model performance over time and retrain with new data as needed"
        ])
        
        if shap_importance is not None and feature_names is not None:
            # Find top features
            top_features = [feature_names[i] for i in np.argsort(shap_importance)[-3:]]
            recommendations.append(f"Focus on top driving features: {', '.join(reversed(top_features))}")
            
    elif quality == "Fair":
        recommendations.extend([
            "⚠️ Model performance is acceptable but has room for improvement",
            "Consider feature engineering to capture additional patterns",
            "Explore hyperparameter tuning or alternative ML algorithms",
            "Use SHAP analysis to identify underperforming or redundant features"
        ])
    else:
        recommendations.extend([
            "⚠️ Model performance needs significant improvement before production use",
            "Review data quality and feature selection process",
            "Consider data preprocessing improvements (outlier handling, feature scaling)",
            "Evaluate if the problem is suitable for the current ML approach"
        ])
    
    # SHAP-specific recommendations
    if shap_importance is not None:
        recommendations.extend([
            "Use SHAP feature importance to guide feature selection and engineering",
            "Analyze SHAP interaction values to understand feature relationships",
            "Create SHAP-based business rules for decision support",
            "Use waterfall plots to explain individual high-impact predictions"
        ])
    else:
        recommendations.append("⚠️ Generate SHAP analysis for better model interpretability and insights")
    
    # Technical recommendations
    recommendations.extend([
        "Implement cross-validation to assess model generalization",
        "Monitor for data drift in production to detect when retraining is needed",
        "Consider ensemble methods to improve robustness",
        "Validate model assumptions and check for bias in predictions"
    ])
    
    # Business value recommendations
    if quality in ["Excellent", "Good"]:
        recommendations.extend([
            "Integrate model predictions into business decision-making processes",
            "Create automated alerts for predictions exceeding certain thresholds",
            "Develop A/B testing framework to measure business impact"
        ])
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_did_metrics(model: Any) -> None:
    """
    Display comprehensive DiD model diagnostics and metrics.
    
    Args:
        model: Trained DiD model (statsmodels OLS result)
    """
    try:
        st.subheader("✨ Difference-in-Differences Analysis")
        
        # Model Summary Section (like OLS output)
        st.markdown("**Model Summary:**")
        st.text(str(model.summary()))
        
        # Key DiD Results Section
        st.markdown("### 🎯 Treatment Effect Results")
        
        # Extract key statistics
        ate = model.params.get("treated:post", float("nan"))
        ci_low, ci_high = model.conf_int().loc["treated:post"]
        pval = model.pvalues.get("treated:post", float("nan"))
        std_err = model.bse.get("treated:post", float("nan"))
        t_stat = model.tvalues.get("treated:post", float("nan"))
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Treatment Effect", f"{ate:.4f}")
        with col2:
            r_squared = model.rsquared if hasattr(model, 'rsquared') else None
            st.metric("R-squared", f"{r_squared:.4f}" if r_squared is not None else "N/A")
        with col3:
            adj_r_squared = model.rsquared_adj if hasattr(model, 'rsquared_adj') else None
            st.metric("Adj R-squared", f"{adj_r_squared:.4f}" if adj_r_squared is not None else "N/A")
        with col4:
            f_stat = model.fvalue if hasattr(model, 'fvalue') else None
            st.metric("F-statistic", f"{f_stat:.2f}" if f_stat is not None else "N/A")
        
        # Coefficient table with color-coded significance
        st.markdown("### 📊 Coefficient Analysis")
        coef_tbl = model.summary2().tables[1].rename(columns={"Coef.": "Coef"})
        coef_tbl["Signif"] = coef_tbl["P>|t|"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        )
        
        # Color-code significance
        styled_coef = coef_tbl.style.applymap(
            lambda p: "color:green; font-weight:bold;" if p < SIGNIFICANCE_LEVEL else "color:red;", 
            subset=["P>|t|"]
        )
        st.dataframe(styled_coef, use_container_width=True)
        
        # Treatment Effect Detail
        st.markdown("### 🔍 Treatment Effect Details")
        
        # Significance indicator with color
        is_significant = pval < SIGNIFICANCE_LEVEL
        significance_color = "🟢" if is_significant else "🔴"
        significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"
        
        effect_direction = "Positive" if ate > 0 else "Negative" if ate < 0 else "No"
        effect_color = "📈" if ate > 0 else "📉" if ate < 0 else "➡️"
        
        st.markdown(f"**{significance_color} {significance_text}** (p = {pval:.4f})")
        st.markdown(f"**{effect_color} {effect_direction} Treatment Effect:** {ate:.4f}")
        st.markdown(f"**📏 Standard Error:** {std_err:.4f}")
        st.markdown(f"**📊 t-statistic:** {t_stat:.4f}")
        st.markdown(f"**🎯 95% Confidence Interval:** [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Model Diagnostics
        st.markdown("### 🔬 Model Diagnostics")
        
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            n_obs = model.nobs if hasattr(model, 'nobs') else None
            st.metric("Observations", f"{int(n_obs)}" if n_obs is not None else "N/A")
            
        with diag_col2:
            aic = model.aic if hasattr(model, 'aic') else None
            st.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")
            
        with diag_col3:
            bic = model.bic if hasattr(model, 'bic') else None
            st.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_did_interpretation(ate, pval, ci_low, ci_high, is_significant)
        
        # Display interpretation with appropriate styling
        if is_significant:
            st.success(interpretation["summary"])
        else:
            st.warning(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced DiD metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying DiD metrics: {str(e)}")
        st.error(f"Error computing DiD metrics: {str(e)}")


def create_did_interpretation(ate: float, pval: float, ci_low: float, ci_high: float, is_significant: bool) -> Dict[str, any]:
    """
    Create business interpretation for DiD results.
    
    Args:
        ate: Average Treatment Effect
        pval: P-value for the treatment effect
        ci_low: Lower bound of confidence interval
        ci_high: Upper bound of confidence interval
        is_significant: Whether the effect is statistically significant
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Determine effect size interpretation
    abs_ate = abs(ate)
    if abs_ate < 0.01:
        effect_size = "negligible"
    elif abs_ate < 0.05:
        effect_size = "small"
    elif abs_ate < 0.1:
        effect_size = "moderate"
    else:
        effect_size = "large"
    
    # Create summary
    if is_significant:
        direction = "increased" if ate > 0 else "decreased"
        summary = f"The intervention had a statistically significant {direction} effect of {ate:.4f} units (p = {pval:.4f}). Effect size is {effect_size}."
    else:
        summary = f"No statistically significant treatment effect detected (p = {pval:.4f}). The intervention does not appear to have a meaningful impact."
    
    # Create detailed explanation
    details = f"""
    **Statistical Interpretation:**
    • The Average Treatment Effect (ATE) of {ate:.4f} represents the average difference between treated and control groups after accounting for baseline differences
    • The 95% confidence interval [{ci_low:.4f}, {ci_high:.4f}] {'excludes' if is_significant else 'includes'} zero
    • With a p-value of {pval:.4f}, we {'can' if is_significant else 'cannot'} reject the null hypothesis of no treatment effect at the 5% significance level
    
    **Effect Size Context:**
    • The effect magnitude is considered {effect_size} ({abs_ate:.4f} units)
    • {'This suggests practical significance in addition to statistical significance' if is_significant and effect_size in ['moderate', 'large'] else 'Consider whether this effect size is practically meaningful for your business context'}
    """
    
    # Create recommendations
    recommendations = []
    
    if is_significant:
        if ate > 0:
            recommendations.extend([
                "The intervention shows positive impact - consider scaling or expanding the treatment",
                "Monitor for sustainability of the effect over longer time periods",
                "Investigate which aspects of the intervention drive the positive results"
            ])
        else:
            recommendations.extend([
                "The intervention shows negative impact - investigate potential causes",
                "Consider discontinuing or modifying the intervention approach",
                "Analyze if the negative effect is due to implementation issues or inherent design flaws"
            ])
    else:
        recommendations.extend([
            "No significant effect detected - the intervention may not be effective as implemented",
            "Consider increasing sample size or treatment intensity for future experiments",
            "Investigate potential moderating factors that might influence treatment effectiveness",
            "Review intervention design and implementation for potential improvements"
        ])
    
    # Add general recommendations
    recommendations.extend([
        "Validate results with additional data or robustness checks",
        "Consider heterogeneous treatment effects across different subgroups",
        "Assess cost-benefit implications of the intervention"
    ])
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_var_metrics(results: Any) -> None:
    """
    Display comprehensive VAR model diagnostics and metrics.
    
    Args:
        results: Trained VAR model results
    """
    try:
        st.subheader("✨ Vector Autoregression (VAR) Analysis")
        
        # Model Summary Section
        st.markdown("**VAR Model Summary:**")
        try:
            summary_str = str(results.summary())
            st.text(summary_str)
        except:
            st.info("Detailed model summary not available for this VAR implementation")
        
        # Key Model Statistics
        st.markdown("### 📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lag Order", f"{results.k_ar}")
        with col2:
            st.metric("AIC", f"{results.aic:.2f}")
        with col3:
            st.metric("BIC", f"{results.bic:.2f}")
        with col4:
            st.metric("FPE", f"{results.fpe:.6f}" if hasattr(results, 'fpe') else "N/A")
        
        # Stability Analysis
        st.markdown("### 🔍 Stability Analysis")
        is_stable = results.is_stable()
        stability_color = "🟢" if is_stable else "🔴"
        stability_text = "System is Stable" if is_stable else "System is Unstable"
        
        st.markdown(f"**{stability_color} {stability_text}**")
        
        if is_stable:
            st.success("The VAR system satisfies the stability condition. All eigenvalues lie inside the unit circle.")
        else:
            st.error("⚠️ The VAR system is unstable! Some eigenvalues lie outside the unit circle. Results may be unreliable.")
        
        # Equation-wise Statistics
        st.markdown("### 📊 Equation-wise Model Statistics")
        
        # Create table for equation statistics
        equation_stats = []
        endog_names = results.names if hasattr(results, 'names') else [f"Equation_{i+1}" for i in range(results.neqs)]
        
        for i, eq_name in enumerate(endog_names):
            try:
                # Extract R-squared and other stats for each equation
                eq_rsquared = results.rsquared[i] if hasattr(results, 'rsquared') and len(results.rsquared) > i else None
                eq_rsquared_adj = results.rsquared_adj[i] if hasattr(results, 'rsquared_adj') and len(results.rsquared_adj) > i else None
                
                equation_stats.append({
                    "Equation": eq_name,
                    "R-squared": f"{eq_rsquared:.4f}" if eq_rsquared is not None else "N/A",
                    "Adj R-squared": f"{eq_rsquared_adj:.4f}" if eq_rsquared_adj is not None else "N/A",
                    "Observations": results.nobs if hasattr(results, 'nobs') else "N/A"
                })
            except Exception as eq_error:
                logger.warning(f"Could not extract stats for equation {eq_name}: {eq_error}")
                equation_stats.append({
                    "Equation": eq_name,
                    "R-squared": "N/A",
                    "Adj R-squared": "N/A",
                    "Observations": "N/A"
                })
        
        if equation_stats:
            eq_df = pd.DataFrame(equation_stats)
            st.dataframe(eq_df, use_container_width=True)
        
        # Granger Causality Tests (if available)
        st.markdown("### 🔍 Granger Causality Analysis")
        try:
            st.info("Granger causality tests help identify which variables predict others in the VAR system.")
            
            # If we have variable names, show potential causality relationships
            if hasattr(results, 'names') and len(results.names) > 1:
                st.markdown("**Variables in the system:**")
                for name in results.names:
                    st.markdown(f"• {name}")
                
                st.markdown("💡 **Interpretation**: Use `results.test_causality()` to test specific Granger causality hypotheses.")
            
        except Exception as causality_error:
            logger.warning(f"Granger causality analysis error: {causality_error}")
        
        # Model Diagnostics
        st.markdown("### 🔬 Model Diagnostics")
        
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            st.metric("Total Observations", f"{results.nobs}" if hasattr(results, 'nobs') else "N/A")
        with diag_col2:
            st.metric("Number of Variables", f"{results.neqs}" if hasattr(results, 'neqs') else "N/A")
        with diag_col3:
            st.metric("Degrees of Freedom", f"{results.df_resid}" if hasattr(results, 'df_resid') else "N/A")
        
        # Residual Analysis
        st.markdown("### 📊 Residual Diagnostics")
        
        try:
            # Check for residual autocorrelation and heteroscedasticity
            st.info("Residual analysis helps validate model assumptions about error terms.")
            
            # If residuals are available, compute basic statistics
            if hasattr(results, 'resid'):
                resid = results.resid
                st.markdown("**Residual Summary Statistics:**")
                
                resid_stats = pd.DataFrame({
                    'Mean': resid.mean(),
                    'Std': resid.std(),
                    'Min': resid.min(),
                    'Max': resid.max()
                })
                st.dataframe(resid_stats.T, use_container_width=True)
                
        except Exception as resid_error:
            logger.warning(f"Residual analysis error: {resid_error}")
            st.info("Detailed residual diagnostics not available for this VAR model.")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_var_interpretation(results, is_stable)
        
        if is_stable:
            st.success(interpretation["summary"])
        else:
            st.warning(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced VAR metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying VAR metrics: {str(e)}")
        st.error(f"Error computing VAR metrics: {str(e)}")


def create_var_interpretation(results: Any, is_stable: bool) -> Dict[str, any]:
    """
    Create business interpretation for VAR results.
    
    Args:
        results: VAR model results
        is_stable: Whether the VAR system is stable
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Extract key information
    lag_order = results.k_ar if hasattr(results, 'k_ar') else None
    aic = results.aic if hasattr(results, 'aic') else None
    bic = results.bic if hasattr(results, 'bic') else None
    n_vars = results.neqs if hasattr(results, 'neqs') else None
    
    # Create summary
    if is_stable:
        summary = f"VAR({lag_order}) model is statistically stable and suitable for analysis. The system captures dynamic relationships between {n_vars} variables with {lag_order} lags."
    else:
        summary = f"⚠️ VAR({lag_order}) model shows instability. Results should be interpreted with caution as the system may not converge to equilibrium."
    
    # Create detailed explanation
    details = f"""
    **Model Specification:**
    • VAR model with {lag_order} lag{'s' if lag_order != 1 else ''} explaining {n_vars} endogenous variables
    • Each variable is regressed on its own lags and lags of all other variables in the system
    • AIC: {aic:.2f}, BIC: {bic:.2f} (lower values indicate better model fit vs complexity trade-off)
    
    **System Properties:**
    • {'Stability condition satisfied' if is_stable else 'Stability condition violated'} - {'reliable' if is_stable else 'potentially unreliable'} for forecasting
    • VAR models capture bidirectional causality and dynamic interactions between variables
    • Useful for understanding how shocks to one variable propagate through the system
    
    **Applications:**
    • Impulse Response Analysis: Trace effects of shocks over time
    • Forecast Error Variance Decomposition: Understand variable contributions to forecast uncertainty
    • Granger Causality: Test predictive relationships between variables
    """
    
    # Create recommendations
    recommendations = []
    
    if is_stable:
        recommendations.extend([
            "Proceed with impulse response analysis to understand shock propagation",
            "Conduct Granger causality tests to identify predictive relationships",
            "Use forecast error variance decomposition to understand variable interactions",
            "Consider the model suitable for forecasting within reasonable time horizons"
        ])
    else:
        recommendations.extend([
            "⚠️ Address stability issues before proceeding with analysis",
            "Consider reducing the lag order or removing problematic variables",
            "Check for unit roots in the data - consider VECM if cointegration is present",
            "Validate model specification and consider alternative modeling approaches"
        ])
    
    # Model selection recommendations
    if aic is not None and bic is not None:
        recommendations.extend([
            f"Current model selection: AIC={aic:.2f}, BIC={bic:.2f}",
            "Consider testing alternative lag orders using information criteria",
            "Compare with simpler models (AR) or more complex specifications if needed"
        ])
    
    # General VAR recommendations
    recommendations.extend([
        "Validate residuals for autocorrelation and heteroscedasticity",
        "Check structural breaks in the data period",
        "Consider economic theory when interpreting causal relationships",
        "Use cross-validation for out-of-sample forecast evaluation"
    ])
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_synthetic_control_metrics(predictions: pd.DataFrame) -> None:
    """
    Display comprehensive Synthetic Control model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame with Actual and Synthetic columns
    """
    try:
        st.subheader("✨ Synthetic Control Method Analysis")
        
        # Data validation
        if "Actual" not in predictions.columns or "Synthetic" not in predictions.columns:
            st.error("Required columns 'Actual' and 'Synthetic' not found in predictions DataFrame.")
            return
        
        actual = predictions["Actual"]
        synthetic = predictions["Synthetic"]
        
        # Comprehensive Model Summary
        st.markdown("### 📊 Model Performance Metrics")
        
        # Calculate multiple fit statistics
        rmspe = np.sqrt(((actual - synthetic) ** 2).mean())
        mspe = ((actual - synthetic) ** 2).mean()
        mae = np.abs(actual - synthetic).mean()
        mape = np.mean(np.abs((actual - synthetic) / actual)) * 100 if (actual != 0).all() else None
        
        # R-squared like measure
        ss_res = np.sum((actual - synthetic) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared_like = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSPE", f"{rmspe:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R²-like", f"{r_squared_like:.4f}" if r_squared_like is not None else "N/A")
        with col4:
            st.metric("MAPE (%)", f"{mape:.2f}%" if mape is not None else "N/A")
        
        # Treatment Effect Analysis
        st.markdown("### 🎯 Treatment Effect Analysis")
        
        # Check if we have post/pre treatment indicators
        treatment_effects = actual - synthetic
        
        if "Post" in predictions.columns and "Treated" in predictions.columns:
            # Split analysis into pre/post treatment
            pre_treatment = predictions[predictions["Post"] == 0]
            post_treatment = predictions[(predictions["Post"] == 1) & (predictions["Treated"] == 1)]
            
            if len(pre_treatment) > 0 and len(post_treatment) > 0:
                pre_rmspe = np.sqrt(((pre_treatment["Actual"] - pre_treatment["Synthetic"]) ** 2).mean())
                post_effects = post_treatment["Actual"] - post_treatment["Synthetic"]
                
                avg_treatment_effect = post_effects.mean()
                treatment_effect_std = post_effects.std()
                cumulative_effect = post_effects.sum()
                
                # Display treatment effect metrics
                effect_col1, effect_col2, effect_col3 = st.columns(3)
                
                with effect_col1:
                    st.metric("Average Treatment Effect", f"{avg_treatment_effect:.4f}")
                with effect_col2:
                    st.metric("Pre-Treatment RMSPE", f"{pre_rmspe:.4f}")
                with effect_col3:
                    st.metric("Cumulative Effect", f"{cumulative_effect:.4f}")
                
                # Effect significance assessment (informal)
                effect_magnitude = abs(avg_treatment_effect)
                pre_treatment_std = pre_treatment["Actual"].std()
                effect_size = effect_magnitude / pre_treatment_std if pre_treatment_std > 0 else None
                
                if effect_size is not None:
                    if effect_size > 0.8:
                        effect_assessment = "Large"
                        effect_color = "🟢"
                    elif effect_size > 0.5:
                        effect_assessment = "Medium"
                        effect_color = "🟡"
                    elif effect_size > 0.2:
                        effect_assessment = "Small"
                        effect_color = "🟠"
                    else:
                        effect_assessment = "Negligible"
                        effect_color = "⚪"
                    
                    st.markdown(f"**{effect_color} Effect Size: {effect_assessment}** (Cohen's d ≈ {effect_size:.2f})")
                
        else:
            # Overall treatment effect without pre/post split
            avg_effect = treatment_effects.mean()
            cumulative_effect = treatment_effects.sum()
            
            st.metric("Average Effect", f"{avg_effect:.4f}")
            st.metric("Cumulative Effect", f"{cumulative_effect:.4f}")
        
        # Model Fit Quality Assessment
        st.markdown("### 🔍 Model Fit Quality")
        
        # Fit quality indicators
        fit_quality = "Excellent" if rmspe < 0.05 else "Good" if rmspe < 0.1 else "Fair" if rmspe < 0.2 else "Poor"
        fit_color = "🟢" if fit_quality == "Excellent" else "🟡" if fit_quality == "Good" else "🟠" if fit_quality == "Fair" else "🔴"
        
        st.markdown(f"**{fit_color} Overall Fit Quality: {fit_quality}** (RMSPE: {rmspe:.4f})")
        
        # Residual analysis
        residuals = actual - synthetic
        residual_autocorr = residuals.autocorr() if hasattr(residuals, 'autocorr') else None
        
        if residual_autocorr is not None:
            autocorr_assessment = "Low" if abs(residual_autocorr) < 0.3 else "Moderate" if abs(residual_autocorr) < 0.7 else "High"
            st.markdown(f"**Residual Autocorrelation: {autocorr_assessment}** ({residual_autocorr:.3f})")
        
        # Weights Analysis (if available)
        if "Treatment_Effect" in predictions.columns or any(col.startswith("Weight_") for col in predictions.columns):
            st.markdown("### ⚙️ Synthetic Control Weights")
            st.info("Weights analysis shows which control units contribute most to the synthetic control.")
        
        # Create actual vs synthetic comparison chart
        st.markdown("### 📈 Actual vs Synthetic Comparison")
        try:
            # Find date column in predictions
            date_cols = [col for col in predictions.columns if pd.api.types.is_datetime64_any_dtype(predictions[col])]
            if not date_cols:
                date_cols = [col for col in predictions.columns if 'date' in col.lower()]
            
            if date_cols:
                date_col = date_cols[0]
                
                # Prepare data for the chart
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
                st.info("Date column not found for time series visualization.")
                
        except Exception as chart_error:
            st.warning(f"Could not create actual vs synthetic chart: {str(chart_error)}")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_synthetic_control_interpretation(
            rmspe, avg_treatment_effect if 'avg_treatment_effect' in locals() else treatment_effects.mean(),
            fit_quality, r_squared_like
        )
        
        if fit_quality in ["Excellent", "Good"]:
            st.success(interpretation["summary"])
        elif fit_quality == "Fair":
            st.warning(interpretation["summary"])
        else:
            st.error(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced Synthetic Control metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying Synthetic Control metrics: {str(e)}")
        st.error(f"Error computing Synthetic Control metrics: {str(e)}")


def create_synthetic_control_interpretation(rmspe: float, treatment_effect: float, fit_quality: str, r_squared_like: float = None) -> Dict[str, any]:
    """
    Create business interpretation for Synthetic Control results.
    
    Args:
        rmspe: Root Mean Squared Prediction Error
        treatment_effect: Average treatment effect
        fit_quality: Quality of the synthetic control fit
        r_squared_like: R-squared like measure
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Determine effect magnitude
    abs_effect = abs(treatment_effect)
    effect_direction = "positive" if treatment_effect > 0 else "negative" if treatment_effect < 0 else "no"
    
    # Create summary based on fit quality and effect
    if fit_quality in ["Excellent", "Good"]:
        if effect_direction != "no":
            summary = f"High-quality synthetic control detected a {effect_direction} treatment effect of {treatment_effect:.4f} units. The synthetic control provides a reliable counterfactual (RMSPE: {rmspe:.4f})."
        else:
            summary = f"High-quality synthetic control found no meaningful treatment effect. The intervention appears to have minimal impact."
    elif fit_quality == "Fair":
        summary = f"Moderate-quality synthetic control suggests a {effect_direction} treatment effect of {treatment_effect:.4f} units. Results should be interpreted with some caution due to moderate fit quality (RMSPE: {rmspe:.4f})."
    else:
        summary = f"⚠️ Poor synthetic control fit (RMSPE: {rmspe:.4f}). Treatment effect estimates may be unreliable. Consider alternative methodologies."
    
    # Create detailed explanation
    details = f"""
    **Synthetic Control Method:**
    • Creates a weighted combination of control units to match the treated unit's pre-treatment characteristics
    • RMSPE of {rmspe:.4f} indicates {'excellent' if rmspe < 0.05 else 'good' if rmspe < 0.1 else 'fair' if rmspe < 0.2 else 'poor'} pre-treatment fit
    • Post-treatment differences are attributed to the intervention (treatment effect)
    
    **Treatment Effect Assessment:**
    • Average effect: {treatment_effect:.4f} units ({effect_direction} direction)
    • Effect interpretation depends on the quality of the synthetic control construction
    • {'Strong' if fit_quality in ['Excellent', 'Good'] else 'Moderate' if fit_quality == 'Fair' else 'Weak'} evidence for causal inference
    
    **Model Assumptions:**
    • No spillover effects between treated and control units
    • Control units not affected by unobserved shocks that also affect the treated unit
    • Sufficient pre-treatment periods to establish baseline patterns
    """
    
    # Create recommendations
    recommendations = []
    
    if fit_quality in ["Excellent", "Good"]:
        if abs_effect > 0.01:  # Meaningful effect threshold
            recommendations.extend([
                f"Strong evidence for {effect_direction} treatment effect - consider {'scaling' if treatment_effect > 0 else 'discontinuing'} the intervention",
                "Conduct robustness checks with alternative donor pools or time periods",
                "Investigate mechanisms driving the treatment effect",
                "Monitor for treatment effect persistence over time"
            ])
        else:
            recommendations.extend([
                "No significant treatment effect detected despite good model fit",
                "Consider intervention modifications or alternative approaches",
                "Examine if the intervention was implemented as designed",
                "Look for heterogeneous effects across subgroups"
            ])
    else:
        recommendations.extend([
            "⚠️ Poor synthetic control fit limits causal inference reliability",
            "Consider expanding the donor pool or extending pre-treatment period",
            "Evaluate alternative matching methods or covariates",
            "Cross-validate results with other causal inference methods (DiD, etc.)"
        ])
    
    # Model-specific recommendations
    recommendations.extend([
        "Examine individual donor unit weights to understand synthetic control composition",
        "Conduct placebo tests using control units as fake treated units",
        "Perform inference using permutation-based statistical tests",
        "Visualize treatment effect over time to assess temporal patterns"
    ])
    
    # Validation recommendations
    if r_squared_like is not None and r_squared_like < 0.7:
        recommendations.append("Consider improving synthetic control fit - low R²-like measure suggests poor counterfactual quality")
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_causal_impact_metrics(predictions: pd.DataFrame, df_original: pd.DataFrame, target: str) -> None:
    """
    Display comprehensive CausalImpact model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame with prediction and optionally confidence intervals
        df_original: Original DataFrame with actual target values
        target: Target variable name
    """
    try:
        st.subheader("✨ Bayesian Causal Impact Analysis")
        
        # Data validation
        if target not in df_original.columns:
            st.error(f"Target column '{target}' not found in original data.")
            return
        
        if "prediction" not in predictions.columns:
            st.error("Prediction column not found in predictions DataFrame.")
            return
        
        actual = df_original[target]
        predicted = predictions["prediction"]
        
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]
        
        # Calculate comprehensive metrics
        treatment_effects = actual - predicted
        
        # Model Performance Metrics
        st.markdown("### 📊 Model Performance Metrics")
        
        cumulative_effect = treatment_effects.sum()
        average_effect = treatment_effects.mean()
        effect_std = treatment_effects.std()
        
        # Calculate confidence metrics if available
        has_confidence_intervals = "prediction_lower" in predictions.columns and "prediction_upper" in predictions.columns
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cumulative Effect", f"{cumulative_effect:.4f}")
        with col2:
            st.metric("Average Effect", f"{average_effect:.4f}")
        with col3:
            st.metric("Effect Std Dev", f"{effect_std:.4f}")
        with col4:
            # Model fit measure
            prediction_accuracy = 1 - (np.mean((actual - predicted) ** 2) / np.var(actual)) if np.var(actual) > 0 else None
            st.metric("Prediction R²", f"{prediction_accuracy:.4f}" if prediction_accuracy is not None else "N/A")
        
        # Posterior Analysis (if confidence intervals available)
        if has_confidence_intervals:
            st.markdown("### 🎯 Posterior Statistics & Uncertainty")
            
            pred_lower = predictions["prediction_lower"].iloc[:min_len]
            pred_upper = predictions["prediction_upper"].iloc[:min_len]
            
            # Calculate credible intervals for effects
            effect_lower = actual - pred_upper  # Note: bounds are flipped for effects
            effect_upper = actual - pred_lower
            
            cumulative_effect_lower = effect_lower.sum()
            cumulative_effect_upper = effect_upper.sum()
            
            # Display credible intervals
            ci_col1, ci_col2, ci_col3 = st.columns(3)
            
            with ci_col1:
                st.metric("Cumulative Effect CI", 
                         f"[{cumulative_effect_lower:.3f}, {cumulative_effect_upper:.3f}]")
            with ci_col2:
                # Check if effect is "significant" (CI doesn't include 0)
                effect_significant = (cumulative_effect_lower > 0 and cumulative_effect_upper > 0) or \
                                   (cumulative_effect_lower < 0 and cumulative_effect_upper < 0)
                significance_color = "🟢" if effect_significant else "🔴"
                significance_text = "Significant" if effect_significant else "Not Significant"
                st.markdown(f"**{significance_color} {significance_text}**")
            
            with ci_col3:
                # Posterior probability of positive effect
                prob_positive = np.mean(treatment_effects > 0) * 100
                st.metric("P(Effect > 0)", f"{prob_positive:.1f}%")
        
        # Effect Magnitude Assessment
        st.markdown("### 🔍 Effect Magnitude & Direction")
        
        # Determine effect size and direction
        baseline_std = actual.std()
        effect_size = abs(average_effect) / baseline_std if baseline_std > 0 else None
        
        if effect_size is not None:
            if effect_size > 0.8:
                magnitude = "Large"
                magnitude_color = "🟢"
            elif effect_size > 0.5:
                magnitude = "Medium"
                magnitude_color = "🟡"
            elif effect_size > 0.2:
                magnitude = "Small"
                magnitude_color = "🟠"
            else:
                magnitude = "Negligible"
                magnitude_color = "⚪"
        else:
            magnitude = "Unknown"
            magnitude_color = "⚪"
        
        direction = "Positive" if average_effect > 0 else "Negative" if average_effect < 0 else "No"
        direction_emoji = "📈" if average_effect > 0 else "📉" if average_effect < 0 else "➡️"
        
        st.markdown(f"**{direction_emoji} Direction: {direction} Effect**")
        st.markdown(f"**{magnitude_color} Magnitude: {magnitude}** (Effect size ≈ {effect_size:.3f})" if effect_size else "**Magnitude: Unknown**")
        
        # Time Series Analysis
        st.markdown("### 📈 Time Series Analysis")
        
        # Check for trend in treatment effects
        if len(treatment_effects) > 3:
            from scipy import stats
            time_index = np.arange(len(treatment_effects))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, treatment_effects)
            
            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            trend_significance = "Significant" if p_value < 0.05 else "Not Significant"
            
            st.markdown(f"**Effect Trend: {trend_direction}** (slope: {slope:.4f}, p-value: {p_value:.4f})")
            st.markdown(f"**Trend Significance: {trend_significance}**")
        
        # Model Diagnostics
        st.markdown("### 🔬 Model Diagnostics")
        
        # Residual analysis
        residuals = actual - predicted
        
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            st.metric("Observations", f"{len(actual)}")
        with diag_col2:
            # Check for autocorrelation in residuals
            residual_autocorr = residuals.autocorr() if hasattr(residuals, 'autocorr') else None
            if residual_autocorr is not None:
                st.metric("Residual Autocorr", f"{residual_autocorr:.3f}")
            else:
                st.metric("Residual Autocorr", "N/A")
        with diag_col3:
            # Residual standard deviation
            residual_std = residuals.std()
            st.metric("Residual Std", f"{residual_std:.4f}")
        
        # Create comprehensive comparison chart
        st.markdown("### 📈 Actual vs Predicted with Treatment Effects")
        
        try:
            # Find date column
            date_cols = [col for col in df_original.columns if pd.api.types.is_datetime64_any_dtype(df_original[col])]
            if not date_cols:
                date_cols = [col for col in df_original.columns if 'date' in col.lower()]
            
            if date_cols:
                date_col = date_cols[0]
                
                fig_comparison = create_actual_vs_predicted_chart(
                    df_original, predictions, date_col, target, 'prediction'
                )
                fig_comparison.update_layout(
                    title='Actual vs CausalImpact Prediction with Treatment Effects',
                    showlegend=True
                )
                
                # Add treatment effect as additional trace if we have dates
                if len(df_original) >= len(treatment_effects):
                    fig_comparison.add_trace(
                        go.Scatter(
                            x=df_original[date_col].iloc[:len(treatment_effects)],
                            y=treatment_effects,
                            mode='lines',
                            name='Treatment Effect',
                            line=dict(color='orange', dash='dot'),
                            yaxis='y2'
                        )
                    )
                    
                    # Add secondary y-axis for treatment effects
                    fig_comparison.update_layout(
                        yaxis2=dict(
                            title="Treatment Effect",
                            overlaying='y',
                            side='right'
                        )
                    )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Date column not found for time series visualization.")
                
        except Exception as chart_error:
            st.warning(f"Could not create comprehensive chart: {str(chart_error)}")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_causal_impact_interpretation(
            cumulative_effect, average_effect, has_confidence_intervals, 
            effect_significant if has_confidence_intervals else None,
            magnitude, direction
        )
        
        if has_confidence_intervals and 'effect_significant' in locals() and effect_significant:
            st.success(interpretation["summary"])
        elif has_confidence_intervals:
            st.warning(interpretation["summary"])
        else:
            st.info(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced CausalImpact metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying CausalImpact metrics: {str(e)}")
        st.error(f"Error computing CausalImpact metrics: {str(e)}")


def create_causal_impact_interpretation(cumulative_effect: float, average_effect: float, has_ci: bool, 
                                      is_significant: bool = None, magnitude: str = "Unknown", 
                                      direction: str = "Unknown") -> Dict[str, any]:
    """
    Create business interpretation for CausalImpact results.
    
    Args:
        cumulative_effect: Total cumulative effect
        average_effect: Average treatment effect per period
        has_ci: Whether confidence intervals are available
        is_significant: Whether effect is statistically significant (if CI available)
        magnitude: Effect size magnitude (Small/Medium/Large/Negligible)
        direction: Effect direction (Positive/Negative/No)
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Create summary based on significance and effect
    if has_ci and is_significant is not None:
        if is_significant:
            summary = f"Bayesian CausalImpact detected a statistically significant {direction.lower()} intervention effect. Cumulative impact: {cumulative_effect:.4f} units, Average per-period effect: {average_effect:.4f} units."
        else:
            summary = f"No statistically significant intervention effect detected. The observed changes may be due to natural variation rather than the intervention."
    else:
        if direction != "No" and magnitude not in ["Negligible", "Unknown"]:
            summary = f"CausalImpact suggests a {direction.lower()} intervention effect (cumulative: {cumulative_effect:.4f}, average: {average_effect:.4f} per period). Confidence intervals not available - interpret with caution."
        else:
            summary = f"Minimal or no intervention effect detected. Cumulative impact: {cumulative_effect:.4f} units."
    
    # Create detailed explanation
    details = f"""
    **Bayesian Structural Time Series (BSTS) Method:**
    • Uses Bayesian inference to model counterfactual outcomes in the absence of intervention
    • Accounts for uncertainty in model parameters through posterior distributions
    • {'Provides credible intervals for causal effect estimates' if has_ci else 'Point estimates only - no uncertainty quantification available'}
    
    **Effect Assessment:**
    • Cumulative effect: {cumulative_effect:.4f} units ({direction.lower()} direction)
    • Average per-period effect: {average_effect:.4f} units
    • Effect magnitude: {magnitude} (relative to baseline variation)
    • {'Statistical significance: ' + ('Yes' if is_significant else 'No') if is_significant is not None else 'Statistical significance: Not assessed'}
    
    **Interpretation:**
    • {f'Strong evidence for causal effect' if has_ci and is_significant else 'Moderate evidence for causal effect' if not has_ci and magnitude in ['Medium', 'Large'] else 'Weak evidence for causal effect' if has_ci and not is_significant else 'Limited evidence for causal effect'}
    • {'The credible interval excludes zero, suggesting the effect is unlikely due to chance' if has_ci and is_significant else 'The credible interval includes zero, suggesting the effect may be due to chance' if has_ci and not is_significant else 'Without confidence intervals, effect significance cannot be formally assessed'}
    """
    
    # Create recommendations
    recommendations = []
    
    if has_ci and is_significant:
        if direction == "Positive":
            recommendations.extend([
                "Strong evidence for positive intervention impact - consider scaling or continuing the intervention",
                "Monitor effect sustainability over longer time horizons",
                "Investigate specific intervention components driving the positive effect",
                "Conduct cost-benefit analysis to assess economic value"
            ])
        else:
            recommendations.extend([
                "Strong evidence for negative intervention impact - investigate causes immediately",
                "Consider discontinuing or substantially modifying the intervention",
                "Analyze implementation issues that may have caused negative effects",
                "Assess if negative effects were anticipated or unexpected"
            ])
    elif has_ci and not is_significant:
        recommendations.extend([
            "No statistically significant effect detected - intervention may be ineffective",
            "Consider increasing intervention intensity or duration for future experiments",
            "Examine if the intervention was implemented as designed",
            "Explore potential moderating factors that might influence effectiveness"
        ])
    else:
        # No confidence intervals available
        if magnitude in ["Medium", "Large"]:
            recommendations.extend([
                f"Observed {direction.lower()} effect appears meaningful but lacks formal significance testing",
                "Implement additional data collection to enable uncertainty quantification",
                "Consider alternative causal inference methods for validation",
                "Proceed with caution given limited statistical evidence"
            ])
        else:
            recommendations.extend([
                "Minimal effect observed - intervention appears to have limited impact",
                "Consider alternative intervention approaches or designs",
                "Assess if current measurement approach captures relevant outcomes",
                "Review intervention theory and implementation quality"
            ])
    
    # Method-specific recommendations
    recommendations.extend([
        "Validate BSTS model assumptions (e.g., structural breaks, seasonality)",
        "Conduct robustness checks with different model specifications",
        "Consider extending analysis period to assess long-term effects",
        "Compare results with other causal inference methods (DiD, Synthetic Control)"
    ])
    
    # Model quality recommendations
    if not has_ci:
        recommendations.append("⚠️ Implement confidence interval estimation for more robust inference")
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_psm_metrics(predictions: pd.DataFrame) -> None:
    """
    Display comprehensive Propensity Score Matching model diagnostics and metrics.
    
    Args:
        predictions: Predictions DataFrame with ATT and potentially balance statistics
    """
    try:
        st.subheader("✨ Propensity Score Matching Analysis")
        
        # Data validation
        if "ATT" not in predictions.columns:
            st.error("ATT (Average Treatment Effect on Treated) column not found in predictions.")
            return
        
        att = predictions["ATT"].iloc[0]
        
        # Main Treatment Effect
        st.markdown("### 🎯 Treatment Effect Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Treatment Effect on Treated (ATT)", f"{att:.4f}")
        
        # Additional metrics if available
        if "ATT_SE" in predictions.columns:
            att_se = predictions["ATT_SE"].iloc[0]
            with col2:
                st.metric("Standard Error", f"{att_se:.4f}")
            
            # Calculate t-statistic and p-value
            t_stat = att / att_se if att_se != 0 else float('inf')
            # Using two-tailed t-test approximation
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            with col3:
                st.metric("t-statistic", f"{t_stat:.3f}")
            with col4:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                significance_color = "🟢" if p_value < 0.05 else "🔴"
                st.markdown(f"**{significance_color} {significance}**")
                st.caption(f"p-value: {p_value:.4f}")
        
        # Effect Assessment
        st.markdown("### 🔍 Effect Assessment")
        
        # Determine effect magnitude and direction
        effect_direction = "Positive" if att > 0 else "Negative" if att < 0 else "No"
        direction_emoji = "📈" if att > 0 else "📉" if att < 0 else "➡️"
        
        # Effect size interpretation (basic threshold-based)
        abs_att = abs(att)
        if abs_att < 0.01:
            effect_size = "Negligible"
            size_color = "⚪"
        elif abs_att < 0.05:
            effect_size = "Small"
            size_color = "🟠"
        elif abs_att < 0.1:
            effect_size = "Medium"
            size_color = "🟡"
        else:
            effect_size = "Large"
            size_color = "🟢"
        
        st.markdown(f"**{direction_emoji} Effect Direction: {effect_direction}**")
        st.markdown(f"**{size_color} Effect Magnitude: {effect_size}** ({abs_att:.4f} units)")
        
        # Matching Quality Assessment
        st.markdown("### ⚙️ Matching Quality Diagnostics")
        
        # Check for balance statistics
        balance_cols = [col for col in predictions.columns if 'balance' in col.lower() or 'bias' in col.lower()]
        
        if balance_cols:
            st.markdown("**Covariate Balance After Matching:**")
            balance_data = []
            
            for col in balance_cols:
                balance_data.append({
                    "Covariate": col.replace('_balance', '').replace('balance_', '').replace('_bias', '').replace('bias_', ''),
                    "Balance Metric": predictions[col].iloc[0] if not predictions[col].empty else "N/A"
                })
            
            if balance_data:
                balance_df = pd.DataFrame(balance_data)
                st.dataframe(balance_df, use_container_width=True)
        else:
            st.info("Detailed balance statistics not available. This is important for validating PSM assumptions.")
        
        # Propensity Score Diagnostics
        st.markdown("### 📊 Propensity Score Diagnostics")
        
        # Check for propensity score information
        ps_cols = [col for col in predictions.columns if 'propensity' in col.lower() or 'pscore' in col.lower()]
        
        if ps_cols:
            st.info("Propensity score distribution analysis helps validate the overlap assumption.")
            
            for col in ps_cols:
                if not predictions[col].empty:
                    ps_values = predictions[col].dropna()
                    if len(ps_values) > 0:
                        ps_mean = ps_values.mean()
                        ps_std = ps_values.std()
                        ps_min = ps_values.min()
                        ps_max = ps_values.max()
                        
                        ps_col1, ps_col2, ps_col3, ps_col4 = st.columns(4)
                        with ps_col1:
                            st.metric("PS Mean", f"{ps_mean:.3f}")
                        with ps_col2:
                            st.metric("PS Std Dev", f"{ps_std:.3f}")
                        with ps_col3:
                            st.metric("PS Range", f"[{ps_min:.3f}, {ps_max:.3f}]")
                        with ps_col4:
                            # Common support assessment
                            overlap_quality = "Good" if ps_min < 0.1 and ps_max > 0.9 else "Limited"
                            overlap_color = "🟢" if overlap_quality == "Good" else "🟠"
                            st.markdown(f"**{overlap_color} Overlap: {overlap_quality}**")
        else:
            st.warning("⚠️ Propensity score diagnostics not available. This limits the ability to validate key PSM assumptions.")
        
        # Matching Statistics
        st.markdown("### 🔢 Matching Statistics")
        
        # Check for matching information
        match_cols = [col for col in predictions.columns if any(keyword in col.lower() for keyword in ['matched', 'pairs', 'ratio', 'caliper'])]
        
        if match_cols:
            st.markdown("**Matching Configuration:**")
            for col in match_cols:
                if not predictions[col].empty:
                    value = predictions[col].iloc[0]
                    st.markdown(f"• {col.replace('_', ' ').title()}: {value}")
        else:
            st.info("Matching configuration details not provided.")
        
        # Sample sizes (if available)
        if "n_treated" in predictions.columns and "n_control" in predictions.columns:
            n_treated = predictions["n_treated"].iloc[0]
            n_control = predictions["n_control"].iloc[0]
            
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            with sample_col1:
                st.metric("Treated Units", f"{int(n_treated)}")
            with sample_col2:
                st.metric("Control Units", f"{int(n_control)}")
            with sample_col3:
                ratio = n_control / n_treated if n_treated > 0 else 0
                st.metric("Control:Treated Ratio", f"{ratio:.2f}:1")
        
        # Model Assumptions Checklist
        st.markdown("### ✅ Key Assumptions Assessment")
        
        assumptions = [
            ("Unconfoundedness", "All relevant confounders are observed and included", "Requires domain expertise - cannot be tested statistically"),
            ("Overlap/Common Support", "Treated and control units have overlapping propensity scores", "Good" if ps_cols and overlap_quality == "Good" else "Unknown" if not ps_cols else "Limited"),
            ("Covariate Balance", "Matched samples are balanced on observed covariates", "Good" if balance_cols else "Unknown - balance diagnostics not available"),
            ("Stable Unit Treatment Value", "No spillover effects between units", "Requires design consideration - assess based on treatment context")
        ]
        
        for assumption, description, assessment in assumptions:
            if "Good" in assessment:
                st.success(f"✅ **{assumption}**: {description} - Status: {assessment}")
            elif "Limited" in assessment or "Unknown" in assessment:
                st.warning(f"⚠️ **{assumption}**: {description} - Status: {assessment}")
            else:
                st.info(f"📝 **{assumption}**: {description} - Status: {assessment}")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_psm_interpretation(
            att, 
            p_value if 'p_value' in locals() else None,
            effect_size, effect_direction,
            bool(balance_cols), bool(ps_cols)
        )
        
        # Display interpretation with appropriate styling
        if 'p_value' in locals() and p_value < 0.05:
            st.success(interpretation["summary"])
        elif 'p_value' in locals():
            st.warning(interpretation["summary"])
        else:
            st.info(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced PSM metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying PSM metrics: {str(e)}")
        st.error(f"Error computing PSM metrics: {str(e)}")


def create_psm_interpretation(att: float, p_value: float = None, effect_size: str = "Unknown", 
                            direction: str = "Unknown", has_balance: bool = False, 
                            has_ps_diag: bool = False) -> Dict[str, any]:
    """
    Create business interpretation for PSM results.
    
    Args:
        att: Average Treatment Effect on Treated
        p_value: Statistical significance p-value
        effect_size: Effect size magnitude (Small/Medium/Large/Negligible)
        direction: Effect direction (Positive/Negative/No)
        has_balance: Whether balance diagnostics are available
        has_ps_diag: Whether propensity score diagnostics are available
        
    Returns:
        Dictionary with summary, details, and recommendations
    """
    
    # Determine statistical significance
    is_significant = p_value is not None and p_value < 0.05
    
    # Create summary based on significance and effect
    if p_value is not None:
        if is_significant:
            summary = f"Propensity Score Matching detected a statistically significant {direction.lower()} treatment effect (ATT = {att:.4f}, p = {p_value:.4f}). Effect size is {effect_size.lower()}."
        else:
            summary = f"No statistically significant treatment effect detected through PSM (ATT = {att:.4f}, p = {p_value:.4f}). Observed differences may be due to chance."
    else:
        if effect_size not in ["Negligible", "Unknown"]:
            summary = f"PSM suggests a {direction.lower()} treatment effect (ATT = {att:.4f}). Effect size appears {effect_size.lower()}, but statistical significance not assessed."
        else:
            summary = f"Minimal treatment effect detected through PSM (ATT = {att:.4f}). Statistical significance assessment not available."
    
    # Create detailed explanation
    details = f"""
    **Propensity Score Matching Method:**
    • Estimates treatment effects by matching treated units with similar control units based on propensity scores
    • ATT (Average Treatment Effect on Treated): {att:.4f} units
    • Effect represents the average impact of treatment on those who actually received it
    • {'Statistical significance: ' + ('Yes' if is_significant else 'No') + f' (p = {p_value:.4f})' if p_value is not None else 'Statistical significance: Not assessed'}
    
    **Method Quality:**
    • {'Balance diagnostics available' if has_balance else 'Balance diagnostics missing - limits assumption validation'}
    • {'Propensity score diagnostics available' if has_ps_diag else 'Propensity score diagnostics missing - limits overlap assessment'}
    • {'High-quality analysis with key diagnostics' if has_balance and has_ps_diag else 'Moderate-quality analysis - some diagnostics missing' if has_balance or has_ps_diag else 'Limited-quality analysis - key diagnostics unavailable'}
    
    **Causal Inference Strength:**
    • {'Strong' if is_significant and has_balance and has_ps_diag else 'Moderate' if is_significant or (has_balance and has_ps_diag) else 'Weak'} evidence for causal effect
    • PSM assumes all confounders are observed and controlled for through matching
    • Effect estimate applies specifically to the treated population (ATT, not ATE)
    """
    
    # Create recommendations
    recommendations = []
    
    if p_value is not None and is_significant:
        if direction == "Positive":
            recommendations.extend([
                "Statistically significant positive treatment effect - consider expanding the intervention",
                "Investigate which treated units benefited most to optimize targeting",
                "Assess cost-effectiveness of scaling the intervention",
                "Monitor treatment effect sustainability over time"
            ])
        else:
            recommendations.extend([
                "Statistically significant negative treatment effect - investigate causes immediately",
                "Assess whether negative effects were anticipated or represent implementation issues",
                "Consider discontinuing or substantially modifying the intervention",
                "Analyze subgroups to identify who was most negatively affected"
            ])
    elif p_value is not None and not is_significant:
        recommendations.extend([
            "No statistically significant effect detected - intervention may be ineffective for treated population",
            "Consider larger sample sizes or longer treatment periods",
            "Examine intervention implementation fidelity",
            "Explore heterogeneous treatment effects across different subgroups"
        ])
    else:
        # No significance testing available
        if effect_size in ["Medium", "Large"]:
            recommendations.extend([
                f"Observed {direction.lower()} effect appears meaningful but lacks significance testing",
                "Implement statistical inference to assess effect reliability",
                "Collect additional data to enable proper hypothesis testing",
                "Consider the practical significance of the observed effect size"
            ])
        else:
            recommendations.extend([
                "Minimal effect observed - intervention impact appears limited",
                "Reassess intervention design and targeting criteria",
                "Consider alternative treatment approaches",
                "Evaluate if outcome measurement captures relevant impacts"
            ])
    
    # Method-specific recommendations
    method_recs = [
        "Validate propensity score model specification and functional form",
        "Conduct sensitivity analysis for unobserved confounding",
        "Compare PSM results with other causal inference methods (DiD, RCT if available)"
    ]
    
    if not has_balance:
        method_recs.append("⚠️ Implement covariate balance checks to validate matching quality")
    
    if not has_ps_diag:
        method_recs.append("⚠️ Add propensity score distribution analysis to assess common support")
    
    recommendations.extend(method_recs)
    
    # Quality improvement recommendations
    quality_recs = [
        "Consider multiple matching algorithms (nearest neighbor, caliper, kernel) for robustness",
        "Examine treatment effect heterogeneity across different matched subgroups",
        "Assess matching quality through standardized bias reduction metrics"
    ]
    
    recommendations.extend(quality_recs)
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


def display_chronos_metrics(model: Any, predictions: pd.DataFrame, df: pd.DataFrame, target: str) -> None:
    """
    Display comprehensive Chronos T5 Large forecasting model diagnostics and metrics.
    
    Args:
        model: Trained Chronos model (or model metadata)
        predictions: Predictions DataFrame with forecasts and confidence intervals
        df: Original DataFrame for validation
        target: Target variable name
    """
    try:
        st.subheader("✨ Chronos T5 Large Forecasting Analysis")
        
        # Data validation
        if predictions.empty:
            st.error("No predictions available for analysis.")
            return
        
        # Forecast Performance Metrics
        st.markdown("### 📊 Forecast Performance Metrics")
        
        # Check if we have actual values for validation
        has_validation = "actual" in predictions.columns or target in predictions.columns
        
        if has_validation:
            actual_col = "actual" if "actual" in predictions.columns else target
            pred_col = "prediction" if "prediction" in predictions.columns else predictions.columns[0]
            
            actual = predictions[actual_col].dropna()
            predicted = predictions[pred_col].iloc[:len(actual)]
            
            if len(actual) > 0 and len(predicted) > 0:
                # Calculate comprehensive forecast metrics
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if (actual != 0).all() else None
                
                # Calculate directional accuracy
                actual_direction = np.diff(actual) > 0
                pred_direction = np.diff(predicted) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual_direction) > 0 else None
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"{mae:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MAPE (%)", f"{mape:.2f}%" if mape is not None else "N/A")
                with col4:
                    st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%" if directional_accuracy is not None else "N/A")
                
                # Forecast quality assessment
                baseline_std = actual.std()
                relative_mae = mae / baseline_std if baseline_std > 0 else None
                
                if relative_mae is not None:
                    if relative_mae < 0.5:
                        quality = "Excellent"
                        quality_color = "🟢"
                    elif relative_mae < 0.8:
                        quality = "Good"
                        quality_color = "🟡"
                    elif relative_mae < 1.2:
                        quality = "Fair"
                        quality_color = "🟠"
                    else:
                        quality = "Poor"
                        quality_color = "🔴"
                    
                    st.markdown(f"**{quality_color} Forecast Quality: {quality}** (Relative MAE: {relative_mae:.3f})")
            else:
                st.warning("Insufficient validation data for accuracy assessment.")
        else:
            st.info("No validation data available. Displaying forecast characteristics only.")
        
        # Model Configuration
        st.markdown("### ⚙️ Model Configuration")
        
        # Extract model metadata if available
        model_info = {}
        if hasattr(model, 'prediction_length'):
            model_info['Prediction Length'] = model.prediction_length
        if hasattr(model, 'model_name'):
            model_info['Model Name'] = model.model_name
        if hasattr(model, 'num_samples'):
            model_info['Number of Samples'] = model.num_samples
        
        # Display model configuration
        if model_info:
            config_col1, config_col2, config_col3 = st.columns(3)
            
            items = list(model_info.items())
            for i, (key, value) in enumerate(items):
                col = [config_col1, config_col2, config_col3][i % 3]
                with col:
                    st.metric(key, str(value))
        else:
            st.info("Model configuration details not available.")
        
        # Forecast Uncertainty Analysis
        st.markdown("### 🎯 Forecast Uncertainty Analysis")
        
        # Check for confidence intervals
        has_intervals = any(col in predictions.columns for col in ['lower', 'upper', 'prediction_lower', 'prediction_upper'])
        
        if has_intervals:
            lower_col = next((col for col in ['lower', 'prediction_lower'] if col in predictions.columns), None)
            upper_col = next((col for col in ['upper', 'prediction_upper'] if col in predictions.columns), None)
            
            if lower_col and upper_col:
                lower_bounds = predictions[lower_col].dropna()
                upper_bounds = predictions[upper_col].dropna()
                
                if len(lower_bounds) > 0 and len(upper_bounds) > 0:
                    # Calculate uncertainty metrics
                    avg_interval_width = np.mean(upper_bounds - lower_bounds)
                    relative_uncertainty = avg_interval_width / np.mean(predictions[pred_col] if 'pred_col' in locals() else predictions.iloc[:, 0])
                    
                    uncertainty_col1, uncertainty_col2 = st.columns(2)
                    
                    with uncertainty_col1:
                        st.metric("Average Prediction Interval Width", f"{avg_interval_width:.4f}")
                    with uncertainty_col2:
                        st.metric("Relative Uncertainty", f"{relative_uncertainty:.3f}")
                    
                    # Coverage assessment if validation data available
                    if has_validation and 'actual' in locals():
                        actual_in_bounds = np.sum((actual >= lower_bounds[:len(actual)]) & (actual <= upper_bounds[:len(actual)]))
                        coverage = actual_in_bounds / len(actual) * 100 if len(actual) > 0 else 0
                        
                        expected_coverage = 95  # Assuming 95% prediction intervals
                        coverage_quality = "Good" if abs(coverage - expected_coverage) < 10 else "Poor"
                        coverage_color = "🟢" if coverage_quality == "Good" else "🔴"
                        
                        st.markdown(f"**{coverage_color} Prediction Interval Coverage: {coverage:.1f}%** (Expected: ~{expected_coverage}%)")
                else:
                    st.info("Confidence interval data is incomplete.")
            else:
                st.info("Confidence interval columns not properly identified.")
        else:
            st.warning("⚠️ No confidence intervals available. Uncertainty assessment limited.")
        
        # Time Series Characteristics
        st.markdown("### 📈 Time Series Forecast Characteristics")
        
        if len(predictions) > 1:
            # Analyze forecast patterns
            pred_values = predictions[pred_col] if 'pred_col' in locals() else predictions.iloc[:, 0]
            
            # Trend analysis
            if len(pred_values) > 2:
                from scipy import stats
                time_index = np.arange(len(pred_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, pred_values)
                
                trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
                trend_strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.3 else "Weak"
                
                trend_col1, trend_col2, trend_col3 = st.columns(3)
                
                with trend_col1:
                    st.metric("Forecast Trend", trend_direction)
                with trend_col2:
                    st.metric("Trend Strength", trend_strength)
                with trend_col3:
                    st.metric("Trend R²", f"{r_value**2:.3f}")
        
        # Business Interpretation
        st.markdown("### 💼 Business Interpretation & Conclusions")
        
        interpretation = create_chronos_interpretation(
            quality if 'quality' in locals() else "Unknown",
            has_validation,
            mape if 'mape' in locals() else None,
            directional_accuracy if 'directional_accuracy' in locals() else None,
            has_intervals,
            trend_direction if 'trend_direction' in locals() else "Unknown"
        )
        
        if 'quality' in locals():
            if quality == "Excellent":
                st.success(interpretation["summary"])
            elif quality == "Good":
                st.success(interpretation["summary"])
            elif quality == "Fair":
                st.warning(interpretation["summary"])
            else:
                st.error(interpretation["summary"])
        else:
            st.info(interpretation["summary"])
        
        st.info(interpretation["details"])
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        for rec in interpretation["recommendations"]:
            st.markdown(f"• {rec}")
        
        logger.info("Enhanced Chronos metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying Chronos metrics: {str(e)}")
        st.error(f"Error computing Chronos metrics: {str(e)}")


def create_chronos_interpretation(quality: str, has_validation: bool, mape: float = None, 
                                directional_accuracy: float = None, has_intervals: bool = False,
                                trend: str = "Unknown") -> Dict[str, any]:
    """Create business interpretation for Chronos forecasting results."""
    
    # Create summary
    if has_validation:
        if quality == "Excellent":
            summary = f"Chronos T5 forecasting model demonstrates excellent predictive performance. MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.1f}%."
        elif quality == "Good":
            summary = f"Chronos T5 shows good forecasting performance with reasonable accuracy. MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.1f}%."
        elif quality == "Fair":
            summary = f"Chronos T5 provides fair forecasting accuracy. Consider model refinement. MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.1f}%."
        else:
            summary = f"Chronos T5 forecasting performance is below expectations. Significant model improvement needed. MAPE: {mape:.2f}%."
    else:
        summary = "Chronos T5 Large has generated forecasts, but validation data is not available to assess accuracy."
    
    # Create detailed explanation
    details = f"""
    **Chronos T5 Large Foundation Model:**
    • Pre-trained transformer model specifically designed for time series forecasting
    • Leverages large-scale training on diverse time series data for robust predictions
    • Zero-shot forecasting capability without domain-specific fine-tuning
    • {'Validation-based assessment available' if has_validation else 'No validation data - forecasts are projections only'}
    
    **Forecast Characteristics:**
    • Trend direction: {trend}
    • {'Uncertainty quantification available through prediction intervals' if has_intervals else 'Point forecasts only - no uncertainty quantification'}
    • {'MAPE of ' + f'{mape:.2f}% indicates ' + ('excellent' if mape < 5 else 'good' if mape < 15 else 'fair' if mape < 25 else 'poor') + ' percentage accuracy' if mape is not None else 'Percentage accuracy not assessed'}
    
    **Model Reliability:**
    • Foundation model approach provides robust forecasting across diverse time series patterns
    • Performance depends on similarity between target series and training data
    • {'Directional accuracy of ' + f'{directional_accuracy:.1f}% indicates ' + ('excellent' if directional_accuracy > 80 else 'good' if directional_accuracy > 65 else 'moderate') + ' trend prediction capability' if directional_accuracy is not None else 'Trend prediction accuracy not assessed'}
    """
    
    # Create recommendations
    recommendations = []
    
    if has_validation:
        if quality in ["Excellent", "Good"]:
            recommendations.extend([
                "Forecasting model shows strong performance - suitable for business planning",
                "Use forecasts for strategic decision-making and resource allocation",
                "Monitor forecast accuracy over time to detect model drift",
                "Consider ensemble methods combining Chronos with other models for robustness"
            ])
        else:
            recommendations.extend([
                "⚠️ Forecasting accuracy below optimal levels - exercise caution in decision-making",
                "Consider data preprocessing improvements (outlier handling, feature engineering)",
                "Evaluate alternative forecasting methods or model ensembles",
                "Assess if training data period is representative of current patterns"
            ])
    else:
        recommendations.extend([
            "⚠️ No validation data available - treat forecasts as projections requiring validation",
            "Collect actual outcomes to assess model performance over time",
            "Implement forecast tracking and accuracy monitoring systems",
            "Consider shorter forecast horizons until validation data becomes available"
        ])
    
    # Technical recommendations
    if not has_intervals:
        recommendations.append("⚠️ Implement prediction intervals for uncertainty quantification")
    
    recommendations.extend([
        "Regularly retrain or update model with new data as it becomes available",
        "Compare Chronos forecasts with traditional statistical methods for validation",
        "Consider domain-specific adjustments based on business knowledge",
        "Implement forecast reconciliation if forecasting hierarchical data"
    ])
    
    return {
        "summary": summary,
        "details": details,
        "recommendations": recommendations
    }


# Model metric display dispatcher
MODEL_METRIC_DISPATCH = {
    "MLR": lambda **kwargs: display_mlr_metrics(
        kwargs["df"], 
        kwargs["target"], 
        kwargs["features"], 
        kwargs.get("model"), 
        kwargs.get("model").client_specific_vars if kwargs.get("model") and hasattr(kwargs.get("model"), "client_specific_vars") else None,
        kwargs.get("predictions")
    ),
    "Distributed Lag": lambda **kwargs: display_distributed_lag_metrics(kwargs["df"], kwargs["target"], kwargs["features"]),
    "ML + SHAP": lambda **kwargs: display_ml_shap_metrics(kwargs["model"], kwargs["df"], kwargs["target"]),
    "DiD": lambda **kwargs: display_did_metrics(kwargs["model"]),
    "VAR": lambda **kwargs: display_var_metrics(kwargs["model"]),
    "Synthetic Control": lambda **kwargs: display_synthetic_control_metrics(kwargs["predictions"]),
    "CausalImpact": lambda **kwargs: display_causal_impact_metrics(kwargs["predictions"], kwargs["df"], kwargs["target"]),
    "PSM": lambda **kwargs: display_psm_metrics(kwargs["predictions"]),
    "Chronos T5 Large": lambda **kwargs: display_chronos_metrics(kwargs["model"], kwargs["predictions"], kwargs["df"], kwargs["target"]),
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
            "client_specific_analysis": {},
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
                ols_model = sm.OLS(y, X_sm).fit(cov_type='HAC', cov_kwds={'maxlags':1})
                
                report["model_diagnostics"] = {
                    "r_squared": float(ols_model.rsquared),
                    "adjusted_r_squared": float(ols_model.rsquared_adj),
                    "f_statistic": float(ols_model.fvalue),
                    "f_pvalue": float(ols_model.f_pvalue),
                    "aic": float(ols_model.aic),
                    "bic": float(ols_model.bic),
                    "log_likelihood": float(ols_model.llf),
                    "durbin_watson": float(sm.stats.stattools.durbin_watson(ols_model.resid)),
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
                        "significance_level": "***" if row["P>|t|"] < 0.001 else "**" if row["P>|t|"] < 0.01 else "*" if row["P>|t|"] < 0.05 else "ns",
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
                        "durbin_watson_statistic": float(sm.stats.stattools.durbin_watson(ols_model.resid)),
                        "autocorrelation_detected": not (1.5 <= sm.stats.stattools.durbin_watson(ols_model.resid) <= 2.5)
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
                
                # Client-specific analysis (if available)
                if hasattr(model, 'client_specific_vars') and model.client_specific_vars:
                    from data_analysis import analyze_client_specific_effects
                    client_analysis = analyze_client_specific_effects(model, df, target, model.client_specific_vars)
                    report["client_specific_analysis"] = client_analysis
                else:
                    report["client_specific_analysis"] = {
                        "note": "No client-specific variables identified for analysis"
                    }
                
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
                ols_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
                
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
                        "residuals_mean": float((df[target] - preds).mean()),
                        "residuals_std": float((df[target] - preds).std()),
                        "residuals_range": {
                            "min": float((df[target] - preds).min()),
                            "max": float((df[target] - preds).max())
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
            ols_model = sm.OLS(y, X_sm).fit(cov_type='HAC', cov_kwds={'maxlags':1})
            
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
                    "significance": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns",
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
    
    with st.expander("Enhanced Model Interpretation Guide", expanded=False):
        st.markdown(f"{model_name} Model Interpretation")
        
        for i, hint in enumerate(hints, 1):
            st.markdown(f"{i}. {hint}")
        
        st.markdown("---")
        st.markdown("### **General Guidelines**")
        st.markdown("""
        - **Statistical Significance**: Look for p-values < 0.05 for reliable effects
        - **Effect Size**: Consider practical significance alongside statistical significance  
        - **Model Fit**: Higher R² indicates better explanatory power
        - **Validation**: Always check model assumptions and robustness
        - **Business Context**: Interpret coefficients in terms of business impact
        """)
        
        if model_name in ["MLR", "Distributed Lag"]:
            st.markdown("### **Linear Model Specifics**")
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
            st.markdown("Causal Inference Specifics")
            st.markdown("""
            - **Causal Assumptions**: Carefully validate identifying assumptions
            - **Treatment Effect**: Consider both statistical and economic significance
            - **Robustness**: Test sensitivity to specification changes
            - **External Validity**: Consider generalizability to other contexts
            """)
        
        st.markdown("---")
        st.info("**Tip**: Your comprehensive JSON report contains detailed diagnostics to help validate these interpretation guidelines.")


def create_did_att_chart(
    df: pd.DataFrame,
    time_col: str,
    att_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    p_value_col: str = None,
    is_post_col: str = None,
    window_size: int = 4  # for rolling average
) -> go.Figure:
    """
    Create a line chart showing ATT over time with confidence intervals.
    
    Args:
        df: DataFrame containing ATT and CI data
        time_col: Column name for time periods
        att_col: Column name for ATT values
        ci_lower_col: Column name for lower CI bound
        ci_upper_col: Column name for upper CI bound
        p_value_col: Column name for p-values (optional)
        is_post_col: Column name indicating post-treatment period (optional)
        window_size: Size of rolling window for smoothing
        
    Returns:
        Plotly figure object
    """
    try:
        # Create copy to avoid modifying original (same as working charts)
        plot_df = df.copy()
        
        # --- unify x-axis dtype ---
        plot_df["raw_time"] = plot_df[time_col].astype(str)      # keep original text
        plot_df[time_col] = pd.to_datetime(plot_df[time_col],
                                           errors="coerce")
        plot_df[time_col] = plot_df[time_col].fillna(
                            pd.to_datetime("1970-01-01"))        # tmp filler
        plot_df = plot_df.sort_values(time_col)                  # stable sort
        plot_df[time_col] = plot_df["raw_time"].astype(str)      # final – pure str
        plot_df.drop(columns=["raw_time"], inplace=True)         # cleanup
        
        # Convert numeric columns
        plot_df[att_col] = pd.to_numeric(plot_df[att_col], errors="coerce").fillna(0.0)
        plot_df[ci_lower_col] = pd.to_numeric(plot_df[ci_lower_col], errors="coerce").fillna(0.0)
        plot_df[ci_upper_col] = pd.to_numeric(plot_df[ci_upper_col], errors="coerce").fillna(0.0)
        if p_value_col and p_value_col in plot_df.columns:
            plot_df[p_value_col] = pd.to_numeric(plot_df[p_value_col], errors="coerce").fillna(1.0)
        
        # Calculate rolling averages (same as working charts)
        plot_df['att_smooth'] = plot_df[att_col].rolling(window=window_size, center=True).mean()
        plot_df['ci_lower_smooth'] = plot_df[ci_lower_col].rolling(window=window_size, center=True).mean()
        plot_df['ci_upper_smooth'] = plot_df[ci_upper_col].rolling(window=window_size, center=True).mean()
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=plot_df[time_col],
            y=plot_df['ci_upper_smooth'],
            mode='lines',
            name='95% CI',
            line=dict(width=0),
            showlegend=True,
            hovertemplate='<b>Upper CI</b><br>Time: %{x}<br>Value: %{y:.1f} visits/store<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=plot_df[time_col],
            y=plot_df['ci_lower_smooth'],
            mode='lines',
            name='95% CI',
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.2)',
            line=dict(width=0),
            showlegend=False,
            hovertemplate='<b>Lower CI</b><br>Time: %{x}<br>Value: %{y:.1f} visits/store<extra></extra>'
        ))
        
        # Add ATT line with color based on significance
        if p_value_col:
            # Split into significant and non-significant segments
            sig_df = plot_df[plot_df[p_value_col] < 0.05]
            nonsig_df = plot_df[plot_df[p_value_col] >= 0.05]
            
            # Add significant points
            if not sig_df.empty:
                fig.add_trace(go.Scatter(
                    x=sig_df[time_col],
                    y=sig_df['att_smooth'],
                    mode='lines+markers',
                    name='Significant ATT (p < 0.05)',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=8),
                    hovertemplate='<b>ATT (Significant)</b><br>Time: %{x}<br>Value: %{y:.1f} visits/store<br>p-value: %{customdata:.3f}<extra></extra>',
                    customdata=sig_df[p_value_col]
                ))
            
            # Add non-significant points
            if not nonsig_df.empty:
                fig.add_trace(go.Scatter(
                    x=nonsig_df[time_col],
                    y=nonsig_df['att_smooth'],
                    mode='lines+markers',
                    name='Non-significant ATT',
                    line=dict(color='#7f7f7f', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='<b>ATT (Non-significant)</b><br>Time: %{x}<br>Value: %{y:.1f} visits/store<br>p-value: %{customdata:.3f}<extra></extra>',
                    customdata=nonsig_df[p_value_col]
                ))
        else:
            # Add single ATT line if no p-values
            fig.add_trace(go.Scatter(
                x=plot_df[time_col],
                y=plot_df['att_smooth'],
                mode='lines+markers',
                name='ATT',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>ATT</b><br>Time: %{x}<br>Value: %{y:.1f} visits/store<extra></extra>'
            ))
        
        # Add treatment start line if post period indicator exists
        if is_post_col and not plot_df.empty:
            # Use string time column directly (no arithmetic)
            treatment_start = plot_df[plot_df[is_post_col]].iloc[0][time_col]
            x_value = str(treatment_start)
            
            fig.add_vline(
                x=x_value,
                line_dash="dash",
                line_color="#ff7f0e",
                annotation_text="Treatment Start",
                annotation_position="top"
            )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Average Treatment Effect Over Time<br><sup>Per-store visits relative to control group, 4-week rolling average</sup>',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Time Period',
            yaxis_title='Average Treatment Effect (visits per store)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
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
        logger.error(f"Error creating ATT chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

def create_cumulative_att_chart(
    df: pd.DataFrame,
    time_col: str,
    att_col: str
) -> go.Figure:
    """
    Create a cumulative line chart showing total ATT over time.
    
    Args:
        df: DataFrame containing ATT data
        time_col: Column name for time periods
        att_col: Column name for ATT values
        
    Returns:
        Plotly figure object
    """
    try:
        # Calculate cumulative ATT
        df = df.copy()
        df['cumulative_att'] = df[att_col].cumsum()
        
        fig = go.Figure()
        
        # Add cumulative ATT line
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df['cumulative_att'],
            mode='lines+markers',
            name='Cumulative ATT',
            line=dict(color='green', width=2),
            hovertemplate='<b>Cumulative ATT</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title='Cumulative ATT Over Time',
            xaxis_title='Time Period',
            yaxis_title='Cumulative Average Treatment Effect',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
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
        logger.error(f"Error creating cumulative ATT chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

def create_att_percentage_chart(
    df: pd.DataFrame,
    time_col: str,
    att_percentage_col: str = 'att_percentage',
    p_value_col: str = None,
    significance_level: float = 0.05,
    window_size: int = 4  # for rolling average
) -> go.Figure:
    """
    Create a line chart showing ATT as percentage of target variable over time.
    
    Args:
        df: DataFrame containing ATT percentage data
        time_col: Column name for time periods
        att_percentage_col: Column name for ATT percentage values
        p_value_col: Column name for p-values (optional)
        significance_level: P-value threshold for significance
        window_size: Size of rolling window for smoothing
        
    Returns:
        Plotly figure object
    """
    try:
        # Create copy to avoid modifying original
        plot_df = df.copy()
        
        # Sort by time column
        try:
            plot_df[time_col] = pd.to_datetime(plot_df[time_col])
        except (ValueError, TypeError):
            pass
        plot_df = plot_df.sort_values(time_col)
        
        # Calculate rolling average
        plot_df['att_pct_smooth'] = plot_df[att_percentage_col].rolling(window=window_size, center=True).mean()
        
        fig = go.Figure()
        
        # Add ATT percentage line with color based on significance
        if p_value_col:
            # Split into significant and non-significant segments
            sig_df = plot_df[plot_df[p_value_col] < significance_level]
            nonsig_df = plot_df[plot_df[p_value_col] >= significance_level]
            
            # Add significant points
            if not sig_df.empty:
                fig.add_trace(go.Scatter(
                    x=sig_df[time_col],
                    y=sig_df['att_pct_smooth'],
                    mode='lines+markers',
                    name='Significant Effect (p < 0.05)',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=8),
                    hovertemplate='<b>ATT %</b><br>Time: %{x}<br>Value: %{y:.1f}%<br>p-value: %{customdata:.3f}<extra></extra>',
                    customdata=sig_df[p_value_col]
                ))
            
            # Add non-significant points
            if not nonsig_df.empty:
                fig.add_trace(go.Scatter(
                    x=nonsig_df[time_col],
                    y=nonsig_df['att_pct_smooth'],
                    mode='lines+markers',
                    name='Non-significant Effect',
                    line=dict(color='#7f7f7f', width=2, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='<b>ATT %</b><br>Time: %{x}<br>Value: %{y:.1f}%<br>p-value: %{customdata:.3f}<extra></extra>',
                    customdata=nonsig_df[p_value_col]
                ))
        else:
            # Add single line if no p-values
            fig.add_trace(go.Scatter(
                x=plot_df[time_col],
                y=plot_df['att_pct_smooth'],
                mode='lines+markers',
                name='ATT %',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>ATT %</b><br>Time: %{x}<br>Value: %{y:.1f}%<extra></extra>'
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Treatment Effect as Percentage of Target<br><sup>Shows campaign impact relative to baseline visits (4-week rolling average)</sup>',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Time Period',
            yaxis_title='ATT as % of Target',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
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
        logger.error(f"Error creating ATT percentage chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig


def create_att_bar_chart(
    df: pd.DataFrame,
    time_col: str,
    att_col: str,
    p_value_col: str,
    significance_level: float = 0.05
) -> go.Figure:
    """
    Create a bar chart showing ATT by time period with significance highlighting.
    
    Args:
        df: DataFrame containing ATT and p-value data
        time_col: Column name for time periods
        att_col: Column name for ATT values
        p_value_col: Column name for p-values
        significance_level: P-value threshold for significance
        
    Returns:
        Plotly figure object
    """
    try:
        # Create color array based on significance
        colors = ['rgba(0, 128, 0, 0.7)' if p < significance_level else 'rgba(128, 128, 128, 0.7)' 
                 for p in df[p_value_col]]
        
        fig = go.Figure()
        
        # Add ATT bars
        fig.add_trace(go.Bar(
            x=df[time_col],
            y=df[att_col],
            marker_color=colors,
            name='ATT',
            hovertemplate='<b>ATT</b><br>Time: %{x}<br>Value: %{y:.2f}<br>P-value: %{customdata:.3f}<extra></extra>',
            customdata=df[p_value_col]
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title='ATT by Week',
            xaxis_title='Time Period',
            yaxis_title='Average Treatment Effect on the Treated (ATT)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=False,
            height=500
        )
        
        # Add legend for color coding
        fig.add_trace(go.Bar(
            x=[None],
            y=[None],
            name='Significant (p < 0.05)',
            marker_color='rgba(0, 128, 0, 0.7)',
            showlegend=True
        ))
        
        fig.add_trace(go.Bar(
            x=[None],
            y=[None],
            name='Not Significant',
            marker_color='rgba(128, 128, 128, 0.7)',
            showlegend=True
        ))
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating ATT bar chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

def create_parallel_trends_chart(
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    treatment_time: Any = None
) -> go.Figure:
    """
    Create a line chart showing pre- and post-treatment trends by group.
    
    Args:
        df: DataFrame containing outcome data for both groups
        time_col: Column name for time periods
        outcome_col: Column name for outcome variable
        treatment_col: Column name indicating treatment/control group
        treatment_time: Time period when treatment began (optional)
        
    Returns:
        Plotly figure object
    """
    try:
        # Create copy to avoid modifying original (same as working charts)
        plot_df = df.copy()
        
        # --- unify x-axis dtype ---
        plot_df["raw_time"] = plot_df[time_col].astype(str)      # keep original text
        plot_df[time_col] = pd.to_datetime(plot_df[time_col],
                                           errors="coerce")
        plot_df[time_col] = plot_df[time_col].fillna(
                            pd.to_datetime("1970-01-01"))        # tmp filler
        plot_df = plot_df.sort_values(time_col)                  # stable sort
        plot_df[time_col] = plot_df["raw_time"].astype(str)      # final – pure str
        plot_df.drop(columns=["raw_time"], inplace=True)         # cleanup
        
        # Convert numeric columns
        plot_df[outcome_col] = pd.to_numeric(plot_df[outcome_col], errors="coerce").fillna(0.0)
        plot_df[treatment_col] = pd.to_numeric(plot_df[treatment_col], errors="coerce").fillna(0.0)
        
        # Calculate mean outcome by time and group (same as working charts)
        grouped = plot_df.groupby([time_col, treatment_col])[outcome_col].mean().reset_index()
        
        fig = go.Figure()
        
        # Add line for treatment group
        treatment_data = grouped[grouped[treatment_col] == 1]
        fig.add_trace(go.Scatter(
            x=treatment_data[time_col],
            y=treatment_data[outcome_col],
            mode='lines+markers',
            name='Treatment Group',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Treatment Group</b><br>Time: %{x}<br>Value: %{y:.1f}<extra></extra>'
        ))
        
        # Add line for control group
        control_data = grouped[grouped[treatment_col] == 0]
        fig.add_trace(go.Scatter(
            x=control_data[time_col],
            y=control_data[outcome_col],
            mode='lines+markers',
            name='Control Group',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Control Group</b><br>Time: %{x}<br>Value: %{y:.1f}<extra></extra>'
        ))
        
        # Add treatment time vertical line if provided (same as working charts)
        if treatment_time is not None:
            fig.add_vline(
                x=treatment_time,
                line_dash="dash",
                line_color="#ff7f0e",
                annotation_text="Treatment Start",
                annotation_position="top"
            )
        
        # Update layout
        fig.update_layout(
            title='Pre- and Post-Treatment Trends by Group',
            xaxis_title='Time Period',
            yaxis_title='Outcome Variable',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
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
        logger.error(f"Error creating parallel trends chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}<br>Please check your data format and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

def display_did_visualizations(
    df: pd.DataFrame,
    time_col: str,
    att_col: str,
    ci_lower_col: str = None,
    ci_upper_col: str = None,
    p_value_col: str = None,
    outcome_col: str = None,
    treatment_col: str = None,
    treatment_time: Any = None
) -> None:
    """
    Display comprehensive DiD analysis visualizations.
    
    Args:
        df: DataFrame containing DiD analysis results
        time_col: Column name for time periods
        att_col: Column name for ATT values
        ci_lower_col: Column name for lower CI bound (optional)
        ci_upper_col: Column name for upper CI bound (optional)
        p_value_col: Column name for p-values (optional)
        outcome_col: Column name for outcome variable (optional)
        treatment_col: Column name indicating treatment/control group (optional)
        treatment_time: Time period when treatment began (optional)
    """
    try:
        st.subheader("Difference-in-Differences (DiD) Analysis")
        
        # Clean the data at entry point to prevent type errors
        clean_df = df.copy()
        
        # Force convert numeric columns to float
        numeric_columns = [att_col, ci_lower_col, ci_upper_col, p_value_col, outcome_col, treatment_col]
        for col in numeric_columns:
            if col and col in clean_df.columns:
                try:
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    logger.warning(f"Error converting {col}: {e}")
                    clean_df[col] = 0.0
                
        # ATT over time with confidence intervals
        if ci_lower_col and ci_upper_col:
            st.markdown("#### ATT Over Time with Confidence Intervals")
            fig_att = create_did_att_chart(
                df=clean_df,
                time_col=time_col,
                att_col=att_col,
                ci_lower_col=ci_lower_col,
                ci_upper_col=ci_upper_col,
                p_value_col=p_value_col,
                is_post_col='is_post',
                window_size=4
            )
            st.plotly_chart(fig_att, use_container_width=True, key="did_att_chart")
            st.caption("""
            This chart shows the per-store impact of the campaign over time. Key features:
            - **Blue line/points**: Statistically significant effects (p < 0.05)
            - **Gray line/points**: Non-significant effects
            - **Shaded area**: 95% confidence interval
            - **Orange line**: Treatment start date
            - **Values**: Average additional visits per store (4-week rolling average)
            """)
        
        # Cumulative ATT
        st.markdown("#### Cumulative ATT Over Time")
        fig_cumulative = create_cumulative_att_chart(clean_df, time_col, att_col)
        st.plotly_chart(fig_cumulative, use_container_width=True, key="did_cumulative_chart")
        st.caption("This chart aggregates the weekly treatment effects to show the total accumulated impact of the campaign. Useful for understanding long-term gain and return.")
        
        # ATT by week with significance
        if p_value_col:
            st.markdown("#### ATT by Week with Significance")
            fig_bar = create_att_bar_chart(clean_df, time_col, att_col, p_value_col)
            st.plotly_chart(fig_bar, use_container_width=True, key="did_bar_chart")
            st.caption("Bar chart version of ATT by week for clearer point comparison. Bars are colored by significance to help stakeholders quickly spot meaningful impact weeks.")
            
            # ATT as percentage of target
            st.markdown("#### Treatment Effect as Percentage of Target")
            fig_pct = create_att_percentage_chart(
                df=clean_df,
                time_col=time_col,
                att_percentage_col='att_percentage',
                p_value_col=p_value_col,
                window_size=4
            )
            st.plotly_chart(fig_pct, use_container_width=True, key="did_percentage_chart")
            st.caption("""
            This chart shows the campaign's impact as a percentage of baseline visits. Key features:
            - **Values**: Treatment effect divided by average visits (as percentage)
            - **Blue line**: Statistically significant effects (p < 0.05)
            - **Gray line**: Non-significant effects
            - **Smoothing**: 4-week rolling average to reduce noise
            """)
        
        # Parallel trends
        if outcome_col and treatment_col:
            st.markdown("#### Parallel Trends Analysis")
            fig_trends = create_parallel_trends_chart(clean_df, time_col, outcome_col, treatment_col, treatment_time)
            st.plotly_chart(fig_trends, use_container_width=True, key="did_trends_chart")
            st.caption("This chart checks the parallel trends assumption by plotting average outcomes over time for treatment and control groups. Useful for validating the DiD model assumptions.")
        
        logger.info("DiD visualizations displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying DiD visualizations: {str(e)}")
        st.error(f"Error displaying DiD visualizations: {str(e)}")