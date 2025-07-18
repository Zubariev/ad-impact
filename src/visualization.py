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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    with st.expander("📊 Dataset Overview", expanded=False):
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
            st.dataframe(desc.style.format("{:.2f}"))

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
            lambda p: "✅" if p < SIGNIFICANCE_LEVEL else "❌"
        )
        styled = coef_tbl.style.applymap(
            lambda p: "color:green;" if p < SIGNIFICANCE_LEVEL else "color:red;", 
            subset=["P>|t|"]
        )
        st.dataframe(styled, height=min(400, 25 * len(coef_tbl)))

        # VIF table
        vif_df = create_vif_table(df, features)
        st.markdown("**Variance Inflation Factor (VIF):**")
        st.dataframe(vif_df.style.format("{:.2f}"))

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
    
    # Re-use MLR diagnostics
    display_mlr_metrics(df, target, features)
    
    # Durbin-Watson test for autocorrelation
    try:
        y = df[target]
        X = sm.add_constant(df[features])
        dw = durbin_watson(y - sm.OLS(y, X).fit().fittedvalues)
        
        dw_status = "✅ No autocorrelation" if DURBIN_WATSON_LOWER <= dw <= DURBIN_WATSON_UPPER else "❌ Autocorrelation detected"
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
        
        # Calculate prediction metrics
        feature_cols = [c for c in df.columns if c != target]
        preds = model.predict(df[feature_cols])
        rmse = mean_squared_error(df[target], preds, squared=False)
        mae = mean_absolute_error(df[target], preds)
        
        st.markdown(f"**RMSE / MAE:** {rmse:.2f} / {mae:.2f}")

        # Global SHAP summary (bar plot)
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer(df[feature_cols])
        shap.summary_plot(shap_values, df[feature_cols], show=False, plot_type="bar")
        st.pyplot(plt.gcf(), clear_figure=True)
        
        logger.info("ML + SHAP metrics displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying ML + SHAP metrics: {str(e)}")
        st.error(f"Error computing ML + SHAP metrics: {str(e)}")


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
        
        significance = "✅ Significant" if pval < SIGNIFICANCE_LEVEL else "❌ Not significant"
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
        
        stability_status = "✅ Stable" if results.is_stable() else "❌ Unstable"
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
            "VIF > 5 may indicate multicollinearity"
        ],
        "Distributed Lag": [
            "Higher lag coefficients indicate longer-lasting effects",
            "Durbin-Watson near 2 indicates no autocorrelation"
        ],
        "ML + SHAP": [
            "Lower RMSE indicates better predictive accuracy",
            "Higher SHAP value indicates a channel's contribution"
        ],
        "DiD": [
            "CI excluding zero indicates significant effect",
            "Placebo test near zero confirms robustness"
        ],
        "VAR": [
            "IRF shows effect of ad spend shocks over time",
            "Stability check confirms forecast reliability"
        ],
        "Synthetic Control": [
            "Low RMSPE indicates a good match",
            "Visible post-gap indicates treatment effect"
        ],
        "CausalImpact": [
            "CI not crossing zero indicates significant impact",
            "High posterior probability confirms effect confidence"
        ],
        "PSM": [
            "SMD < 0.1 indicates good balance",
            "ATT p-value < 0.05 indicates significance"
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
    
    with st.expander("💡 Interpretation Hints", expanded=False):
        for hint in hints:
            st.markdown(f"• {hint}") 