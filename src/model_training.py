"""
Model training functions for the Ad Impact Modeling Dashboard.
Contains all model training logic with caching and error handling.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR

from config import MODEL_DIR, MODEL_HYPERPARAMS, PREDICTIONS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def train_mlr(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[LinearRegression, pd.DataFrame, Figure]:
    """
    Train Multiple Linear Regression model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        X = df[features]
        y = df[target]
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Create contributions DataFrame
        contributions = pd.DataFrame({date_col: df[date_col]})
        for coef, feat in zip(reg.coef_, features):
            contributions[feat] = df[feat] * coef
        
        # Create visualization
        fig = px.area(
            contributions,
            x=date_col,
            y=features,
            title="MLR Estimated Channel Contributions",
        )
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": reg.predict(X)
        })
        
        logger.info(f"MLR model trained successfully with {len(features)} features")
        return reg, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training MLR model: {str(e)}")
        raise


@st.cache_resource
def train_distributed_lag(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    features: List[str],
    max_lag: int = None,
) -> Tuple[LinearRegression, pd.DataFrame, Figure]:
    """
    Train Distributed Lag model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        max_lag: Maximum lag to consider
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        if max_lag is None:
            max_lag = MODEL_HYPERPARAMS["Distributed Lag"]["max_lag"]
        
        # Create lagged features
        lagged_cols = []
        for lag in range(1, max_lag + 1):
            for feat in features:
                col = f"{feat}_lag{lag}"
                df[col] = df[feat].shift(lag)
                lagged_cols.append(col)
        
        df = df.dropna()
        X = df[lagged_cols]
        y = df[target]
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Create contributions DataFrame
        contributions = pd.DataFrame({date_col: df[date_col]})
        for coef, feat in zip(reg.coef_, lagged_cols):
            contributions[feat] = df[feat] * coef
        
        # Create visualization
        fig = px.area(
            contributions,
            x=date_col,
            y=lagged_cols,
            title="Distributed Lag Model Contributions",
        )
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": reg.predict(X)
        })
        
        logger.info(f"Distributed Lag model trained successfully with {len(lagged_cols)} lagged features")
        return reg, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training Distributed Lag model: {str(e)}")
        raise


@st.cache_resource
def train_ml_shap(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Any, pd.DataFrame, Figure]:
    """
    Train ML model with SHAP analysis.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        import xgboost as xgb
        import shap
        from sklearn.preprocessing import LabelEncoder
        
        X = df[features].copy()
        y = df[target]
        
        # Handle categorical columns
        categorical_cols = []
        label_encoders = {}
        
        for col in features:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Get hyperparameters
        params = MODEL_HYPERPARAMS["ML + SHAP"]
        
        # Configure XGBoost with categorical support if needed
        if categorical_cols:
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                enable_categorical=True
            )
            # Convert categorical columns to category dtype for XGBoost
            for col in categorical_cols:
                X[col] = X[col].astype('category')
        else:
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"]
            )
        
        model.fit(X, y)
        
        # Create SHAP explainer and values
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Create SHAP DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=features)
        shap_df[date_col] = df[date_col]
        
        # Create visualization
        fig = px.area(
            shap_df,
            x=date_col,
            y=features,
            title="ML + SHAP Channel Contributions",
        )
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": model.predict(X)
        })
        
        logger.info(f"ML + SHAP model trained successfully with {len(features)} features ({len(categorical_cols)} categorical)")
        return model, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training ML + SHAP model: {str(e)}")
        raise


@st.cache_resource
def train_did(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Any, pd.DataFrame, Figure]:
    """
    Train Difference-in-Differences model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        if "treated" not in df.columns or "post" not in df.columns:
            # Provide suggestions for creating these columns
            available_cols = list(df.columns)
            error_msg = (
                "DiD requires 'treated' and 'post' indicator columns in the dataset.\n\n"
                "To use DiD analysis, you need to create these columns:\n"
                "• 'treated': Binary indicator (0/1) for treatment vs control groups\n"
                "• 'post': Binary indicator (0/1) for pre vs post treatment periods\n\n"
                f"Available columns in your dataset: {available_cols}\n\n"
                "Suggestions:\n"
                "• If you have a city/location column, you could create 'treated' based on specific cities\n"
                "• If you have a time column, you could create 'post' based on a specific date cutoff\n"
                "• Use the data preparation utilities to add these columns before training"
            )
            raise ValueError(error_msg)
        
        # Validate that treated and post are binary
        if not df['treated'].isin([0, 1]).all():
            raise ValueError("'treated' column must contain only 0 and 1 values")
        if not df['post'].isin([0, 1]).all():
            raise ValueError("'post' column must contain only 0 and 1 values")
        
        formula = f"{target} ~ treated * post + " + " + ".join(features)
        model = smf.ols(formula, data=df).fit()
        
        df["prediction"] = model.predict(df)
        
        # Create aggregated data for visualization
        agg = (
            df.groupby(["treated", "post"])[target]
            .mean()
            .reset_index()
            .pivot(index="post", columns="treated", values=target)
        )
        
        # Create visualization
        fig = px.bar(
            agg, 
            barmode="group", 
            title="DiD Estimated Effects", 
            labels={"value": target}
        )
        
        # Create predictions DataFrame
        predictions = df[[date_col, "prediction"]]
        
        logger.info("DiD model trained successfully")
        return model, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training DiD model: {str(e)}")
        raise


@st.cache_resource
def train_var(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Any, pd.DataFrame, Figure]:
    """
    Train Vector Autoregression model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        params = MODEL_HYPERPARAMS["VAR"]
        vars_used = [target] + features
        model = VAR(df[vars_used])
        results = model.fit(maxlags=params["maxlags"], ic=params["ic"])
        
        # Create impulse response functions
        irf = results.irf(10)
        irf_df = irf.cum_effects[vars_used.index(target)]
        irf_df = pd.DataFrame(irf_df, columns=vars_used)
        
        # Handle date column for IRF
        try:
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                irf_df[date_col] = pd.date_range(
                    start=df[date_col].min(), 
                    periods=len(irf_df)
                )
            else:
                irf_df[date_col] = range(len(irf_df))
        except:
            irf_df[date_col] = range(len(irf_df))
        
        # Create visualization
        fig = px.line(
            irf_df, 
            x=date_col, 
            y=vars_used[1:], 
            title="VAR Impulse Response Functions"
        )
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": results.fittedvalues[target]
        })
        
        logger.info(f"VAR model trained successfully with {len(vars_used)} variables")
        return results, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training VAR model: {str(e)}")
        raise


@st.cache_resource
def train_synthetic_control(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Dict, pd.DataFrame, Figure]:
    """
    Train Synthetic Control model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        if "treated" not in df.columns:
            available_cols = list(df.columns)
            error_msg = (
                "Synthetic Control requires a 'treated' column.\n\n"
                "To use Synthetic Control analysis, you need to create a 'treated' column:\n"
                "• Binary indicator (0/1) where 1 = treated unit, 0 = control units\n"
                "• Typically only one or few units should be treated (treated=1)\n"
                "• Most units should be controls (treated=0) to create the synthetic control\n\n"
                f"Available columns in your dataset: {available_cols}\n\n"
                "Suggestions:\n"
                "• If you have a city/location column, create 'treated' based on the specific location of interest\n"
                "• Use the data preparation utilities to add this column before training"
            )
            raise ValueError(error_msg)
        
        # Validate that treated is binary
        if not df['treated'].isin([0, 1]).all():
            raise ValueError("'treated' column must contain only 0 and 1 values")
        
        # Check if we have both treated and control units
        treated_count = df['treated'].sum()
        control_count = (df['treated'] == 0).sum()
        
        if treated_count == 0:
            raise ValueError("No treated units found (all 'treated' values are 0)")
        if control_count == 0:
            raise ValueError("No control units found (all 'treated' values are 1)")
        
        treated = df[df["treated"] == 1]
        control = df[df["treated"] == 0]
        
        if len(control) < len(features):
            raise ValueError(f"Not enough control units ({len(control)}) for the number of features ({len(features)})")
        
        # Fit weights using linear regression
        weights = LinearRegression().fit(control[features], control[target]).coef_
        
        # Create synthetic control for all periods
        synthetic_values = []
        for idx in df.index:
            if df.loc[idx, 'treated'] == 0:
                # For control units, use actual values
                synthetic_values.append(df.loc[idx, target])
            else:
                # For treated units, compute synthetic value
                feature_values = df.loc[idx, features].values
                synthetic_val = np.dot(feature_values, weights)
                synthetic_values.append(synthetic_val)
        
        # Create comparison DataFrame
        comp_df = pd.DataFrame({
            date_col: df[date_col],
            "Actual": df[target],
            "Synthetic": synthetic_values,
        })
        
        # Create visualization
        fig = px.line(
            comp_df,
            x=date_col,
            y=["Actual", "Synthetic"],
            title="Synthetic Control: Actual vs Synthetic",
        )
        
        # Create predictions DataFrame
        predictions = comp_df
        model = {"weights": weights, "treated_count": treated_count, "control_count": control_count}
        
        logger.info(f"Synthetic Control model trained successfully ({treated_count} treated, {control_count} control units)")
        return model, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training Synthetic Control model: {str(e)}")
        raise


@st.cache_resource
def train_causal_impact(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Any, pd.DataFrame, Figure]:
    """
    Train CausalImpact model (ARIMA proxy).
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        from pmdarima import auto_arima
        
        train = df[target]
        arima_model = auto_arima(train, seasonal=False, suppress_warnings=True)
        prediction = arima_model.predict_in_sample()
        
        # Create predictions DataFrame
        ci_df = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": prediction
        })
        
        # Create visualization
        fig = px.line(df, x=date_col, y=target, title="CausalImpact (ARIMA proxy)")
        fig.add_scatter(
            x=ci_df[date_col], 
            y=ci_df["prediction"], 
            mode="lines", 
            name="Counterfactual"
        )
        
        logger.info("CausalImpact model trained successfully")
        return arima_model, ci_df, fig
        
    except Exception as e:
        logger.error(f"Error training CausalImpact model: {str(e)}")
        raise


@st.cache_resource
def train_psm(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str]
) -> Tuple[Dict, pd.DataFrame, Figure]:
    """
    Train Propensity Score Matching model.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        if "treated" not in df.columns:
            available_cols = list(df.columns)
            error_msg = (
                "PSM requires a 'treated' column in the dataset.\n\n"
                "To use Propensity Score Matching, you need to create a 'treated' column:\n"
                "• Binary indicator (0/1) for treatment vs control groups\n"
                "• Should have reasonable balance between treated and control groups\n\n"
                f"Available columns in your dataset: {available_cols}\n\n"
                "Suggestions:\n"
                "• If you have a city/location column, create 'treated' based on specific cities\n"
                "• If you have a time-based intervention, create based on before/after periods\n"
                "• Use the data preparation utilities to add this column before training"
            )
            raise ValueError(error_msg)
        
        # Validate that treated is binary
        if not df['treated'].isin([0, 1]).all():
            raise ValueError("'treated' column must contain only 0 and 1 values")
        
        # Check balance
        treated_count = df['treated'].sum()
        control_count = (df['treated'] == 0).sum()
        
        if treated_count == 0:
            raise ValueError("No treated units found (all 'treated' values are 0)")
        if control_count == 0:
            raise ValueError("No control units found (all 'treated' values are 1)")
        
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate propensity scores
        logits = LinearRegression().fit(X_scaled, df["treated"]).predict(X_scaled)
        df["propensity"] = logits
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1).fit(logits.reshape(-1, 1))
        distances, indices = nn.kneighbors(logits.reshape(-1, 1))
        matched_idx = indices[df["treated"] == 1].flatten()
        control_matches = df.iloc[matched_idx]
        treated = df[df["treated"] == 1]
        
        # Calculate ATT
        att = treated[target].mean() - control_matches[target].mean()
        effect_df = pd.DataFrame({"ATT": [att]})
        
        # Create visualization
        fig = px.bar(effect_df, y="ATT", title="Propensity Score Matching ATT")
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "ATT": att
        })
        
        model = {
            "scaler": scaler, 
            "logits": logits,
            "treated_count": treated_count,
            "control_count": control_count,
            "att": att
        }
        
        logger.info(f"PSM model trained successfully ({treated_count} treated, {control_count} control units)")
        return model, predictions, fig
        
    except Exception as e:
        logger.error(f"Error training PSM model: {str(e)}")
        raise


# Model training function dispatcher
TRAIN_FUNCTIONS = {
    "MLR": train_mlr,
    "Distributed Lag": train_distributed_lag,
    "ML + SHAP": train_ml_shap,
    "DiD": train_did,
    "VAR": train_var,
    "Synthetic Control": train_synthetic_control,
    "CausalImpact": train_causal_impact,
    "PSM": train_psm,
}


def save_model_and_predictions(
    model: Any, 
    predictions: pd.DataFrame, 
    model_name: str
) -> Tuple[str, str]:
    """
    Save trained model and predictions to disk.
    
    Args:
        model: Trained model object
        predictions: Predictions DataFrame
        model_name: Name of the model
        
    Returns:
        Tuple of (model file path, predictions file path)
    """
    try:
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        
        # Save predictions
        pred_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv")
        predictions.to_csv(pred_path, index=False)
        
        logger.info(f"Model and predictions saved for {model_name}")
        return model_path, pred_path
        
    except Exception as e:
        logger.error(f"Error saving model and predictions: {str(e)}")
        raise 