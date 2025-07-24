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
        
        # Store preprocessing info with the model
        model.label_encoders_ = label_encoders
        model.categorical_cols_ = categorical_cols
        model.feature_names_ = features
        
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


@st.cache_resource
def train_chronos(
    df: pd.DataFrame, 
    date_col: str, 
    target: str, 
    features: List[str],
    prediction_length: int = None,
    test_percentage: int = None
) -> Tuple[Any, pd.DataFrame, Figure]:
    """
    Train Chronos T5 Large model for time series forecasting.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target: Target variable name
        features: Feature variable names (for context)
        prediction_length: Number of future points to forecast (optional)
        test_percentage: Percentage of data to use for testing (optional, default 20%)
        
    Returns:
        Tuple of (trained model, predictions DataFrame, plotly figure)
    """
    try:
        logger.info("Loading Chronos T5 Large model for forecasting")
        
        # Import Chronos pipeline
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(f"Required libraries not installed: {e}. Please install with: pip install chronos-forecasting torch")
        
        # Load the Chronos pipeline with optimal settings
        with st.spinner("Loading Chronos T5 Large model (this may take a few minutes on first run)..."):
            # Detect best available device and dtype
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype = torch.bfloat16
                st.info("Using GPU acceleration with bfloat16 for optimal performance")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = "mps"  # Apple Silicon
                dtype = torch.float32
                st.info("Using Apple Silicon MPS acceleration")
            else:
                device_map = "cpu"
                dtype = torch.float32
                st.info("Using CPU (consider GPU for faster inference)")
            
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map=device_map,
                torch_dtype=dtype,
            )
        
        # Prepare data
        df_clean = df.dropna(subset=[date_col, target])
        df_sorted = df_clean.sort_values(date_col).reset_index(drop=True)
        
        if len(df_sorted) < 10:
            raise ValueError("Need at least 10 observations for Chronos forecasting")
        
        # Only convert to datetime if it looks like actual date/time data
        is_datetime_like = False
        if df_sorted[date_col].dtype == 'object':
            # Check if string data looks like dates
            sample_values = df_sorted[date_col].dropna().head(3).astype(str)
            if any(len(str(val)) > 8 and ('-' in str(val) or '/' in str(val) or ':' in str(val)) for val in sample_values):
                try:
                    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
                    is_datetime_like = True
                except (ValueError, TypeError):
                    pass
        elif pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]):
            is_datetime_like = True
        
        # Provide user feedback about column type
        if is_datetime_like:
            st.success(f"Using datetime column '{date_col}' for time series analysis")
        elif pd.api.types.is_numeric_dtype(df_sorted[date_col]):
            st.info(f"Using numeric column '{date_col}' as sequence index. Consider using a date column for better time series analysis.")
        else:
            st.warning(f"Column '{date_col}' is not numeric or date/time. Chronos works best with temporal data.")
        
        # Prepare time series data - use ALL data as context (Chronos is zero-shot)
        target_series = df_sorted[target].values
        
        # Split data based on user-specified test percentage or default
        if test_percentage is not None:
            test_ratio = test_percentage / 100.0
            train_size = int((1 - test_ratio) * len(target_series))
            st.info(f"Using user-specified {test_percentage}% ({len(target_series) - train_size} points) for testing")
        else:
            train_size = int(0.8 * len(target_series))
            st.info(f"Using default 20% ({len(target_series) - train_size} points) for testing")
        
        test_data = target_series[train_size:] if train_size < len(target_series) else []
        
        # Use user-specified prediction length or calculate default
        if prediction_length is None:
            prediction_length = len(test_data) if len(test_data) > 0 else min(12, len(target_series) // 4)
            prediction_length = max(1, prediction_length)  # Ensure at least 1 prediction
            st.info(f"Using {len(target_series)} observations as context to forecast {prediction_length} future points (auto-calculated)")
        else:
            prediction_length = max(1, min(prediction_length, 100))  # Ensure reasonable bounds
            st.success(f"Using {len(target_series)} observations as context to forecast {prediction_length} future points (user-specified)")
        
        # Generate forecasts using Chronos (following HF example pattern)
        with st.spinner("Generating forecasts using Chronos T5 Large..."):
            # Use full historical data as context (like HF example)
            context = torch.tensor(target_series, dtype=dtype)
            
            # Generate forecast (using defaults similar to HF example)
            forecast = pipeline.predict(context, prediction_length)
            
            # Extract forecast statistics
            # forecast shape: [num_series, num_samples, prediction_length]
            forecast_numpy = forecast[0].numpy()  # First (and only) series
            
            # Calculate quantiles for uncertainty estimation
            forecast_median = np.median(forecast_numpy, axis=0)
            forecast_lower = np.quantile(forecast_numpy, 0.1, axis=0)  # 10th percentile
            forecast_upper = np.quantile(forecast_numpy, 0.9, axis=0)  # 90th percentile
        
        # Create future dates/indices for forecasts
        last_value = df_sorted[date_col].iloc[-1]
        
        if is_datetime_like and pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]):
            # Handle datetime columns with proper pandas methods
            try:
                freq = pd.infer_freq(df_sorted[date_col])
                if freq is not None:
                    # Use pandas date_range with inferred frequency
                    future_dates = pd.date_range(start=last_value, periods=prediction_length + 1, freq=freq)[1:]
                else:
                    # Calculate most common time difference
                    time_diffs = df_sorted[date_col].diff().dropna()
                    if len(time_diffs) > 0:
                        # Use the most common time difference
                        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else pd.Timedelta(days=1)
                        future_dates = [last_value + most_common_diff * (i + 1) for i in range(prediction_length)]
                    else:
                        # Fallback to daily frequency
                        future_dates = pd.date_range(start=last_value, periods=prediction_length + 1, freq='D')[1:]
            except Exception as e:
                st.warning(f"⚠️ Could not generate future dates: {e}. Using sequential indices.")
                future_dates = [f"forecast_{i+1}" for i in range(prediction_length)]
        else:
            # Handle numeric/ID columns or other types
            if pd.api.types.is_numeric_dtype(df_sorted[date_col]):
                # For numeric columns, increment by 1 (works for ID columns like 0,1,2...113)
                try:
                    future_dates = [last_value + (i + 1) for i in range(prediction_length)]
                except Exception:
                    # Fallback if arithmetic fails
                    future_dates = [f"{last_value}_+{i+1}" for i in range(prediction_length)]
            else:
                # For other types, create simple sequential labels
                future_dates = [f"{last_value}_forecast_{i+1}" for i in range(prediction_length)]
        
        # Create predictions DataFrame following HF example pattern
        # Historical period: actual values (no fitting needed for zero-shot model)
        predictions_df = pd.DataFrame({
            date_col: list(df_sorted[date_col]) + list(future_dates),
            f'{target}_actual': list(df_sorted[target]) + [np.nan] * prediction_length,
            f'{target}_predicted': list(df_sorted[target]) + list(forecast_median),
            f'{target}_lower': list(df_sorted[target]) + list(forecast_lower),
            f'{target}_upper': list(df_sorted[target]) + list(forecast_upper),
            'data_type': ['historical'] * len(df_sorted) + ['forecast'] * prediction_length
        })
        
        # Calculate metrics on out-of-sample forecast if test data available
        metrics = {}
        if len(test_data) > 0:
            # Compare forecast with actual test data
            test_forecast = forecast_median[:len(test_data)]
            mae = mean_absolute_error(test_data, test_forecast)
            mse = mean_squared_error(test_data, test_forecast)
            rmse = np.sqrt(mse)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'test_size': len(test_data),
                'context_length': len(target_series),
                'forecast_horizon': prediction_length
            }
            
            logger.info(f"Chronos forecast metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            st.success(f"📈 Forecast validation - MAPE: {mape:.2f}%, RMSE: {rmse:.4f}")
        
        # Create visualization
        fig = go.Figure()
        
        # Historical data
        historical_data = predictions_df[predictions_df['data_type'] == 'historical']
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=historical_data[f'{target}_actual'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Forecasts
        forecast_data = predictions_df[predictions_df['data_type'] == 'forecast']
        if len(forecast_data) > 0:
            fig.add_trace(go.Scatter(
                x=forecast_data[date_col],
                y=forecast_data[f'{target}_predicted'],
                mode='lines',
                name='Chronos Forecast (Median)',
                line=dict(color='red', width=2)
            ))
            
            # Confidence intervals (only for forecasts)
            fig.add_trace(go.Scatter(
                x=list(forecast_data[date_col]) + list(forecast_data[date_col][::-1]),
                y=list(forecast_data[f'{target}_upper']) + list(forecast_data[f'{target}_lower'][::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='80% Confidence Interval'
            ))
        
        # Add test data if available (overlay on historical period)
        if len(test_data) > 0:
            test_dates = df_sorted[date_col].iloc[train_size:train_size + len(test_data)]
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_data,
                mode='markers',
                name='Test Data (Actual)',
                marker=dict(color='green', size=8, symbol='circle-open')
            ))
        
        # Add vertical line to separate historical and forecast
        fig.add_vline(
            x=df_sorted[date_col].iloc[-1],
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start"
        )
        
        fig.update_layout(
            title=f"Chronos T5 Large Time Series Forecast: {target}",
            xaxis_title=date_col,
            yaxis_title=target,
            hovermode='x unified',
            showlegend=True,
            height=600
        )
        
        # Create model info object for consistency (exclude pipeline for pickling)
        model_info = {
            'model_type': 'Chronos T5 Large',
            'target_variable': target,
            'features': features,
            'prediction_length': prediction_length,
            'model_name': 'amazon/chronos-t5-large',
            'metrics': metrics,
            'context_length': len(target_series),
            'device': device_map,
            'dtype': str(dtype),
            'note': 'Chronos pipeline not saved due to accelerate hooks - can be reloaded using model_name'
        }
        
        logger.info("Chronos T5 Large forecasting completed successfully")
        return model_info, predictions_df, fig
        
    except Exception as e:
        logger.error(f"Error with Chronos model: {str(e)}")
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
    "Chronos T5 Large": train_chronos,
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
        # Save predictions first (always works)
        pred_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv")
        predictions.to_csv(pred_path, index=False)
        
        # Handle model saving based on type
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        
        if model_name == "Chronos T5 Large":
            # Chronos models can't be pickled due to accelerate hooks
            # Save only the metadata, pipeline can be reloaded from model_name
            model_metadata = {
                'model_type': model.get('model_type', 'Chronos T5 Large'),
                'model_name': model.get('model_name', 'amazon/chronos-t5-large'),
                'target_variable': model.get('target_variable'),
                'features': model.get('features'),
                'prediction_length': model.get('prediction_length'),
                'context_length': model.get('context_length'),
                'device': model.get('device'),
                'dtype': model.get('dtype'),
                'metrics': model.get('metrics'),
                'saved_timestamp': pd.Timestamp.now().isoformat(),
                'note': 'Model pipeline not saved - reload using ChronosPipeline.from_pretrained(model_name)'
            }
            joblib.dump(model_metadata, model_path)
            logger.info(f"Chronos model metadata saved (pipeline excluded due to accelerate hooks)")
        else:
            # Regular models can be pickled normally
            joblib.dump(model, model_path)
            logger.info(f"Model saved for {model_name}")
        
        logger.info(f"Predictions saved for {model_name}")
        return model_path, pred_path
        
    except Exception as e:
        logger.error(f"Error saving model and predictions: {str(e)}")
        # For Chronos, try to save at least the predictions
        if model_name == "Chronos T5 Large":
            try:
                pred_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv")
                predictions.to_csv(pred_path, index=False)
                logger.info(f"At least predictions saved for {model_name}")
                return "metadata_save_failed", pred_path
            except:
                pass
        raise 