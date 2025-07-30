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
from scipy import stats
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
        for coef, lagged_col in zip(reg.coef_, lagged_cols):
            contributions[lagged_col] = df[lagged_col] * coef
        
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
        
        # Perform proper model evaluation with train/test split
        test_size = min(int(0.2 * len(X)), 50)  # Use 20% or max 50 observations for testing
        if test_size > 5:  # Only do evaluation if we have enough test data
            X_train = X.iloc[:-test_size]
            X_test = X.iloc[-test_size:]
            y_train = y.iloc[:-test_size]
            y_test = y.iloc[-test_size:]
            
            # Retrain model on training data only for evaluation
            eval_model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                enable_categorical=True if categorical_cols else False
            )
            eval_model.fit(X_train, y_train)
            
            # Evaluate using the utility function
            evaluation_results = evaluate_model_with_shap(
                eval_model, X_test, y_test, features
            )
            
            # Store evaluation results in the main model
            model.evaluation_results_ = evaluation_results
            logger.info(f"ML+SHAP evaluation - R²: {evaluation_results['metrics']['r2']:.3f}, "
                       f"RMSE: {evaluation_results['metrics']['rmse']:.3f}")
        else:
            model.evaluation_results_ = {
                'metrics': {'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0},
                'note': 'Insufficient data for evaluation'
            }
        
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
        
        # Apply group balancing if severely unbalanced
        original_df = df.copy()
        df = balance_treatment_groups(df, 'treated', balance_threshold=0.2)
        
        # Validate parallel trends assumption
        pre_treatment_data = df[df['post'] == 0]
        treatment_start = df[df['post'] == 1][date_col].min() if len(df[df['post'] == 1]) > 0 else None
        
        parallel_trends_result = validate_parallel_trends(
            pre_treatment_data, 
            target, 
            'treated', 
            date_col, 
            treatment_start
        )
        
        if not parallel_trends_result['parallel_trends_holds']:
            logger.warning(f"Parallel trends assumption may be violated: {parallel_trends_result.get('reason', 'Unknown')}")
            logger.warning(f"P-value: {parallel_trends_result['p_value']:.4f}, Slope difference: {parallel_trends_result['slope_diff']:.4f}")
        else:
            logger.info(f"Parallel trends assumption holds (p-value: {parallel_trends_result['p_value']:.4f})")
        
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
        
        # Create predictions DataFrame with additional metrics for visualization
        predictions = df[[date_col, "prediction"]].copy()
        
        # Calculate ATT and confidence intervals for each time period
        att_results = []
        
        # Ensure consistent type for date_col
        original_type = df[date_col].dtype
        logger.info(f"Original date_col type: {original_type}")
        
        # Get all unique times and treatment start
        all_times = df[date_col].unique()
        treatment_start = df[df['post'] == 1][date_col].min()
        
        # Sort based on type
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            all_times = sorted(all_times)
            start_idx = list(all_times).index(treatment_start)
        elif pd.api.types.is_numeric_dtype(df[date_col]):
            all_times = sorted([float(x) for x in all_times])
            start_idx = list(all_times).index(float(treatment_start))
        else:
            # For strings or other types, sort as strings
            all_times = sorted([str(x) for x in all_times])
            start_idx = list(all_times).index(str(treatment_start))
        
        # Take 4 periods before treatment start if available
        pre_periods = 4
        start_idx = max(0, start_idx - pre_periods)
        relevant_times = all_times[start_idx:]
        
        for t in relevant_times:
            period_data = df[df[date_col] == t]
            if len(period_data) > 0:
                # Get store counts for normalization
                treated_stores = len(period_data[period_data['treated'] == 1])
                control_stores = len(period_data[period_data['treated'] == 0])
                
                # Calculate normalized treatment effect
                treated_data = period_data[period_data['treated'] == 1][target]
                control_data = period_data[period_data['treated'] == 0][target]
                
                # Convert to numeric if needed
                try:
                    treated_data = pd.to_numeric(treated_data, errors='coerce')
                    control_data = pd.to_numeric(control_data, errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting data to numeric: {str(e)}")
                
                # Calculate means
                treated_mean = float(treated_data.mean()) if not treated_data.empty else 0
                control_mean = float(control_data.mean()) if not control_data.empty else 0
                treated_effect = treated_mean - control_mean
                
                # Normalize by store count
                att = float(treated_effect / treated_stores if treated_stores > 0 else 0)
                
                # Calculate ATT as percentage of target
                target_mean = period_data[target].mean()
                att_percentage = (att / target_mean * 100) if target_mean != 0 else 0
                
                # Calculate standard error for this period
                try:
                    treated_data = period_data[period_data['treated'] == 1][target]
                    control_data = period_data[period_data['treated'] == 0][target]
                    
                    # Calculate standard deviations
                    treated_std = float(treated_data.std()) if len(treated_data) > 1 else 0
                    control_std = float(control_data.std()) if len(control_data) > 1 else 0
                    
                    # Calculate pooled standard error
                    if treated_stores > 0 and control_stores > 0:
                        se = np.sqrt((treated_std**2/treated_stores) + (control_std**2/control_stores))
                    else:
                        se = float(model.bse['treated:post'])  # fallback to model SE
                    
                    # Calculate confidence intervals
                    ci_margin = 1.96 * se
                    ci_lower = att - ci_margin
                    ci_upper = att + ci_margin
                    
                    # Calculate t-statistic and p-value
                    t_stat = att / se if se > 0 else 0
                    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                    
                except Exception as e:
                    logger.warning(f"Error calculating CI for period {t}: {str(e)}")
                    # Fallback: use model's global standard error
                    se = float(model.bse['treated:post'])
                    ci_margin = 1.96 * se
                    ci_lower = att - ci_margin
                    ci_upper = att + ci_margin
                    p_value = float(model.pvalues['treated:post'])
                
                # Ensure all values are float
                att = float(att)
                ci_lower = float(ci_lower)
                ci_upper = float(ci_upper)
                p_value = float(p_value)
                
                att_results.append({
                    'time': t,
                    'att': att,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'treated_stores': treated_stores,
                    'control_stores': control_stores,
                    'is_post': t >= treatment_start,
                    'att_percentage': float(att_percentage),
                    'target_mean': float(target_mean)
                })
        
        # Add ATT results to predictions
        att_df = pd.DataFrame(att_results)
        
        # Convert time column to match original type
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            att_df['time'] = pd.to_datetime(att_df['time'])
        elif pd.api.types.is_numeric_dtype(df[date_col]):
            att_df['time'] = pd.to_numeric(att_df['time'])
        else:
            att_df['time'] = att_df['time'].astype(str)
            predictions[date_col] = predictions[date_col].astype(str)
        
        # Merge with consistent types
        predictions = predictions.merge(att_df, left_on=date_col, right_on='time', how='left')
        
        # Add original data columns needed for parallel trends
        predictions['outcome'] = df[target]
        predictions['treated'] = df['treated']
        predictions['post'] = df['post']
        
        # Add parallel trends validation results to model
        model.parallel_trends_result_ = parallel_trends_result
        
        # Add balancing information
        balancing_info = {
            'was_balanced': len(df) == len(original_df),
            'original_sample_size': len(original_df),
            'balanced_sample_size': len(df),
            'original_treatment_distribution': original_df['treated'].value_counts().to_dict(),
            'balanced_treatment_distribution': df['treated'].value_counts().to_dict()
        }
        model.balancing_info_ = balancing_info
        
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
        
        # Create predictions DataFrame with proper evaluation
        fitted_values = results.fittedvalues[target]
        
        # Generate out-of-sample forecasts for evaluation
        test_size = min(10, len(df) // 4)  # Use last 25% or 10 periods for testing
        train_data = df[vars_used].iloc[:-test_size] if test_size > 0 else df[vars_used]
        test_data = df[vars_used].iloc[-test_size:] if test_size > 0 else pd.DataFrame()
        
        # Fit model on training data for evaluation
        if len(test_data) > 0:
            train_model = VAR(train_data)
            train_results = train_model.fit(maxlags=params["maxlags"], ic=params["ic"])
            
            # Generate forecasts
            lag_order = train_results.k_ar
            if len(train_data) >= lag_order:
                forecast = train_results.forecast(train_data.values[-lag_order:], steps=test_size)
                forecast_df = pd.DataFrame(forecast, columns=vars_used)
                
                # Calculate evaluation metrics for target variable
                actual_test = test_data[target].values
                predicted_test = forecast_df[target].values
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                r2 = r2_score(actual_test, predicted_test)
                mae = mean_absolute_error(actual_test, predicted_test)
                rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((actual_test - predicted_test) / np.where(actual_test != 0, actual_test, 1))) * 100
                
                evaluation_metrics = {
                    'r2': float(r2),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'test_periods': test_size,
                    'lag_order': lag_order
                }
            else:
                evaluation_metrics = {
                    'r2': 0.0,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'mape': 0.0,
                    'test_periods': 0,
                    'lag_order': 0,
                    'note': 'Insufficient data for evaluation'
                }
        else:
            evaluation_metrics = {
                'r2': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'test_periods': 0,
                'lag_order': results.k_ar,
                'note': 'No test data available'
            }
        
        # Add evaluation metrics to results object
        results.evaluation_metrics_ = evaluation_metrics
        
        # Create comprehensive predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col], 
            "prediction": fitted_values,
            "actual": df[target],
            "residuals": df[target] - fitted_values
        })
        
        # Add impulse response data
        predictions['cumulative_effect'] = 0.0
        if len(irf_df) > 0:
            # Add cumulative effects from features to target
            for i, feature in enumerate(features):
                if feature in irf_df.columns:
                    effect_col = f'{feature}_cumulative_effect'
                    # Repeat IRF values for the length of the data
                    irf_values = irf_df[feature].values
                    repeated_values = np.tile(irf_values, (len(predictions) // len(irf_values)) + 1)[:len(predictions)]
                    predictions[effect_col] = repeated_values
        
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
        
        # Split into pre and post treatment periods
        if "post" not in df.columns:
            # Create a default treatment period (last 30% of data)
            n_obs = len(df)
            treatment_start_idx = int(0.7 * n_obs)
            df = df.copy()
            df['post'] = 0
            df.iloc[treatment_start_idx:, df.columns.get_loc('post')] = 1
            logger.info(f"Created default treatment period starting at observation {treatment_start_idx}")
        
        # Get pre-treatment data for weight optimization
        pre_treatment = df[df['post'] == 0]
        treated_pre = pre_treatment[pre_treatment['treated'] == 1]
        control_pre = pre_treatment[pre_treatment['treated'] == 0]
        
        if len(treated_pre) == 0:
            raise ValueError("No treated units in pre-treatment period")
        if len(control_pre) < len(features):
            raise ValueError(f"Not enough control units in pre-treatment period ({len(control_pre)}) for features ({len(features)})")
        
        # Optimize weights to minimize pre-treatment prediction error
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Calculate synthetic control for treated unit in pre-period
            synthetic_values = []
            for _, treated_row in treated_pre.iterrows():
                synthetic_val = 0
                for i, control_row in control_pre.iterrows():
                    synthetic_val += weights[control_pre.index.get_loc(i)] * control_row[target]
                synthetic_values.append(synthetic_val)
            
            # Calculate mean squared error
            actual_values = treated_pre[target].values
            mse = np.mean((actual_values - synthetic_values) ** 2)
            return mse
        
        # Initialize weights
        n_control = len(control_pre)
        initial_weights = np.ones(n_control) / n_control
        
        # Constraints: weights must be non-negative and sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_control)]
        
        # Optimize weights
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            optimal_weights = result.x
        except:
            logger.warning("Weight optimization failed, using equal weights")
            optimal_weights = initial_weights
        
        # Create synthetic control for all periods
        synthetic_values = []
        treatment_effects = []
        
        for _, row in df.iterrows():
            if row['treated'] == 0:
                # For control units, use actual values
                synthetic_values.append(row[target])
                treatment_effects.append(0)
            else:
                # For treated units, compute synthetic value using optimal weights
                synthetic_val = 0
                control_data = df[(df['treated'] == 0) & (df[date_col] == row[date_col])]
                
                if len(control_data) > 0:
                    # Use available control units for this time period
                    for i, (_, control_row) in enumerate(control_data.iterrows()):
                        if i < len(optimal_weights):
                            synthetic_val += optimal_weights[i] * control_row[target]
                else:
                    # Fallback if no control data for this period
                    synthetic_val = row[target]
                
                synthetic_values.append(synthetic_val)
                treatment_effects.append(row[target] - synthetic_val)
        
        # Create comparison DataFrame
        comp_df = pd.DataFrame({
            date_col: df[date_col],
            "Actual": df[target],
            "Synthetic": synthetic_values,
            "Treatment_Effect": treatment_effects,
            "Post": df['post'],
            "Treated": df['treated']
        })
        
        # Calculate treatment effect statistics
        post_treatment_effects = comp_df[(comp_df['Post'] == 1) & (comp_df['Treated'] == 1)]['Treatment_Effect'].values
        
        if len(post_treatment_effects) > 0:
            cumulative_effect = np.sum(post_treatment_effects)
            mean_effect = np.mean(post_treatment_effects)
            
            # Calculate Root Mean Squared Prediction Error (RMSPE) for pre-treatment fit
            pre_effects = comp_df[(comp_df['Post'] == 0) & (comp_df['Treated'] == 1)]['Treatment_Effect'].values
            rmspe = np.sqrt(np.mean(pre_effects ** 2)) if len(pre_effects) > 0 else 0
        else:
            cumulative_effect = mean_effect = rmspe = 0
        
        # Create visualization
        fig = go.Figure()
        
        # Plot actual and synthetic for treated unit only
        treated_data = comp_df[comp_df['Treated'] == 1]
        
        fig.add_trace(go.Scatter(
            x=treated_data[date_col],
            y=treated_data["Actual"],
            mode='lines+markers',
            name='Treated Unit (Actual)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=treated_data[date_col],
            y=treated_data["Synthetic"],
            mode='lines',
            name='Synthetic Control',
            line=dict(color='red', dash='dash')
        ))
        
        # Add vertical line for treatment start
        if len(treated_data[treated_data['Post'] == 1]) > 0:
            treatment_start = treated_data[treated_data['Post'] == 1][date_col].iloc[0]
            fig.add_vline(
                x=treatment_start, 
                line_dash="dot", 
                line_color="gray",
                annotation_text="Treatment Start"
            )
        
        fig.update_layout(
            title="Synthetic Control Method: Treated Unit vs Synthetic Control",
            xaxis_title=date_col,
            yaxis_title=target,
            hovermode='x unified'
        )
        
        # Create predictions DataFrame with additional metrics
        predictions = comp_df.copy()
        predictions["cumulative_effect"] = predictions["Treatment_Effect"].cumsum()
        
        # Model summary
        model = {
            "weights": optimal_weights,
            "control_units": control_pre.index.tolist(),
            "treated_count": treated_count,
            "control_count": control_count,
            "cumulative_effect": float(cumulative_effect),
            "mean_effect": float(mean_effect),
            "rmspe": float(rmspe),
            "pre_treatment_periods": len(pre_treatment),
            "post_treatment_periods": len(df[df['post'] == 1])
        }
        
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
    Train CausalImpact model with proper pre/post period analysis.
    
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
        from sklearn.metrics import r2_score
        
        # Check for 'post' column to define treatment period
        if "post" not in df.columns:
            # Create a default treatment period (last 30% of data)
            n_obs = len(df)
            treatment_start_idx = int(0.7 * n_obs)
            df = df.copy()
            df['post'] = 0
            df.iloc[treatment_start_idx:, df.columns.get_loc('post')] = 1
            logger.info(f"Created default treatment period starting at observation {treatment_start_idx}")
        
        # Split into pre and post periods
        pre_period_data = df[df['post'] == 0].copy()
        post_period_data = df[df['post'] == 1].copy()
        
        if len(pre_period_data) < 10:
            raise ValueError("Insufficient pre-treatment data (need at least 10 observations)")
        if len(post_period_data) < 1:
            raise ValueError("No post-treatment data found")
        
        # Prepare data for modeling
        if features:
            # Use features as control variables
            pre_X = pre_period_data[features]
            pre_y = pre_period_data[target]
            post_X = post_period_data[features]
            post_y = post_period_data[target]
            
            # Train regression model on pre-period
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(pre_X, pre_y)
            
            # Generate counterfactual predictions for post-period
            counterfactual_post = model.predict(post_X)
            
            # Fit ARIMA to residuals for better modeling
            residuals = pre_y - model.predict(pre_X)
            try:
                arima_residuals = auto_arima(residuals, seasonal=False, suppress_warnings=True)
                residual_forecast = arima_residuals.forecast(steps=len(post_period_data))
                counterfactual_post += residual_forecast
            except:
                logger.warning("ARIMA residual modeling failed, using simple regression")
        else:
            # Use ARIMA model on target variable only
            model = auto_arima(pre_period_data[target], seasonal=False, suppress_warnings=True)
            counterfactual_post = model.forecast(steps=len(post_period_data))
        
        # Prevent negative predictions for count data
        counterfactual_post = np.maximum(counterfactual_post, 0)
        
        # Calculate treatment effects
        actual_post = post_period_data[target].values
        treatment_effects = actual_post - counterfactual_post
        
        # Create full predictions DataFrame
        predictions = pd.DataFrame({
            date_col: df[date_col],
            "actual": df[target],
            "prediction": np.concatenate([
                pre_period_data[target].values,  # Use actual values for pre-period
                counterfactual_post  # Use counterfactual for post-period
            ]),
            "post": df['post']
        })
        
        # Add treatment effect calculations
        predictions['treatment_effect'] = 0.0
        predictions.loc[predictions['post'] == 1, 'treatment_effect'] = treatment_effects
        
        # Calculate cumulative effects
        predictions['cumulative_effect'] = predictions['treatment_effect'].cumsum()
        
        # Calculate confidence intervals (simplified approach)
        post_residuals = treatment_effects
        se = np.std(post_residuals) if len(post_residuals) > 1 else 0
        predictions['ci_lower'] = predictions['prediction'] - 1.96 * se
        predictions['ci_upper'] = predictions['prediction'] + 1.96 * se
        
        # Calculate summary statistics
        total_effect = np.sum(treatment_effects)
        avg_effect = np.mean(treatment_effects)
        post_mean = np.mean(actual_post)
        relative_effect = (avg_effect / post_mean * 100) if post_mean != 0 else 0
        
        # Add summary to model
        model_summary = {
            'model': model,
            'total_effect': float(total_effect),
            'avg_effect': float(avg_effect),
            'relative_effect': float(relative_effect),
            'p_value': 0.05,  # Simplified - would need proper Bayesian calculation
            'pre_periods': len(pre_period_data),
            'post_periods': len(post_period_data)
        }
        
        # Create visualization
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=df[date_col], 
            y=df[target],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Plot counterfactual (only for post-period)
        fig.add_trace(go.Scatter(
            x=post_period_data[date_col],
            y=counterfactual_post,
            mode='lines',
            name='Counterfactual',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence interval
        if len(post_period_data) > 0:
            post_ci_lower = counterfactual_post - 1.96 * se
            post_ci_upper = counterfactual_post + 1.96 * se
            
            fig.add_trace(go.Scatter(
                x=post_period_data[date_col],
                y=post_ci_lower,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=post_period_data[date_col],
                y=post_ci_upper,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='95% CI',
                hoverinfo='skip'
            ))
        
        # Add vertical line for treatment start
        treatment_start_date = post_period_data[date_col].iloc[0]
        fig.add_vline(
            x=treatment_start_date, 
            line_dash="dot", 
            line_color="gray",
            annotation_text="Treatment Start"
        )
        
        fig.update_layout(
            title="CausalImpact Analysis: Actual vs Counterfactual",
            xaxis_title=date_col,
            yaxis_title=target,
            hovermode='x unified'
        )
        
        logger.info(f"CausalImpact model trained successfully. Total effect: {total_effect:.2f}, Avg effect: {avg_effect:.2f}")
        return model_summary, predictions, fig
        
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


# ============================================================================
# Utility Functions for Model Improvements
# ============================================================================

def validate_parallel_trends(
    df: pd.DataFrame, 
    outcome_col: str, 
    group_col: str = 'treated', 
    time_col: str = 'time',
    treatment_start: Any = None
) -> Dict:
    """
    Validate parallel trends assumption for DiD analysis.
    
    Args:
        df: DataFrame with pre-treatment data
        outcome_col: Name of outcome variable
        group_col: Name of treatment group column
        time_col: Name of time column
        treatment_start: Start of treatment period
        
    Returns:
        Dictionary with parallel trends test results
    """
    try:
        import statsmodels.api as sm
        
        # Filter to pre-treatment period if treatment_start is provided
        if treatment_start is not None:
            pre_data = df[df[time_col] < treatment_start].copy()
        else:
            pre_data = df.copy()
        
        if len(pre_data) < 10:
            return {
                'parallel_trends_holds': False,
                'reason': 'Insufficient pre-treatment data',
                'slope_diff': 0,
                'p_value': 1.0
            }
        
        # Create numeric time variable
        pre_data['time_numeric'] = pd.factorize(pre_data[time_col])[0]
        
        # Fit separate regressions for treatment and control groups
        control_data = pre_data[pre_data[group_col] == 0]
        treatment_data = pre_data[pre_data[group_col] == 1]
        
        if len(control_data) < 3 or len(treatment_data) < 3:
            return {
                'parallel_trends_holds': False,
                'reason': 'Insufficient observations in treatment or control group',
                'slope_diff': 0,
                'p_value': 1.0
            }
        
        # Fit trend models
        control_trend = sm.OLS(
            control_data[outcome_col],
            sm.add_constant(control_data['time_numeric'])
        ).fit()
        
        treatment_trend = sm.OLS(
            treatment_data[outcome_col],
            sm.add_constant(treatment_data['time_numeric'])
        ).fit()
        
        # Calculate slope difference
        control_slope = control_trend.params.iloc[1] if len(control_trend.params) > 1 else 0
        treatment_slope = treatment_trend.params.iloc[1] if len(treatment_trend.params) > 1 else 0
        slope_diff = treatment_slope - control_slope
        
        # Calculate standard error of difference
        control_se = control_trend.bse.iloc[1] if len(control_trend.bse) > 1 else 0
        treatment_se = treatment_trend.bse.iloc[1] if len(treatment_trend.bse) > 1 else 0
        slope_diff_se = np.sqrt(control_se**2 + treatment_se**2)
        
        # Calculate t-statistic and p-value
        if slope_diff_se > 0:
            t_stat = slope_diff / slope_diff_se
            # Use minimum degrees of freedom
            df_min = min(len(control_data) - 2, len(treatment_data) - 2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_min))
        else:
            p_value = 1.0
            
        return {
            'parallel_trends_holds': p_value > 0.05,
            'slope_diff': float(slope_diff),
            'p_value': float(p_value),
            'control_slope': float(control_slope),
            'treatment_slope': float(treatment_slope),
            'control_observations': len(control_data),
            'treatment_observations': len(treatment_data)
        }
        
    except Exception as e:
        logger.warning(f"Error in parallel trends validation: {str(e)}")
        return {
            'parallel_trends_holds': False,
            'reason': f'Error: {str(e)}',
            'slope_diff': 0,
            'p_value': 1.0
        }


def balance_treatment_groups(
    df: pd.DataFrame, 
    treatment_col: str = 'treated',
    balance_threshold: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance treatment and control groups to improve DiD design.
    
    Args:
        df: Input DataFrame
        treatment_col: Name of treatment indicator column
        balance_threshold: Minimum acceptable balance ratio
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced DataFrame
    """
    try:
        # Check current distribution
        treatment_counts = df[treatment_col].value_counts()
        
        if len(treatment_counts) < 2:
            logger.warning("Only one group present, no balancing needed")
            return df
        
        minority_class = treatment_counts.idxmin()
        majority_class = treatment_counts.idxmax()
        
        # Calculate imbalance ratio
        imbalance_ratio = treatment_counts.min() / treatment_counts.max()
        
        if imbalance_ratio >= balance_threshold:
            logger.info(f"Groups already balanced (ratio: {imbalance_ratio:.3f})")
            return df
        
        logger.info(f"Balancing groups (current ratio: {imbalance_ratio:.3f})")
        
        # Undersample majority class
        n_samples = treatment_counts.min()
        
        balanced_df = pd.concat([
            df[df[treatment_col] == minority_class],
            df[df[treatment_col] == majority_class].sample(
                n=n_samples, random_state=random_state
            )
        ])
        
        # Verify balancing
        new_counts = balanced_df[treatment_col].value_counts()
        new_ratio = new_counts.min() / new_counts.max()
        
        logger.info(f"Balanced groups - new ratio: {new_ratio:.3f}")
        logger.info(f"Sample sizes: {dict(new_counts)}")
        
        return balanced_df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error in group balancing: {str(e)}")
        return df


def calculate_treatment_effects_with_ci(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = 'treated',
    time_col: str = 'time',
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate treatment effects with confidence intervals for each time period.
    
    Args:
        df: DataFrame with outcome, treatment, and time data
        outcome_col: Name of outcome variable
        treatment_col: Name of treatment indicator
        time_col: Name of time variable
        confidence_level: Confidence level for intervals
        
    Returns:
        DataFrame with treatment effects and confidence intervals
    """
    try:
        results = []
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        for time_point in df[time_col].unique():
            period_data = df[df[time_col] == time_point]
            
            if len(period_data) < 2:
                continue
                
            treated_data = period_data[period_data[treatment_col] == 1][outcome_col]
            control_data = period_data[period_data[treatment_col] == 0][outcome_col]
            
            if len(treated_data) == 0 or len(control_data) == 0:
                continue
            
            # Calculate means
            treated_mean = treated_data.mean()
            control_mean = control_data.mean()
            effect = treated_mean - control_mean
            
            # Calculate standard error
            treated_var = treated_data.var() / len(treated_data) if len(treated_data) > 1 else 0
            control_var = control_data.var() / len(control_data) if len(control_data) > 1 else 0
            se = np.sqrt(treated_var + control_var)
            
            # Calculate confidence interval
            ci_margin = z_score * se
            ci_lower = effect - ci_margin
            ci_upper = effect + ci_margin
            
            # Calculate t-statistic and p-value
            if se > 0:
                t_stat = effect / se
                df_combined = len(treated_data) + len(control_data) - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_combined))
            else:
                t_stat = 0
                p_value = 1.0
            
            results.append({
                time_col: time_point,
                'treatment_effect': effect,
                'standard_error': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                't_statistic': t_stat,
                'treated_n': len(treated_data),
                'control_n': len(control_data),
                'treated_mean': treated_mean,
                'control_mean': control_mean
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error calculating treatment effects: {str(e)}")
        return pd.DataFrame()


def evaluate_model_with_shap(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    feature_names: List[str] = None
) -> Dict:
    """
    Evaluate ML model with SHAP explanations and proper variable handling.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        feature_names: Names of features
        
    Returns:
        Dictionary with metrics, SHAP values, and diagnostics
    """
    try:
        import shap
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics (fix the 'actual_values' undefined variable issue)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        }
        
        # Generate SHAP values
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values.values).mean(0)
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            shap_values = None
            importance_df = pd.DataFrame()
        
        # Create diagnostics DataFrame
        diagnostics = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'residuals': y_test - y_pred,
            'abs_residuals': np.abs(y_test - y_pred)
        })
        
        return {
            'metrics': metrics,
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'diagnostics': diagnostics,
            'n_test_samples': len(y_test)
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return {
            'metrics': {'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0},
            'shap_values': None,
            'feature_importance': pd.DataFrame(),
            'diagnostics': pd.DataFrame(),
            'error': str(e)
        } 