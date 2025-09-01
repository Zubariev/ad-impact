"""
Time Series Utilities for Ad Impact Modeling Dashboard.

Contains functions for stationarity tests, autocorrelation checks, and multicollinearity analysis
specifically focused on time series data.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_stationarity(series: pd.Series, method: str = 'adf', 
                       significance: float = 0.05) -> Dict[str, Any]:
    """
    Check stationarity of a time series using ADF or KPSS test.
    
    Args:
        series: Time series data to test
        method: Test method ('adf' or 'kpss')
        significance: Significance level for the test
    
    Returns:
        Dictionary with test results
    """
    results = {
        'series_name': series.name,
        'is_stationary': False,
        'p_value': None,
        'test_statistic': None,
        'critical_values': {},
        'method': method,
        'message': "",
        'suggested_transformation': None
    }
    
    try:
        if method.lower() == 'adf':
            # Augmented Dickey-Fuller test
            # Null hypothesis: series has a unit root (not stationary)
            test_result = adfuller(series.dropna(), autolag='AIC')
            results['test_statistic'] = test_result[0]
            results['p_value'] = test_result[1]
            results['critical_values'] = test_result[4]
            results['is_stationary'] = results['p_value'] < significance
            
            # Interpretation
            if results['is_stationary']:
                results['message'] = "Series is stationary (ADF test: reject null hypothesis)"
                results['suggested_transformation'] = None
            else:
                results['message'] = "Series is non-stationary (ADF test: fail to reject null hypothesis)"
                results['suggested_transformation'] = "differencing"
                
        elif method.lower() == 'kpss':
            # KPSS test
            # Null hypothesis: series is stationary
            test_result = kpss(series.dropna())
            results['test_statistic'] = test_result[0]
            results['p_value'] = test_result[1]
            results['critical_values'] = test_result[3]
            results['is_stationary'] = results['p_value'] > significance
            
            # Interpretation
            if results['is_stationary']:
                results['message'] = "Series is stationary (KPSS test: fail to reject null hypothesis)"
                results['suggested_transformation'] = None
            else:
                results['message'] = "Series is non-stationary (KPSS test: reject null hypothesis)"
                results['suggested_transformation'] = "differencing"
                
        else:
            raise ValueError(f"Unsupported stationarity test method: {method}")
            
    except Exception as e:
        results['message'] = f"Error testing stationarity: {str(e)}"
        logger.error(f"Stationarity test failed: {str(e)}")
    
    return results


def apply_differencing(series: pd.Series, order: int = 1) -> pd.Series:
    """
    Apply differencing to a time series.
    
    Args:
        series: Original time series
        order: Differencing order
    
    Returns:
        Differenced time series
    """
    result = series.copy()
    for i in range(order):
        result = result.diff().dropna()
    
    return result


def check_autocorrelation(residuals: np.ndarray, lags: int = 20) -> Dict[str, Any]:
    """
    Check autocorrelation in model residuals.
    
    Args:
        residuals: Model residuals
        lags: Number of lags to test
    
    Returns:
        Dictionary with autocorrelation test results
    """
    results = {
        'durbin_watson': None,
        'positive_autocorr': False,
        'negative_autocorr': False,
        'ljung_box': {
            'statistic': None,
            'p_value': None,
            'autocorrelation_present': None
        },
        'acf': [],
        'pacf': [],
        'message': "",
        'suggested_action': None
    }
    
    try:
        # Durbin-Watson test
        dw_stat = durbin_watson(residuals)
        results['durbin_watson'] = dw_stat
        
        # Interpret Durbin-Watson
        if dw_stat < 1.5:
            results['positive_autocorr'] = True
            results['message'] = f"Positive autocorrelation detected (DW={dw_stat:.3f})"
            results['suggested_action'] = "Add lagged dependent variable or use robust standard errors"
        elif dw_stat > 2.5:
            results['negative_autocorr'] = True
            results['message'] = f"Negative autocorrelation detected (DW={dw_stat:.3f})"
            results['suggested_action'] = "Check for model misspecification or alternating patterns"
        else:
            results['message'] = f"No significant autocorrelation (DW={dw_stat:.3f})"
        
        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[lags])
        results['ljung_box']['statistic'] = float(lb_test.iloc[0, 0])
        results['ljung_box']['p_value'] = float(lb_test.iloc[0, 1])
        results['ljung_box']['autocorrelation_present'] = results['ljung_box']['p_value'] < 0.05
        
        # ACF and PACF
        acf_values = acf(residuals, nlags=lags)
        pacf_values = pacf(residuals, nlags=lags, method='ols')
        
        results['acf'] = acf_values.tolist()
        results['pacf'] = pacf_values.tolist()
        
    except Exception as e:
        results['message'] = f"Error testing autocorrelation: {str(e)}"
        logger.error(f"Autocorrelation test failed: {str(e)}")
    
    return results


def multicollinearity_check_with_vif(X: pd.DataFrame, 
                                     threshold: float = 10.0,
                                     include_detailed: bool = False) -> Dict[str, Any]:
    """
    Enhanced multicollinearity check using VIF with detailed analysis.
    
    Args:
        X: Feature matrix
        threshold: VIF threshold for high multicollinearity
        include_detailed: Whether to include detailed analysis
    
    Returns:
        Dictionary with VIF analysis results
    """
    results = {
        'high_multicollinearity': False,
        'vif_values': {},
        'problematic_variables': [],
        'condition_number': None,
        'correlation_matrix': {},
        'suggested_actions': [],
        'severity': 'low'
    }
    
    try:
        # Add constant for proper VIF calculation
        X_with_const = sm.add_constant(X)
        
        # Calculate VIF for each feature
        vif_data = {}
        for i in range(1, X_with_const.shape[1]):  # Skip constant
            feature_name = X_with_const.columns[i]
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_data[feature_name] = float(vif)
            
            if vif > threshold:
                results['problematic_variables'].append(feature_name)
        
        results['vif_values'] = vif_data
        results['high_multicollinearity'] = len(results['problematic_variables']) > 0
        
        # Calculate correlation matrix
        if include_detailed:
            corr_matrix = X.corr().round(3)
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find highly correlated pairs
            upper_triangle = np.triu(corr_matrix.values, k=1)
            high_corr_pairs = []
            for i in range(len(X.columns)):
                for j in range(i+1, len(X.columns)):
                    if abs(upper_triangle[i, j]) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'var1': X.columns[i],
                            'var2': X.columns[j],
                            'correlation': float(upper_triangle[i, j])
                        })
            
            results['high_correlation_pairs'] = high_corr_pairs
        
        # Calculate condition number
        try:
            from numpy.linalg import svd
            X_centered = X.values - X.mean().values
            _, s, _ = svd(X_centered)
            results['condition_number'] = float(max(s) / min(s))
        except:
            results['condition_number'] = None
        
        # Suggest actions
        if results['high_multicollinearity']:
            if len(results['problematic_variables']) > len(X.columns) // 2:
                results['severity'] = 'high'
                results['suggested_actions'] = [
                    "Use PCA to reduce dimensionality",
                    "Apply Ridge regression to stabilize coefficients",
                    "Consider removing the most problematic variables"
                ]
            else:
                results['severity'] = 'medium'
                results['suggested_actions'] = [
                    "Remove the most problematic variables one by one",
                    "Consider using Ridge regression"
                ]
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"VIF calculation failed: {str(e)}")
    
    return results


def perform_complete_time_series_analysis(data: pd.DataFrame,
                                         target_col: str,
                                         feature_cols: List[str]) -> Dict[str, Any]:
    """
    Perform complete time series analysis including stationarity, autocorrelation, and multicollinearity.
    
    Args:
        data: DataFrame with target and features
        target_col: Target column name
        feature_cols: Feature column names
    
    Returns:
        Dictionary with comprehensive analysis results
    """
    analysis_results = {
        'stationarity': {},
        'autocorrelation': {},
        'multicollinearity': {},
        'summary': {
            'stationary_series': 0,
            'non_stationary_series': 0,
            'has_autocorrelation': False,
            'has_multicollinearity': False,
            'suggested_transformations': []
        }
    }
    
    # 1. Stationarity analysis
    for col in [target_col] + feature_cols:
        adf_result = check_stationarity(data[col], method='adf')
        kpss_result = check_stationarity(data[col], method='kpss')
        
        # Combined interpretation based on both tests
        is_stationary = adf_result['is_stationary'] and kpss_result['is_stationary']
        
        analysis_results['stationarity'][col] = {
            'adf': adf_result,
            'kpss': kpss_result,
            'conclusion': {
                'is_stationary': is_stationary,
                'suggested_transformation': None if is_stationary else "differencing"
            }
        }
        
        if is_stationary:
            analysis_results['summary']['stationary_series'] += 1
        else:
            analysis_results['summary']['non_stationary_series'] += 1
            analysis_results['summary']['suggested_transformations'].append(
                f"Apply differencing to '{col}'"
            )
    
    # 2. Fit a basic model to check autocorrelation
    try:
        # Filter only complete observations
        model_data = data[[target_col] + feature_cols].dropna()
        
        # Fit OLS model
        X = sm.add_constant(model_data[feature_cols])
        y = model_data[target_col]
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        
        # Check autocorrelation in residuals
        autocorr_results = check_autocorrelation(model.resid)
        analysis_results['autocorrelation'] = autocorr_results
        
        # Update summary
        analysis_results['summary']['has_autocorrelation'] = (
            autocorr_results['positive_autocorr'] or 
            autocorr_results['negative_autocorr']
        )
        
        if analysis_results['summary']['has_autocorrelation']:
            analysis_results['summary']['suggested_transformations'].append(
                "Add lagged dependent variable or use Newey-West standard errors"
            )
        
        # 3. Check multicollinearity
        multicol_results = multicollinearity_check_with_vif(
            model_data[feature_cols], include_detailed=True
        )
        analysis_results['multicollinearity'] = multicol_results
        
        # Update summary
        analysis_results['summary']['has_multicollinearity'] = multicol_results['high_multicollinearity']
        
        if analysis_results['summary']['has_multicollinearity']:
            analysis_results['summary']['suggested_transformations'].extend(
                multicol_results['suggested_actions']
            )
        
        # 4. Model summary info
        analysis_results['model_summary'] = {
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'p_value': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        analysis_results['error'] = str(e)
    
    return analysis_results


def fit_optimal_model(data: pd.DataFrame,
                     target_col: str,
                     feature_cols: List[str],
                     time_col: str = None,
                     handle_autocorr: bool = True,
                     handle_multicoll: bool = True,
                     use_decomposition: bool = False,
                     decomposition_method: str = 'STL',
                     decomposition_period: int = None) -> Dict[str, Any]:
    """
    Fit optimal time series model based on diagnostics.

    Args:
        data: DataFrame with target and features
        target_col: Target column name
        feature_cols: Feature column names
        time_col: Time column name (optional)
        handle_autocorr: Whether to handle autocorrelation
        handle_multicoll: Whether to handle multicollinearity
        use_decomposition: Whether to apply seasonal decomposition to the target variable
        decomposition_method: Method for seasonal decomposition
        decomposition_period: Seasonal period for decomposition

    Returns:
        Dictionary with model results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    import copy
    from src.seasonal_decomposition import apply_seasonal_decomposition

    logger.info("fit_optimal_model: Starting to fit optimal model.")
    logger.info(f"fit_optimal_model: target_col={target_col}, feature_cols={feature_cols}, time_col={time_col}, use_decomposition={use_decomposition}")

    results = {
        'models': {},
        'best_model': None,
        'feature_importance': {},
        'diagnostics': {},
        'suggested_model': None,
        'model_comparison': {}
    }
    
    try:
        # Prepare data
        cols_to_include = [target_col] + feature_cols
        if use_decomposition and time_col and time_col not in cols_to_include:
            cols_to_include.append(time_col)
        model_data = data[cols_to_include].dropna()
        
        # Check if time_col is provided and is a valid datetime column for decomposition
        if use_decomposition and time_col and pd.api.types.is_datetime64_any_dtype(model_data[time_col]):
            logger.info("fit_optimal_model: Applying seasonal decomposition to target variable.")
            try:
                model_data, _, _ = apply_seasonal_decomposition(
                    model_data,
                    target_col,
                    time_col,
                    method=decomposition_method,
                    period=decomposition_period
                )
                logger.info("fit_optimal_model: Seasonal decomposition applied successfully.")
            except Exception as e:
                logger.warning(f"fit_optimal_model: Seasonal decomposition failed: {e}. Using original data.")

        # Analyze data characteristics
        logger.info("fit_optimal_model: Performing complete time series analysis.")
        analysis = perform_complete_time_series_analysis(model_data, target_col, feature_cols)
        results['diagnostics'] = analysis

        # 1. Base OLS model
        logger.info("fit_optimal_model: Fitting base OLS model.")
        try:
            X = sm.add_constant(model_data[feature_cols])
            y = model_data[target_col]
            ols_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})

            results['models']['base_ols'] = {
                'model_type': 'OLS',
                'r_squared': float(ols_model.rsquared),
                'adj_r_squared': float(ols_model.rsquared_adj),
                'aic': float(ols_model.aic),
                'coefficients': {name: float(coef) for name, coef in zip(['const'] + feature_cols, ols_model.params)},
                'p_values': {name: float(p) for name, p in zip(['const'] + feature_cols, ols_model.pvalues)},
                'durbin_watson': float(durbin_watson(ols_model.resid))
            }
            logger.info("fit_optimal_model: Base OLS model fitted successfully.")
        except Exception as e:
            results['models']['base_ols'] = {'error': str(e)}
            logger.error(f"fit_optimal_model: Base OLS model failed: {str(e)}")

        # 2. Handling autocorrelation if detected
        has_autocorr = analysis['summary'].get('has_autocorrelation', False)
        logger.info(f"fit_optimal_model: Autocorrelation detected: {has_autocorr}")

        if has_autocorr and handle_autocorr:
            # 2.1. Add lagged dependent variable
            logger.info("fit_optimal_model: Fitting OLS with lagged dependent variable.")
            try:
                lagged_data = model_data.copy()
                lagged_data['lagged_target'] = lagged_data[target_col].shift(1).fillna(0)
                lagged_data = lagged_data.dropna()

                X_lagged = sm.add_constant(lagged_data[feature_cols + ['lagged_target']])
                y_lagged = lagged_data[target_col]

                lagged_model = sm.OLS(y_lagged, X_lagged).fit(cov_type='HAC', cov_kwds={'maxlags':1})
                results['models']['lagged_dependent'] = {
                    'model_type': 'OLS with lagged dependent variable',
                    'r_squared': float(lagged_model.rsquared),
                    'adj_r_squared': float(lagged_model.rsquared_adj),
                    'aic': float(lagged_model.aic),
                    'coefficients': {name: float(coef) for name, coef in 
                                   zip(['const'] + feature_cols + ['lagged_target'], lagged_model.params)},
                    'durbin_watson': float(durbin_watson(lagged_model.resid))
                }
                logger.info("fit_optimal_model: Lagged dependent variable model fitted successfully.")
            except Exception as e:
                results['models']['lagged_dependent'] = {'error': str(e)}
                logger.error(f"fit_optimal_model: Lagged dependent variable model failed: {str(e)}")

            # 2.2. Use Newey-West robust standard errors
            logger.info("fit_optimal_model: Fitting OLS with Newey-West robust errors.")
            try:
                nw_cov = sm.stats.sandwich_covariance.cov_hac(
                    ols_model, nlags=int(np.sqrt(len(model_data)))
                )
                nw_results = ols_model.get_robustcov_results(cov=nw_cov)

                results['models']['newey_west'] = {
                    'model_type': 'OLS with Newey-West robust errors',
                    'r_squared': float(nw_results.rsquared),
                    'adj_r_squared': float(nw_results.rsquared_adj),
                    'aic': float(nw_results.aic),
                    'coefficients': {name: float(coef) for name, coef in zip(['const'] + feature_cols, nw_results.params)},
                    'p_values': {name: float(p) for name, p in zip(['const'] + feature_cols, nw_results.pvalues)}
                }
                logger.info("fit_optimal_model: Newey-West model fitted successfully.")
            except Exception as e:
                results['models']['newey_west'] = {'error': str(e)}
                logger.error(f"fit_optimal_model: Newey-West model failed: {str(e)}")

        # 3. Handling multicollinearity if detected
        has_multicol = analysis['summary'].get('has_multicollinearity', False)
        logger.info(f"fit_optimal_model: Multicollinearity detected: {has_multicol}")

        if has_multicol and handle_multicoll:
            # 3.1. Ridge regression
            logger.info("fit_optimal_model: Fitting Ridge Regression.")
            try:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(model_data[feature_cols])

                # Find optimal alpha through cross-validation
                from sklearn.linear_model import RidgeCV
                alphas = np.logspace(-3, 3, 20)
                ridge_cv = RidgeCV(alphas=alphas, cv=5)
                ridge_cv.fit(X_scaled, model_data[target_col])

                # Final Ridge model
                best_alpha = ridge_cv.alpha_
                ridge = Ridge(alpha=best_alpha)
                ridge.fit(X_scaled, model_data[target_col])

                # Get coefficients and rescale
                ridge_coefs = ridge.coef_
                ridge_intercept = ridge.intercept_

                results['models']['ridge'] = {
                    'model_type': 'Ridge Regression',
                    'alpha': float(best_alpha),
                    'r_squared': float(ridge.score(X_scaled, model_data[target_col])),
                    'coefficients': {name: float(coef) for name, coef in zip(feature_cols, ridge_coefs)},
                    'intercept': float(ridge_intercept)
                }
                logger.info("fit_optimal_model: Ridge Regression fitted successfully.")
            except Exception as e:
                results['models']['ridge'] = {'error': str(e)}
                logger.error(f"fit_optimal_model: Ridge regression failed: {str(e)}")

            # 3.2. PCA regression
            logger.info("fit_optimal_model: Fitting PCA Regression.")
            try:
                # Standardize features
                X_scaled = scaler.fit_transform(model_data[feature_cols])

                # Run PCA with enough components to explain 95% variance
                pca = PCA(n_components=0.95)
                X_pca = pca.fit_transform(X_scaled)

                # Fit OLS on principal components
                X_pca_with_const = sm.add_constant(X_pca)
                pca_model = sm.OLS(model_data[target_col], X_pca_with_const).fit(cov_type='HAC', cov_kwds={'maxlags':1})

                # Transform coefficients back to original features
                # NOTE: This is approximate and for interpretation only
                pca_coefs = np.dot(pca.components_.T, pca_model.params[1:])

                results['models']['pca_regression'] = {
                    'model_type': 'Principal Component Regression',
                    'n_components': int(pca.n_components_),
                    'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
                    'r_squared': float(pca_model.rsquared),
                    'component_coefficients': {f'PC{i+1}': float(coef) for i, coef in enumerate(pca_model.params[1:])},
                    'transformed_coefficients': {name: float(coef) for name, coef in zip(feature_cols, pca_coefs)},
                    'durbin_watson': float(durbin_watson(pca_model.resid))
                }
                logger.info("fit_optimal_model: PCA Regression fitted successfully.")
            except Exception as e:
                results['models']['pca_regression'] = {'error': str(e)}
                logger.error(f"fit_optimal_model: PCA regression failed: {str(e)}")

        # 4. Determine best model
        logger.info("fit_optimal_model: Determining best model.")
        valid_models = {name: model for name, model in results['models'].items() 
                       if 'error' not in model}

        if valid_models:
            # Compare by adjusted R-squared or AIC
            if all('adj_r_squared' in model for model in valid_models.values()):
                best_model = max(valid_models.items(), key=lambda x: x[1]['adj_r_squared'])
                results['best_model'] = best_model[0]
            elif all('aic' in model for model in valid_models.values()):
                best_model = min(valid_models.items(), key=lambda x: x[1]['aic'])
                results['best_model'] = best_model[0]
            logger.info(f"fit_optimal_model: Best model determined: {results['best_model']}")

        # 5. Model comparison
        if len(valid_models) > 1:
            logger.info("fit_optimal_model: Generating model comparison.")
            comparison = []
            for name, model in valid_models.items():
                entry = {
                    'model_name': name,
                    'model_type': model.get('model_type', name),
                    'r_squared': model.get('r_squared', None),
                    'adj_r_squared': model.get('adj_r_squared', None),
                    'aic': model.get('aic', None)
                }
                comparison.append(entry)

            # Sort by adjusted R-squared
            comparison = sorted(comparison, key=lambda x: x.get('adj_r_squared', 0) if x.get('adj_r_squared') is not None else 0, reverse=True)
            results['model_comparison'] = comparison
            logger.info("fit_optimal_model: Model comparison generated.")

        # 6. Feature importance for the best model
        if results['best_model'] in valid_models:
            logger.info("fit_optimal_model: Calculating feature importance for the best model.")
            best = valid_models[results['best_model']]
            if 'coefficients' in best:
                # Normalize coefficients for feature importance
                coefs = {k: v for k, v in best['coefficients'].items() if k != 'const'}
                abs_coefs = {k: abs(v) for k, v in coefs.items()}
                total = sum(abs_coefs.values())
                if total > 0:
                    results['feature_importance'] = {k: v/total for k, v in abs_coefs.items()}
            logger.info("fit_optimal_model: Feature importance calculated.")

        # 7. Suggested model based on diagnostics
        logger.info("fit_optimal_model: Determining suggested model based on diagnostics.")
        has_autocorr = analysis['summary'].get('has_autocorrelation', False)
        has_multicol = analysis['summary'].get('has_multicollinearity', False)

        if has_autocorr and has_multicol:
            results['suggested_model'] = 'ridge'
        elif has_autocorr:
            results['suggested_model'] = 'lagged_dependent'
        elif has_multicol:
            if analysis['multicollinearity'].get('severity', 'low') == 'high':
                results['suggested_model'] = 'pca_regression'
            else:
                results['suggested_model'] = 'ridge'
        else:
            results['suggested_model'] = 'base_ols'
        logger.info(f"fit_optimal_model: Suggested model: {results['suggested_model']}")
        
        logger.info("fit_optimal_model: Optimal model fitting completed.")
        return results

    except Exception as e:
        error_message = f"fit_optimal_model: An error occurred during optimal model fitting: {str(e)}"
        logger.error(error_message, exc_info=True) # Log full traceback
        return {'error': error_message, 'status': 'failed'}
