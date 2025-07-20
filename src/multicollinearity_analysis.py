"""
Multicollinearity Analysis Pipeline for Ad Impact Models

This module provides comprehensive analysis to improve predictive models by:
- Detecting and visualizing correlations
- Identifying multicollinearity through VIF analysis
- Suggesting variable reduction strategies
- Testing for overfitting
- Generating structured reports for dashboard integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MulticollinearityAnalyzer:
    """
    Comprehensive multicollinearity and overfitting analysis for ad impact models.
    """
    
    def __init__(self, target_column: str = 'Visits', correlation_threshold: float = 0.8, 
                 vif_threshold: float = 10.0, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the analyzer with configurable parameters.
        
        Args:
            target_column: Name of the target variable
            correlation_threshold: Threshold for high correlation detection
            vif_threshold: Threshold for VIF-based multicollinearity detection
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.test_size = test_size
        self.random_state = random_state
        
        # Storage for analysis results
        self.data = None
        self.numeric_data = None
        self.correlation_matrix = None
        self.high_corr_pairs = []
        self.vif_results = None
        self.suggested_removals = []
        self.reduced_features = []
        self.model_results = {}
        self.overfitting_alerts = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        print(f"Loading data from {data_path}...")
        
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Extract numeric columns only
        self.numeric_data = self.data.select_dtypes(include=[np.number])
        print(f"Numeric columns: {len(self.numeric_data.columns)}")
        
        # Verify target column exists
        if self.target_column not in self.numeric_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in numeric data")
            
        return self.data
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation matrix for all numeric variables."""
        print("Calculating correlation matrix...")
        
        self.correlation_matrix = self.numeric_data.corr(method='pearson')
        return self.correlation_matrix
    
    def plot_correlation_heatmap(self, save_path: str = 'correlation_heatmap.png', 
                                interactive: bool = False) -> None:
        """
        Generate and save correlation heatmap visualization.
        
        Args:
            save_path: Path to save the heatmap image
            interactive: Whether to create an interactive Plotly heatmap
        """
        print(f"Generating correlation heatmap...")
        
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        if interactive:
            # Create interactive Plotly heatmap
            fig = px.imshow(
                self.correlation_matrix,
                labels=dict(x="Variables", y="Variables", color="Correlation"),
                title="Correlation Matrix Heatmap",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig.update_layout(
                width=1200,
                height=1000,
                title_x=0.5,
                font=dict(size=10)
            )
            fig.write_html(save_path.replace('.png', '_interactive.html'))
            
        # Create static matplotlib heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8},
            mask=mask,
            annot_kws={'size': 8}
        )
        
        plt.title('Correlation Matrix Heatmap\n(Lower Triangle)', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation heatmap saved as {save_path}")
    
    def identify_high_correlations(self) -> List[Dict[str, Any]]:
        """
        Identify variable pairs with high correlation (above threshold).
        
        Returns:
            List of dictionaries containing high correlation pairs and their correlations
        """
        print(f"Identifying correlations above {self.correlation_threshold}...")
        
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        self.high_corr_pairs = []
        
        # Get upper triangle of correlation matrix to avoid duplicates
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                if upper_triangle[i, j]:
                    var1 = self.correlation_matrix.columns[i]
                    var2 = self.correlation_matrix.columns[j]
                    corr_value = self.correlation_matrix.iloc[i, j]
                    
                    if abs(corr_value) > self.correlation_threshold:
                        # Determine which variable to keep (stronger correlation with target)
                        target_corr1 = abs(self.correlation_matrix.loc[var1, self.target_column])
                        target_corr2 = abs(self.correlation_matrix.loc[var2, self.target_column])
                        
                        keep_var = var1 if target_corr1 > target_corr2 else var2
                        remove_var = var2 if target_corr1 > target_corr2 else var1
                        
                        self.high_corr_pairs.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value),
                            'target_correlation_var1': target_corr1,
                            'target_correlation_var2': target_corr2,
                            'recommended_keep': keep_var,
                            'recommended_remove': remove_var
                        })
        
        # Sort by absolute correlation value
        self.high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"Found {len(self.high_corr_pairs)} high correlation pairs")
        return self.high_corr_pairs
    
    def calculate_vif(self, exclude_target: bool = True) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for all variables.
        
        Args:
            exclude_target: Whether to exclude target variable from VIF calculation
            
        Returns:
            DataFrame with VIF values for each variable
        """
        print("Calculating Variance Inflation Factors...")
        
        # Prepare data for VIF calculation
        vif_data = self.numeric_data.copy()
        
        if exclude_target and self.target_column in vif_data.columns:
            vif_data = vif_data.drop(columns=[self.target_column])
        
        # Remove any columns with zero variance
        vif_data = vif_data.loc[:, vif_data.var() != 0]
        
        # Handle missing values
        vif_data = vif_data.dropna()
        
        if vif_data.empty:
            print("Warning: No valid data for VIF calculation")
            return pd.DataFrame()
        
        # Add constant term for VIF calculation
        vif_data_with_const = add_constant(vif_data)
        
        # Calculate VIF for each variable
        vif_results = []
        for i, column in enumerate(vif_data.columns):
            try:
                vif_value = variance_inflation_factor(vif_data_with_const.values, i + 1)  # +1 to skip constant
                vif_results.append({
                    'Variable': column,
                    'VIF': vif_value,
                    'High_Multicollinearity': vif_value > self.vif_threshold
                })
            except Exception as e:
                print(f"Warning: Could not calculate VIF for {column}: {e}")
                vif_results.append({
                    'Variable': column,
                    'VIF': np.inf,
                    'High_Multicollinearity': True
                })
        
        self.vif_results = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
        
        high_vif_count = sum(self.vif_results['High_Multicollinearity'])
        print(f"Variables with VIF > {self.vif_threshold}: {high_vif_count}")
        
        return self.vif_results
    
    def generate_reduction_suggestions(self) -> Dict[str, Any]:
        """
        Generate comprehensive variable reduction suggestions based on correlation and VIF analysis.
        
        Returns:
            Dictionary containing detailed reduction suggestions
        """
        print("Generating variable reduction suggestions...")
        
        # Ensure analyses are complete
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        if not self.high_corr_pairs:
            self.identify_high_correlations()
        if self.vif_results is None:
            self.calculate_vif()
        
        # Collect variables to remove
        removal_candidates = set()
        
        # From high correlation analysis
        correlation_removals = [pair['recommended_remove'] for pair in self.high_corr_pairs]
        removal_candidates.update(correlation_removals)
        
        # From VIF analysis
        if self.vif_results is not None and not self.vif_results.empty:
            high_vif_vars = self.vif_results[
                self.vif_results['High_Multicollinearity']
            ]['Variable'].tolist()
            
            # For VIF removals, prioritize those with weakest target correlation
            vif_removals = []
            for var in high_vif_vars:
                if var in self.correlation_matrix.columns:
                    target_corr = abs(self.correlation_matrix.loc[var, self.target_column])
                    vif_removals.append((var, target_corr))
            
            # Sort by target correlation (ascending) and take the weakest correlations
            vif_removals.sort(key=lambda x: x[1])
            vif_removal_vars = [var for var, _ in vif_removals]
            removal_candidates.update(vif_removal_vars)
        
        # Ensure target variable is not removed
        removal_candidates.discard(self.target_column)
        
        # Create final feature set
        all_features = set(self.numeric_data.columns)
        self.reduced_features = list(all_features - removal_candidates)
        
        self.suggested_removals = list(removal_candidates)
        
        reduction_summary = {
            'original_feature_count': len(self.numeric_data.columns),
            'suggested_removals': self.suggested_removals,
            'reduced_feature_set': self.reduced_features,
            'reduced_feature_count': len(self.reduced_features),
            'reduction_percentage': (len(self.suggested_removals) / len(self.numeric_data.columns)) * 100,
            'high_correlation_removals': correlation_removals,
            'high_vif_removals': vif_removal_vars if 'vif_removal_vars' in locals() else [],
            'removal_reasons': {}
        }
        
        # Add detailed removal reasons
        for var in self.suggested_removals:
            reasons = []
            if var in correlation_removals:
                reasons.append(f"High correlation (>{self.correlation_threshold})")
            if self.vif_results is not None and var in self.vif_results['Variable'].values:
                vif_val = self.vif_results[self.vif_results['Variable'] == var]['VIF'].iloc[0]
                if vif_val > self.vif_threshold:
                    reasons.append(f"High VIF ({vif_val:.2f} > {self.vif_threshold})")
            reduction_summary['removal_reasons'][var] = reasons
        
        print(f"Suggested removing {len(self.suggested_removals)} variables ({reduction_summary['reduction_percentage']:.1f}% reduction)")
        
        return reduction_summary
    
    def run_baseline_models(self, use_reduced_features: bool = True) -> Dict[str, Any]:
        """
        Run baseline predictive models with original and reduced feature sets.
        
        Args:
            use_reduced_features: Whether to use the reduced feature set
            
        Returns:
            Dictionary containing model performance results
        """
        print("Running baseline predictive models...")
        
        # Prepare feature sets
        if use_reduced_features and self.reduced_features:
            feature_columns = [col for col in self.reduced_features if col != self.target_column]
            print(f"Using reduced feature set: {len(feature_columns)} features")
        else:
            feature_columns = [col for col in self.numeric_data.columns if col != self.target_column]
            print(f"Using all features: {len(feature_columns)} features")
        
        # Prepare data
        X = self.numeric_data[feature_columns].dropna()
        y = self.numeric_data.loc[X.index, self.target_column]
        
        if X.empty or y.empty:
            print("Warning: No valid data for modeling")
            return {}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'XGBoost': xgb.XGBRegressor(random_state=self.random_state, verbosity=0),
            'Random Forest': RandomForestRegressor(random_state=self.random_state, n_estimators=100)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            try:
                # Use scaled data for Linear Regression, original for tree-based models
                if model_name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Calculate adjusted R² for training set
                n_train = len(y_train)
                k = len(feature_columns)
                adj_r2_train = 1 - (1 - train_r2) * (n_train - 1) / (n_train - k - 1)
                
                results[model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_adj_r2': adj_r2_train,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'r2_difference': train_r2 - test_r2,
                    'mae_ratio': test_mae / train_mae if train_mae > 0 else np.inf,
                    'feature_count': len(feature_columns)
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.model_results = results
        print("Baseline modeling completed")
        return results
    
    def detect_overfitting(self, r2_threshold: float = 0.1, mae_ratio_threshold: float = 1.3) -> List[Dict[str, Any]]:
        """
        Detect potential overfitting in trained models.
        
        Args:
            r2_threshold: Threshold for R² difference between train and test
            mae_ratio_threshold: Threshold for MAE ratio (test/train)
            
        Returns:
            List of overfitting alerts
        """
        print("Detecting overfitting...")
        
        if not self.model_results:
            print("Warning: No model results available for overfitting detection")
            return []
        
        self.overfitting_alerts = []
        
        for model_name, results in self.model_results.items():
            if 'error' in results:
                continue
                
            alerts = []
            
            # Check R² difference
            r2_diff = results.get('r2_difference', 0)
            if r2_diff > r2_threshold:
                alerts.append({
                    'type': 'high_r2_difference',
                    'message': f"R² difference ({r2_diff:.3f}) exceeds threshold ({r2_threshold})",
                    'severity': 'high' if r2_diff > 0.2 else 'medium'
                })
            
            # Check MAE ratio
            mae_ratio = results.get('mae_ratio', 1.0)
            if mae_ratio > mae_ratio_threshold:
                alerts.append({
                    'type': 'high_mae_ratio',
                    'message': f"Test MAE is {mae_ratio:.2f}x higher than train MAE (threshold: {mae_ratio_threshold})",
                    'severity': 'high' if mae_ratio > 2.0 else 'medium'
                })
            
            # Check for very high training performance
            train_r2 = results.get('train_r2', 0)
            test_r2 = results.get('test_r2', 0)
            if train_r2 > 0.95 and test_r2 < 0.8:
                alerts.append({
                    'type': 'suspiciously_high_train_performance',
                    'message': f"Very high train R² ({train_r2:.3f}) with much lower test R² ({test_r2:.3f})",
                    'severity': 'high'
                })
            
            if alerts:
                self.overfitting_alerts.append({
                    'model': model_name,
                    'alerts': alerts,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'r2_difference': r2_diff,
                    'mae_ratio': mae_ratio
                })
        
        print(f"Overfitting alerts generated for {len(self.overfitting_alerts)} models")
        return self.overfitting_alerts
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report with all findings.
        
        Returns:
            Dictionary containing complete analysis results
        """
        print("Generating comprehensive report...")
        
        # Ensure all analyses are complete
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        if not self.high_corr_pairs:
            self.identify_high_correlations()
        if self.vif_results is None:
            self.calculate_vif()
        
        reduction_suggestions = self.generate_reduction_suggestions()
        
        # Run models if not already done
        if not self.model_results:
            self.run_baseline_models(use_reduced_features=True)
        
        # Detect overfitting if not already done
        if not self.overfitting_alerts:
            self.detect_overfitting()
        
        # Compile comprehensive report
        report = {
            'analysis_metadata': {
                'dataset_shape': self.data.shape if self.data is not None else None,
                'numeric_columns_count': len(self.numeric_data.columns) if self.numeric_data is not None else 0,
                'target_variable': self.target_column,
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold,
                'test_size': self.test_size,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            
            'correlation_analysis': {
                'high_correlation_pairs_count': len(self.high_corr_pairs),
                'high_correlation_pairs': self.high_corr_pairs[:10],  # Top 10 for readability
                'correlation_matrix_summary': {
                    'mean_absolute_correlation': float(np.abs(self.correlation_matrix.values).mean()) if self.correlation_matrix is not None else None,
                    'max_correlation': float(np.abs(self.correlation_matrix.values).max()) if self.correlation_matrix is not None else None
                }
            },
            
            'vif_analysis': {
                'high_vif_variables_count': int(self.vif_results['High_Multicollinearity'].sum()) if self.vif_results is not None else 0,
                'vif_results': self.vif_results.to_dict('records') if self.vif_results is not None else [],
                'mean_vif': float(self.vif_results['VIF'].mean()) if self.vif_results is not None else None,
                'max_vif': float(self.vif_results['VIF'].max()) if self.vif_results is not None else None
            },
            
            'variable_reduction': reduction_suggestions,
            
            'baseline_models': {
                'model_performance': self.model_results,
                'best_model': self._identify_best_model(),
                'performance_summary': self._summarize_model_performance()
            },
            
            'overfitting_analysis': {
                'overfitting_alerts_count': len(self.overfitting_alerts),
                'overfitting_alerts': self.overfitting_alerts,
                'overall_overfitting_risk': self._assess_overall_overfitting_risk()
            },
            
            'recommendations': self._generate_final_recommendations()
        }
        
        return report
    
    def _identify_best_model(self) -> Dict[str, Any]:
        """Identify the best performing model based on test performance."""
        if not self.model_results:
            return {}
        
        valid_models = {name: results for name, results in self.model_results.items() 
                       if 'error' not in results}
        
        if not valid_models:
            return {}
        
        # Sort by test R² (descending)
        best_model_name = max(valid_models.keys(), 
                             key=lambda x: valid_models[x].get('test_r2', 0))
        
        return {
            'model_name': best_model_name,
            'performance': valid_models[best_model_name]
        }
    
    def _summarize_model_performance(self) -> Dict[str, Any]:
        """Summarize overall model performance statistics."""
        if not self.model_results:
            return {}
        
        valid_results = [results for results in self.model_results.values() 
                        if 'error' not in results]
        
        if not valid_results:
            return {}
        
        test_r2_values = [r.get('test_r2', 0) for r in valid_results]
        r2_differences = [r.get('r2_difference', 0) for r in valid_results]
        
        return {
            'mean_test_r2': float(np.mean(test_r2_values)),
            'std_test_r2': float(np.std(test_r2_values)),
            'mean_r2_difference': float(np.mean(r2_differences)),
            'models_count': len(valid_results)
        }
    
    def _assess_overall_overfitting_risk(self) -> str:
        """Assess overall overfitting risk level."""
        if not self.overfitting_alerts:
            return 'low'
        
        high_severity_count = sum(1 for alert in self.overfitting_alerts 
                                 for a in alert['alerts'] if a['severity'] == 'high')
        
        if high_severity_count >= 2:
            return 'high'
        elif high_severity_count >= 1 or len(self.overfitting_alerts) >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final actionable recommendations."""
        recommendations = []
        
        # Variable reduction recommendations
        if self.suggested_removals:
            recommendations.append(
                f"Remove {len(self.suggested_removals)} variables with high multicollinearity: "
                f"{', '.join(self.suggested_removals[:5])}{'...' if len(self.suggested_removals) > 5 else ''}"
            )
        
        # Model performance recommendations
        best_model = self._identify_best_model()
        if best_model:
            test_r2 = best_model['performance'].get('test_r2', 0)
            if test_r2 < 0.3:
                recommendations.append("Consider feature engineering or additional data collection - current predictive power is low")
            elif test_r2 > 0.8:
                recommendations.append("Good predictive performance achieved - monitor for overfitting in production")
        
        # Overfitting recommendations
        overfitting_risk = self._assess_overall_overfitting_risk()
        if overfitting_risk == 'high':
            recommendations.append("High overfitting risk detected - consider regularization, cross-validation, or more data")
        elif overfitting_risk == 'medium':
            recommendations.append("Moderate overfitting detected - monitor model performance on new data")
        
        # Multicollinearity recommendations
        if self.vif_results is not None and not self.vif_results.empty:
            high_vif_count = self.vif_results['High_Multicollinearity'].sum()
            if high_vif_count > 0:
                recommendations.append(f"Address {high_vif_count} variables with VIF > {self.vif_threshold} to improve model stability")
        
        return recommendations
    
    def save_report(self, filepath: str = 'model_improvement_report.json') -> None:
        """Save the comprehensive report to a JSON file."""
        print(f"Saving report to {filepath}...")
        
        report = self.generate_comprehensive_report()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_numpy_types(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved successfully to {filepath}")
    
    def save_reduced_dataset(self, filepath: str = 'reduced_dataset.csv') -> None:
        """Save dataset with only the reduced feature set."""
        if not self.reduced_features:
            print("No reduced feature set available - skipping dataset save")
            return
        
        print(f"Saving reduced dataset to {filepath}...")
        
        reduced_data = self.data[self.reduced_features].copy()
        reduced_data.to_csv(filepath, index=False)
        
        print(f"Reduced dataset saved: {len(self.reduced_features)} columns, {len(reduced_data)} rows")
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the analysis results."""
        print("\n" + "="*80)
        print("MULTICOLLINEARITY ANALYSIS SUMMARY")
        print("="*80)
        
        if self.data is not None:
            print(f"Dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            print(f"Numeric columns: {len(self.numeric_data.columns)}")
            print(f"Target variable: {self.target_column}")
        
        print(f"\nCORRELATION ANALYSIS:")
        print(f"  - High correlation pairs (>{self.correlation_threshold}): {len(self.high_corr_pairs)}")
        if self.high_corr_pairs:
            print(f"  - Strongest correlation: {self.high_corr_pairs[0]['abs_correlation']:.3f}")
            print(f"    between {self.high_corr_pairs[0]['variable_1']} and {self.high_corr_pairs[0]['variable_2']}")
        
        print(f"\nVIF ANALYSIS:")
        if self.vif_results is not None and not self.vif_results.empty:
            high_vif_count = self.vif_results['High_Multicollinearity'].sum()
            print(f"  - Variables with VIF > {self.vif_threshold}: {high_vif_count}")
            if high_vif_count > 0:
                max_vif_var = self.vif_results.iloc[0]
                print(f"  - Highest VIF: {max_vif_var['Variable']} (VIF = {max_vif_var['VIF']:.2f})")
        
        print(f"\nVARIABLE REDUCTION:")
        print(f"  - Suggested removals: {len(self.suggested_removals)}")
        if self.suggested_removals:
            print(f"  - Variables to remove: {', '.join(self.suggested_removals[:5])}")
            if len(self.suggested_removals) > 5:
                print(f"    ... and {len(self.suggested_removals) - 5} more")
        print(f"  - Reduced feature set: {len(self.reduced_features)} variables")
        
        print(f"\nMODEL PERFORMANCE:")
        if self.model_results:
            for model_name, results in self.model_results.items():
                if 'error' not in results:
                    print(f"  - {model_name}:")
                    print(f"    Train R²: {results.get('train_r2', 0):.3f}, Test R²: {results.get('test_r2', 0):.3f}")
                    print(f"    R² difference: {results.get('r2_difference', 0):.3f}")
        
        print(f"\nOVERFITTING ANALYSIS:")
        print(f"  - Models with overfitting alerts: {len(self.overfitting_alerts)}")
        print(f"  - Overall overfitting risk: {self._assess_overall_overfitting_risk().upper()}")
        
        print(f"\nRECOMMENDATIONS:")
        recommendations = self._generate_final_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("="*80)


def run_complete_analysis(data_path: str, target_column: str = 'Visits', 
                         correlation_threshold: float = 0.8, vif_threshold: float = 10.0,
                         save_outputs: bool = True) -> MulticollinearityAnalyzer:
    """
    Run the complete multicollinearity analysis pipeline.
    
    Args:
        data_path: Path to the dataset file
        target_column: Name of the target variable
        correlation_threshold: Threshold for high correlation detection
        vif_threshold: Threshold for VIF-based multicollinearity detection
        save_outputs: Whether to save output files
    
    Returns:
        Configured MulticollinearityAnalyzer instance with completed analysis
    """
    print("Starting complete multicollinearity analysis pipeline...")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MulticollinearityAnalyzer(
        target_column=target_column,
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold
    )
    
    # Load data
    analyzer.load_data(data_path)
    
    # Correlation analysis
    analyzer.calculate_correlation_matrix()
    analyzer.identify_high_correlations()
    
    # VIF analysis
    analyzer.calculate_vif()
    
    # Variable reduction
    analyzer.generate_reduction_suggestions()
    
    # Baseline modeling
    analyzer.run_baseline_models(use_reduced_features=True)
    
    # Overfitting detection
    analyzer.detect_overfitting()
    
    # Save outputs
    if save_outputs:
        analyzer.plot_correlation_heatmap('correlation_heatmap.png', interactive=True)
        analyzer.save_report('model_improvement_report.json')
        analyzer.save_reduced_dataset('reduced_dataset.csv')
    
    # Print summary
    analyzer.print_summary()
    
    print("\nAnalysis pipeline completed successfully!")
    return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = run_complete_analysis('dataset_3.csv') 