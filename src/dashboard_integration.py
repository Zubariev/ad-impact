"""
Dashboard Integration for Multicollinearity Analysis

This module provides seamless integration between the multicollinearity analysis pipeline 
and dashboard systems, delivering comprehensive model improvement insights directly to 
the user interface after model training completion.

Features:
- Automatic analysis trigger after model training
- Dashboard-ready data structures and visualizations
- Comprehensive JSON export for frontend consumption
- Interactive correlation visualizations for web display
- Real-time auto-suggestions and recommendations
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import base64
import io
from pathlib import Path

from multicollinearity_analysis import MulticollinearityAnalyzer


class DashboardAnalysisIntegrator:
    """
    Integration layer for displaying multicollinearity analysis results in dashboard.
    
    This class handles the conversion of analysis results into dashboard-friendly formats,
    generates interactive visualizations, and provides comprehensive JSON exports for
    frontend consumption.
    """
    
    def __init__(self, analysis_config: Optional[Dict] = None):
        """
        Initialize dashboard integrator.
        
        Args:
            analysis_config: Configuration for analysis parameters
        """
        self.config = analysis_config or {
            'correlation_threshold': 0.8,
            'vif_threshold': 10.0,
            'test_size': 0.2,
            'random_state': 42
        }
        
        self.analyzer = None
        self.dashboard_data = {}
        self.visualizations = {}
        
    def run_post_training_analysis(self, data_path: str, target_column: str, 
                                 model_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive analysis after model training and prepare dashboard data.
        
        Args:
            data_path: Path to the dataset used for training
            target_column: Name of the target variable
            model_metadata: Metadata from the trained model
            
        Returns:
            Comprehensive dashboard data structure
        """
        print("🔍 Running post-training multicollinearity analysis for dashboard...")
        
        # Initialize analyzer
        self.analyzer = MulticollinearityAnalyzer(
            target_column=target_column,
            correlation_threshold=self.config['correlation_threshold'],
            vif_threshold=self.config['vif_threshold'],
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Load data and run analysis
        self.analyzer.load_data(data_path)
        self.analyzer.calculate_correlation_matrix()
        self.analyzer.identify_high_correlations()
        self.analyzer.calculate_vif()
        reduction_suggestions = self.analyzer.generate_reduction_suggestions()
        self.analyzer.run_baseline_models(use_reduced_features=True)
        self.analyzer.detect_overfitting()
        
        # Generate dashboard-specific data structures
        self.dashboard_data = self._prepare_dashboard_data(model_metadata)
        
        # Generate interactive visualizations
        self.visualizations = self._generate_dashboard_visualizations()
        
        print(" Dashboard analysis completed successfully!")
        
        return self.dashboard_data
    
    def _prepare_dashboard_data(self, model_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepare comprehensive data structure optimized for dashboard consumption.
        
        Args:
            model_metadata: Metadata from the trained model
            
        Returns:
            Dashboard-ready data structure
        """
        # Get base analysis report
        base_report = self.analyzer.generate_comprehensive_report()
        
        # Enhance for dashboard display
        dashboard_data = {
            'analysis_summary': {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'dataset_info': {
                    'total_rows': self.analyzer.data.shape[0],
                    'total_columns': self.analyzer.data.shape[1],
                    'numeric_columns': len(self.analyzer.numeric_data.columns),
                    'target_variable': self.analyzer.target_column
                },
                'analysis_config': self.config,
                'model_metadata': model_metadata or {}
            },
            
            'multicollinearity_insights': {
                'severity_level': self._assess_multicollinearity_severity(),
                'total_high_correlations': len(self.analyzer.high_corr_pairs),
                'total_high_vif_variables': int(self.analyzer.vif_results['High_Multicollinearity'].sum()) if self.analyzer.vif_results is not None else 0,
                'reduction_impact': {
                    'variables_to_remove': len(self.analyzer.suggested_removals),
                    'reduction_percentage': round((len(self.analyzer.suggested_removals) / len(self.analyzer.numeric_data.columns)) * 100, 1),
                    'final_feature_count': len(self.analyzer.reduced_features)
                }
            },
            
            'correlation_analysis': self._format_correlation_analysis(),
            'vif_analysis': self._format_vif_analysis(),
            'variable_reduction': self._format_variable_reduction(),
            'model_performance': self._format_model_performance(),
            'overfitting_analysis': self._format_overfitting_analysis(),
            'auto_suggestions': self._generate_auto_suggestions(),
            'visualizations': self._prepare_visualization_data(),
            
            # Dashboard-specific sections
            'dashboard_alerts': self._generate_dashboard_alerts(),
            'action_items': self._generate_action_items(),
            'model_improvement_plan': self._generate_improvement_plan(),
            'export_data': self._prepare_export_data()
        }
        
        return dashboard_data
    
    def _assess_multicollinearity_severity(self) -> str:
        """Assess overall multicollinearity severity for dashboard display."""
        high_corr_count = len(self.analyzer.high_corr_pairs)
        high_vif_count = int(self.analyzer.vif_results['High_Multicollinearity'].sum()) if self.analyzer.vif_results is not None else 0
        reduction_percentage = (len(self.analyzer.suggested_removals) / len(self.analyzer.numeric_data.columns)) * 100
        
        if reduction_percentage > 80 or high_corr_count > 100:
            return 'critical'
        elif reduction_percentage > 50 or high_corr_count > 50:
            return 'high'
        elif reduction_percentage > 20 or high_corr_count > 20:
            return 'medium'
        else:
            return 'low'
    
    def _format_correlation_analysis(self) -> Dict[str, Any]:
        """Format correlation analysis for dashboard display."""
        if not self.analyzer.high_corr_pairs:
            return {'status': 'no_high_correlations'}
        
        # Group correlations by severity
        perfect_correlations = [p for p in self.analyzer.high_corr_pairs if p['abs_correlation'] >= 0.99]
        very_high_correlations = [p for p in self.analyzer.high_corr_pairs if 0.95 <= p['abs_correlation'] < 0.99]
        high_correlations = [p for p in self.analyzer.high_corr_pairs if 0.8 <= p['abs_correlation'] < 0.95]
        
        return {
            'summary': {
                'total_pairs': len(self.analyzer.high_corr_pairs),
                'perfect_correlations': len(perfect_correlations),
                'very_high_correlations': len(very_high_correlations),
                'high_correlations': len(high_correlations),
                'strongest_correlation': {
                    'value': self.analyzer.high_corr_pairs[0]['abs_correlation'],
                    'variables': [self.analyzer.high_corr_pairs[0]['variable_1'], self.analyzer.high_corr_pairs[0]['variable_2']]
                }
            },
            'top_correlations': self.analyzer.high_corr_pairs[:20],  # Top 20 for dashboard display
            'correlation_groups': {
                'perfect': perfect_correlations[:10],
                'very_high': very_high_correlations[:10],
                'high': high_correlations[:10]
            },
            'matrix_stats': {
                'mean_abs_correlation': float(np.abs(self.analyzer.correlation_matrix.values).mean()),
                'max_correlation': float(np.abs(self.analyzer.correlation_matrix.values).max()),
                'min_correlation': float(np.abs(self.analyzer.correlation_matrix.values).min())
            }
        }
    
    def _format_vif_analysis(self) -> Dict[str, Any]:
        """Format VIF analysis for dashboard display."""
        if self.analyzer.vif_results is None or self.analyzer.vif_results.empty:
            return {'status': 'no_vif_data'}
        
        # Categorize VIF values
        vif_categories = {
            'severe': self.analyzer.vif_results[self.analyzer.vif_results['VIF'] > 50],
            'high': self.analyzer.vif_results[(self.analyzer.vif_results['VIF'] > 10) & (self.analyzer.vif_results['VIF'] <= 50)],
            'moderate': self.analyzer.vif_results[(self.analyzer.vif_results['VIF'] > 5) & (self.analyzer.vif_results['VIF'] <= 10)],
            'acceptable': self.analyzer.vif_results[self.analyzer.vif_results['VIF'] <= 5]
        }
        
        return {
            'summary': {
                'total_variables': len(self.analyzer.vif_results),
                'high_multicollinearity_count': int(self.analyzer.vif_results['High_Multicollinearity'].sum()),
                'severe_vif_count': len(vif_categories['severe']),
                'high_vif_count': len(vif_categories['high']),
                'mean_vif': float(self.analyzer.vif_results['VIF'].replace([np.inf, -np.inf], np.nan).mean()),
                'max_vif': float(self.analyzer.vif_results['VIF'].replace([np.inf, -np.inf], np.nan).max())
            },
            'vif_categories': {
                'severe': vif_categories['severe'].to_dict('records'),
                'high': vif_categories['high'].to_dict('records'),
                'moderate': vif_categories['moderate'].to_dict('records'),
                'acceptable': vif_categories['acceptable'].to_dict('records')
            },
            'top_problematic_variables': self.analyzer.vif_results.head(15).to_dict('records')
        }
    
    def _format_variable_reduction(self) -> Dict[str, Any]:
        """Format variable reduction suggestions for dashboard display."""
        # Group removals by reason
        correlation_removals = set()
        vif_removals = set()
        
        for var in self.analyzer.suggested_removals:
            if var in [pair['recommended_remove'] for pair in self.analyzer.high_corr_pairs]:
                correlation_removals.add(var)
            if self.analyzer.vif_results is not None and var in self.analyzer.vif_results['Variable'].values:
                vif_val = self.analyzer.vif_results[self.analyzer.vif_results['Variable'] == var]['VIF'].iloc[0]
                if vif_val > self.analyzer.vif_threshold:
                    vif_removals.add(var)
        
        return {
            'summary': {
                'total_removals': len(self.analyzer.suggested_removals),
                'reduction_percentage': round((len(self.analyzer.suggested_removals) / len(self.analyzer.numeric_data.columns)) * 100, 1),
                'final_feature_count': len(self.analyzer.reduced_features),
                'original_feature_count': len(self.analyzer.numeric_data.columns)
            },
            'removal_breakdown': {
                'correlation_based': list(correlation_removals),
                'vif_based': list(vif_removals),
                'both_criteria': list(correlation_removals & vif_removals)
            },
            'suggested_removals': self.analyzer.suggested_removals,
            'reduced_feature_set': self.analyzer.reduced_features,
            'variable_importance_ranking': self._rank_variables_by_importance()
        }
    
    def _rank_variables_by_importance(self) -> List[Dict[str, Any]]:
        """Rank variables by their importance for keeping."""
        rankings = []
        
        for var in self.analyzer.numeric_data.columns:
            if var == self.analyzer.target_column:
                continue
                
            # Calculate importance score
            target_corr = abs(self.analyzer.correlation_matrix.loc[var, self.analyzer.target_column])
            
            # VIF score (lower is better)
            vif_score = 0
            if self.analyzer.vif_results is not None and var in self.analyzer.vif_results['Variable'].values:
                vif_val = self.analyzer.vif_results[self.analyzer.vif_results['Variable'] == var]['VIF'].iloc[0]
                vif_score = min(vif_val, 100)  # Cap at 100 for scoring
            
            # Combined importance score (higher is better)
            importance_score = target_corr * 100 - (vif_score / 10)
            
            rankings.append({
                'variable': var,
                'target_correlation': round(target_corr, 4),
                'vif': round(vif_score, 2),
                'importance_score': round(importance_score, 2),
                'recommended_action': 'remove' if var in self.analyzer.suggested_removals else 'keep'
            })
        
        # Sort by importance score (descending)
        rankings.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return rankings
    
    def _format_model_performance(self) -> Dict[str, Any]:
        """Format model performance analysis for dashboard display."""
        if not self.analyzer.model_results:
            return {'status': 'no_model_results'}
        
        # Prepare performance comparison
        performance_comparison = []
        
        for model_name, results in self.analyzer.model_results.items():
            if 'error' not in results:
                performance_comparison.append({
                    'model': model_name,
                    'train_r2': round(results.get('train_r2', 0), 4),
                    'test_r2': round(results.get('test_r2', 0), 4),
                    'r2_difference': round(results.get('r2_difference', 0), 4),
                    'train_mae': round(results.get('train_mae', 0), 2),
                    'test_mae': round(results.get('test_mae', 0), 2),
                    'mae_ratio': round(results.get('mae_ratio', 1), 2),
                    'feature_count': results.get('feature_count', 0),
                    'performance_grade': self._grade_model_performance(results)
                })
        
        # Identify best and worst models
        best_model = max(performance_comparison, key=lambda x: x['test_r2']) if performance_comparison else None
        
        return {
            'summary': {
                'total_models_tested': len(performance_comparison),
                'best_model': best_model['model'] if best_model else None,
                'best_test_r2': best_model['test_r2'] if best_model else 0,
                'average_test_r2': round(np.mean([m['test_r2'] for m in performance_comparison]), 4) if performance_comparison else 0,
                'features_used': performance_comparison[0]['feature_count'] if performance_comparison else 0
            },
            'model_comparison': performance_comparison,
            'performance_insights': self._generate_performance_insights(performance_comparison)
        }
    
    def _grade_model_performance(self, results: Dict) -> str:
        """Grade model performance for dashboard display."""
        test_r2 = results.get('test_r2', 0)
        r2_diff = results.get('r2_difference', 0)
        mae_ratio = results.get('mae_ratio', 1)
        
        # Grade based on test performance and overfitting indicators
        if test_r2 > 0.8 and r2_diff < 0.1 and mae_ratio < 1.2:
            return 'excellent'
        elif test_r2 > 0.6 and r2_diff < 0.15 and mae_ratio < 1.3:
            return 'good'
        elif test_r2 > 0.3 and r2_diff < 0.2 and mae_ratio < 1.5:
            return 'fair'
        elif test_r2 > 0.1:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_performance_insights(self, performance_comparison: List[Dict]) -> List[str]:
        """Generate performance insights for dashboard display."""
        insights = []
        
        if not performance_comparison:
            return ['No model performance data available.']
        
        best_r2 = max(m['test_r2'] for m in performance_comparison)
        avg_r2_diff = np.mean([m['r2_difference'] for m in performance_comparison])
        
        if best_r2 < 0.1:
            insights.append("Very low predictive power detected. Consider feature engineering or data collection.")
        elif best_r2 < 0.3:
            insights.append("Low predictive power. Multicollinearity reduction may help but additional features needed.")
        elif best_r2 > 0.7:
            insights.append("Good predictive power achieved with reduced feature set.")
        
        if avg_r2_diff > 0.2:
            insights.append("High overfitting risk across models. Consider regularization or cross-validation.")
        elif avg_r2_diff > 0.1:
            insights.append("Moderate overfitting detected. Monitor model performance on new data.")
        
        return insights
    
    def _format_overfitting_analysis(self) -> Dict[str, Any]:
        """Format overfitting analysis for dashboard display."""
        if not self.analyzer.overfitting_alerts:
            return {
                'status': 'no_overfitting_detected',
                'overall_risk': 'low',
                'summary': 'No significant overfitting detected across tested models.'
            }
        
        # Categorize alerts by severity
        high_severity_alerts = []
        medium_severity_alerts = []
        
        for alert in self.analyzer.overfitting_alerts:
            for a in alert['alerts']:
                alert_data = {
                    'model': alert['model'],
                    'type': a['type'],
                    'message': a['message'],
                    'severity': a['severity'],
                    'train_r2': alert.get('train_r2', 0),
                    'test_r2': alert.get('test_r2', 0),
                    'r2_difference': alert.get('r2_difference', 0)
                }
                
                if a['severity'] == 'high':
                    high_severity_alerts.append(alert_data)
                else:
                    medium_severity_alerts.append(alert_data)
        
        return {
            'status': 'overfitting_detected',
            'overall_risk': self.analyzer._assess_overall_overfitting_risk(),
            'summary': {
                'models_with_alerts': len(self.analyzer.overfitting_alerts),
                'high_severity_count': len(high_severity_alerts),
                'medium_severity_count': len(medium_severity_alerts),
                'total_alerts': len(high_severity_alerts) + len(medium_severity_alerts)
            },
            'alerts': {
                'high_severity': high_severity_alerts,
                'medium_severity': medium_severity_alerts
            },
            'mitigation_strategies': self._generate_overfitting_mitigation_strategies()
        }
    
    def _generate_overfitting_mitigation_strategies(self) -> List[Dict[str, str]]:
        """Generate overfitting mitigation strategies for dashboard display."""
        strategies = [
            {
                'strategy': 'Cross-Validation',
                'description': 'Implement k-fold cross-validation to better assess model generalization.',
                'implementation': 'Use sklearn.model_selection.cross_val_score with cv=5 or cv=10.'
            },
            {
                'strategy': 'Regularization',
                'description': 'Add L1/L2 penalties to prevent overfitting in linear models.',
                'implementation': 'Use Ridge (L2) or Lasso (L1) regression instead of plain LinearRegression.'
            },
            {
                'strategy': 'Feature Selection',
                'description': 'Use the reduced feature set suggested by multicollinearity analysis.',
                'implementation': 'Apply the recommended variable removals to all models.'
            },
            {
                'strategy': 'Early Stopping',
                'description': 'For tree-based models, implement early stopping based on validation performance.',
                'implementation': 'Use validation sets and monitor performance during training.'
            },
            {
                'strategy': 'Ensemble Methods',
                'description': 'Combine multiple simple models to reduce overfitting risk.',
                'implementation': 'Use voting classifiers or stacking with regularized base models.'
            }
        ]
        
        return strategies
    
    def _generate_auto_suggestions(self) -> Dict[str, Any]:
        """Generate comprehensive auto-suggestions for dashboard display."""
        suggestions = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_strategies': [],
            'dashboard_model_updates': {}
        }
        
        # Immediate actions
        if self.analyzer.suggested_removals:
            suggestions['immediate_actions'].append({
                'action': 'Remove Highly Correlated Variables',
                'description': f'Remove {len(self.analyzer.suggested_removals)} variables with high multicollinearity',
                'variables': self.analyzer.suggested_removals[:10],  # Show top 10
                'priority': 'high',
                'expected_impact': 'Improved model stability and interpretability'
            })
        
        if self.analyzer.overfitting_alerts:
            suggestions['immediate_actions'].append({
                'action': 'Address Overfitting',
                'description': f'Implement regularization for {len(self.analyzer.overfitting_alerts)} models showing overfitting',
                'priority': 'high',
                'expected_impact': 'Better generalization to new data'
            })
        
        # Short-term improvements
        suggestions['short_term_improvements'].extend([
            {
                'improvement': 'Feature Engineering',
                'description': 'Create composite advertising indices instead of individual metrics',
                'implementation': 'Combine GRP, OTS, and Budget into single advertising intensity measures',
                'timeline': '1-2 weeks'
            },
            {
                'improvement': 'Cross-Validation',
                'description': 'Implement robust model validation',
                'implementation': 'Add k-fold cross-validation to all model training processes',
                'timeline': '1 week'
            }
        ])
        
        # Long-term strategies
        suggestions['long_term_strategies'].extend([
            {
                'strategy': 'Data Collection Enhancement',
                'description': 'Collect qualitatively different variables to reduce multicollinearity',
                'details': 'Add creative quality scores, targeting precision, channel mix data',
                'timeline': '1-3 months'
            },
            {
                'strategy': 'Advanced Modeling',
                'description': 'Implement ensemble methods with regularization',
                'details': 'Develop stacked models with cross-validated base learners',
                'timeline': '2-4 weeks'
            }
        ])
        
        # Dashboard model updates
        suggestions['dashboard_model_updates'] = {
            'MLR': {
                'recommended_features': [f for f in self.analyzer.reduced_features if f != self.analyzer.target_column],
                'suggested_changes': 'Use only non-correlated features, add regularization'
            },
            'Distributed_Lag': {
                'recommended_features': [f for f in self.analyzer.reduced_features if f != self.analyzer.target_column],
                'suggested_changes': 'Focus on temporal features, add lagged target variables'
            },
            'ML_SHAP': {
                'recommended_features': [f for f in self.analyzer.reduced_features if f != self.analyzer.target_column],
                'suggested_changes': 'Start with reduced set, gradually add engineered features'
            },
            'CausalImpact': {
                'recommended_features': [f for f in self.analyzer.reduced_features if f != self.analyzer.target_column],
                'suggested_changes': 'Use as pre-regressors, ensure temporal ordering'
            }
        }
        
        return suggestions
    
    def _generate_dashboard_alerts(self) -> List[Dict[str, Any]]:
        """Generate dashboard alerts for immediate attention."""
        alerts = []
        
        # Critical multicollinearity alert
        severity = self._assess_multicollinearity_severity()
        if severity in ['critical', 'high']:
            alerts.append({
                'type': 'multicollinearity',
                'severity': severity,
                'title': 'Severe Multicollinearity Detected',
                'message': f'Found {len(self.analyzer.high_corr_pairs)} high correlation pairs requiring immediate attention',
                'action_required': True,
                'suggested_action': 'Apply variable reduction recommendations'
            })
        
        # Model performance alert
        if self.analyzer.model_results:
            best_r2 = max([r.get('test_r2', 0) for r in self.analyzer.model_results.values() if 'error' not in r])
            if best_r2 < 0.1:
                alerts.append({
                    'type': 'poor_performance',
                    'severity': 'high',
                    'title': 'Low Predictive Power',
                    'message': f'Best model achieves only {best_r2:.3f} R² - consider feature engineering',
                    'action_required': True,
                    'suggested_action': 'Review data quality and add meaningful features'
                })
        
        # Overfitting alert
        if self.analyzer.overfitting_alerts:
            high_risk_models = [a for a in self.analyzer.overfitting_alerts 
                              if any(alert['severity'] == 'high' for alert in a['alerts'])]
            if high_risk_models:
                alerts.append({
                    'type': 'overfitting',
                    'severity': 'medium',
                    'title': 'Overfitting Risk Detected',
                    'message': f'{len(high_risk_models)} models show high overfitting risk',
                    'action_required': True,
                    'suggested_action': 'Implement regularization and cross-validation'
                })
        
        return alerts
    
    def _generate_action_items(self) -> List[Dict[str, Any]]:
        """Generate prioritized action items for dashboard display."""
        action_items = []
        
        # Priority 1: Critical multicollinearity
        if len(self.analyzer.suggested_removals) > 0:
            action_items.append({
                'priority': 1,
                'category': 'Data Quality',
                'title': 'Remove Highly Correlated Variables',
                'description': f'Remove {len(self.analyzer.suggested_removals)} variables to eliminate multicollinearity',
                'estimated_time': '30 minutes',
                'expected_impact': 'High - Improved model stability and interpretability',
                'implementation_notes': 'Apply suggested removals to all dashboard models'
            })
        
        # Priority 2: Model performance
        if self.analyzer.model_results:
            action_items.append({
                'priority': 2,
                'category': 'Model Improvement',
                'title': 'Implement Cross-Validation',
                'description': 'Add robust validation to all models to prevent overfitting',
                'estimated_time': '2-3 hours',
                'expected_impact': 'Medium - Better model generalization',
                'implementation_notes': 'Use stratified k-fold cross-validation'
            })
        
        # Priority 3: Feature engineering
        action_items.append({
            'priority': 3,
            'category': 'Feature Engineering',
            'title': 'Create Composite Advertising Metrics',
            'description': 'Combine correlated advertising variables into meaningful indices',
            'estimated_time': '4-6 hours',
            'expected_impact': 'Medium - Reduced multicollinearity with preserved information',
            'implementation_notes': 'Create total_advertising_intensity and competitive_pressure_index'
        })
        
        return action_items
    
    def _generate_improvement_plan(self) -> Dict[str, Any]:
        """Generate comprehensive model improvement plan."""
        current_issues = []
        if len(self.analyzer.suggested_removals) > 0:
            current_issues.append('High multicollinearity between advertising variables')
        if self.analyzer.overfitting_alerts:
            current_issues.append('Overfitting detected in multiple models')
        if self.analyzer.model_results:
            best_r2 = max([r.get('test_r2', 0) for r in self.analyzer.model_results.values() if 'error' not in r])
            if best_r2 < 0.3:
                current_issues.append('Low predictive power across all models')
        
        return {
            'current_issues': current_issues,
            'improvement_phases': [
                {
                    'phase': 1,
                    'title': 'Immediate Fixes',
                    'duration': '1-2 days',
                    'actions': [
                        'Apply variable reduction recommendations',
                        'Update dashboard model configurations',
                        'Implement basic cross-validation'
                    ],
                    'expected_outcomes': [
                        'Eliminated multicollinearity',
                        'More stable model predictions',
                        'Improved model interpretability'
                    ]
                },
                {
                    'phase': 2,
                    'title': 'Model Enhancement',
                    'duration': '1-2 weeks',
                    'actions': [
                        'Implement regularization techniques',
                        'Create composite advertising features',
                        'Add ensemble methods'
                    ],
                    'expected_outcomes': [
                        'Reduced overfitting risk',
                        'Better feature representation',
                        'Improved prediction accuracy'
                    ]
                },
                {
                    'phase': 3,
                    'title': 'Advanced Improvements',
                    'duration': '1-3 months',
                    'actions': [
                        'Collect additional non-correlated features',
                        'Implement advanced validation strategies',
                        'Develop automated model monitoring'
                    ],
                    'expected_outcomes': [
                        'Higher predictive power',
                        'Robust model performance',
                        'Continuous model improvement'
                    ]
                }
            ],
            'success_metrics': [
                'Reduction in VIF values below 10',
                'Elimination of correlation pairs > 0.8',
                'R² difference between train/test < 0.1',
                'Test R² improvement > 20%'
            ]
        }
    
    def _prepare_export_data(self) -> Dict[str, Any]:
        """Prepare data for export/download from dashboard."""
        return {
            'reduced_dataset_preview': self.analyzer.data[self.analyzer.reduced_features].head(10).to_dict('records'),
            'correlation_matrix_data': self.analyzer.correlation_matrix.round(3).to_dict(),
            'vif_results_data': self.analyzer.vif_results.to_dict('records') if self.analyzer.vif_results is not None else [],
            'model_performance_data': self.analyzer.model_results,
            'suggested_removals_list': self.analyzer.suggested_removals,
            'export_files': {
                'reduced_dataset_csv': 'reduced_dataset.csv',
                'correlation_heatmap_png': 'correlation_heatmap.png',
                'comprehensive_report_json': 'model_improvement_report.json',
                'interactive_heatmap_html': 'correlation_heatmap_interactive.html'
            }
        }
    
    def _prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare visualization data for dashboard rendering."""
        return {
            'correlation_heatmap': self._create_interactive_correlation_heatmap(),
            'vif_bar_chart': self._create_vif_bar_chart(),
            'model_performance_chart': self._create_model_performance_chart(),
            'reduction_impact_chart': self._create_reduction_impact_chart(),
            'correlation_distribution': self._create_correlation_distribution()
        }
    
    def _generate_dashboard_visualizations(self) -> Dict[str, Any]:
        """Generate all visualizations for dashboard display."""
        visualizations = {}
        
        # Interactive correlation heatmap
        visualizations['correlation_heatmap'] = self._create_interactive_correlation_heatmap()
        
        # VIF analysis chart
        visualizations['vif_analysis'] = self._create_vif_bar_chart()
        
        # Model performance comparison
        visualizations['model_performance'] = self._create_model_performance_chart()
        
        # Variable reduction impact
        visualizations['reduction_impact'] = self._create_reduction_impact_chart()
        
        # Correlation distribution
        visualizations['correlation_distribution'] = self._create_correlation_distribution()
        
        return visualizations
    
    def _create_interactive_correlation_heatmap(self) -> str:
        """Create interactive correlation heatmap for dashboard."""
        if self.analyzer.correlation_matrix is None:
            return ""
        
        # Create interactive heatmap with Plotly
        fig = px.imshow(
            self.analyzer.correlation_matrix,
            labels=dict(x="Variables", y="Variables", color="Correlation"),
            title="Interactive Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            width=800,
            height=700,
            title_x=0.5,
            font=dict(size=10),
            xaxis=dict(side="bottom"),
            yaxis=dict(side="left")
        )
        
        # Convert to JSON for dashboard embedding
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def _create_vif_bar_chart(self) -> str:
        """Create VIF analysis bar chart for dashboard."""
        if self.analyzer.vif_results is None or self.analyzer.vif_results.empty:
            return ""
        
        # Limit to top 20 variables and cap VIF at 50 for visualization
        vif_display = self.analyzer.vif_results.head(20).copy()
        vif_display['VIF_capped'] = vif_display['VIF'].apply(lambda x: min(x, 50) if x != np.inf else 50)
        
        fig = px.bar(
            vif_display,
            x='Variable',
            y='VIF_capped',
            color='High_Multicollinearity',
            title='Variance Inflation Factor (VIF) Analysis',
            labels={'VIF_capped': 'VIF (capped at 50)', 'Variable': 'Variables'},
            color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'}
        )
        
        # Add threshold line
        fig.add_hline(y=10, line_dash="dash", line_color="red", 
                     annotation_text="VIF Threshold (10)")
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def _create_model_performance_chart(self) -> str:
        """Create model performance comparison chart for dashboard."""
        if not self.analyzer.model_results:
            return ""
        
        # Prepare data for visualization
        models = []
        train_r2 = []
        test_r2 = []
        
        for model_name, results in self.analyzer.model_results.items():
            if 'error' not in results:
                models.append(model_name)
                train_r2.append(results.get('train_r2', 0))
                test_r2.append(results.get('test_r2', 0))
        
        # Create grouped bar chart
        fig = go.Figure(data=[
            go.Bar(name='Training R²', x=models, y=train_r2, marker_color='lightblue'),
            go.Bar(name='Test R²', x=models, y=test_r2, marker_color='darkblue')
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='R² Score',
            barmode='group',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def _create_reduction_impact_chart(self) -> str:
        """Create variable reduction impact visualization for dashboard."""
        # Create pie chart showing variable reduction
        labels = ['Variables to Keep', 'Variables to Remove']
        values = [len(self.analyzer.reduced_features), len(self.analyzer.suggested_removals)]
        colors = ['#2ECC71', '#E74C3C']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title='Variable Reduction Impact',
            height=400,
            showlegend=True
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def _create_correlation_distribution(self) -> str:
        """Create correlation distribution histogram for dashboard."""
        if self.analyzer.correlation_matrix is None:
            return ""
        
        # Get upper triangle correlations (excluding diagonal)
        mask = np.triu(np.ones_like(self.analyzer.correlation_matrix, dtype=bool), k=1)
        correlations = self.analyzer.correlation_matrix.values[mask]
        
        fig = px.histogram(
            x=correlations,
            nbins=50,
            title='Distribution of Correlation Coefficients',
            labels={'x': 'Correlation Coefficient', 'y': 'Frequency'}
        )
        
        # Add threshold lines
        fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                     annotation_text="High Correlation Threshold")
        fig.add_vline(x=-0.8, line_dash="dash", line_color="red")
        
        fig.update_layout(height=400)
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
    
    def export_comprehensive_json(self, filepath: str = 'comprehensive_dashboard_report.json') -> None:
        """Export comprehensive analysis results for dashboard consumption."""
        print(f"Exporting comprehensive dashboard report to {filepath}...")
        
        # Convert numpy types for JSON serialization
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
        
        # Prepare final export data
        export_data = convert_numpy_types(self.dashboard_data)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f" Comprehensive dashboard report exported successfully!")
        
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get concise summary for dashboard header display."""
        if not self.dashboard_data:
            return {}
        
        return {
            'severity': self._assess_multicollinearity_severity(),
            'variables_to_remove': len(self.analyzer.suggested_removals) if self.analyzer else 0,
            'reduction_percentage': round((len(self.analyzer.suggested_removals) / len(self.analyzer.numeric_data.columns)) * 100, 1) if self.analyzer else 0,
            'best_model_r2': max([r.get('test_r2', 0) for r in self.analyzer.model_results.values() if 'error' not in r]) if self.analyzer and self.analyzer.model_results else 0,
            'overfitting_risk': self.analyzer._assess_overall_overfitting_risk() if self.analyzer else 'unknown',
            'total_alerts': len(self.dashboard_data.get('dashboard_alerts', [])),
            'analysis_timestamp': self.dashboard_data.get('analysis_summary', {}).get('timestamp', '')
        } 