"""
Streamlit Integration for Multicollinearity Analysis
Integrates seamlessly with your existing dashboard in src/app.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile
import os

# Import our analysis modules
try:
    from multicollinearity_analysis import MulticollinearityAnalyzer
    from dashboard_integration import DashboardAnalysisIntegrator
except ImportError:
    # Fallback if modules aren't found
    st.error("Multicollinearity analysis modules not found. Please ensure all analysis files are present.")
    st.stop()


def add_multicollinearity_analysis(
    df: pd.DataFrame,
    target_column: str, 
    feature_columns: List[str],
    model_name: str,
    model_results: Optional[Dict] = None
) -> None:
    """
    Add multicollinearity analysis to existing model training workflow.
    
    This function integrates directly with your existing train_model_safely function.
    
    Args:
        df: The DataFrame used for training
        target_column: Name of the target variable
        feature_columns: List of feature columns used in training
        model_name: Name of the trained model (e.g., "MLR", "XGBoost", etc.)
        model_results: Optional dict containing model performance metrics
    """
    
    # Debug information
    st.write("🔍 **Multicollinearity Analysis Debug Info:**")
    st.write(f"- Model: {model_name}")
    st.write(f"- Dataset shape: {df.shape}")
    st.write(f"- Target: {target_column}")
    st.write(f"- Features: {len(feature_columns)} ({feature_columns[:5]}...)")
    
    # Create temporary data file for analysis
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Include target and features in the analysis dataset
            analysis_columns = [target_column] + feature_columns
            analysis_df = df[analysis_columns].copy()
            analysis_df.to_csv(temp_file.name, index=False)
            temp_data_path = temp_file.name
        
        st.success(f" Temporary analysis file created: {len(analysis_df)} rows, {len(analysis_columns)} columns")
        
    except Exception as e:
        st.error(f" Failed to create analysis dataset: {e}")
        return
    
    # Create analysis section
    st.markdown("---")
    st.header("🔍 Multicollinearity Analysis")
    st.caption(f"Automatic analysis triggered after {model_name} training completion")
    
    # Analysis toggle
    if st.checkbox(" Run Multicollinearity Analysis", value=True, key=f"analysis_toggle_{model_name}"):
        
        # Run analysis with progress indicator
        with st.spinner("Running comprehensive multicollinearity analysis..."):
            
            try:
                # Initialize integrator
                integrator = DashboardAnalysisIntegrator(
                    analysis_config={
                        'correlation_threshold': 0.8,
                        'vif_threshold': 10.0,
                        'test_size': 0.2,
                        'random_state': 42
                    }
                )
                
                # Prepare model metadata
                model_metadata = {
                    'model_type': model_name,
                    'training_time': datetime.now().isoformat(),
                    'features_used': feature_columns,
                    'model_results': model_results or {}
                }
                
                # Run analysis
                dashboard_data = integrator.run_post_training_analysis(
                    data_path=temp_data_path,
                    target_column=target_column,
                    model_metadata=model_metadata
                )
                
                # Export comprehensive report
                report_path = f"multicollinearity_analysis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                integrator.export_comprehensive_json(report_path)
                
                st.success(" Multicollinearity analysis completed successfully!")
                
                # Display results
                _display_analysis_results(dashboard_data, integrator, model_name, report_path)
                
            except Exception as e:
                st.error(f" Analysis failed: {str(e)}")
                st.write("**Debug Info:**")
                st.write(f"- Temp file exists: {os.path.exists(temp_data_path)}")
                st.write(f"- Target column in data: {target_column in df.columns}")
                st.write(f"- Feature columns in data: {all(col in df.columns for col in feature_columns)}")
                
        # Cleanup
        try:
            os.unlink(temp_data_path)
        except:
            pass


def _display_analysis_results(dashboard_data: Dict, integrator: DashboardAnalysisIntegrator, 
                            model_name: str, report_path: str) -> None:
    """Display the multicollinearity analysis results in Streamlit."""
    
    # Get summary data
    summary = integrator.get_dashboard_summary()
    insights = dashboard_data.get('multicollinearity_insights', {})
    
    # Display severity header
    st.markdown("###  Analysis Summary")
    
    # Severity indicator
    severity = summary.get('severity', 'unknown')
    severity_colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'unknown': '⚪'}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Severity Level",
            value=f"{severity_colors.get(severity, '⚪')} {severity.upper()}"
        )
    
    with col2:
        st.metric(
            label="High Correlations Found",
            value=insights.get('total_high_correlations', 0)
        )
    
    with col3:
        st.metric(
            label="Variables to Remove",
            value=insights.get('reduction_impact', {}).get('variables_to_remove', 0),
            delta=f"-{insights.get('reduction_impact', {}).get('reduction_percentage', 0):.1f}%"
        )
    
    with col4:
        st.metric(
            label="Optimized Features",
            value=insights.get('reduction_impact', {}).get('final_feature_count', 0)
        )
    
    # Create analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚠️ Alerts", "🔗 Correlations", "💡 Suggestions", "📁 Downloads"])
    
    with tab1:
        _display_alerts_tab(dashboard_data)
    
    with tab2:
        _display_correlations_tab(dashboard_data, integrator)
    
    with tab3:
        _display_suggestions_tab(dashboard_data, model_name)
    
    with tab4:
        _display_downloads_tab(integrator, report_path, model_name)


def _display_alerts_tab(dashboard_data: Dict) -> None:
    """Display critical alerts."""
    
    alerts = dashboard_data.get('dashboard_alerts', [])
    
    if not alerts:
        st.success(" No critical multicollinearity issues detected!")
        return
    
    st.markdown("### ⚠️ Critical Issues Detected")
    
    for i, alert in enumerate(alerts):
        severity_emoji = "🔴" if alert.get('severity') == 'critical' else "🟠"
        
        with st.expander(f"{severity_emoji} {alert.get('title')}", expanded=i == 0):
            st.write(f"**Message:** {alert.get('message')}")
            st.write(f"**Severity:** {alert.get('severity', 'unknown').upper()}")
            
            if alert.get('action_required'):
                st.info(f"**Recommended Action:** {alert.get('suggested_action')}")
                
                if st.button(f"Mark as Acknowledged", key=f"alert_ack_{i}"):
                    st.success(" Alert acknowledged!")


def _display_correlations_tab(dashboard_data: Dict, integrator: DashboardAnalysisIntegrator) -> None:
    """Display correlation analysis."""
    
    correlation_data = dashboard_data.get('correlation_analysis', {})
    
    if correlation_data.get('status') == 'no_high_correlations':
        st.success(" No problematic correlations detected!")
        return
    
    # Show correlation summary
    summary = correlation_data.get('summary', {})
    
    st.markdown("### 🔗 Correlation Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total High Correlations", summary.get('total_pairs', 0))
    with col2:
        st.metric("Perfect Correlations", summary.get('perfect_correlations', 0))
    with col3:
        st.metric("Very High (>0.95)", summary.get('very_high_correlations', 0))
    
    # Show interactive heatmap if available
    try:
        if hasattr(integrator, 'visualizations') and integrator.visualizations.get('correlation_heatmap'):
            st.markdown("###  Interactive Correlation Heatmap")
            heatmap_json = integrator.visualizations['correlation_heatmap']
            if heatmap_json:
                st.plotly_chart(
                    json.loads(heatmap_json), 
                    use_container_width=True,
                    config={'displayModeBar': True}
                )
    except Exception as e:
        st.warning(f"Could not display interactive heatmap: {e}")
    
    # Show top correlations table
    if correlation_data.get('top_correlations'):
        st.markdown("### 🔍 Top Problematic Correlations")
        
        top_corr_data = []
        for pair in correlation_data['top_correlations'][:15]:
            top_corr_data.append({
                'Variable 1': pair['variable_1'],
                'Variable 2': pair['variable_2'], 
                'Correlation': f"{pair['correlation']:.3f}",
                'Action': f"Keep: {pair['recommended_keep']}, Remove: {pair['recommended_remove']}"
            })
        
        st.dataframe(pd.DataFrame(top_corr_data), use_container_width=True)


def _display_suggestions_tab(dashboard_data: Dict, model_name: str) -> None:
    """Display auto-suggestions."""
    
    suggestions = dashboard_data.get('auto_suggestions', {})
    
    # Immediate actions
    immediate_actions = suggestions.get('immediate_actions', [])
    if immediate_actions:
        st.markdown("### 🚨 Immediate Actions Required")
        
        for i, action in enumerate(immediate_actions):
            with st.expander(f"🔥 Priority {i+1}: {action.get('action')}", expanded=i == 0):
                st.write(f"**Description:** {action.get('description')}")
                st.write(f"**Expected Impact:** {action.get('expected_impact')}")
                st.write(f"**Priority:** {action.get('priority', 'medium').upper()}")
                
                if action.get('variables'):
                    st.write("**Variables to Remove:**")
                    variables = action['variables'][:20]  # Show first 20
                    
                    # Display variables in a nice format
                    for j in range(0, len(variables), 4):
                        cols = st.columns(4)
                        for k, var in enumerate(variables[j:j+4]):
                            with cols[k]:
                                st.code(var, language=None)
                    
                    if len(action['variables']) > 20:
                        st.info(f"... and {len(action['variables']) - 20} more variables")
                
                if st.button(f" Apply {action.get('action')}", key=f"apply_action_{i}"):
                    st.success(f" {action.get('action')} marked for implementation!")
                    st.balloons()
    
    # Model-specific recommendations
    model_updates = suggestions.get('dashboard_model_updates', {})
    if model_updates:
        st.markdown("### 🔧 Model-Specific Recommendations")
        
        # Highlight current model
        if model_name in model_updates:
            current_model = model_updates[model_name]
            st.markdown(f"#### 🎯 Current Model ({model_name}) Recommendations")
            
            with st.container():
                st.info(f"**Suggested Changes:** {current_model.get('suggested_changes', 'No specific changes recommended')}")
                
                recommended_features = current_model.get('recommended_features', [])
                if recommended_features:
                    st.write(f"**Recommended Features ({len(recommended_features)}):**")
                    features_text = ", ".join(recommended_features)
                    st.code(features_text, language=None)
                    
                    # Show implementation code
                    st.write("**Implementation Example:**")
                    code_example = f"""
# Updated {model_name} Model Configuration
features = {recommended_features}
X = data[features]
y = data['{dashboard_data.get("analysis_summary", {}).get("dataset_info", {}).get("target_variable", "target")}']

# Apply to your {model_name} model training
model.fit(X, y)
"""
                    st.code(code_example, language='python')
        
        # Other model recommendations
        other_models = {k: v for k, v in model_updates.items() if k != model_name}
        if other_models:
            st.markdown("#### 📋 Other Model Recommendations")
            
            for model_type, updates in other_models.items():
                with st.expander(f" {model_type.replace('_', ' ')} Model"):
                    st.write(f"**Recommended Features:** {len(updates.get('recommended_features', []))}")
                    st.write(f"**Suggested Changes:** {updates.get('suggested_changes')}")


def _display_downloads_tab(integrator: DashboardAnalysisIntegrator, report_path: str, model_name: str) -> None:
    """Display download section."""
    
    st.markdown("### 📁 Download Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Analysis Reports")
        
        # Comprehensive JSON report
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                json_data = f.read()
            
            st.download_button(
                label="📋 Download Complete Analysis (JSON)",
                data=json_data,
                file_name=f"{model_name}_multicollinearity_analysis.json",
                mime="application/json",
                help="Download comprehensive analysis report with all findings and recommendations"
            )
        else:
            st.warning("Analysis report not available for download.")
    
    with col2:
        st.markdown("#### 🗂️ Optimized Data")
        
        # Optimized dataset
        if hasattr(integrator, 'analyzer') and hasattr(integrator.analyzer, 'reduced_features'):
            try:
                if integrator.analyzer.reduced_features and integrator.analyzer.data is not None:
                    reduced_data = integrator.analyzer.data[integrator.analyzer.reduced_features]
                    csv_data = reduced_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📁 Download Optimized Dataset (CSV)",
                        data=csv_data,
                        file_name=f"{model_name}_optimized_dataset.csv",
                        mime="text/csv",
                        help="Download dataset with only the recommended variables"
                    )
                    
                    # Show preview
                    st.write("**Optimized Dataset Preview:**")
                    st.dataframe(reduced_data.head(3), use_container_width=True)
                else:
                    st.info("No optimized dataset available.")
                    
            except Exception as e:
                st.error(f"Could not generate optimized dataset: {e}")
        
        # Variable removal list
        if hasattr(integrator, 'analyzer') and hasattr(integrator.analyzer, 'suggested_removals'):
            if integrator.analyzer.suggested_removals:
                removals_text = "\n".join(integrator.analyzer.suggested_removals)
                
                st.download_button(
                    label="🗑️ Variables to Remove (TXT)",
                    data=removals_text,
                    file_name=f"{model_name}_variables_to_remove.txt",
                    mime="text/plain",
                    help="List of variables recommended for removal"
                )
