"""
Background Processes Manager for Ad Impact Modeling Dashboard.

Manages concurrent background processes for model diagnostics and advanced analysis.
Provides real-time progress tracking and UI integration.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
import json
import os
from pathlib import Path
import uuid

import pandas as pd
import numpy as np
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry for background processes (now using st.session_state for persistence)
def _init_process_states():
    if 'background_processes_data' not in st.session_state:
        st.session_state['background_processes_data'] = {
            'processes': {},
            'results': {}
        }

def get_processes_state() -> Dict:
    _init_process_states()
    return st.session_state['background_processes_data']


class BackgroundProcess:
    """
    Class for managing a background process with progress tracking.
    """
    
    def __init__(self, name: str, target_func: Callable, args: tuple = None, kwargs: Dict = None):
        """
        Initialize a background process.
        
        Args:
            name: Process name
            target_func: Target function to run
            args: Positional arguments for target function
            kwargs: Keyword arguments for target function
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.target_func = target_func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0.0
        self.message = f"Process {name} initialized"
        self.result = None
        self.error = None
        self.logs = []
        
        # For UI integration
        self.ui_key = f"process_{self.id}"
    
    def start(self):
        """Start the background process in a separate thread."""
        if self.status == "running":
            logger.warning(f"Process '{self.name}' is already running")
            return False
        
        self.start_time = datetime.now()
        self.status = "running"
        self.progress = 0.0
        self.message = f"Started {self.name}"
        self.add_log(f"Process started at {self.start_time}")
        
        # Create wrapper function that captures results/errors
        def wrapper():
            try:
                # Call the target function with progress callback
                kwargs = self.kwargs.copy()
                # Only inject progress_callback if it's not explicitly set to something else
                if kwargs.get('progress_callback', 1) is None: # Check if it was explicitly set to None
                    kwargs['progress_callback'] = self.update_progress
                elif 'progress_callback' not in kwargs: # If not present at all, inject it
                    kwargs['progress_callback'] = self.update_progress

                self.result = self.target_func(*self.args, **kwargs)
                self.status = "completed"
                self.progress = 1.0
                self.message = f"Completed {self.name}"
            except Exception as e:
                self.error = str(e)
                self.status = "failed"
                self.message = f"Failed: {str(e)}"
                logger.error(f"Process '{self.name}' failed: {str(e)}")
            finally:
                self.end_time = datetime.now()
                self.add_log(f"Process ended at {self.end_time}")
                get_processes_state()['results'][self.id] = self.get_result()
        
        # Start thread
        self.thread = threading.Thread(target=wrapper)
        self.thread.daemon = True
        self.thread.start()
        
        # Register process
        get_processes_state()['processes'][self.id] = self
        
        return True
    
    def update_progress(self, progress: float, message: str = None):
        """
        Update progress status.
        
        Args:
            progress: Progress value between 0 and 1
            message: Progress message
        """
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message
            self.add_log(message)
    
    def add_log(self, message: str):
        """
        Add log message with timestamp.
        
        Args:
            message: Log message
        """
        timestamp = datetime.now()
        self.logs.append({
            'timestamp': timestamp,
            'message': message
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status information."""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'has_error': self.error is not None,
            'error': self.error
        }
    
    def get_result(self) -> Dict[str, Any]:
        """Get process result."""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'logs': self.logs,
            'duration': (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
    
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.status == "running" and (self.thread and self.thread.is_alive())
    
    def is_complete(self) -> bool:
        """Check if process has completed."""
        return self.status == "completed"
    
    def has_failed(self) -> bool:
        """Check if process has failed."""
        return self.status == "failed"


def run_background_process(name: str, target_func: Callable, 
                          args: tuple = None, kwargs: Dict = None) -> str:
    """
    Run a function in a background thread and return its ID.
    
    Args:
        name: Process name
        target_func: Target function to run
        args: Positional arguments for target function
        kwargs: Keyword arguments for target function
    
    Returns:
        Process ID
    """
    process = BackgroundProcess(name, target_func, args, kwargs)
    process.start()
    return process.id


def get_process_status(process_id: str) -> Dict[str, Any]:
    """
    Get status of a background process.
    
    Args:
        process_id: Process ID
    
    Returns:
        Process status dictionary
    """
    if process_id not in get_processes_state()['processes']:
        return {'error': 'Process not found', 'status': 'unknown'}
    
    return get_processes_state()['processes'][process_id].get_status()


def get_process_result(process_id: str) -> Dict[str, Any]:
    """
    Get result of a completed background process.
    
    Args:
        process_id: Process ID
    
    Returns:
        Process result dictionary
    """
    if process_id in get_processes_state()['results']:
        return get_processes_state()['results'][process_id]
    
    if process_id not in get_processes_state()['processes']:
        return {'error': 'Process not found', 'status': 'unknown'}
    
    if get_processes_state()['processes'][process_id].status != "completed":
        return {'error': 'Process not completed', 'status': get_processes_state()['processes'][process_id].status}
    
    return get_processes_state()['processes'][process_id].get_result()


def get_active_processes() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all active background processes.
    
    Returns:
        Dictionary of process statuses
    """
    active_processes = {}
    for process_id, process in get_processes_state()['processes'].items():
        if process.is_running():
            active_processes[process_id] = process.get_status()
    
    return active_processes


def display_process_ui(process_id: str, 
                   title: str = "Background Process Status", 
                   on_complete_callback: Callable = None, 
                   result_key: str = None, 
                   refresh_interval: float = 0.5):
    """
    Display a UI element for tracking process progress.

    Args:
        process_id: Process ID
        title: Title to display for the process UI
        on_complete_callback: Function to call when process completes successfully
        result_key: Key to store the result in Streamlit session state
        refresh_interval: How often to refresh the UI (in seconds)
    """
    if process_id not in get_processes_state()['processes']:
        st.error(f"Process {process_id} not found")
        return

    process = get_processes_state()['processes'][process_id]

    # Create a placeholder
    placeholder = st.empty()

    # Update status periodically
    with placeholder.container():
        # Display process information
        st.markdown(f"##### {title}") # Use the provided title
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Check status until complete or failed
        while process.is_running():
            progress_bar.progress(process.progress)
            status_text.text(process.message)
            time.sleep(refresh_interval) # Use the provided refresh interval

        # Final update
        if process.is_complete():
            progress_bar.progress(1.0)
            status_text.success(f"✅ {process.name} completed successfully")
            if on_complete_callback and process.result is not None:
                # Store result in session state before calling callback
                if result_key:
                    st.session_state[result_key] = process.result
                on_complete_callback(process.result)
        elif process.has_failed():
            status_text.error(f"❌ {process.name} failed: {process.error}")

        # Display logs if requested
        if st.checkbox(f"Show logs for {process.name}", key=f"{process.ui_key}_logs"):
            logs_text = "\n".join([
                f"{log['timestamp'].strftime('%H:%M:%S')} - {log['message']}"
                for log in process.logs
            ])
            st.text_area("Process logs", logs_text, height=150)

        # Display results if completed and no callback was used or if callback didn't display all
        if process.is_complete():
            if st.checkbox(f"Show raw results for {process.name}", key=f"{process.ui_key}_raw_results"):
                st.json(process.get_result()) # Display raw result


# Example background process functions
def stationarity_analysis_process(data: pd.DataFrame, target: str, features: List[str], 
                               progress_callback: Callable = None) -> Dict[str, Any]:
    """
    Background process function for stationarity analysis.
    
    Args:
        data: DataFrame with target and features
        target: Target column name
        features: Feature column names
        progress_callback: Progress callback function
    
    Returns:
        Dictionary with stationarity analysis results
    """
    from src.time_series_utils import check_stationarity
    
    if progress_callback:
        progress_callback(0.1, "Starting stationarity analysis")
    
    results = {}
    series_to_check = [target] + features
    
    for i, col in enumerate(series_to_check):
        if progress_callback:
            progress = (i + 1) / len(series_to_check)
            progress_callback(progress * 0.8 + 0.1, f"Checking stationarity of {col}")
        
        # Check with ADF
        adf_result = check_stationarity(data[col], method='adf')
        
        # Check with KPSS
        kpss_result = check_stationarity(data[col], method='kpss')
        
        # Combined interpretation
        is_stationary = adf_result['is_stationary'] and kpss_result['is_stationary']
        
        results[col] = {
            'adf_test': adf_result,
            'kpss_test': kpss_result,
            'is_stationary': is_stationary,
            'suggested_transformation': None if is_stationary else "differencing"
        }
    
    if progress_callback:
        progress_callback(1.0, "Stationarity analysis completed")
    
    return {
        'stationarity_results': results,
        'stationary_series': sum(1 for r in results.values() if r['is_stationary']),
        'non_stationary_series': sum(1 for r in results.values() if not r['is_stationary']),
        'summary': {
            'stationary_series': [col for col, r in results.items() if r['is_stationary']],
            'non_stationary_series': [col for col, r in results.items() if not r['is_stationary']]
        }
    }


def multicollinearity_analysis_process(data: pd.DataFrame, features: List[str], 
                                    progress_callback: Callable = None) -> Dict[str, Any]:
    """
    Background process function for multicollinearity analysis.
    
    Args:
        data: DataFrame with features
        features: Feature column names
        progress_callback: Progress callback function
    
    Returns:
        Dictionary with multicollinearity analysis results
    """
    from src.time_series_utils import multicollinearity_check_with_vif
    
    if progress_callback:
        progress_callback(0.1, "Starting multicollinearity analysis")
    
    # Check correlation matrix
    if progress_callback:
        progress_callback(0.2, "Calculating correlation matrix")
    
    corr_matrix = data[features].corr()
    
    # Find high correlation pairs
    if progress_callback:
        progress_callback(0.3, "Finding high correlation pairs")
    
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.7:  # Threshold for high correlation
                high_corr_pairs.append({
                    'var1': features[i],
                    'var2': features[j],
                    'correlation': float(corr)
                })
    
    # Check VIF
    if progress_callback:
        progress_callback(0.6, "Calculating VIF values")
    
    vif_result = multicollinearity_check_with_vif(data[features], include_detailed=True)
    
    # Final assessment
    if progress_callback:
        progress_callback(0.9, "Finalizing analysis")
    
    has_multicollinearity = (len(high_corr_pairs) > 0 or 
                          len(vif_result.get('problematic_variables', [])) > 0)
    
    result = {
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlation_pairs': high_corr_pairs,
        'vif_analysis': vif_result,
        'has_multicollinearity': has_multicollinearity,
        'problematic_variables': vif_result.get('problematic_variables', []),
        'suggested_actions': vif_result.get('suggested_actions', [])
    }
    
    if progress_callback:
        progress_callback(1.0, "Multicollinearity analysis completed")
    
    return result


def autocorrelation_analysis_process(data: pd.DataFrame, target: str, features: List[str], 
                                 progress_callback: Callable = None) -> Dict[str, Any]:
    """
    Background process function for autocorrelation analysis.
    
    Args:
        data: DataFrame with target and features
        target: Target column name
        features: Feature column names
        progress_callback: Progress callback function
    
    Returns:
        Dictionary with autocorrelation analysis results
    """
    from src.time_series_utils import check_autocorrelation
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    
    if progress_callback:
        progress_callback(0.1, "Starting autocorrelation analysis")
    
    # Fit base model
    if progress_callback:
        progress_callback(0.3, "Fitting base OLS model")
    
    try:
        # Prepare data
        model_data = data[[target] + features].dropna()
        
        # Add constant
        X = sm.add_constant(model_data[features])
        y = model_data[target]
        
        # Fit OLS model
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        
        # Check autocorrelation
        if progress_callback:
            progress_callback(0.6, "Analyzing residuals for autocorrelation")
        
        autocorr_result = check_autocorrelation(model.resid)
        
        # Get model diagnostics
        if progress_callback:
            progress_callback(0.8, "Calculating model diagnostics")
        
        result = {
            'autocorrelation': autocorr_result,
            'model_diagnostics': {
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'aic': float(model.aic),
                'bic': float(model.bic),
                'f_statistic': float(model.fvalue),
                'durbin_watson': float(durbin_watson(model.resid))
            },
            'has_autocorrelation': (
                autocorr_result['positive_autocorr'] or 
                autocorr_result['negative_autocorr']
            ),
            'suggested_action': autocorr_result['suggested_action']
        }
        
    except Exception as e:
        result = {
            'error': str(e),
            'has_autocorrelation': None
        }
    
    if progress_callback:
        progress_callback(1.0, "Autocorrelation analysis completed")
    
    return result


def optimal_model_selection_process(data: pd.DataFrame, 
                                 target: str, 
                                 features: List[str], 
                                 progress_callback: Callable = None,
                                 date_col: str = None, 
                                 use_decomposition: bool = False, 
                                 decomposition_method: str = 'STL', 
                                 decomposition_period: int = None) -> Dict[str, Any]:
    """
    Background process function for optimal model selection.

    Args:
        data: DataFrame with target and features
        target: Target column name
        features: Feature column names
        progress_callback: Progress callback function
        date_col: Date column name (needed for decomposition)
        use_decomposition: Whether to apply seasonal decomposition
        decomposition_method: Method for seasonal decomposition
        decomposition_period: Seasonal period for decomposition

    Returns:
        Dictionary with optimal model selection results
    """
    from src.time_series_utils import fit_optimal_model

    logger.info("optimal_model_selection_process: Starting optimal model selection process.")

    if progress_callback:
        progress_callback(0.1, "Starting model selection process")

    # Run full model selection analysis
    if progress_callback:
        progress_callback(0.3, "Analyzing data characteristics")
    
    try:
        # Fit optimal model
        if progress_callback:
            progress_callback(0.5, "Fitting candidate models")
        
        logger.info(f"optimal_model_selection_process: Calling fit_optimal_model with target={target}, features={features}, date_col={date_col}, use_decomposition={use_decomposition}, method={decomposition_method}, period={decomposition_period}")

        result = fit_optimal_model(
            data,
            target_col=target,
            feature_cols=features,
            time_col=date_col,
            handle_autocorr=True,
            handle_multicoll=True,
            use_decomposition=use_decomposition,
            decomposition_method=decomposition_method,
            decomposition_period=decomposition_period
        )

        if progress_callback:
            progress_callback(0.9, "Finalizing model selection")
        
        logger.info("optimal_model_selection_process: Optimal model selection completed successfully.")

        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()

        if progress_callback:
            progress_callback(1.0, "Model selection completed")
        
        return result

    except Exception as e:
        error_message = f"optimal_model_selection_process: Error during optimal model selection: {str(e)}"
        logger.error(error_message)
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        raise # Re-raise to ensure the BackgroundProcess catches it


# UI component for displaying background processes
def render_background_process_ui():
    """Render UI component for background processes."""
    # Get active processes
    active_processes = get_active_processes()
    
    if not active_processes:
        st.info("No background processes currently running.")
        return
    
    st.subheader("Background Processes")
    
    for process_id, process_status in active_processes.items():
        with st.expander(f"{process_status['name']} - {process_status['status']}"):
            st.progress(process_status['progress'])
            st.text(process_status['message'])
            
            # Show logs if available
            if process_id in get_processes_state()['processes']:
                process = get_processes_state()['processes'][process_id]
                if process.logs:
                    logs_text = "\n".join([
                        f"{log['timestamp'].strftime('%H:%M:%S')} - {log['message']}"
                        for log in process.logs[-5:]  # Show last 5 logs
                    ])
                    st.text_area("Recent logs", logs_text, height=100)
