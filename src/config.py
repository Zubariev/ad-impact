"""
Configuration settings for the Ad Impact Modeling Dashboard.
Centralizes all constants, model parameters, and file paths.
"""

import os
from typing import Dict, List, Tuple

# Directory paths
MODEL_DIR = "saved_models"
PREDICTIONS_DIR = "saved_predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Model configurations
MODEL_TABLE = [
    {
        "name": "MLR",
        "chart": "Stacked Area Chart",
        "variables_allowed": (1, 50),
    },
    {
        "name": "Distributed Lag",
        "chart": "Stacked Area Chart (with lags)",
        "variables_allowed": (1, 20),
    },
    {
        "name": "ML + SHAP",
        "chart": "Stacked Area Chart (from SHAP)",
        "variables_allowed": (1, 500),
    },
    {
        "name": "DiD",
        "chart": "Slope Graph / Bar Chart with Error Bars",
        "variables_allowed": (1, 10),
    },
    {
        "name": "VAR",
        "chart": "IRF Line Plot / Cumulative Bar Chart",
        "variables_allowed": (1, 6),
    },
    {
        "name": "Synthetic Control",
        "chart": "Line Chart (Actual vs Synthetic)",
        "variables_allowed": (1, 10),
    },
    {
        "name": "CausalImpact",
        "chart": "Line Chart with CI / Cumulative Bar Chart",
        "variables_allowed": (1, 20),
    },
    {
        "name": "PSM",
        "chart": "Bar Chart (Matched Groups) / Density Plot",
        "variables_allowed": (1, 20),
    },
]

MODEL_CONFIGS = {m["name"]: m for m in MODEL_TABLE}

# Model hyperparameters
MODEL_HYPERPARAMS = {
    "ML + SHAP": {
        "n_estimators": 200,
        "learning_rate": 0.05,
    },
    "Distributed Lag": {
        "max_lag": 7,
    },
    "VAR": {
        "maxlags": 5,
        "ic": "aic",
    },
}

# File upload settings
SUPPORTED_FILE_TYPES = [".csv", ".xls", ".xlsx"]
MAX_FILE_SIZE_MB = 100

# Data processing settings
MISSING_DATA_METHODS = [
    "Remove rows with missing data (Listwise deletion)",
    "Fill with zeros",
    "Fill with column mean",
    "Fill with column median",
    "Forward fill (carry last observation forward)",
    "Backward fill (carry next observation backward)",
    "Linear interpolation",
    "Fill with mode (most frequent value)",
]

# UI settings
DATE_KEYWORDS = ['date', 'time', 'timestamp', 'created', 'updated', 'day', 'month', 'year']
MIN_OBSERVATIONS_FOR_TRAINING = 5
OUTLIER_THRESHOLD = 0.99  # 99th percentile

# Statistical significance thresholds
SIGNIFICANCE_LEVEL = 0.05
VIF_THRESHOLD = 5.0
DURBIN_WATSON_LOWER = 1.5
DURBIN_WATSON_UPPER = 2.5 