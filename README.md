# Ad Impact Modeling Dashboard

A comprehensive Streamlit dashboard for analyzing advertising impact on supermarket foot traffic using multiple econometric and machine learning models.

## Features

- **9 Analysis Tools**: 8 different models + Report Combination functionality
- **Interactive Data Upload**: Support for CSV, Excel files with automatic column detection
- **Advanced Data Preparation**: Automated creation of 'treated' and 'post' columns for causal inference models
- **Smart Range Selection**: Automatic detection of datetime, numeric, and categorical columns
- **Missing Data Handling**: Multiple strategies for handling missing values
- **Comprehensive Multicollinearity Analysis**: VIF analysis, correlation matrices, overfitting detection, and variable reduction suggestions
- **Comprehensive Diagnostics**: Model-specific metrics, VIF analysis, residual plots, SHAP explanations
- **JSON Reports**: Enterprise-grade analysis reports with dataset overview, model diagnostics, and results
- **Report Combination**: Upload and combine multiple analysis reports and predictions into unified JSON reports
- **Interpretation Hints**: User-friendly guidance for understanding results
- **Download Capabilities**: Export trained models, predictions, and comprehensive JSON reports

## Project Structure

```
econ/
├── app.py                          # Main entry point (imports from src/app.py)
├── data_prep.py                    # Data preprocessing utilities
├── analyze_combined_report.py      # Standalone report analysis script
├── multicollinearity_analysis_MLR_20250724_1540.json  # Example analysis file
├── src/                            # Core application modules
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit application with 9 tabs
│   ├── config.py                   # Configuration and constants
│   ├── data_utils.py               # Data processing utilities
│   ├── model_training.py           # Model training functions
│   ├── visualization.py            # Charts and diagnostic displays
│   ├── data_analysis.py            # Advanced data analysis functions
│   ├── combine_reports.py          # Report combination functionality
│   ├── dashboard_integration.py    # Dashboard integration utilities
│   ├── multicollinearity_analysis.py      # VIF and correlation analysis
│   ├── multicollinearity_streamlit.py     # Streamlit multicollinearity UI
│   └── debug_multicollinearity.py # Debugging utilities
├── econ/                           # Legacy dashboard (alternative interface)
│   ├── ad_impact_dashboard.py      # Standalone dashboard version
│   └── requirements.txt
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_model_training.py
│   └── test_visualization.py
├── saved_models/                   # Trained model storage
│   ├── MLR.pkl
│   ├── ML + SHAP.pkl
│   ├── VAR.pkl
│   ├── CausalImpact.pkl
│   └── Distributed Lag.pkl
├── saved_predictions/              # Prediction results storage
│   ├── MLR_predictions.csv
│   ├── ML + SHAP_predictions.csv
│   ├── VAR_predictions.csv
│   ├── CausalImpact_predictions.csv
│   └── Distributed Lag_predictions.csv
├── reports/                        # Generated comprehensive reports
│   ├── combined_comprehensive_report.json
│   ├── MLR_comprehensive_report.json
│   ├── ML + SHAP_comprehensive_report.json
│   ├── VAR_comprehensive_report.json
│   ├── CausalImpact_comprehensive_report.json
│   └── Distributed Lag_comprehensive_report.json
├── requirements.txt                # Python dependencies
├── .flake8                        # Linting configuration
├── pyproject.toml                 # Black formatting configuration
├── mypy.ini                       # Type checking configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd econ
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Main Dashboard Workflow

1. **Upload Data**: Drag and drop CSV or Excel files containing your supermarket foot traffic data
2. **Data Preparation**: Use the advanced data preparation section to create 'treated' and 'post' columns for causal inference models
3. **Select Columns**: Choose your target variable and feature variables
4. **Filter Data**: Use the smart range selector to filter observations
5. **Handle Missing Data**: Choose how to handle any missing values
6. **Train Model**: Click "Apply & Train Model" to run the selected model
7. **Review Results**: Examine the interactive charts, diagnostics, and multicollinearity analysis
8. **Download Results**: Save the trained model, predictions, and comprehensive JSON report

### Combine Reports Workflow

1. **Navigate to "📊 Combine Reports" Tab**: The 9th tab in the dashboard
2. **Upload Analysis Reports**: Upload JSON files (multicollinearity analysis, model reports)
3. **Upload Prediction Files**: Upload CSV files from your saved_predictions folder
4. **Combine Reports**: Click to merge all files into a unified comprehensive report
5. **Download Combined JSON**: Get a timestamped combined report file

## Model Descriptions

### MLR (Multiple Linear Regression)
- **Use Case**: Basic attribution modeling
- **Features**: 1-50 variables
- **Output**: Coefficient estimates, R², VIF analysis, residual plots

### Distributed Lag
- **Use Case**: Modeling delayed advertising effects
- **Features**: 1-20 variables
- **Output**: Lag coefficients, Durbin-Watson test for autocorrelation

### ML + SHAP
- **Use Case**: Complex non-linear relationships
- **Features**: 1-500 variables
- **Output**: SHAP explanations, feature importance, RMSE/MAE

### DiD (Difference-in-Differences)
- **Use Case**: Natural experiments, policy evaluation
- **Features**: 1-10 variables
- **Requirements**: 'treated' and 'post' indicator columns
- **Output**: Average Treatment Effect, confidence intervals

### VAR (Vector Autoregression)
- **Use Case**: Time series analysis, impulse response functions
- **Features**: 1-6 variables
- **Output**: IRF plots, stability checks, AIC/BIC

### Synthetic Control
- **Use Case**: Single treatment unit analysis
- **Features**: 1-10 variables
- **Requirements**: 'treated' indicator column
- **Output**: Actual vs synthetic comparison, RMSPE

### CausalImpact
- **Use Case**: Bayesian structural time series
- **Features**: 1-20 variables
- **Output**: Cumulative effects, counterfactual predictions

### PSM (Propensity Score Matching)
- **Use Case**: Observational studies, treatment effect estimation
- **Features**: 1-20 variables
- **Requirements**: 'treated' indicator column
- **Output**: ATT estimates, balance diagnostics

## Advanced Features

### Enhanced Multicollinearity Analysis
- **Comprehensive VIF Analysis**: Variance inflation factors with severity classification
- **Advanced Correlation Analysis**: Interactive heatmaps, correlation distribution analysis
- **Overfitting Detection**: Training vs. test performance gap analysis, severity alerts
- **Variable Reduction Recommendations**: Smart feature selection suggestions
- **Model Performance Comparison**: Baseline model benchmarking (Linear Regression, XGBoost, Random Forest)
- **Dashboard Integration**: Actionable alerts and model-specific recommendations
- **Export Capabilities**: JSON reports with visualizations and improvement suggestions

### Data Preparation
- **Automated 'treated' column creation**: Based on location/unit identifiers
- **Automated 'post' column creation**: Based on date cutoffs or numeric periods
- **Complete DiD dataset preparation**: Creates both treatment and time indicators
- **Synthetic Control preparation**: Designates one treated unit vs. controls

### Report Combination System
- **Multi-File Upload**: Support for JSON reports and CSV prediction files
- **Smart File Recognition**: Automatic model name extraction from filenames and metadata
- **Common Section Deduplication**: Dataset overview appears only once across combined reports
- **Model-Specific Preservation**: Individual analysis sections maintained per model
- **Prediction Data Integration**: Full prediction datasets with summary statistics
- **Timestamped Downloads**: Organized file naming with generation timestamps

### Comprehensive JSON Reports
- **Dataset Overview**: Observations, date coverage, missing data analysis, outlier detection
- **Detailed Column Analysis**: Statistics for numeric, categorical, and datetime columns
- **Model Diagnostics**: R², coefficients, p-values, confidence intervals, cross-validation results
- **Advanced VIF Analysis**: Multicollinearity metrics with removal recommendations
- **Overfitting Analysis**: Performance gaps, risk assessment, mitigation strategies
- **Residual Diagnostics**: Model validation metrics, assumption testing
- **Channel Contributions**: Individual feature impact analysis with statistical significance
- **Full Predictions Dataset**: Complete model output with timestamps and confidence intervals
- **Executive Summary**: Priority actions, severity levels, performance benchmarks

## Alternative Interfaces

### Legacy Dashboard
Located in `econ/ad_impact_dashboard.py` - a standalone version with similar functionality.

### Standalone Scripts
- **combine_reports.py**: Can be run independently to combine reports from the reports/ directory
- **analyze_combined_report.py**: Analyzes existing combined reports
- **data_prep.py**: Standalone data preprocessing utilities

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality Checks
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure Benefits

- **Modularity**: Each component has a single responsibility
- **Testability**: Comprehensive unit tests for all modules
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new models or features
- **Configuration**: Centralized settings in `src/config.py`
- **Report System**: Enterprise-grade analysis and combination capabilities

## Configuration

Key settings can be modified in `src/config.py`:

- Model hyperparameters
- Feature limits per model
- File upload settings
- Statistical significance thresholds
- UI behavior options
- Multicollinearity analysis thresholds

## Dependencies

- **Core**: streamlit, pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, shap
- **Statistics**: statsmodels, pmdarima
- **Visualization**: plotly, matplotlib
- **File Processing**: openpyxl, xlrd
- **Model Persistence**: joblib
- **Analysis**: VIF calculation, correlation analysis

## File Naming Conventions

### Model Files
- **Saved Models**: `{ModelName}.pkl` (e.g., `MLR.pkl`, `ML + SHAP.pkl`)
- **Predictions**: `{ModelName}_predictions.csv`
- **Reports**: `{ModelName}_comprehensive_report.json`

### Analysis Files
- **Multicollinearity**: `multicollinearity_analysis_{ModelName}_{YYYYMMDD}_{HHMM}.json`
- **Combined Reports**: `combined_comprehensive_report_{YYYYMMDD}_{HHMMSS}.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please [create an issue](link-to-issues) in the repository. 