# Ad Impact Modeling Dashboard

A comprehensive Streamlit dashboard for analyzing advertising impact on supermarket foot traffic using multiple econometric and machine learning models.

## Features

- **8 Different Models**: MLR, Distributed Lag, ML + SHAP, DiD, VAR, Synthetic Control, CausalImpact, PSM
- **Interactive Data Upload**: Support for CSV, Excel files with automatic column detection
- **Advanced Data Preparation**: Automated creation of 'treated' and 'post' columns for causal inference models
- **Smart Range Selection**: Automatic detection of datetime, numeric, and categorical columns
- **Missing Data Handling**: Multiple strategies for handling missing values
- **Multicollinearity Analysis**: VIF analysis, correlation matrices, and overfitting detection
- **Comprehensive Diagnostics**: Model-specific metrics, VIF analysis, residual plots, SHAP explanations
- **JSON Reports**: Comprehensive analysis reports with dataset overview, model diagnostics, and results
- **Interpretation Hints**: User-friendly guidance for understanding results
- **Download Capabilities**: Export trained models, predictions, and comprehensive JSON reports

## Project Structure

```
econ/
├── app.py                  # Entry point (imports from src/app.py)
├── data_prep.py           # Data preprocessing utilities
├── src/
│   ├── __init__.py
│   ├── app.py              # Main Streamlit application
│   ├── config.py           # Configuration and constants
│   ├── data_utils.py       # Data processing utilities
│   ├── model_training.py   # Model training functions
│   ├── visualization.py    # Charts and diagnostic displays
│   ├── data_analysis.py    # Advanced data analysis functions
│   ├── dashboard_integration.py # Dashboard integration utilities
│   ├── multicollinearity_analysis.py # VIF and correlation analysis
│   ├── multicollinearity_streamlit.py # Streamlit multicollinearity UI
│   └── debug_multicollinearity.py # Debugging utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_model_training.py
│   └── test_visualization.py
├── saved_models/           # Trained model storage
├── saved_predictions/      # Prediction results storage
├── requirements.txt        # Python dependencies
├── .flake8                # Linting configuration
├── pyproject.toml         # Black formatting configuration
├── mypy.ini              # Type checking configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
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

1. **Upload Data**: Drag and drop CSV or Excel files containing your supermarket foot traffic data
2. **Data Preparation**: Use the advanced data preparation section to create 'treated' and 'post' columns for causal inference models
3. **Select Columns**: Choose your target variable and feature variables
4. **Filter Data**: Use the smart range selector to filter observations
5. **Handle Missing Data**: Choose how to handle any missing values
6. **Train Model**: Click "Apply & Train Model" to run the selected model
7. **Review Results**: Examine the interactive charts, diagnostics, and multicollinearity analysis
8. **Download Results**: Save the trained model, predictions, and comprehensive JSON report

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

### Data Preparation
- **Automated 'treated' column creation**: Based on location/unit identifiers
- **Automated 'post' column creation**: Based on date cutoffs or numeric periods
- **Complete DiD dataset preparation**: Creates both treatment and time indicators
- **Synthetic Control preparation**: Designates one treated unit vs. controls

### Multicollinearity Analysis
- **VIF (Variance Inflation Factor)**: Detects multicollinearity issues
- **Correlation matrices**: Visualizes feature relationships
- **Overfitting detection**: Identifies potential model reliability issues
- **Feature reduction suggestions**: Recommendations for improving model stability

### Comprehensive JSON Reports
- **Dataset Overview**: Observations, date coverage, missing data analysis
- **Detailed Column Analysis**: Statistics for numeric, categorical, and datetime columns
- **Model Diagnostics**: R², coefficients, p-values, confidence intervals
- **VIF Analysis**: Multicollinearity metrics for each feature
- **Residual Diagnostics**: Model validation metrics
- **Channel Contributions**: Individual feature impact analysis
- **Full Predictions Dataset**: Complete model output with timestamps

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
- **Configuration**: Centralized settings in `config.py`

## Configuration

Key settings can be modified in `src/config.py`:

- Model hyperparameters
- Feature limits per model
- File upload settings
- Statistical significance thresholds
- UI behavior options

## Dependencies

- **Core**: streamlit, pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, shap
- **Statistics**: statsmodels, pmdarima
- **Visualization**: plotly, matplotlib
- **File Processing**: openpyxl, xlrd
- **Model Persistence**: joblib

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