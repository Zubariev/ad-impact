# Ad Impact Modeling Dashboard

A comprehensive Streamlit dashboard for analyzing advertising impact on supermarket foot traffic using multiple econometric and machine learning models.

## Features

- **8 Different Models**: MLR, Distributed Lag, ML + SHAP, DiD, VAR, Synthetic Control, CausalImpact, PSM
- **Interactive Data Upload**: Support for CSV, Excel files with automatic column detection
- **Smart Range Selection**: Automatic detection of datetime, numeric, and categorical columns
- **Missing Data Handling**: Multiple strategies for handling missing values
- **Comprehensive Diagnostics**: Model-specific metrics, VIF analysis, residual plots, SHAP explanations
- **Interpretation Hints**: User-friendly guidance for understanding results
- **Download Capabilities**: Export trained models and predictions

## Project Structure

```
econ/
├── src/
│   ├── __init__.py
│   ├── app.py              # Main Streamlit application
│   ├── config.py           # Configuration and constants
│   ├── data_utils.py       # Data processing utilities
│   ├── model_training.py   # Model training functions
│   └── visualization.py    # Charts and diagnostic displays
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
2. **Select Columns**: Choose your target variable and feature variables
3. **Filter Data**: Use the smart range selector to filter observations
4. **Handle Missing Data**: Choose how to handle any missing values
5. **Train Model**: Click "Apply & Train Model" to run the selected model
6. **Review Results**: Examine the interactive charts and diagnostic metrics
7. **Download Results**: Save the trained model and predictions

## Model Descriptions

### MLR (Multiple Linear Regression)
- **Use Case**: Basic attribution modeling
- **Features**: 5-50 variables
- **Output**: Coefficient estimates, R², VIF analysis, residual plots

### Distributed Lag
- **Use Case**: Modeling delayed advertising effects
- **Features**: 5-20 variables
- **Output**: Lag coefficients, Durbin-Watson test for autocorrelation

### ML + SHAP
- **Use Case**: Complex non-linear relationships
- **Features**: 10-500 variables
- **Output**: SHAP explanations, feature importance, RMSE/MAE

### DiD (Difference-in-Differences)
- **Use Case**: Natural experiments, policy evaluation
- **Features**: 1-10 variables
- **Requirements**: 'treated' and 'post' indicator columns
- **Output**: Average Treatment Effect, confidence intervals

### VAR (Vector Autoregression)
- **Use Case**: Time series analysis, impulse response functions
- **Features**: 2-6 variables
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