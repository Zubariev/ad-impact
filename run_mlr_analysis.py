#!/usr/bin/env python3
"""
Run MLR analysis with advanced time series diagnostics.
This script provides a command-line interface to the MLR_VAR analysis tools.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Parse command-line arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MLR analysis with advanced diagnostics")
    
    parser.add_argument(
        "--file", "-f", 
        default="data/main.csv", 
        help="Path to input data file (CSV or Excel)"
    )
    
    parser.add_argument(
        "--date-col", "-d", 
        default="Y&W", 
        help="Name of date column"
    )
    
    parser.add_argument(
        "--target", "-t", 
        default="Visits_dynamics", 
        help="Name of target variable"
    )
    
    parser.add_argument(
        "--features", "-x", 
        default=None, 
        help="Comma-separated list of feature variables (if not provided, all numeric columns except target are used)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="mlr_var_analysis_results.json", 
        help="Path for output JSON file"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Import MLR_VAR module
        from MLR_VAR import load_data, preprocess_data, run_complete_analysis, save_results
        
        # Load data
        logger.info(f"Loading data from {args.file}")
        data = load_data(args.file)
        
        # Parse features
        if args.features:
            features = [f.strip() for f in args.features.split(",")]
        else:
            # Use all numeric columns except target as features
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            features = [col for col in numeric_cols if col != args.target]
            logger.info(f"Using {len(features)} numeric columns as features")
        
        # Preprocess data
        data = preprocess_data(data, args.date_col, args.target)
        
        # Run analysis
        logger.info(f"Running analysis for target: {args.target}")
        results = run_complete_analysis(data, args.target, features, args.date_col)
        
        # Save results
        save_results(results, args.output)
        
        logger.info(f"Analysis completed. Results saved to {args.output}")
        
        # Print short summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        
        # Get suggested model
        suggested_model = results.get('optimal_model', {}).get('suggested_model')
        if suggested_model:
            print(f"Suggested model: {suggested_model}")
        
        # Print model comparison
        model_comparison = results.get('optimal_model', {}).get('model_comparison', [])
        if model_comparison:
            print("\nModel Comparison:")
            print("-" * 50)
            print(f"{'Model':<20} {'R²':<10} {'Adj R²':<10} {'AIC':<10}")
            print("-" * 50)
            
            for model in model_comparison[:3]:  # Show top 3 models
                model_name = model.get('model_name', 'Unknown')
                r2 = model.get('r_squared', None)
                adj_r2 = model.get('adj_r_squared', None)
                aic = model.get('aic', None)
                
                r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
                adj_r2_str = f"{adj_r2:.3f}" if adj_r2 is not None else "N/A"
                aic_str = f"{aic:.1f}" if aic is not None else "N/A"
                
                print(f"{model_name:<20} {r2_str:<10} {adj_r2_str:<10} {aic_str:<10}")
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
