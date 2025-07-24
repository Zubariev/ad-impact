#!/usr/bin/env python3
"""
Script to combine all JSON model reports into a single comprehensive report.
Ensures that common sections like 'dataset_overview' and 'comprehensive_dataset_analysis' 
appear only once, while preserving model-specific sections.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union


def load_json_report(file_path: Path) -> Dict[str, Any]:
    """Load a JSON report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def load_json_from_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Load JSON data from uploaded file."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error loading JSON file {uploaded_file.name}: {e}")
        return {}


def combine_uploaded_files(json_files: List = None, prediction_files: List = None) -> Dict[str, Any]:
    """
    Combine uploaded JSON reports and prediction files into a single comprehensive report.
    
    Args:
        json_files: List of uploaded JSON files (file objects)
        prediction_files: List of uploaded CSV prediction files (file objects)
        
    Returns:
        Combined report dictionary
    """
    
    if json_files is None:
        json_files = []
    if prediction_files is None:
        prediction_files = []
    
    # Load all JSON reports
    reports = {}
    for json_file in json_files:
        try:
            json_file.seek(0)  # Reset file pointer
            report_data = load_json_from_uploaded_file(json_file)
            if report_data:
                # Extract model name from metadata or filename
                model_name = report_data.get('report_metadata', {}).get('model_name', 
                                           json_file.name.replace('.json', '').replace('_comprehensive_report', ''))
                # Handle multicollinearity analysis files
                if 'multicollinearity_analysis' in json_file.name.lower():
                    model_name = report_data.get('analysis_summary', {}).get('model_metadata', {}).get('model_type', 
                                               json_file.name.split('_')[2] if len(json_file.name.split('_')) > 2 else 'Unknown')
                reports[model_name] = report_data
                print(f"✓ Loaded {model_name} from {json_file.name}")
        except Exception as e:
            print(f"Error processing JSON file {json_file.name}: {e}")
    
    # Load prediction files
    predictions_data = {}
    for pred_file in prediction_files:
        try:
            pred_file.seek(0)  # Reset file pointer
            df = pd.read_csv(pred_file)
            # Extract model name from filename
            model_name = pred_file.name.replace('_predictions.csv', '').replace('.csv', '')
            predictions_data[model_name] = {
                "predictions_data": df.to_dict('records'),
                "total_predictions": len(df),
                "file_name": pred_file.name,
                "prediction_summary": {
                    "mean": float(df.iloc[:, -1].mean()) if len(df.columns) > 0 else 0,
                    "std": float(df.iloc[:, -1].std()) if len(df.columns) > 0 else 0,
                    "min": float(df.iloc[:, -1].min()) if len(df.columns) > 0 else 0,
                    "max": float(df.iloc[:, -1].max()) if len(df.columns) > 0 else 0,
                }
            }
            print(f"✓ Loaded predictions for {model_name} ({len(df)} predictions)")
        except Exception as e:
            print(f"Error processing prediction file {pred_file.name}: {e}")
    
    if not reports and not predictions_data:
        print("No valid data loaded")
        return {}
    
    # Start building the combined report
    combined_report = {
        "combined_report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "combined_from": list(reports.keys()),
            "prediction_files": list(predictions_data.keys()),
            "total_models": len(reports),
            "total_prediction_sets": len(predictions_data),
            "report_version": "1.0",
            "combination_method": "uploaded_files"
        }
    }
    
    # Use the first report as the source for common sections
    if reports:
        first_report = next(iter(reports.values()))
        
        # Add common sections (these should be identical across all reports)
        if "dataset_overview" in first_report:
            combined_report["dataset_overview"] = first_report["dataset_overview"]
            print("✓ Added dataset_overview (common section)")
        
        if "comprehensive_dataset_analysis" in first_report:
            combined_report["comprehensive_dataset_analysis"] = first_report["comprehensive_dataset_analysis"]
            print("✓ Added comprehensive_dataset_analysis (common section)")
    
    # Add model-specific sections
    combined_report["models"] = {}
    
    # Define sections that are model-specific (everything except the common ones)
    common_sections = {"dataset_overview", "comprehensive_dataset_analysis"}
    
    for model_name, report_data in reports.items():
        print(f"\nProcessing model: {model_name}")
        
        model_sections = {}
        
        # Copy all sections except the common ones
        for section_name, section_data in report_data.items():
            if section_name not in common_sections:
                model_sections[section_name] = section_data
                print(f"  ✓ Added {section_name}")
        
        combined_report["models"][model_name] = model_sections
    
    # Add prediction data
    if predictions_data:
        combined_report["predictions"] = predictions_data
        print(f"\n✓ Added prediction data for {len(predictions_data)} models")
    
    return combined_report


def combine_reports() -> Dict[str, Any]:
    """Combine all JSON reports from reports directory into a single comprehensive report."""
    
    # Handle path relative to current location (src/ or root)
    reports_dir = Path("reports")
    if not reports_dir.exists():
        reports_dir = Path("../reports")  # Look in parent directory if not found
    json_files = list(reports_dir.glob("*_comprehensive_report.json"))
    
    if not json_files:
        print("No JSON report files found in reports directory")
        return {}
    
    print(f"Found {len(json_files)} report files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Load all reports
    reports = {}
    for json_file in json_files:
        report_data = load_json_report(json_file)
        if report_data:
            # Extract model name from metadata or filename
            model_name = report_data.get('report_metadata', {}).get('model_name', 
                                       json_file.stem.replace('_comprehensive_report', ''))
            reports[model_name] = report_data
    
    if not reports:
        print("No valid reports loaded")
        return {}
    
    # Start building the combined report
    combined_report = {
        "combined_report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "combined_from": list(reports.keys()),
            "total_models": len(reports),
            "report_version": "1.0",
            "combination_method": "directory_scan"
        }
    }
    
    # Use the first report as the source for common sections
    first_report = next(iter(reports.values()))
    
    # Add common sections (these should be identical across all reports)
    if "dataset_overview" in first_report:
        combined_report["dataset_overview"] = first_report["dataset_overview"]
        print("✓ Added dataset_overview (common section)")
    
    if "comprehensive_dataset_analysis" in first_report:
        combined_report["comprehensive_dataset_analysis"] = first_report["comprehensive_dataset_analysis"]
        print("✓ Added comprehensive_dataset_analysis (common section)")
    
    # Add model-specific sections
    combined_report["models"] = {}
    
    # Define sections that are model-specific (everything except the common ones)
    common_sections = {"dataset_overview", "comprehensive_dataset_analysis"}
    
    for model_name, report_data in reports.items():
        print(f"\nProcessing model: {model_name}")
        
        model_sections = {}
        
        # Copy all sections except the common ones
        for section_name, section_data in report_data.items():
            if section_name not in common_sections:
                model_sections[section_name] = section_data
                print(f"  ✓ Added {section_name}")
        
        combined_report["models"][model_name] = model_sections
    
    return combined_report


def main():
    """Main function to combine reports and save the result."""
    print("Starting report combination process...")
    print("=" * 50)
    
    # Combine all reports
    combined_report = combine_reports()
    
    if not combined_report:
        print("Failed to combine reports")
        return
    
    # Save the combined report
    # Handle path relative to current location (src/ or root)
    output_file = Path("reports/combined_comprehensive_report.json")
    if not output_file.parent.exists():
        output_file = Path("../reports/combined_comprehensive_report.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 50)
        print(f" Successfully combined all reports!")
        print(f" Output file: {output_file}")
        print(f" Total models: {combined_report['combined_report_metadata']['total_models']}")
        print(f" Models included: {', '.join(combined_report['combined_report_metadata']['combined_from'])}")
        
        # Print file size info
        file_size = output_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"📏 Output file size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error saving combined report: {e}")


if __name__ == "__main__":
    main() 