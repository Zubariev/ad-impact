#!/usr/bin/env python3
"""
Script to combine all JSON model reports into a single comprehensive report.
Ensures that common sections like 'dataset_overview' and 'comprehensive_dataset_analysis' 
appear only once, while preserving model-specific sections.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_json_report(file_path: Path) -> Dict[str, Any]:
    """Load a JSON report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def combine_reports() -> Dict[str, Any]:
    """Combine all JSON reports into a single comprehensive report."""
    
    reports_dir = Path("reports")
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
            "report_version": "1.0"
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
    output_file = Path("reports/combined_comprehensive_report.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 50)
        print(f"✅ Successfully combined all reports!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Total models: {combined_report['combined_report_metadata']['total_models']}")
        print(f"🔗 Models included: {', '.join(combined_report['combined_report_metadata']['combined_from'])}")
        
        # Print file size info
        file_size = output_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"📏 Output file size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error saving combined report: {e}")


if __name__ == "__main__":
    main() 