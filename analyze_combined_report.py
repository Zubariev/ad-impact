#!/usr/bin/env python3
"""
Utility script to analyze the structure and content of the combined comprehensive report.
"""

import json
from pathlib import Path
from typing import Dict, Any


def analyze_combined_report():
    """Analyze the combined report structure and provide summary."""
    
    report_file = Path("reports/combined_comprehensive_report.json")
    
    if not report_file.exists():
        print("Combined report not found. Run combine_reports.py first.")
        return
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error loading combined report: {e}")
        return
    
    print("📊 COMBINED REPORT ANALYSIS")
    print("=" * 50)
    
    # Metadata analysis
    if "combined_report_metadata" in report:
        metadata = report["combined_report_metadata"]
        print(f"📅 Generated: {metadata.get('generated_at', 'Unknown')}")
        print(f"🔢 Total Models: {metadata.get('total_models', 0)}")
        print(f"📋 Models: {', '.join(metadata.get('combined_from', []))}")
        print()
    
    # Common sections analysis
    print("🔗 COMMON SECTIONS (shared across all models):")
    common_sections = ["dataset_overview", "comprehensive_dataset_analysis"]
    for section in common_sections:
        if section in report:
            print(f"  ✅ {section}")
            if section == "dataset_overview":
                overview = report[section]
                print(f"     📊 Observations: {overview.get('total_observations', 'N/A')}")
                print(f"     📊 Columns: {overview.get('columns_count', 'N/A')}")
        else:
            print(f" {section} - Missing")
    print()
    
    # Models analysis
    if "models" in report:
        models = report["models"]
        print(f"🤖 MODEL-SPECIFIC SECTIONS ({len(models)} models):")
        
        for model_name, model_data in models.items():
            print(f"\n  📈 {model_name}:")
            
            sections = list(model_data.keys())
            for section in sections:
                print(f"     ✅ {section}")
                
                # Special analysis for key sections
                if section == "model_configuration":
                    config = model_data[section]
                    target = config.get('target_variable', 'N/A')
                    features_count = len(config.get('feature_variables', []))
                    print(f"        🎯 Target: {target}")
                    print(f"        🔧 Features: {features_count}")
                
                elif section == "model_diagnostics":
                    diag = model_data[section]
                    if isinstance(diag, dict):
                        r2 = diag.get('r_squared', diag.get('R²', 'N/A'))
                        print(f"        📈 R²: {r2}")
                
                elif section == "model_predictions":
                    pred = model_data[section]
                    if isinstance(pred, dict) and 'predictions' in pred:
                        pred_count = len(pred['predictions'])
                        print(f"        🔮 Predictions: {pred_count}")
    
    # File size info
    file_size = report_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"\n📏 File Size: {file_size_mb:.2f} MB")
    
    # JSON structure depth analysis
    def get_max_depth(obj, depth=0):
        if isinstance(obj, dict):
            return max([get_max_depth(v, depth + 1) for v in obj.values()], default=depth)
        elif isinstance(obj, list) and obj:
            return max([get_max_depth(item, depth + 1) for item in obj], default=depth)
        return depth
    
    max_depth = get_max_depth(report)
    print(f"🏗️  JSON Structure Depth: {max_depth} levels")
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")


if __name__ == "__main__":
    analyze_combined_report() 