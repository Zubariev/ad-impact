"""
Entry point for the Ad Impact Modeling Dashboard.
Run this file with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main() 