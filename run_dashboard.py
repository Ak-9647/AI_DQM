#!/usr/bin/env python3
"""
Quick start script for AI-Powered Data Quality Monitor Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import duckdb
        import sklearn
        import yaml
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/profiles",
        "data/anomalies"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure verified")

def main():
    print("🤖 AI-Powered Data Quality Monitor")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if sample data exists
    sample_data = "data/sample/customer_data.csv"
    if os.path.exists(sample_data):
        print(f"✅ Sample data available at {sample_data}")
    else:
        print(f"⚠️  Sample data not found at {sample_data}")
        print("   You can still upload your own CSV files through the dashboard")
    
    # Run Streamlit dashboard
    print("\n🚀 Starting dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 