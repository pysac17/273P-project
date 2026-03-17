#!/usr/bin/env python3
"""
Setup script for Image Matching Challenge 2025 Pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version OK: {sys.version}")

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found!")
        sys.exit(1)
    
    print("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def check_sample_data():
    """Check if sample data exists."""
    sample_dir = Path("sample_data/ETs_sample")
    if sample_dir.exists():
        images = list(sample_dir.glob("*.png"))
        print(f"Sample data found: {len(images)} images")
        return True
    else:
        print("Warning: Sample data not found in sample_data/ETs_sample/")
        return False

def setup_environment():
    """Set up environment variables for Mac stability."""
    if sys.platform == "darwin":  # macOS
        print("Setting up Mac-specific environment...")
        env_vars = {
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        }
        for var, value in env_vars.items():
            os.environ[var] = value
        print("Environment variables set for Mac stability.")

def main():
    """Main setup function."""
    print("=== Image Matching Challenge 2025 Setup ===\n")
    
    # Check Python version
    check_python_version()
    print()
    
    # Install requirements
    install_requirements()
    print()
    
    # Check sample data
    has_sample = check_sample_data()
    print()
    
    # Set up environment
    setup_environment()
    print()
    
    # Instructions
    print("=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Run: jupyter notebook ml-project.ipynb")
    print("2. Execute cells sequentially")
    if has_sample:
        print("3. Sample data will be used for testing")
    else:
        print("3. Download full dataset from Kaggle for complete testing")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
