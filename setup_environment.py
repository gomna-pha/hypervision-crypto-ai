#!/usr/bin/env python3
"""
Complete Environment Setup Script for Hyperbolic CNN Trading System
This script installs all required packages and verifies the installation
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("="*80)
    print("HYPERBOLIC CNN TRADING SYSTEM - COMPLETE ENVIRONMENT SETUP")
    print("="*80)
    
    # Core packages that must be installed
    essential_packages = [
        # Core
        "numpy<2.0.0",  # Ensure compatibility
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        
        # PyTorch (CPU version for compatibility)
        "torch",
        "torchvision",
        
        # Machine Learning
        "scikit-learn>=1.3.0",
        "imbalanced-learn>=0.11.0",
        
        # Ensemble Methods
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        
        # Financial Data
        "yfinance>=0.2.28",
        "ta>=0.10.2",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        
        # Utilities
        "tqdm>=4.66.0",
        "tabulate>=0.9.0",
        "colorama>=0.4.6",
    ]
    
    # Optional but recommended packages
    optional_packages = [
        "optuna>=3.3.0",  # Hyperparameter optimization
        "shap>=0.42.0",  # Model interpretability
        "pandas-ta>=0.3.14b0",  # Additional technical indicators
        "rich>=13.5.0",  # Better console output
        "python-dotenv>=1.0.0",  # Environment variables
    ]
    
    print("\n📦 Installing Essential Packages...")
    print("-" * 40)
    
    failed_essential = []
    for i, package in enumerate(essential_packages, 1):
        package_name = package.split('>=')[0].split('<')[0].split('[')[0]
        print(f"[{i}/{len(essential_packages)}] Installing {package_name}...", end=" ")
        if install_package(package):
            print("✅")
        else:
            print("❌")
            failed_essential.append(package)
    
    print("\n📦 Installing Optional Packages...")
    print("-" * 40)
    
    failed_optional = []
    for i, package in enumerate(optional_packages, 1):
        package_name = package.split('>=')[0].split('<')[0].split('[')[0]
        print(f"[{i}/{len(optional_packages)}] Installing {package_name}...", end=" ")
        if install_package(package):
            print("✅")
        else:
            print("❌")
            failed_optional.append(package)
    
    # Special handling for TA-Lib (requires system libraries)
    print("\n📦 Attempting TA-Lib Installation...")
    print("-" * 40)
    
    # Try to install TA-Lib (might fail on some systems)
    try:
        # For Google Colab
        if 'google.colab' in str(get_ipython()):
            print("Detected Google Colab environment")
            subprocess.check_call(["wget", "-q", "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"])
            subprocess.check_call(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"])
            os.chdir("ta-lib")
            subprocess.check_call(["./configure", "--prefix=/usr"])
            subprocess.check_call(["make"])
            subprocess.check_call(["make", "install"])
            os.chdir("..")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ta-lib"])
            print("TA-Lib installed successfully ✅")
    except:
        print("TA-Lib installation skipped (requires system libraries) ⚠️")
        print("The system will work without TA-Lib, using pandas-ta instead")
    
    # Verification
    print("\n🔍 Verifying Installation...")
    print("-" * 40)
    
    verification_imports = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("torch", "torch"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgb"),
        ("lightgbm", "lgb"),
        ("catboost", "catboost"),
        ("yfinance", "yf"),
        ("imblearn.over_sampling", "ADASYN"),
        ("plotly", "plotly"),
    ]
    
    all_verified = True
    for module, import_name in verification_imports:
        try:
            if "." in module:
                # Handle submodule imports
                parts = module.split(".")
                exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            all_verified = False
    
    # Summary
    print("\n" + "="*80)
    print("INSTALLATION SUMMARY")
    print("="*80)
    
    if failed_essential:
        print("\n⚠️  Failed Essential Packages:")
        for package in failed_essential:
            print(f"   - {package}")
        print("\n   Please install these manually using:")
        print("   pip install", " ".join(failed_essential))
    else:
        print("\n✅ All essential packages installed successfully!")
    
    if failed_optional:
        print("\n⚠️  Failed Optional Packages:")
        for package in failed_optional:
            print(f"   - {package}")
        print("\n   These are optional and won't affect core functionality.")
    
    if all_verified:
        print("\n✅ All core imports verified successfully!")
        print("\n🎉 Environment setup complete! You can now run the Hyperbolic CNN trading system.")
    else:
        print("\n⚠️  Some imports could not be verified. Please check the errors above.")
    
    print("\n" + "="*80)
    
    return all_verified and len(failed_essential) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)