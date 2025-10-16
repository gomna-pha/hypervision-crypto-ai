"""
READY-TO-RUN SCRIPT FOR GOOGLE COLAB
This handles all compatibility issues automatically
"""

import subprocess
import sys
import os

def fix_numpy():
    """Fix NumPy version compatibility"""
    try:
        import numpy as np
        version = np.__version__
        if version.startswith('2.'):
            print(f"⚠️ Detected NumPy {version} - fixing compatibility...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "-q"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "-q"])
            print("✅ NumPy fixed! Restarting...")
            return True
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "-q"])
    return False

def install_packages():
    """Install all required packages with compatible versions"""
    packages = [
        ("numpy==1.24.3", "NumPy"),
        ("pandas==2.0.3", "Pandas"),
        ("scipy==1.11.4", "SciPy"),
        ("torch", "PyTorch"),
        ("scikit-learn==1.3.2", "Scikit-learn"),
        ("imbalanced-learn==0.11.0", "Imbalanced-learn"),
        ("xgboost==2.0.3", "XGBoost"),
        ("lightgbm==4.1.0", "LightGBM"),
        ("catboost==1.2.2", "CatBoost"),
        ("yfinance==0.2.33", "yFinance"),
        ("matplotlib==3.7.2", "Matplotlib"),
        ("seaborn==0.13.0", "Seaborn"),
        ("plotly==5.18.0", "Plotly"),
        ("tqdm", "TQDM"),
        ("tabulate", "Tabulate"),
    ]
    
    print("📦 Installing packages...")
    for package, name in packages:
        print(f"  Installing {name}...", end=" ")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"], 
                                stderr=subprocess.DEVNULL)
            print("✅")
        except:
            print(f"⚠️ (retrying...)", end=" ")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package.split("==")[0], "-q"])
            print("✅")
    
    print("\n✅ All packages installed!")

def download_and_run():
    """Download and execute the fixed implementation"""
    import urllib.request
    
    print("\n📥 Downloading implementation...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/gomna-pha/hypervision-crypto-ai/main/FINAL_HYPERBOLIC_CNN_FULLY_FIXED.py',
        'hyperbolic_cnn.py'
    )
    print("✅ Downloaded!")
    
    # Import and verify
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy as np
    print(f"\nNumPy version: {np.__version__}")
    
    print("\n" + "="*80)
    print("🚀 RUNNING HYPERBOLIC CNN TRADING SYSTEM")
    print("="*80 + "\n")
    
    # Execute
    exec(open('hyperbolic_cnn.py').read())

def main():
    """Main execution"""
    print("🔧 HYPERBOLIC CNN - COLAB SETUP")
    print("="*40)
    
    # Check if we need to fix NumPy
    if fix_numpy():
        print("\n⚠️ IMPORTANT: Runtime will restart now.")
        print("After restart, run this script again to continue.")
        os.kill(os.getpid(), 9)
    
    # Install packages
    install_packages()
    
    # Download and run
    download_and_run()

if __name__ == "__main__":
    main()