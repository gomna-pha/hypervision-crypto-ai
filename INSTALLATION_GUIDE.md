# 📦 Complete Installation Guide for GOMNA Platform

## Table of Contents
1. [Quick Install (Essential Only)](#quick-install-essential-only)
2. [Full Installation](#full-installation)
3. [Platform-Specific Instructions](#platform-specific-instructions)
4. [Google Colab Setup](#google-colab-setup)
5. [Troubleshooting](#troubleshooting)
6. [Verification Script](#verification-script)

---

## 🚀 Quick Install (Essential Only)

### Minimal Requirements for Running Experiments

```bash
# Essential packages only - will run the experiments
pip install torch numpy pandas scikit-learn imbalanced-learn ccxt yfinance scipy matplotlib
```

### One-Line Install Command
```bash
pip install torch>=2.0.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 imbalanced-learn>=0.11.0 ccxt>=4.0.0 yfinance>=0.2.28 scipy>=1.10.0 matplotlib>=3.7.0
```

---

## 📋 Full Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv gomna_env

# Activate it
# On Mac/Linux:
source gomna_env/bin/activate
# On Windows:
gomna_env\Scripts\activate
```

### Step 2: Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Core Requirements
```bash
# Clone repository first
git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
cd hypervision-crypto-ai

# Install all requirements
pip install -r requirements_advanced.txt
```

### Step 4: Install TA-Lib (Optional but Recommended)

**Mac:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**Windows:**
```bash
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Install the .whl file:
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

---

## 🖥️ Platform-Specific Instructions

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python@3.10

# Install system dependencies
brew install gcc openblas lapack

# Install packages
pip3 install -r requirements_advanced.txt
```

### Ubuntu/Debian Linux

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.10 python3-pip python3-venv
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Install scientific computing libraries
sudo apt-get install -y libopenblas-dev liblapack-dev gfortran

# Install packages
pip3 install -r requirements_advanced.txt
```

### Windows

```powershell
# Install Python from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH"

# Open PowerShell as Administrator
# Install Visual C++ Build Tools (required for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install packages
pip install -r requirements_advanced.txt

# If torch installation fails, use:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🌐 Google Colab Setup

### Complete Setup in Colab

```python
# Cell 1: Install all packages
!pip install torch numpy pandas scikit-learn imbalanced-learn ccxt yfinance scipy matplotlib seaborn
!pip install ta websocket-client python-dotenv tqdm
!pip install tensorboard wandb mlflow optuna

# Cell 2: Clone repository
!git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
%cd hypervision-crypto-ai

# Cell 3: Import and verify
import torch
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
import ccxt
import yfinance as yf

print("✅ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Cell 4: Run the advanced training
!python advanced_model_training.py
```

### Colab Pro Features (If Using)
```python
# Enable GPU
from google.colab import runtime
runtime.change_runtime_type('GPU')

# Check GPU
!nvidia-smi

# Mount Google Drive for saving results
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🔍 Verification Script

Create a file `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
Verify all required packages are installed correctly
"""

import sys
import importlib
from packaging import version

def check_package(package_name, min_version=None, import_name=None):
    """Check if a package is installed and meets version requirements"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        installed_version = getattr(module, '__version__', 'Unknown')
        
        if min_version and installed_version != 'Unknown':
            if version.parse(installed_version) < version.parse(min_version):
                return f"❌ {package_name}: {installed_version} (need >= {min_version})"
        
        return f"✅ {package_name}: {installed_version}"
    except ImportError:
        return f"❌ {package_name}: NOT INSTALLED"

# Check all required packages
packages = [
    ('torch', '2.0.0', None),
    ('numpy', '1.24.0', None),
    ('pandas', '2.0.0', None),
    ('sklearn', '1.3.0', 'sklearn'),
    ('imblearn', '0.11.0', None),
    ('ccxt', '4.0.0', None),
    ('yfinance', '0.2.28', None),
    ('scipy', '1.10.0', None),
    ('matplotlib', '3.7.0', None),
    ('seaborn', '0.12.0', None),
]

print("="*50)
print("PACKAGE VERIFICATION")
print("="*50)

all_ok = True
for package in packages:
    result = check_package(*package)
    print(result)
    if '❌' in result:
        all_ok = False

print("="*50)

# Check CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
    else:
        print("ℹ️ CUDA: Not available (CPU mode)")
except:
    print("ℹ️ CUDA: Cannot check")

print("="*50)

if all_ok:
    print("✅ ALL ESSENTIAL PACKAGES INSTALLED CORRECTLY!")
    print("You can now run: python advanced_model_training.py")
else:
    print("⚠️ Some packages are missing or outdated.")
    print("Run: pip install -r requirements_advanced.txt")

# Test SMOTE import specifically
print("\nTesting SMOTE import...")
try:
    from imblearn.over_sampling import SMOTE
    print("✅ SMOTE imported successfully!")
except ImportError as e:
    print(f"❌ SMOTE import failed: {e}")
    print("Fix: pip install imbalanced-learn")

# Test exchange connections
print("\nTesting exchange connections...")
try:
    import ccxt
    binance = ccxt.binance()
    print(f"✅ Binance API: {binance.has['fetchOHLCV']}")
except Exception as e:
    print(f"⚠️ Exchange test failed: {e}")
```

Run verification:
```bash
python verify_installation.py
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **ImportError: No module named 'imblearn'**
```bash
# Solution
pip install imbalanced-learn
# or
conda install -c conda-forge imbalanced-learn
```

#### 2. **PyTorch CUDA Issues**
```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. **TA-Lib Installation Failed**
```bash
# Alternative: Use pandas-ta instead
pip install pandas-ta
# It's pure Python and doesn't need C dependencies
```

#### 4. **Memory Issues**
```python
# Reduce batch size in config
config = {
    'batch_size': 32,  # Reduce from 64
    'embed_dim': 64,   # Reduce from 128
}
```

#### 5. **SSL Certificate Errors**
```bash
# Temporary fix (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package_name>

# Better solution: Update certificates
pip install --upgrade certifi
```

---

## 📊 Minimal vs Full Installation

### Minimal (2.5GB) - Essential for experiments
```bash
pip install torch numpy pandas scikit-learn imbalanced-learn ccxt yfinance scipy matplotlib
```

### Full (8GB) - Everything including monitoring/deployment
```bash
pip install -r requirements_advanced.txt
```

---

## ✅ Quick Test After Installation

```python
# test_setup.py
import torch
from imblearn.over_sampling import SMOTE
import ccxt
import yfinance as yf

# Test data fetching
btc = yf.download('BTC-USD', period='1d')
print(f"✅ Downloaded {len(btc)} BTC data points")

# Test SMOTE
import numpy as np
X = np.random.randn(100, 10)
y = np.random.choice([0, 1, 2], 100, p=[0.1, 0.7, 0.2])
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"✅ SMOTE working: {len(X)} → {len(X_balanced)} samples")

# Test PyTorch
model = torch.nn.Linear(10, 3)
print(f"✅ PyTorch model created: {model}")

print("\n🎉 Setup complete! You can now run experiments.")
```

---

## 🚨 CRITICAL Packages for Paper Results

These MUST be installed for accurate results:

1. **imbalanced-learn** - For SMOTE/ADASYN
2. **scikit-learn** - For metrics and preprocessing  
3. **ccxt + yfinance** - For real market data
4. **torch** - For deep learning model
5. **scipy** - For statistical tests

Without these, the experiments won't produce valid results for publication!

---

**After installation, run:**
```bash
python advanced_model_training.py
```

This will confirm everything works with balanced data and regularization!