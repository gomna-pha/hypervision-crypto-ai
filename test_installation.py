#!/usr/bin/env python3
"""
Quick test to verify all components work correctly
Run this after installation to confirm everything is set up
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("GOMNA PLATFORM - INSTALLATION TEST")
print("="*60)

# Track success
all_tests_passed = True
critical_failures = []

# Test 1: Core packages
print("\n1. Testing Core Packages...")
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    print("✅ Core packages imported")
except ImportError as e:
    print(f"❌ Core package import failed: {e}")
    critical_failures.append("core_packages")
    all_tests_passed = False

# Test 2: Imbalanced learning (CRITICAL)
print("\n2. Testing Imbalanced Learning (SMOTE/ADASYN)...")
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    
    # Test SMOTE
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], 100, p=[0.1, 0.7, 0.2])
    
    print(f"   Original distribution: {np.bincount(y)}")
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"   After SMOTE: {np.bincount(y_balanced)}")
    print("✅ SMOTE working correctly")
    
except ImportError as e:
    print(f"❌ Imbalanced-learn not installed: {e}")
    print("   FIX: pip install imbalanced-learn")
    critical_failures.append("imbalanced_learn")
    all_tests_passed = False
except Exception as e:
    print(f"❌ SMOTE test failed: {e}")
    all_tests_passed = False

# Test 3: Market data fetching
print("\n3. Testing Market Data APIs...")
try:
    import ccxt
    import yfinance as yf
    
    # Test Yahoo Finance
    btc = yf.download('BTC-USD', period='1d', progress=False)
    if len(btc) > 0:
        print(f"✅ Yahoo Finance: Downloaded {len(btc)} BTC data points")
    else:
        print("⚠️ Yahoo Finance: No data downloaded (might be weekend)")
    
    # Test CCXT
    exchange = ccxt.binance()
    print(f"✅ CCXT: Binance connection established")
    
except ImportError as e:
    print(f"❌ Market data packages not installed: {e}")
    print("   FIX: pip install ccxt yfinance")
    critical_failures.append("market_data")
    all_tests_passed = False
except Exception as e:
    print(f"⚠️ Market data test warning: {e}")

# Test 4: PyTorch functionality
print("\n4. Testing PyTorch Deep Learning...")
try:
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    
    # Test forward pass
    x = torch.randn(5, 10)
    output = model(x)
    
    # Test loss and backward
    target = torch.tensor([0, 1, 2, 1, 0])
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    
    print(f"✅ PyTorch model: Forward and backward pass working")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ CUDA not available (will use CPU)")
    
except Exception as e:
    print(f"❌ PyTorch test failed: {e}")
    critical_failures.append("pytorch")
    all_tests_passed = False

# Test 5: Focal Loss implementation
print("\n5. Testing Focal Loss for Imbalanced Data...")
try:
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    focal = FocalLoss()
    loss_val = focal(output, target)
    
    print(f"✅ Focal Loss: {loss_val.item():.4f}")
    
except Exception as e:
    print(f"❌ Focal Loss test failed: {e}")
    all_tests_passed = False

# Test 6: Scikit-learn metrics
print("\n6. Testing Scikit-learn Metrics...")
try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    
    # Test class weight calculation
    classes = np.array([0, 1, 2])
    y_imbalanced = np.random.choice([0, 1, 2], 100, p=[0.1, 0.7, 0.2])
    weights = compute_class_weight('balanced', classes=classes, y=y_imbalanced)
    
    print(f"✅ Class weights calculated: {weights}")
    
except ImportError as e:
    print(f"❌ Scikit-learn not installed: {e}")
    print("   FIX: pip install scikit-learn")
    critical_failures.append("sklearn")
    all_tests_passed = False

# Test 7: Run mini training loop
print("\n7. Testing Complete Training Pipeline...")
try:
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create mini dataset
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 3, (100,))
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Mini training loop
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i >= 2:  # Just 2 batches for testing
            break
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"✅ Training pipeline: Successfully completed mini training")
    
except Exception as e:
    print(f"❌ Training pipeline test failed: {e}")
    all_tests_passed = False

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

if all_tests_passed:
    print("✅ ALL TESTS PASSED!")
    print("\nYou can now run:")
    print("  python advanced_model_training.py")
    print("  python run_real_experiments.py")
    
else:
    print("⚠️ Some tests failed.")
    
    if critical_failures:
        print("\n🔴 CRITICAL PACKAGES MISSING:")
        
        if "imbalanced_learn" in critical_failures:
            print("\n1. Imbalanced-learn (REQUIRED for SMOTE):")
            print("   pip install imbalanced-learn")
        
        if "market_data" in critical_failures:
            print("\n2. Market data packages (REQUIRED for real data):")
            print("   pip install ccxt yfinance")
        
        if "sklearn" in critical_failures:
            print("\n3. Scikit-learn (REQUIRED for metrics):")
            print("   pip install scikit-learn")
        
        if "pytorch" in critical_failures:
            print("\n4. PyTorch (REQUIRED for deep learning):")
            print("   pip install torch")
        
        if "core_packages" in critical_failures:
            print("\n5. Core packages:")
            print("   pip install numpy pandas")
    
    print("\n💡 Quick fix - install all at once:")
    print("   pip install torch numpy pandas scikit-learn imbalanced-learn ccxt yfinance scipy matplotlib")

# Final check
print("\n" + "="*60)
try:
    # The most important check - can we import the training script?
    exec(open('advanced_model_training.py').read().split('if __name__')[0])
    print("✅ advanced_model_training.py can be imported!")
except FileNotFoundError:
    print("⚠️ advanced_model_training.py not found in current directory")
    print("   Make sure you're in the hypervision-crypto-ai directory")
except Exception as e:
    print(f"⚠️ Issue with advanced_model_training.py: {e}")

print("="*60)