# 🔬 Complete Guide to Running Real Experiments

## Where to Run the Experiments

### Option 1: Your Local Computer (Windows/Mac/Linux)

#### Step 1: Install Python (if not already installed)
- **Windows**: Download from https://www.python.org/downloads/
- **Mac**: Use Homebrew: `brew install python3`
- **Linux**: Use package manager: `sudo apt install python3 python3-pip`

#### Step 2: Clone Your Repository
```bash
# Open Terminal (Mac/Linux) or Command Prompt (Windows)
git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
cd hypervision-crypto-ai
```

#### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 4: Install Required Packages
```bash
pip install -r requirements_experiments.txt
```

If requirements_experiments.txt doesn't exist, install manually:
```bash
pip install numpy pandas torch scipy ccxt yfinance matplotlib seaborn jupyter
```

#### Step 5: Run the Experiments
```bash
python run_real_experiments.py
```

---

### Option 2: Google Colab (FREE Cloud Option) ⭐ EASIEST

#### Step 1: Open Google Colab
Go to: https://colab.research.google.com/

#### Step 2: Create New Notebook
Click "New Notebook"

#### Step 3: Clone Repository in Colab
Run this in the first cell:
```python
!git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
%cd hypervision-crypto-ai
```

#### Step 4: Install Dependencies
Run in the next cell:
```python
!pip install ccxt yfinance scipy
import torch
print(f"PyTorch version: {torch.__version__}")
```

#### Step 5: Run Experiments
```python
!python run_real_experiments.py
```

Or run interactively:
```python
# Import the experiment module
import sys
sys.path.append('/content/hypervision-crypto-ai')
from run_real_experiments import ExperimentRunner, ExperimentConfig

# Run experiments
config = ExperimentConfig()
runner = ExperimentRunner(config)
results = runner.run_experiments(['BTC/USDT', 'ETH/USDT'])

# Display results
print(f"Results: {results}")
```

---

### Option 3: Kaggle Notebooks (FREE with GPU)

1. Go to https://www.kaggle.com/
2. Create account (free)
3. Click "Code" → "New Notebook"
4. Enable GPU: Settings → Accelerator → GPU
5. Run same code as Google Colab above

---

### Option 4: AWS SageMaker (Professional)

```python
# In SageMaker Notebook
!git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
%cd hypervision-crypto-ai
!pip install -r requirements_experiments.txt
!python run_real_experiments.py
```

---

## 📊 What the Experiments Will Do

### 1. Data Collection Phase
```
Collecting data for BTC/USDT from 2021-01-01 to 2024-06-30
✓ Training data: 2021-01-01 to 2023-06-30 (21,900 hours)
✓ Validation data: 2023-07-01 to 2023-12-31 (4,380 hours)
✓ Test data: 2024-01-01 to 2024-06-30 (4,380 hours)
```

### 2. Model Training Phase
```
Training Hyperbolic CNN Model...
Epoch 1/100: Loss=0.892, Val_Accuracy=52.3%
Epoch 10/100: Loss=0.623, Val_Accuracy=61.7%
Epoch 50/100: Loss=0.342, Val_Accuracy=68.9%
Epoch 100/100: Loss=0.213, Val_Accuracy=71.2%
✓ Model training complete
```

### 3. Backtesting Phase
```
Running backtest on test data...
Initial Capital: $100,000
Processing 4,380 time periods...
✓ Executed 247 trades
✓ Win Rate: 64.7%
✓ Total Return: 18.3%
```

### 4. Results Output
```
========================================
EXPERIMENTAL RESULTS SUMMARY
========================================
Mean Return: 18.3% ± 4.2%
Mean Sharpe: 1.67 ± 0.31
Mean Drawdown: 8.7% ± 2.1%
Mean Win Rate: 64.7% ± 5.8%
========================================
Results saved to: experimental_results.json
```

---

## 🔍 Verifying Your Results

### Check the Output Files

After running, you'll have these files:

1. **experimental_results.json** - Main results file
```json
{
  "timestamp": "2024-01-15T10:23:45",
  "config": {
    "train_period": "2021-01-01 to 2023-06-30",
    "test_period": "2024-01-01 to 2024-06-30"
  },
  "results": {
    "mean_return": 0.183,
    "mean_sharpe": 1.67,
    "mean_drawdown": 0.087,
    "mean_win_rate": 0.647,
    "std_return": 0.042,
    "std_sharpe": 0.31
  }
}
```

2. **experiment_log.txt** - Detailed log
3. **model_checkpoint.pth** - Trained model
4. **backtest_trades.csv** - All trades executed

---

## ⚠️ Important Checks Before Publishing

### 1. Sanity Check Your Results
```python
# Are results realistic?
assert 0 < win_rate < 1.0, "Win rate should be between 0 and 1"
assert -1 < sharpe_ratio < 5, "Sharpe ratio seems unrealistic"
assert 0 < max_drawdown < 0.5, "Drawdown seems wrong"
```

### 2. Compare with Baselines
The script should also run baseline models:
- Buy & Hold strategy
- Simple Moving Average
- Random predictions

Your model should outperform random but not by unrealistic margins.

### 3. Statistical Significance
Check p-values in results:
```python
if p_value > 0.05:
    print("⚠️ Results not statistically significant!")
    print("Need more data or different approach")
```

---

## 📈 Expected Realistic Results

Based on academic literature, realistic results for crypto trading:

| Metric | Realistic Range | Suspicious If |
|--------|----------------|---------------|
| Accuracy | 55-70% | > 80% |
| Sharpe Ratio | 0.5-2.0 | > 3.0 |
| Annual Return | 10-40% | > 100% |
| Max Drawdown | 10-30% | < 5% |
| Win Rate | 45-65% | > 75% |

---

## 🚀 Quick Start Commands

### Fastest Way (Copy-Paste Ready):

```bash
# 1. Clone repository
git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
cd hypervision-crypto-ai

# 2. Install dependencies
pip install numpy pandas torch scipy ccxt yfinance

# 3. Run experiments
python run_real_experiments.py

# 4. View results
cat experimental_results.json
```

---

## 📝 Updating Documentation with Real Results

After experiments complete:

1. Open `JOURNAL_PUBLICATION.md`
2. Find all placeholder values (94.7%, 2.89, etc.)
3. Replace with your actual results from `experimental_results.json`
4. Include confidence intervals and p-values
5. Update all tables with real data

Example replacement:
```markdown
<!-- BEFORE (Placeholder) -->
| H-CNN (Ours) | 94.7% | 2.89 | 1.2M | 125ms |

<!-- AFTER (Real Results) -->
| H-CNN (Ours) | 68.3% ± 2.1% | 1.67 ± 0.31 | 1.2M | 125ms |
```

---

## 🆘 Troubleshooting

### If you get "Module not found" error:
```bash
pip install [missing-module-name]
```

### If data download fails:
- Check internet connection
- Try using VPN if exchange blocked
- Use Yahoo Finance fallback (already in code)

### If results seem wrong:
- Check data quality
- Verify no look-ahead bias
- Ensure proper train/test split
- Check for bugs in backtesting logic

---

## 📧 Getting Help

If experiments fail or results seem incorrect:

1. Check `experiment_log.txt` for errors
2. Review the backtesting logic in `run_real_experiments.py`
3. Ensure data is being downloaded correctly
4. Verify your Python environment has all dependencies

---

## ✅ Checklist Before Publication

- [ ] Experiments run successfully
- [ ] Results are realistic (not too good to be true)
- [ ] Statistical tests show significance (p < 0.05)
- [ ] Results include error bars (± standard deviation)
- [ ] Compared against multiple baselines
- [ ] Transaction costs included
- [ ] No look-ahead bias in backtesting
- [ ] Results reproducible with fixed seed
- [ ] All placeholder values replaced with real results
- [ ] Raw experimental data saved as evidence

---

**Remember**: It's better to report honest 65% accuracy than fake 95% accuracy. 
Academic integrity is paramount!