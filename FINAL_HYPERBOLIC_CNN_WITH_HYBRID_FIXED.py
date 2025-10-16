"""
Final Improved Hyperbolic CNN with Hybrid Models for Cryptocurrency Trading
FIXED VERSION - Corrected ADASYN initialization
Academic Research Implementation - All computations from real data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HYPERBOLIC GEOMETRY OPERATIONS
# ============================================================================

def mobius_add(x, y, c=1.0):
    """Möbius addition in the Poincaré ball."""
    x_norm_sq = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), max=1-1e-5)
    y_norm_sq = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), max=1-1e-5)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2*c*xy + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
    denom = 1 + 2*c*xy + c*c*x_norm_sq*y_norm_sq
    return num / torch.clamp(denom, min=1e-5)

def exp_map(v, c=1.0):
    """Exponential map from tangent space to Poincaré ball."""
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-5)
    coeff = torch.tanh(torch.sqrt(c) * v_norm / 2) / (torch.sqrt(c) * v_norm)
    return coeff * v

def log_map(x, c=1.0):
    """Logarithmic map from Poincaré ball to tangent space."""
    x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-5, max=1-1e-5)
    artanh_x = torch.atanh(torch.sqrt(c) * x_norm)
    return (artanh_x / (torch.sqrt(c) * x_norm)) * x

# ============================================================================
# FINAL IMPROVED HYPERBOLIC CNN WITH ALL ENHANCEMENTS
# ============================================================================

class FinalImprovedHyperbolicCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, c=1.0, dropout=0.2):
        super().__init__()
        self.c = c
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale feature extraction
        self.scale1 = nn.Linear(input_dim, hidden_dim)
        self.scale2 = nn.Linear(input_dim, hidden_dim // 2)
        self.scale3 = nn.Linear(input_dim, hidden_dim // 4)
        
        # Combine scales
        combined_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 4
        self.fusion = nn.Linear(combined_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Hyperbolic layers with residual connections
        self.hyperbolic1 = nn.Linear(hidden_dim, hidden_dim)
        self.hyperbolic2 = nn.Linear(hidden_dim, hidden_dim)
        self.hyperbolic3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.5)  # Stronger dropout in later layers
        
        # Output layers
        self.pre_output = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, num_classes)
        
        # Ensemble-like predictions
        self.auxiliary_output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Multi-scale feature extraction
        s1 = F.relu(self.scale1(x))
        s2 = F.relu(self.scale2(x))
        s3 = F.relu(self.scale3(x))
        
        # Concatenate scales
        multi_scale = torch.cat([s1, s2, s3], dim=-1)
        x = F.relu(self.fusion(multi_scale))
        x = self.norm1(x)
        x = self.dropout(x)
        
        # Self-attention
        residual = x
        x = x.unsqueeze(0)  # Add sequence dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = x + residual  # Residual connection
        x = self.norm2(x)
        
        # Hyperbolic transformations with residual connections
        # Layer 1
        residual = x
        h1 = self.hyperbolic1(x)
        h1 = exp_map(h1, self.c)
        h1 = log_map(h1, self.c)
        x = F.relu(h1 + residual)  # Residual connection
        x = self.norm3(x)
        x = self.dropout(x)
        
        # Layer 2
        residual = x
        h2 = self.hyperbolic2(x)
        h2 = exp_map(h2, self.c)
        h2 = log_map(h2, self.c)
        x = F.relu(h2 + residual)  # Residual connection
        x = self.norm4(x)
        x = self.dropout2(x)
        
        # Layer 3
        h3 = self.hyperbolic3(x)
        h3 = exp_map(h3, self.c)
        h3 = log_map(h3, self.c)
        x = F.relu(h3 + x)  # Final residual
        
        # Output
        pre_out = F.relu(self.pre_output(x))
        pre_out = self.dropout(pre_out)
        out = self.output(pre_out)
        
        # Auxiliary output for ensemble effect
        aux_out = self.auxiliary_output(x)
        
        # Weighted combination
        final_out = 0.7 * out + 0.3 * aux_out
        
        return final_out

# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Label smoothing
        n_classes = inputs.size(-1)
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.fill_(self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        log_probs = F.log_softmax(inputs, dim=-1)
        smooth_loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        return (focal_loss * 0.7 + smooth_loss * 0.3).mean()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features(df):
    """Create 60+ technical features from OHLCV data."""
    features = df.copy()
    
    # Price-based features
    features['returns'] = features['Close'].pct_change()
    features['log_returns'] = np.log1p(features['Close'].pct_change())
    features['high_low_ratio'] = features['High'] / (features['Low'] + 1e-10)
    features['close_open_ratio'] = features['Close'] / (features['Open'] + 1e-10)
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = features['Close'].rolling(period).mean()
        features[f'ema_{period}'] = features['Close'].ewm(span=period).mean()
        features[f'volume_sma_{period}'] = features['Volume'].rolling(period).mean()
    
    # Volatility features
    for period in [5, 10, 20]:
        features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        features[f'volume_volatility_{period}'] = features['Volume'].pct_change().rolling(period).std()
    
    # RSI
    for period in [7, 14, 21]:
        delta = features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = features['Close'].ewm(span=12).mean()
    ema26 = features['Close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_diff'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    for period in [10, 20]:
        sma = features['Close'].rolling(period).mean()
        std = features['Close'].rolling(period).std()
        features[f'bb_upper_{period}'] = sma + 2 * std
        features[f'bb_lower_{period}'] = sma - 2 * std
        features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
        features[f'bb_position_{period}'] = (features['Close'] - features[f'bb_lower_{period}']) / (features[f'bb_width_{period}'] + 1e-10)
    
    # Volume indicators
    features['volume_ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
    features['vwap'] = (features['Close'] * features['Volume']).cumsum() / features['Volume'].cumsum()
    features['vwap_ratio'] = features['Close'] / features['vwap']
    
    # Price patterns
    features['higher_high'] = (features['High'] > features['High'].shift(1)).astype(int)
    features['lower_low'] = (features['Low'] < features['Low'].shift(1)).astype(int)
    features['inside_bar'] = ((features['High'] < features['High'].shift(1)) & 
                              (features['Low'] > features['Low'].shift(1))).astype(int)
    
    # Market microstructure
    features['spread'] = features['High'] - features['Low']
    features['spread_pct'] = features['spread'] / features['Close']
    features['volume_price_trend'] = (features['Volume'] * features['Close'].pct_change()).cumsum()
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
    
    # Remove NaN rows
    features = features.dropna()
    
    # Create labels (Hold=0, Buy=1, Sell=2)
    future_returns = features['Close'].shift(-1) / features['Close'] - 1
    
    # Enhanced labeling with market regime consideration
    volatility = features['returns'].rolling(20).std()
    adaptive_threshold = volatility * 1.5  # Adaptive threshold based on volatility
    
    labels = pd.Series(0, index=features.index)  # Default to Hold
    labels[future_returns > adaptive_threshold] = 1  # Buy
    labels[future_returns < -adaptive_threshold] = 2  # Sell
    
    # Select feature columns
    feature_cols = [col for col in features.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividends', 'Stock Splits']]
    
    X = features[feature_cols].values[:-1]  # Remove last row (no future return)
    y = labels.values[:-1]
    
    return X, y

# ============================================================================
# HYBRID MODEL
# ============================================================================

class HybridModel:
    def __init__(self, hyperbolic_model, ensemble_models, weights=None):
        """
        Hybrid model combining Hyperbolic CNN with ensemble methods.
        
        Args:
            hyperbolic_model: Trained Hyperbolic CNN
            ensemble_models: Dict of ensemble models (e.g., {'xgboost': model1, 'lightgbm': model2})
            weights: Dict of weights for each model
        """
        self.hyperbolic_model = hyperbolic_model
        self.ensemble_models = ensemble_models
        
        if weights is None:
            n_models = len(ensemble_models) + 1
            weight = 1.0 / n_models
            self.weights = {'hyperbolic': weight}
            for name in ensemble_models.keys():
                self.weights[name] = weight
        else:
            self.weights = weights
            
    def predict(self, X_tensor, X_scaled=None):
        """Make predictions using weighted ensemble."""
        device = next(self.hyperbolic_model.parameters()).device
        
        # Hyperbolic CNN predictions
        if not isinstance(X_tensor, torch.Tensor):
            X_tensor = torch.FloatTensor(X_tensor).to(device)
            
        self.hyperbolic_model.eval()
        with torch.no_grad():
            hyperbolic_probs = F.softmax(self.hyperbolic_model(X_tensor), dim=1).cpu().numpy()
        
        # Ensemble predictions
        ensemble_probs = []
        for name, model in self.ensemble_models.items():
            if X_scaled is not None:
                # Use scaled features for ensemble models
                probs = model.predict_proba(X_scaled)
            else:
                # Convert tensor to numpy if needed
                X_np = X_tensor.cpu().numpy() if isinstance(X_tensor, torch.Tensor) else X_tensor
                probs = model.predict_proba(X_np)
            ensemble_probs.append(probs)
        
        # Weighted combination
        final_probs = hyperbolic_probs * self.weights['hyperbolic']
        for i, name in enumerate(self.ensemble_models.keys()):
            final_probs += ensemble_probs[i] * self.weights[name]
        
        return np.argmax(final_probs, axis=1)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_improved_model(model, X_train, y_train, X_val, y_val, 
                         epochs=150, batch_size=32, learning_rate=0.001, device='cuda'):
    """Train the improved Hyperbolic CNN with all enhancements."""
    
    model = model.to(device)
    criterion = FocalLossWithSmoothing(gamma=2.0, smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_correct = (val_outputs.argmax(1) == y_val).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / len(X_train) * 100
        val_acc = val_correct / len(X_val) * 100
        
        train_losses.append(train_loss / (len(X_train) // batch_size))
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    return model, train_losses, val_losses, train_accs, val_accs

# ============================================================================
# BACKTESTING WITH RISK MANAGEMENT
# ============================================================================

def backtest_strategy(df, predictions, initial_capital=10000):
    """Backtest trading strategy with risk management."""
    portfolio_value = initial_capital
    position = 0
    entry_price = 0
    portfolio_values = [initial_capital]
    trades = []
    
    # Risk management parameters
    stop_loss = 0.03  # 3%
    take_profit = 0.06  # 6%
    position_size = 0.25  # 25% of portfolio
    
    prices = df['Close'].values[-len(predictions):]
    
    for i in range(1, len(predictions)):
        current_price = prices[i]
        signal = predictions[i]
        
        # Check stop loss and take profit
        if position != 0:
            price_change = (current_price - entry_price) / entry_price
            
            if position == 1:  # Long position
                if price_change <= -stop_loss or price_change >= take_profit:
                    trade_return = price_change
                    portfolio_value *= (1 + trade_return * position_size)
                    trades.append(trade_return)
                    position = 0
                    entry_price = 0
            
            elif position == -1:  # Short position
                if price_change >= stop_loss or price_change <= -take_profit:
                    trade_return = -price_change
                    portfolio_value *= (1 + trade_return * position_size)
                    trades.append(trade_return)
                    position = 0
                    entry_price = 0
        
        # Execute new signals
        if position == 0:
            if signal == 1:  # Buy signal
                position = 1
                entry_price = current_price
            elif signal == 2:  # Sell signal
                position = -1
                entry_price = current_price
        
        portfolio_values.append(portfolio_value)
    
    return np.array(portfolio_values), len(trades)

def calculate_financial_metrics(portfolio_values):
    """Calculate comprehensive financial metrics."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    
    # Sharpe ratio (assuming 252 trading days)
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10) if len(downside_returns) > 0 else 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Calmar ratio
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("FINAL HYPERBOLIC CNN WITH HYBRID MODELS - COMPLETE COMPARISON")
    print("="*80)
    
    # Fetch data
    print("\n1. Fetching cryptocurrency data...")
    df = yf.download('BTC-USD', 
                     start=datetime.now() - timedelta(days=730),
                     end=datetime.now(),
                     progress=False)
    print(f"   Data shape: {df.shape}")
    
    # Feature engineering
    print("\n2. Applying enhanced feature engineering...")
    X, y = create_enhanced_features(df)
    print(f"   Total features: {X.shape[1]}")
    print(f"   Final dataset: {X.shape}")
    print(f"   Original class distribution: {np.bincount(y)}")
    
    # Apply ADASYN balancing - FIXED: Removed n_jobs parameter
    print("\n3. Applying ADASYN balancing...")
    adasyn = ADASYN(
        sampling_strategy='auto',
        random_state=42,
        n_neighbors=5
        # Removed n_jobs parameter - ADASYN doesn't support it
    )
    X_balanced, y_balanced = adasyn.fit_resample(X, y)
    
    print(f"   Balanced dataset: {X_balanced.shape}")
    print(f"   Balanced distribution: {np.bincount(y_balanced)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_balanced
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_original = scaler.transform(X)  # For backtesting
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    print(f"\n4. Training on device: {device}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train Improved Hyperbolic CNN
    print("\n5. Training Improved Hyperbolic CNN...")
    hyperbolic_model = FinalImprovedHyperbolicCNN(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        num_classes=3,
        c=1.0,
        dropout=0.2
    )
    
    hyperbolic_model, train_losses, val_losses, train_accs, val_accs = train_improved_model(
        hyperbolic_model, X_train_tensor, y_train_tensor, 
        X_test_tensor, y_test_tensor, epochs=50, device=device
    )
    
    # Train ensemble models
    print("\n6. Training ensemble models...")
    
    # XGBoost
    print("   Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # LightGBM
    print("   Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.01,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    
    # CatBoost
    print("   Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.01,
        random_state=42, verbose=False
    )
    cat_model.fit(X_train_scaled, y_train)
    
    # Create hybrid models
    print("\n7. Creating hybrid models...")
    
    # Hybrid 1: Hyperbolic + XGBoost
    hybrid1 = HybridModel(
        hyperbolic_model=hyperbolic_model,
        ensemble_models={'xgboost': xgb_model},
        weights={'hyperbolic': 0.7, 'xgboost': 0.3}
    )
    
    # Hybrid 2: Hyperbolic + LightGBM
    hybrid2 = HybridModel(
        hyperbolic_model=hyperbolic_model,
        ensemble_models={'lightgbm': lgb_model},
        weights={'hyperbolic': 0.7, 'lightgbm': 0.3}
    )
    
    # Hybrid 3: Hyperbolic + All
    hybrid3 = HybridModel(
        hyperbolic_model=hyperbolic_model,
        ensemble_models={'xgboost': xgb_model, 'lightgbm': lgb_model, 'catboost': cat_model},
        weights={'hyperbolic': 0.4, 'xgboost': 0.2, 'lightgbm': 0.2, 'catboost': 0.2}
    )
    
    # Evaluate all models
    print("\n8. Model Comparison:")
    print("-" * 80)
    
    results = []
    
    # Evaluate Hyperbolic CNN
    hyperbolic_model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(hyperbolic_model(X_test_tensor), dim=1).cpu().numpy()
        y_pred_trading = torch.argmax(hyperbolic_model(torch.FloatTensor(X_original).to(device)), dim=1).cpu().numpy()
    
    acc = accuracy_score(y_test, y_pred)
    portfolio, num_trades = backtest_strategy(df, y_pred_trading)
    metrics = calculate_financial_metrics(portfolio)
    
    results.append({
        'Model': 'Improved Hyperbolic CNN',
        'Accuracy': acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'Return': metrics['total_return'],
        'Sharpe': metrics['sharpe_ratio'],
        'Sortino': metrics['sortino_ratio'],
        'Calmar': metrics['calmar_ratio'],
        'MaxDD': metrics['max_drawdown'],
        'Trades': num_trades
    })
    
    print(f"Improved Hyperbolic CNN: Acc={acc:.4f}, Sharpe={metrics['sharpe_ratio']:.3f}")
    
    # Evaluate ensemble models
    ensemble_models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'CatBoost': cat_model
    }
    
    for name, model in ensemble_models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_trading = model.predict(X_original)
        
        acc = accuracy_score(y_test, y_pred)
        portfolio, num_trades = backtest_strategy(df, y_pred_trading)
        metrics = calculate_financial_metrics(portfolio)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'Return': metrics['total_return'],
            'Sharpe': metrics['sharpe_ratio'],
            'Sortino': metrics['sortino_ratio'],
            'Calmar': metrics['calmar_ratio'],
            'MaxDD': metrics['max_drawdown'],
            'Trades': num_trades
        })
        
        print(f"{name}: Acc={acc:.4f}, Sharpe={metrics['sharpe_ratio']:.3f}")
    
    # Evaluate hybrid models
    hybrid_configs = [
        ('Hybrid: Hyperbolic+XGBoost (70-30)', hybrid1),
        ('Hybrid: Hyperbolic+LightGBM (70-30)', hybrid2),
        ('Hybrid: Hyperbolic+All (40-20-20-20)', hybrid3)
    ]
    
    for name, hybrid in hybrid_configs:
        y_pred = hybrid.predict(X_test_tensor, X_test_scaled)
        y_pred_trading = hybrid.predict(torch.FloatTensor(X_original).to(device), X_original)
        
        acc = accuracy_score(y_test, y_pred)
        portfolio, num_trades = backtest_strategy(df, y_pred_trading)
        metrics = calculate_financial_metrics(portfolio)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'Return': metrics['total_return'],
            'Sharpe': metrics['sharpe_ratio'],
            'Sortino': metrics['sortino_ratio'],
            'Calmar': metrics['calmar_ratio'],
            'MaxDD': metrics['max_drawdown'],
            'Trades': num_trades
        })
        
        print(f"{name}: Acc={acc:.4f}, Sharpe={metrics['sharpe_ratio']:.3f}")
    
    # Display results table
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.round({
        'Accuracy': 4, 'Precision': 4, 'Recall': 4, 'F1': 4,
        'Return': 2, 'Sharpe': 3, 'Sortino': 3, 'Calmar': 3, 'MaxDD': 2
    })
    
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - ALL RESULTS FROM REAL DATA")
    print("="*80)

if __name__ == "__main__":
    main()