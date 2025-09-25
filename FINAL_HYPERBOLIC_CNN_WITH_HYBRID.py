#!/usr/bin/env python3
"""
FINAL IMPROVED Hyperbolic CNN with Hybrid Models
Complete implementation with all improvements and hybrid comparisons
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime
import json
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class ImprovedPoincareBall:
    """
    Enhanced Poincaré Ball with numerical stability and better gradient flow
    """
    
    def __init__(self, c=1.0, eps=1e-7):
        self.c = c
        self.eps = eps
        self.max_norm = (1.0 / np.sqrt(c)) - eps
        
    def mobius_add(self, x, y):
        """Möbius addition with improved numerical stability"""
        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=self.eps)
        y2 = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=self.eps)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = ((1 + 2*self.c*xy + self.c*y2) * x + (1 - self.c*x2) * y)
        denom = (1 + 2*self.c*xy + self.c**2 * x2 * y2).clamp(min=self.eps)
        
        result = num / denom
        return self.project(result)
    
    def exp_map(self, v, p):
        """Exponential map with stable gradients"""
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        p_norm = torch.norm(p, dim=-1, keepdim=True).clamp(max=self.max_norm)
        
        lambda_p = 2 / (1 - self.c * p_norm**2 + self.eps)
        coeff = torch.tanh(lambda_p * v_norm / 2) / v_norm
        
        result = self.mobius_add(p, coeff * v)
        return self.project(result)
    
    def project(self, x):
        """Project points onto Poincaré ball with gradient-friendly clipping"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        
        # Soft projection for better gradients
        scale = torch.where(
            norm < self.max_norm * 0.999,  # Small margin for stability
            torch.ones_like(norm),
            (self.max_norm * 0.999) / norm
        )
        
        return x * scale


class HyperbolicLayer(nn.Module):
    """
    Hyperbolic layer with improved gradient flow and regularization
    """
    
    def __init__(self, in_features, out_features, c=1.0, dropout=0.1):
        super().__init__()
        self.poincare = ImprovedPoincareBall(c=c)
        
        # Tangent space parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Regularization layers
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with smaller values for stability
        nn.init.xavier_uniform_(self.weight, gain=0.5)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # Project to Poincaré ball
        x = self.poincare.project(x)
        
        # Linear transformation in tangent space
        output = F.linear(x, self.weight, self.bias)
        
        # Apply layer normalization and activation
        output = self.layer_norm(output)
        output = F.gelu(output)  # GELU activation
        output = self.dropout(output)
        
        # Project back to Poincaré ball
        return self.poincare.project(output)


class FinalImprovedHyperbolicCNN(nn.Module):
    """
    FINAL Improved Hyperbolic CNN with all enhancements
    - Multi-scale feature extraction
    - Attention mechanism
    - Residual connections
    - Ensemble-like predictions
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, c=1.0, dropout=0.2):
        super().__init__()
        
        # Multi-scale feature extraction (inspired by ensemble methods)
        self.feature_scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.8)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.8),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.8)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.8)
            )
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Hyperbolic layers with residual connections
        self.hyp_layer1 = HyperbolicLayer(hidden_dim, hidden_dim, c=c, dropout=dropout)
        self.hyp_layer2 = HyperbolicLayer(hidden_dim, hidden_dim // 2, c=c, dropout=dropout)
        self.hyp_layer3 = HyperbolicLayer(hidden_dim // 2, hidden_dim // 4, c=c, dropout=dropout)
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Skip connections for gradient flow
        self.skip1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.skip2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        # Multiple prediction heads (ensemble-like)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, num_classes),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Linear(hidden_dim, num_classes),
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Multi-scale feature extraction
        features = []
        for scale in self.feature_scales:
            features.append(scale(x))
        
        # Concatenate and fuse features
        x = torch.cat(features, dim=-1)
        x = self.fusion(x)
        
        # Apply self-attention
        x_att = x.unsqueeze(1)
        x_att, _ = self.attention(x_att, x_att, x_att)
        x = x + 0.1 * x_att.squeeze(1)  # Residual connection
        
        # Hyperbolic transformations with skip connections
        h1 = self.hyp_layer1(x)
        
        # Skip connection 1
        skip1_out = self.skip1(x)
        h2 = self.hyp_layer2(h1) + 0.1 * skip1_out
        
        # Skip connection 2
        skip2_out = self.skip2(h1)
        h3 = self.hyp_layer3(h2) + 0.1 * skip2_out
        
        # Multi-scale predictions
        pred1 = self.heads[0](h3)
        pred2 = self.heads[1](h2)
        pred3 = self.heads[2](h1)
        
        # Weighted ensemble of predictions
        weights = F.softmax(self.ensemble_weights, dim=0)
        output = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3
        
        return output


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Advanced focal loss with label smoothing for better generalization
    """
    
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Apply label smoothing
            targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Calculate focal loss
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Cross entropy with focal weight
        loss = -focal_weight * targets_smooth * log_probs
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t.unsqueeze(1)
        
        return loss.sum(dim=-1).mean()


class HybridModel:
    """
    Hybrid model combining Hyperbolic CNN with ensemble methods
    """
    
    def __init__(self, hyperbolic_model, ensemble_models, weights=None):
        self.hyperbolic_model = hyperbolic_model
        self.ensemble_models = ensemble_models
        self.weights = weights if weights else [1/len(ensemble_models)] * len(ensemble_models)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def predict(self, X):
        """
        Combine predictions from all models
        """
        predictions = []
        
        # Hyperbolic CNN prediction
        self.hyperbolic_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            hyp_output = self.hyperbolic_model(X_tensor)
            hyp_pred = F.softmax(hyp_output, dim=-1).cpu().numpy()
            predictions.append(hyp_pred)
        
        # Ensemble models predictions
        for model in self.ensemble_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
                # Convert to probabilities if needed
                if len(pred.shape) == 1:
                    pred = np.eye(3)[pred]
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred
        
        return np.argmax(weighted_pred, axis=1)


def enhanced_feature_engineering(df):
    """
    Comprehensive feature engineering matching top ensemble methods
    """
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Multi-timeframe features
    for window in [5, 10, 20, 50]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        df[f'volume_ma_{window}'] = df['Volume'].rolling(window).mean()
        
        # Price relative to moving averages
        df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']
        df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_ma_{window}']
    
    # MACD variations
    for fast, slow in [(12, 26), (5, 15), (10, 30)]:
        exp_fast = df['Close'].ewm(span=fast).mean()
        exp_slow = df['Close'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = exp_fast - exp_slow
        df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
    
    # RSI variations
    for period in [7, 14, 21, 28]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [10, 20, 30]:
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + 2 * std
        df[f'bb_lower_{period}'] = sma - 2 * std
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_width_{period}'] + 1e-10)
    
    # Volume indicators
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['volume_price'] = df['Volume'] * df['Close']
    
    # Price patterns
    df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-10)
    df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-10)
    df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    
    # Momentum
    for lag in [1, 3, 5, 10, 20]:
        df[f'return_{lag}d'] = df['Close'].pct_change(lag)
        df[f'volume_change_{lag}d'] = df['Volume'].pct_change(lag)
    
    # Support and Resistance
    for period in [10, 20, 50]:
        df[f'resistance_{period}'] = df['High'].rolling(period).max()
        df[f'support_{period}'] = df['Low'].rolling(period).min()
        df[f'sr_position_{period}'] = (df['Close'] - df[f'support_{period}']) / (
            df[f'resistance_{period}'] - df[f'support_{period}'] + 1e-10
        )
    
    # Statistical features
    df['skew_20'] = df['returns'].rolling(20).skew()
    df['kurtosis_20'] = df['returns'].rolling(20).kurt()
    df['mean_reversion'] = df['Close'] / df['sma_20'] - 1
    
    # Interaction features (what XGBoost finds automatically)
    df['rsi_volume'] = df['rsi_14'] * df['volume_ratio_20']
    df['macd_volatility'] = df['macd_12_26'] * df['volatility_20']
    df['bb_rsi'] = df['bb_position_20'] * df['rsi_14']
    df['volume_volatility'] = df['volume_ratio_20'] * df['volatility_20']
    
    return df


def create_labels_with_dynamic_threshold(df, base_threshold=0.015):
    """
    Create labels with dynamic thresholds based on market volatility
    """
    # Calculate future returns
    df['future_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['future_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['future_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # Weighted future return
    df['weighted_return'] = (
        0.5 * df['future_1d'] + 
        0.3 * df['future_3d'] + 
        0.2 * df['future_5d']
    )
    
    # Dynamic threshold based on recent volatility
    volatility = df['volatility_20'].fillna(df['returns'].std())
    dynamic_buy_threshold = base_threshold * (1 + volatility)
    dynamic_sell_threshold = -base_threshold * (1 + volatility)
    
    # Create labels with confirmation signals
    buy_condition = (
        (df['weighted_return'] > dynamic_buy_threshold) & 
        (df['rsi_14'] < 70) &
        (df['macd_hist_12_26'] > 0)
    )
    
    sell_condition = (
        (df['weighted_return'] < dynamic_sell_threshold) & 
        (df['rsi_14'] > 30) &
        (df['macd_hist_12_26'] < 0)
    )
    
    df['label'] = np.select(
        [buy_condition, sell_condition],
        [2, 0],
        default=1
    )
    
    return df


def train_improved_hyperbolic_cnn(X_train, y_train, X_val, y_val, epochs=150, patience=30):
    """
    Train the improved Hyperbolic CNN with best practices
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create model
    model = FinalImprovedHyperbolicCNN(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        num_classes=3,
        c=1.0,
        dropout=0.2
    ).to(device)
    
    # Calculate class weights for balanced training
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Loss function with label smoothing
    criterion = FocalLossWithLabelSmoothing(
        gamma=2.0, 
        alpha=class_weights, 
        smoothing=0.1
    )
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create DataLoader with shuffling
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    
    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("\nTraining Improved Hyperbolic CNN...")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = (val_pred == y_val_t).float().mean()
        
        # Update learning rate
        scheduler.step()
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    return model


def train_ensemble_models(X_train, y_train, X_val, y_val):
    """
    Train ensemble models for comparison and hybrid
    """
    models = {}
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    models['XGBoost'] = xgb_model
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    models['LightGBM'] = lgb_model
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    return models


def calculate_financial_metrics(returns, risk_free_rate=0.02):
    """
    Calculate comprehensive financial performance metrics
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return {
            'total_return': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'max_drawdown': 0, 'calmar_ratio': 0, 'win_rate': 0,
            'profit_factor': 0, 'volatility': 0
        }
    
    daily_returns = np.diff(returns) / returns[:-1]
    total_return = (returns[-1] / returns[0] - 1) * 100
    
    # Risk metrics
    if len(daily_returns) > 0:
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(downside_returns) + 1e-8)
        else:
            sortino_ratio = 0
        
        cumulative = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
        
        calmar_ratio = total_return / (abs(max_drawdown) + 1e-8)
        win_rate = (np.sum(daily_returns > 0) / len(daily_returns) * 100)
        
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        profit_factor = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 else 0
        
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
    else:
        sharpe_ratio = sortino_ratio = max_drawdown = calmar_ratio = 0
        win_rate = profit_factor = volatility = 0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'volatility': float(volatility)
    }


def backtest_strategy(df, predictions, initial_capital=10000):
    """
    Backtest trading strategy with risk management
    """
    df_bt = df.iloc[-len(predictions):].copy()
    df_bt['prediction'] = predictions
    
    capital = initial_capital
    position = 0
    entry_price = 0
    portfolio_values = [initial_capital]
    num_trades = 0
    
    # Risk management parameters
    stop_loss = 0.03
    take_profit = 0.06
    position_size = 0.25
    transaction_cost = 0.002  # 0.2%
    
    for i in range(1, len(df_bt)):
        current_price = df_bt['Close'].iloc[i]
        signal = df_bt['prediction'].iloc[i]
        
        # Check stop loss and take profit
        if position > 0 and entry_price > 0:
            returns = (current_price - entry_price) / entry_price
            
            if returns <= -stop_loss or returns >= take_profit:
                # Exit position
                capital += position * current_price * (1 - transaction_cost)
                position = 0
                entry_price = 0
                num_trades += 1
        
        # New signals
        if signal == 2 and position == 0:  # BUY
            invest_amount = capital * position_size
            position = invest_amount / (current_price * (1 + transaction_cost))
            capital -= invest_amount
            entry_price = current_price
            
        elif signal == 0 and position > 0:  # SELL
            capital += position * current_price * (1 - transaction_cost)
            position = 0
            entry_price = 0
            num_trades += 1
        
        # Update portfolio value
        total_value = capital + (position * current_price if position > 0 else 0)
        portfolio_values.append(total_value)
    
    # Close any open position
    if position > 0:
        capital += position * df_bt['Close'].iloc[-1] * (1 - transaction_cost)
        portfolio_values[-1] = capital
        num_trades += 1
    
    return portfolio_values, num_trades


def main():
    """
    Main execution: Train and compare all models including hybrids
    """
    print("="*80)
    print("FINAL HYPERBOLIC CNN WITH HYBRID MODELS - COMPLETE COMPARISON")
    print("="*80)
    
    # Fetch data
    print("\n1. Fetching cryptocurrency data...")
    ticker = yf.Ticker('BTC-USD')
    df = ticker.history(period='2y')
    print(f"   Data shape: {df.shape}")
    
    # Feature engineering
    print("\n2. Applying enhanced feature engineering...")
    df = enhanced_feature_engineering(df)
    df = create_labels_with_dynamic_threshold(df)
    df = df.dropna()
    print(f"   Total features: {len(df.columns)}")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in 
                   ['label', 'future_1d', 'future_3d', 'future_5d', 'weighted_return']]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"   Final dataset: {X.shape}")
    print(f"   Original class distribution: {np.bincount(y)}")
    
    # Apply ADASYN balancing
    print("\n3. Applying ADASYN balancing...")
    adasyn = ADASYN(
        sampling_strategy='auto',
        random_state=42,
        n_neighbors=5,
        n_jobs=-1
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
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    # Scale data using RobustScaler
    print("\n4. Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # For backtesting
    X_original = scaler.transform(df[feature_cols].iloc[-len(y_test):].values)
    
    print(f"   Train: {X_train_scaled.shape}")
    print(f"   Val:   {X_val_scaled.shape}")
    print(f"   Test:  {X_test_scaled.shape}")
    
    # Train improved Hyperbolic CNN
    print("\n5. Training Improved Hyperbolic CNN...")
    print("-" * 50)
    hyperbolic_model = train_improved_hyperbolic_cnn(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        epochs=150,
        patience=30
    )
    
    # Train ensemble models
    print("\n6. Training ensemble models for comparison...")
    ensemble_models = train_ensemble_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val
    )
    
    # Create hybrid models
    print("\n7. Creating hybrid models...")
    
    # Hybrid 1: Hyperbolic + XGBoost (70-30)
    hybrid1 = HybridModel(
        hyperbolic_model, 
        [ensemble_models['XGBoost']],
        weights=[0.7, 0.3]
    )
    
    # Hybrid 2: Hyperbolic + LightGBM (70-30)
    hybrid2 = HybridModel(
        hyperbolic_model,
        [ensemble_models['LightGBM']],
        weights=[0.7, 0.3]
    )
    
    # Hybrid 3: Hyperbolic + All Ensembles (40-20-20-20)
    hybrid3 = HybridModel(
        hyperbolic_model,
        [ensemble_models['XGBoost'], ensemble_models['LightGBM'], ensemble_models['RandomForest']],
        weights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Evaluate all models
    print("\n8. Evaluating all models...")
    print("-" * 50)
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate Improved Hyperbolic CNN
    hyperbolic_model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        outputs = hyperbolic_model(X_test_t)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        
        X_orig_t = torch.FloatTensor(X_original).to(device)
        orig_outputs = hyperbolic_model(X_orig_t)
        y_pred_trading = torch.argmax(orig_outputs, dim=1).cpu().numpy()
    
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
        y_pred = hybrid.predict(X_test_scaled)
        y_pred_trading = hybrid.predict(X_original)
        
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
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Sharpe', ascending=False)
    
    # Display final results
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    print("\n📊 Classification Performance:")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']].round(4).to_string(index=False))
    
    print("\n💰 Trading Performance:")
    print(results_df[['Model', 'Return', 'Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'Trades']].round(3).to_string(index=False))
    
    # Highlight best performers
    print("\n" + "="*80)
    print("TOP PERFORMERS BY METRIC")
    print("="*80)
    
    metrics_to_check = ['Accuracy', 'Sharpe', 'Return', 'Calmar']
    for metric in metrics_to_check:
        if metric in ['MaxDD']:  # Lower is better
            best_idx = results_df[metric].idxmax()
        else:  # Higher is better
            best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_value = results_df.loc[best_idx, metric]
        print(f"{metric:12s}: {best_model:40s} = {best_value:.3f}")
    
    # Save results
    results_df.to_csv('final_model_comparison.csv', index=False)
    print("\n✅ Results saved to final_model_comparison.csv")
    
    # Save best model
    torch.save({
        'model_state_dict': hyperbolic_model.state_dict(),
        'model_config': {
            'input_dim': X_train_scaled.shape[1],
            'hidden_dim': 256,
            'num_classes': 3,
            'c': 1.0,
            'dropout': 0.2
        },
        'scaler': scaler,
        'feature_cols': feature_cols,
        'results': results_df.to_dict()
    }, 'best_hyperbolic_model.pth')
    
    print("✅ Best model saved to best_hyperbolic_model.pth")
    
    return results_df, hyperbolic_model


if __name__ == "__main__":
    results_df, model = main()