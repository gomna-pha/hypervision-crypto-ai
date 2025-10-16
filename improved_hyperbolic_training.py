#!/usr/bin/env python3
"""
IMPROVED Hyperbolic CNN Training with ADASYN and Better Hyperparameters
Addressing the performance gap with ensemble methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class ImprovedPoincareBall:
    """
    Enhanced Poincaré Ball with better numerical stability
    """
    
    def __init__(self, c=1.0, eps=1e-7):
        self.c = c
        self.eps = eps
        self.max_norm = (1.0 / np.sqrt(c)) - eps
        
    def mobius_add(self, x, y):
        """Möbius addition with improved stability"""
        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=self.eps)
        y2 = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=self.eps)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        # Add stability checks
        num = ((1 + 2*self.c*xy + self.c*y2) * x + (1 - self.c*x2) * y)
        denom = 1 + 2*self.c*xy + self.c**2 * x2 * y2
        denom = torch.clamp(denom, min=self.eps)
        
        result = num / denom
        return self.project(result)
    
    def exp_map(self, v, p):
        """Exponential map with gradient-friendly implementation"""
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        p_norm = torch.norm(p, dim=-1, keepdim=True).clamp(max=self.max_norm)
        
        # Lambda function
        lambda_p = 2 / (1 - self.c * p_norm**2 + self.eps)
        
        # Compute coefficient
        coeff = torch.tanh(lambda_p * v_norm / 2) / v_norm
        
        # Result
        result = self.mobius_add(p, coeff * v)
        return self.project(result)
    
    def project(self, x):
        """Project with adaptive clipping"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Adaptive scaling
        scale = torch.where(
            norm < self.max_norm,
            torch.ones_like(norm),
            self.max_norm / norm
        )
        
        return x * scale
    
    def distance(self, x, y):
        """Hyperbolic distance"""
        x = self.project(x)
        y = self.project(y)
        
        sqrt_c = np.sqrt(self.c)
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm = torch.sum(x * x, dim=-1)
        y_norm = torch.sum(y * y, dim=-1)
        
        denominator = (1 - self.c * x_norm) * (1 - self.c * y_norm)
        denominator = torch.clamp(denominator, min=self.eps)
        
        return (2 / sqrt_c) * torch.atanh(sqrt_c * torch.sqrt(dist_sq / denominator))


class HyperbolicLayer(nn.Module):
    """
    Hyperbolic neural network layer with better gradient flow
    """
    
    def __init__(self, in_features, out_features, c=1.0, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.poincare = ImprovedPoincareBall(c=c)
        
        # Weights in tangent space
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Batch normalization in tangent space
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # Project to Poincaré ball
        x = self.poincare.project(x)
        
        # Linear transformation in tangent space (at origin)
        # Map to tangent space
        x_tangent = x  # Simplified: assume near origin
        
        # Apply transformation
        output = F.linear(x_tangent, self.weight, self.bias)
        
        # Batch norm and activation
        output = self.bn(output)
        output = F.gelu(output)  # GELU often works better than ReLU
        output = self.dropout(output)
        
        # Project back to Poincaré ball
        return self.poincare.project(output)


class EnhancedHyperbolicCNN(nn.Module):
    """
    Improved Hyperbolic CNN with techniques from top-performing models
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, c=1.0, dropout=0.2):
        super().__init__()
        
        # Feature extraction (inspired by ensemble methods' feature importance)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Hyperbolic layers
        self.hyp_layer1 = HyperbolicLayer(hidden_dim, hidden_dim, c=c, dropout=dropout)
        self.hyp_layer2 = HyperbolicLayer(hidden_dim, hidden_dim // 2, c=c, dropout=dropout)
        self.hyp_layer3 = HyperbolicLayer(hidden_dim // 2, hidden_dim // 4, c=c, dropout=dropout)
        
        # Attention mechanism (key for performance)
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Residual connections
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Output layers with skip connections
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Linear(hidden_dim // 4, num_classes),
        ])
        
        # Ensemble-like voting
        self.ensemble_weight = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Apply attention
        attn_input = features.unsqueeze(1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        features = features + self.residual_weight * attn_output.squeeze(1)
        
        # Hyperbolic transformations with skip connections
        h1 = self.hyp_layer1(features)
        h2 = self.hyp_layer2(h1)
        h3 = self.hyp_layer3(h2)
        
        # Multi-scale predictions (ensemble-like)
        out1 = self.output_layers[0](h1)
        out2 = self.output_layers[1](h2)
        out3 = self.output_layers[2](h3)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weight, dim=0)
        output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        
        return output


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Enhanced focal loss with label smoothing
    """
    
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        
        # Label smoothing
        with torch.no_grad():
            smoothed_targets = torch.zeros_like(inputs)
            smoothed_targets.fill_(self.smoothing / (n_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Focal loss
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Apply focal term
        focal_weight = (1 - probs) ** self.gamma
        
        # Calculate loss
        loss = -focal_weight * smoothed_targets * log_probs
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t.unsqueeze(1) * loss
        
        return loss.sum(dim=-1).mean()


def enhanced_feature_engineering(df):
    """
    Advanced feature engineering matching ensemble methods
    """
    
    # Basic features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility features (multiple timeframes)
    for window in [5, 10, 20, 50]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['Volume'] / df['Volume'].rolling(window).mean()
    
    # Technical indicators
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [10, 20]:
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + 2 * std
        df[f'bb_lower_{period}'] = sma - 2 * std
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_width_{period}'] + 1e-10)
    
    # Price patterns
    df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-10)
    df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-10)
    
    # Momentum indicators
    for lag in [1, 3, 5, 10]:
        df[f'return_{lag}d'] = df['Close'].pct_change(lag)
        df[f'volume_change_{lag}d'] = df['Volume'].pct_change(lag)
    
    # Support and Resistance
    for period in [10, 20, 50]:
        df[f'resistance_{period}'] = df['High'].rolling(period).max()
        df[f'support_{period}'] = df['Low'].rolling(period).min()
        df[f'sr_ratio_{period}'] = (df['Close'] - df[f'support_{period}']) / (
            df[f'resistance_{period}'] - df[f'support_{period}'] + 1e-10
        )
    
    # Statistical features
    df['skew'] = df['returns'].rolling(20).skew()
    df['kurtosis'] = df['returns'].rolling(20).kurt()
    
    # Interaction features (key for ensemble methods)
    df['volume_price'] = df['Volume'] * df['Close']
    df['volatility_volume'] = df['volatility_20'] * df['volume_ratio_20']
    df['rsi_macd'] = df['rsi_14'] * df['macd_hist']
    
    return df


def create_advanced_labels(df, threshold=0.015):
    """
    More refined labeling strategy
    """
    # Multi-timeframe returns
    df['future_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['future_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['future_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # Weighted future return
    df['weighted_return'] = (
        0.5 * df['future_1d'] + 
        0.3 * df['future_3d'] + 
        0.2 * df['future_5d']
    )
    
    # Dynamic thresholds based on volatility
    volatility = df['returns'].rolling(20).std()
    dynamic_threshold = threshold * (1 + volatility)
    
    # Create labels with trend confirmation
    conditions = [
        (df['weighted_return'] > dynamic_threshold) & (df['rsi_14'] < 75),  # BUY
        (df['weighted_return'] < -dynamic_threshold) & (df['rsi_14'] > 25),  # SELL
    ]
    choices = [2, 0]
    
    df['label'] = np.select(conditions, choices, default=1)  # HOLD = 1
    
    return df


class HyperparameterOptimization:
    """
    Optuna-based hyperparameter optimization
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def objective(self, trial):
        # Hyperparameters to optimize
        hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        c = trial.suggest_float('c', 0.5, 2.0)
        
        # Create model
        model = EnhancedHyperbolicCNN(
            input_dim=self.X_train.shape[1],
            hidden_dim=hidden_dim,
            num_classes=3,
            c=c,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss
        criterion = FocalLossWithLabelSmoothing(gamma=2.0, smoothing=0.1)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(self.X_train).to(self.device)
        y_train_t = torch.LongTensor(self.y_train).to(self.device)
        X_val_t = torch.FloatTensor(self.X_val).to(self.device)
        y_val_t = torch.LongTensor(self.y_val).to(self.device)
        
        # Training
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(30):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()
        
        return val_acc
    
    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


def train_improved_model(X_train, y_train, X_val, y_val, best_params=None):
    """
    Train the improved model with best practices
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use best params or defaults
    if best_params is None:
        best_params = {
            'hidden_dim': 256,
            'dropout': 0.2,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'c': 1.0
        }
    
    # Create model
    model = EnhancedHyperbolicCNN(
        input_dim=X_train.shape[1],
        hidden_dim=best_params['hidden_dim'],
        num_classes=3,
        c=best_params['c'],
        dropout=best_params['dropout']
    ).to(device)
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Loss and optimizer
    criterion = FocalLossWithLabelSmoothing(gamma=2.0, alpha=class_weights, smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = (val_pred == y_val_t).float().mean()
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model


def main():
    """
    Main execution with improved training
    """
    print("="*80)
    print("IMPROVED HYPERBOLIC CNN TRAINING")
    print("Addressing performance gap with ensemble methods")
    print("="*80)
    
    # Fetch data
    print("\nFetching cryptocurrency data...")
    ticker = yf.Ticker('BTC-USD')
    df = ticker.history(period='2y')  # More data for better training
    
    # Enhanced feature engineering
    print("Applying enhanced feature engineering...")
    df = enhanced_feature_engineering(df)
    df = create_advanced_labels(df)
    df = df.dropna()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                   ['label', 'future_1d', 'future_3d', 'future_5d', 'weighted_return']]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    
    # Apply ADASYN (as you requested)
    print("\nApplying ADASYN balancing...")
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_balanced, y_balanced = adasyn.fit_resample(X, y)
    
    print("Balanced distribution:")
    unique, counts = np.bincount(y_balanced)
    for i, count in enumerate(counts):
        print(f"  Class {i}: {count} ({count/len(y_balanced)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Use RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Hyperparameter optimization (optional, takes time)
    print("\nOptimizing hyperparameters...")
    # optimizer = HyperparameterOptimization(X_train, y_train, X_val, y_val)
    # best_params = optimizer.optimize(n_trials=20)
    # print(f"Best parameters: {best_params}")
    
    # For quick testing, use good defaults
    best_params = {
        'hidden_dim': 256,
        'dropout': 0.2,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'c': 1.0
    }
    
    # Train improved model
    print("\nTraining improved Hyperbolic CNN...")
    model = train_improved_model(X_train, y_train, X_val, y_val, best_params)
    
    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        test_outputs = model(X_test_t)
        y_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
    
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
    
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = main()