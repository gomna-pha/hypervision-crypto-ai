#!/usr/bin/env python3
"""
Comprehensive Model Comparison Framework
Compares Hyperbolic CNN with Traditional Trading Models
Preserves all methodologies while adding comparative analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GRU
from keras.optimizers import Adam as KerasAdam
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class PoincareBall:
    """Your Poincaré Ball model - preserved exactly"""
    
    def __init__(self, c=1.0, eps=1e-5):
        self.c = c
        self.eps = eps
        
    def mobius_add(self, x, y):
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = ((1 + 2 * self.c * xy + self.c * y_norm_sq) * x + 
               (1 - self.c * x_norm_sq) * y)
        denom = 1 + 2 * self.c * xy + (self.c ** 2) * x_norm_sq * y_norm_sq
        
        return num / torch.clamp(denom, min=self.eps)
    
    def project(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        
        max_norm = 1.0 / np.sqrt(self.c) - self.eps
        max_norm_tensor = torch.tensor(max_norm, device=x.device, dtype=x.dtype)
        
        scale = torch.where(
            norm < max_norm_tensor,
            torch.ones_like(norm),
            max_norm_tensor / norm
        )
        
        return x * scale


class HyperbolicCNN(nn.Module):
    """Your Hyperbolic CNN model - preserved exactly"""
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//4),
            nn.Dropout(dropout),
        )
        
        self.poincare = PoincareBall(c=1.0)
        self.output = nn.Linear(hidden_dim//4, num_classes)
        self.attention_weight = nn.Parameter(torch.randn(1, input_dim))
        
    def forward(self, x):
        attention_scores = torch.sigmoid(torch.matmul(x, self.attention_weight.t()))
        x = x * attention_scores
        x = self.features(x)
        x = self.poincare.project(x)
        return self.output(x)


class StandardCNN:
    """Standard CNN for comparison"""
    
    def build_model(self, input_shape, num_classes=3):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(input_shape, 1)),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=KerasAdam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class LSTMModel:
    """LSTM model for time series comparison"""
    
    def build_model(self, input_shape, num_classes=3):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(input_shape, 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=KerasAdam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class GRUModel:
    """GRU model for comparison"""
    
    def build_model(self, input_shape, num_classes=3):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(input_shape, 1)),
            Dropout(0.2),
            GRU(64),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=KerasAdam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class TransformerModel(nn.Module):
    """Transformer-based model for comparison"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, num_classes=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x.transpose(0, 1))
        x = x.mean(dim=0)  # Global average pooling
        return self.fc(x)


def calculate_financial_metrics(returns, risk_free_rate=0.02):
    """Your financial metrics calculation - preserved exactly"""
    
    returns = np.array(returns)
    daily_returns = np.diff(returns) / returns[:-1]
    
    if len(daily_returns) == 0:
        return {
            'total_return': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'max_drawdown': 0, 'calmar_ratio': 0, 'win_rate': 0,
            'profit_factor': 0, 'volatility': 0
        }
    
    total_return = (returns[-1] / returns[0] - 1) * 100
    
    # Sharpe Ratio
    excess_returns = daily_returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(downside_returns) + 1e-8)
    else:
        sortino_ratio = 0
    
    # Maximum Drawdown
    cumulative = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = total_return / (abs(max_drawdown) + 1e-8)
    
    # Win Rate
    win_rate = (np.sum(daily_returns > 0) / len(daily_returns) * 100) if len(daily_returns) > 0 else 0
    
    # Profit Factor
    gains = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
        profit_factor = np.sum(gains) / np.abs(np.sum(losses))
    else:
        profit_factor = 0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'volatility': float(np.std(daily_returns) * np.sqrt(252) * 100) if len(daily_returns) > 0 else 0
    }


def backtest_with_risk_management(df, predictions, initial_capital=10000):
    """Your backtesting function - preserved exactly"""
    
    df = df.iloc[-len(predictions):].copy()
    df['prediction'] = predictions
    
    capital = initial_capital
    position = 0
    entry_price = 0
    portfolio = [initial_capital]
    trades = []
    
    # Risk parameters (your exact settings)
    stop_loss = 0.03
    take_profit = 0.06
    position_size = 0.25
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        signal = df['prediction'].iloc[i]
        
        # Check exit conditions
        if position > 0 and entry_price > 0:
            returns = (price - entry_price) / entry_price
            
            if returns <= -stop_loss or returns >= take_profit:
                capital += position * price * 0.998
                trades.append(returns)
                position = 0
                entry_price = 0
        
        # New trades
        if signal == 2 and position == 0:  # BUY
            invest = capital * position_size
            position = invest / price * 0.998
            capital -= invest
            entry_price = price
            
        elif signal == 0 and position > 0:  # SELL
            capital += position * price * 0.998
            if entry_price > 0:
                trades.append((price - entry_price) / entry_price)
            position = 0
            entry_price = 0
        
        portfolio.append(capital + position * price)
    
    # Close final position
    if position > 0:
        capital += position * df['Close'].iloc[-1] * 0.998
        portfolio[-1] = capital
    
    return portfolio, trades


class ModelComparison:
    """Main comparison framework"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def add_traditional_models(self):
        """Add all traditional models for comparison"""
        
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', random_state=42, use_label_encoder=False
        )
        
        self.models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multiclass', random_state=42, verbose=-1
        )
        
        self.models['SVM'] = SVC(
            kernel='rbf', C=1.0, gamma='scale', random_state=42
        )
        
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        
        self.models['MLP Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=500, random_state=42
        )
        
        self.models['Logistic Regression'] = LogisticRegression(
            multi_class='multinomial', max_iter=1000, random_state=42
        )
        
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10, random_state=42
        )
        
        self.models['Naive Bayes'] = GaussianNB()
        
    def train_hyperbolic_cnn(self, X_train, y_train, X_val, y_val):
        """Train your Hyperbolic CNN model"""
        
        model = HyperbolicCNN(
            input_dim=X_train.shape[1],
            hidden_dim=128,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_t).float().mean()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break
        
        model.load_state_dict(best_model_state)
        return model
    
    def train_lstm_model(self, X_train, y_train, X_val, y_val):
        """Train LSTM model"""
        
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        lstm = LSTMModel()
        model = lstm.build_model(X_train.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=50, batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model
    
    def train_standard_cnn(self, X_train, y_train, X_val, y_val):
        """Train standard CNN"""
        
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        cnn = StandardCNN()
        model = cnn.build_model(X_train.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=50, batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model
    
    def train_transformer(self, X_train, y_train, X_val, y_val):
        """Train Transformer model"""
        
        model = TransformerModel(
            input_dim=X_train.shape[1],
            d_model=128,
            nhead=8,
            num_layers=2,
            num_classes=3
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_t).float().mean()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
        
        model.load_state_dict(best_model_state)
        return model
    
    def evaluate_model(self, model, model_name, X_test, y_test, X_original, main_df, scaler):
        """Evaluate a single model"""
        
        # Get predictions based on model type
        if model_name == 'Hyperbolic CNN (Ours)':
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(self.device)
                outputs = model(X_test_t)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # For trading
                X_orig_t = torch.FloatTensor(X_original).to(self.device)
                orig_outputs = model(X_orig_t)
                y_pred_trading = torch.argmax(orig_outputs, dim=1).cpu().numpy()
                
        elif model_name == 'Transformer':
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(self.device)
                outputs = model(X_test_t)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                
                X_orig_t = torch.FloatTensor(X_original).to(self.device)
                orig_outputs = model(X_orig_t)
                y_pred_trading = torch.argmax(orig_outputs, dim=1).cpu().numpy()
                
        elif model_name in ['LSTM', 'Standard CNN', 'GRU']:
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            y_pred_proba = model.predict(X_test_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            X_orig_reshaped = X_original.reshape((X_original.shape[0], X_original.shape[1], 1))
            y_pred_trading_proba = model.predict(X_orig_reshaped)
            y_pred_trading = np.argmax(y_pred_trading_proba, axis=1)
            
        else:  # Traditional ML models
            y_pred = model.predict(X_test)
            y_pred_trading = model.predict(X_original)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Run backtesting
        portfolio, trades = backtest_with_risk_management(main_df, y_pred_trading)
        
        # Calculate financial metrics
        financial_metrics = calculate_financial_metrics(portfolio)
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_return': financial_metrics['total_return'],
            'sharpe_ratio': financial_metrics['sharpe_ratio'],
            'sortino_ratio': financial_metrics['sortino_ratio'],
            'calmar_ratio': financial_metrics['calmar_ratio'],
            'max_drawdown': financial_metrics['max_drawdown'],
            'win_rate': financial_metrics['win_rate'],
            'profit_factor': financial_metrics['profit_factor'],
            'volatility': financial_metrics['volatility'],
            'num_trades': len(trades)
        }
    
    def run_comparison(self, X_train, y_train, X_val, y_val, X_test, y_test, X_original, main_df, scaler):
        """Run full comparison"""
        
        results = []
        
        print("="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # 1. Train and evaluate Hyperbolic CNN (Your model)
        print("\n1. Training Hyperbolic CNN (Your Model)...")
        hyperbolic_cnn = self.train_hyperbolic_cnn(X_train, y_train, X_val, y_val)
        result = self.evaluate_model(hyperbolic_cnn, 'Hyperbolic CNN (Ours)', 
                                    X_test, y_test, X_original, main_df, scaler)
        results.append(result)
        print(f"   ✓ Accuracy: {result['accuracy']:.4f}, Sharpe: {result['sharpe_ratio']:.3f}")
        
        # 2. Train deep learning models
        print("\n2. Training LSTM...")
        lstm_model = self.train_lstm_model(X_train, y_train, X_val, y_val)
        result = self.evaluate_model(lstm_model, 'LSTM', 
                                    X_test, y_test, X_original, main_df, scaler)
        results.append(result)
        print(f"   ✓ Accuracy: {result['accuracy']:.4f}, Sharpe: {result['sharpe_ratio']:.3f}")
        
        print("\n3. Training Standard CNN...")
        cnn_model = self.train_standard_cnn(X_train, y_train, X_val, y_val)
        result = self.evaluate_model(cnn_model, 'Standard CNN', 
                                    X_test, y_test, X_original, main_df, scaler)
        results.append(result)
        print(f"   ✓ Accuracy: {result['accuracy']:.4f}, Sharpe: {result['sharpe_ratio']:.3f}")
        
        print("\n4. Training Transformer...")
        transformer = self.train_transformer(X_train, y_train, X_val, y_val)
        result = self.evaluate_model(transformer, 'Transformer', 
                                    X_test, y_test, X_original, main_df, scaler)
        results.append(result)
        print(f"   ✓ Accuracy: {result['accuracy']:.4f}, Sharpe: {result['sharpe_ratio']:.3f}")
        
        # 3. Train traditional ML models
        self.add_traditional_models()
        
        for i, (name, model) in enumerate(self.models.items(), 5):
            print(f"\n{i}. Training {name}...")
            model.fit(X_train, y_train)
            result = self.evaluate_model(model, name, 
                                        X_test, y_test, X_original, main_df, scaler)
            results.append(result)
            print(f"   ✓ Accuracy: {result['accuracy']:.4f}, Sharpe: {result['sharpe_ratio']:.3f}")
        
        return pd.DataFrame(results)


def create_comparison_visualizations(results_df):
    """Create comprehensive comparison charts"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    
    # Sort by accuracy for better visualization
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # 1. Accuracy Comparison
    ax = axes[0, 0]
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df['model']]
    bars = ax.bar(range(len(results_df)), results_df['accuracy'], color=colors)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_title('Classification Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.axhline(y=0.778, color='r', linestyle='--', label='Your Model (77.8%)')
    
    # Add value labels on bars
    for bar, val in zip(bars, results_df['accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom')
    
    # 2. Sharpe Ratio Comparison
    ax = axes[0, 1]
    results_df_sorted = results_df.sort_values('sharpe_ratio', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['sharpe_ratio'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Sharpe Ratio (Risk-Adjusted Returns)', fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(y=3.133, color='r', linestyle='--', label='Your Model (3.133)')
    
    for bar, val in zip(bars, results_df_sorted['sharpe_ratio']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # 3. Total Return Comparison
    ax = axes[0, 2]
    results_df_sorted = results_df.sort_values('total_return', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#e74c3c' if r < 0 else '#3498db' 
              for m, r in zip(results_df_sorted['model'], results_df_sorted['total_return'])]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['total_return'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Total Return (%)', fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=7.54, color='r', linestyle='--', label='Your Model (7.54%)')
    
    for bar, val in zip(bars, results_df_sorted['total_return']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # 4. Max Drawdown Comparison (lower is better)
    ax = axes[1, 0]
    results_df_sorted = results_df.sort_values('max_drawdown', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['max_drawdown'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Maximum Drawdown (Lower is Better)', fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.axhline(y=-0.96, color='r', linestyle='--', label='Your Model (-0.96%)')
    
    for bar, val in zip(bars, results_df_sorted['max_drawdown']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='top' if val < 0 else 'bottom')
    
    # 5. Sortino Ratio Comparison
    ax = axes[1, 1]
    results_df_sorted = results_df.sort_values('sortino_ratio', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['sortino_ratio'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Sortino Ratio (Downside Risk)', fontweight='bold')
    ax.set_ylabel('Sortino Ratio')
    ax.axhline(y=5.675, color='r', linestyle='--', label='Your Model (5.675)')
    
    for bar, val in zip(bars, results_df_sorted['sortino_ratio']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # 6. Calmar Ratio Comparison
    ax = axes[1, 2]
    results_df_sorted = results_df.sort_values('calmar_ratio', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['calmar_ratio'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Calmar Ratio (Return/Drawdown)', fontweight='bold')
    ax.set_ylabel('Calmar Ratio')
    ax.axhline(y=7.837, color='r', linestyle='--', label='Your Model (7.837)')
    
    for bar, val in zip(bars, results_df_sorted['calmar_ratio']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # 7. F1 Score Comparison
    ax = axes[2, 0]
    results_df_sorted = results_df.sort_values('f1_score', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['f1_score'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('F1 Score', fontweight='bold')
    ax.set_ylabel('F1 Score')
    
    for bar, val in zip(bars, results_df_sorted['f1_score']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 8. Win Rate Comparison
    ax = axes[2, 1]
    results_df_sorted = results_df.sort_values('win_rate', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['win_rate'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Win Rate (%)', fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    
    for bar, val in zip(bars, results_df_sorted['win_rate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # 9. Number of Trades
    ax = axes[2, 2]
    results_df_sorted = results_df.sort_values('num_trades', ascending=False)
    colors = ['#2ecc71' if m == 'Hyperbolic CNN (Ours)' else '#3498db' for m in results_df_sorted['model']]
    bars = ax.bar(range(len(results_df_sorted)), results_df_sorted['num_trades'], color=colors)
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels(results_df_sorted['model'], rotation=45, ha='right')
    ax.set_title('Number of Trades', fontweight='bold')
    ax.set_ylabel('Trades')
    
    for bar, val in zip(bars, results_df_sorted['num_trades']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Main execution"""
    
    print("="*80)
    print("MODEL COMPARISON FRAMEWORK")
    print("Comparing Hyperbolic CNN with Traditional Trading Models")
    print("="*80)
    
    # Your exact data preparation process
    from model_comparison_framework import enhanced_feature_engineering
    
    # Fetch data
    print("\nFetching cryptocurrency data...")
    ticker = yf.Ticker('BTC-USD')
    df = ticker.history(period='1y')
    
    # Feature engineering (your exact process)
    df = enhanced_feature_engineering(df)
    
    # Create labels (your exact process)
    df['return_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['weighted_return'] = df['return_1d'] * 0.7 + df['return_3d'] * 0.3
    
    conditions = [
        (df['weighted_return'] > 0.02) & (df['rsi'] < 70),  # BUY
        (df['weighted_return'] < -0.02) & (df['rsi'] > 30),  # SELL
    ]
    choices = [2, 0]
    df['label'] = np.select(conditions, choices, default=1)
    
    df = df.dropna()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                   ['label', 'return_1d', 'return_3d', 'weighted_return']]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Apply SMOTE (your exact process)
    print("\nApplying SMOTE balancing...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y))-1))
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Original data for backtesting
    X_original = scaler.transform(df[feature_cols].iloc[-len(y_test):].values)
    
    # Run comparison
    comparison = ModelComparison()
    results_df = comparison.run_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        X_original, df, scaler
    )
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    print("\n📊 Classification Performance:")
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    print("\n💰 Trading Performance:")
    print(results_df[['model', 'total_return', 'sharpe_ratio', 'sortino_ratio', 
                      'calmar_ratio', 'max_drawdown']].to_string(index=False))
    
    print("\n📈 Risk Metrics:")
    print(results_df[['model', 'win_rate', 'profit_factor', 'volatility', 
                      'num_trades']].to_string(index=False))
    
    # Highlight your model's superiority
    print("\n" + "="*80)
    print("YOUR MODEL'S RANKING")
    print("="*80)
    
    your_model = results_df[results_df['model'] == 'Hyperbolic CNN (Ours)'].iloc[0]
    
    for metric in ['accuracy', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return']:
        rank = (results_df[metric] >= your_model[metric]).sum()
        print(f"{metric.replace('_', ' ').title()}: Rank {rank}/{len(results_df)}")
    
    # Create visualizations
    fig = create_comparison_visualizations(results_df)
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✅ Results saved to model_comparison_results.csv")
    
    # Generate publication-ready table
    publication_table = results_df[['model', 'accuracy', 'total_return', 'sharpe_ratio', 
                                   'max_drawdown', 'calmar_ratio']].round(3)
    publication_table.to_latex('comparison_table.tex', index=False)
    print("✅ LaTeX table saved to comparison_table.tex")
    
    return results_df


def enhanced_feature_engineering(df):
    """Your feature engineering - preserved"""
    
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + 2 * bb_std
    df['bb_lower'] = df['sma_20'] - 2 * bb_std
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


if __name__ == "__main__":
    results = main()