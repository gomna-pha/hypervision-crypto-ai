#!/usr/bin/env python3
"""
ADVANCED MODEL TRAINING WITH BALANCING AND REGULARIZATION
Addresses: Class imbalance, Overfitting, Underfitting

Author: GOMNA Research Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: CLASS BALANCING TECHNIQUES
# ============================================================================

class BalancedTradingDataset(Dataset):
    """
    Custom dataset with built-in balancing techniques
    Handles imbalanced Buy/Hold/Sell signals
    """
    
    def __init__(self, data, labels, balance_method='smote'):
        """
        Args:
            data: Feature matrix
            labels: Trading signals (0=Sell, 1=Hold, 2=Buy)
            balance_method: 'smote', 'adasyn', 'weighted', 'focal'
        """
        self.balance_method = balance_method
        
        # Apply balancing
        if balance_method == 'smote':
            self.data, self.labels = self.apply_smote(data, labels)
        elif balance_method == 'adasyn':
            self.data, self.labels = self.apply_adasyn(data, labels)
        elif balance_method == 'undersample':
            self.data, self.labels = self.apply_undersampling(data, labels)
        elif balance_method == 'oversample':
            self.data, self.labels = self.apply_oversampling(data, labels)
        else:
            self.data, self.labels = data, labels
            
        # Calculate class weights for weighted loss
        self.class_weights = self.calculate_class_weights()
        
    def apply_smote(self, X, y):
        """
        SMOTE: Synthetic Minority Over-sampling Technique
        Creates synthetic examples for minority classes
        """
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"SMOTE Balancing:")
        print(f"  Original: {np.bincount(y)}")
        print(f"  Balanced: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def apply_adasyn(self, X, y):
        """
        ADASYN: Adaptive Synthetic Sampling
        Generates more synthetic data for harder to learn minority samples
        """
        from imblearn.over_sampling import ADASYN
        
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
        
        print(f"ADASYN Balancing:")
        print(f"  Original: {np.bincount(y)}")
        print(f"  Balanced: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def apply_undersampling(self, X, y):
        """
        Random Under-sampling of majority class
        """
        from imblearn.under_sampling import RandomUnderSampler
        
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
        
        return X_balanced, y_balanced
    
    def apply_oversampling(self, X, y):
        """
        Random Over-sampling of minority classes
        """
        from imblearn.over_sampling import RandomOverSampler
        
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X, y)
        
        return X_balanced, y_balanced
    
    def calculate_class_weights(self):
        """
        Calculate class weights for weighted loss function
        """
        classes = np.unique(self.labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.labels
        )
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# ============================================================================
# PART 2: FOCAL LOSS FOR IMBALANCED CLASSIFICATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# PART 3: ADVANCED REGULARIZATION TECHNIQUES
# ============================================================================

class HyperbolicCNNWithRegularization(nn.Module):
    """
    Enhanced Hyperbolic CNN with multiple regularization techniques
    to prevent overfitting and underfitting
    """
    
    def __init__(self, input_dim, embed_dim=128, curvature=1.0, 
                 dropout_rate=0.3, use_batch_norm=True, use_layer_norm=True,
                 use_spectral_norm=False, use_mixup=True):
        super().__init__()
        
        self.curvature = curvature
        self.use_mixup = use_mixup
        
        # Encoder with regularization
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        if use_layer_norm:
            layers.append(nn.LayerNorm(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layer
        layers.append(nn.Linear(256, embed_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(embed_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.8))  # Reduce dropout in deeper layers
        
        self.encoder = nn.Sequential(*layers)
        
        # Hyperbolic layers with Spectral Normalization
        if use_spectral_norm:
            self.h_conv1 = nn.utils.spectral_norm(
                nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
            )
            self.h_conv2 = nn.utils.spectral_norm(
                nn.Conv1d(64, 128, kernel_size=3, padding=1)
            )
            self.h_conv3 = nn.utils.spectral_norm(
                nn.Conv1d(128, 256, kernel_size=3, padding=1)
            )
        else:
            self.h_conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
            self.h_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.h_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization for convolutions
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Attention mechanism with dropout
        self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout_rate * 0.5)
        
        # Output layers with regularization
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(64, 3)  # 3 classes: Buy, Hold, Sell
        )
        
        # Weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        He initialization for ReLU networks
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def exponential_map(self, v):
        """Map from tangent space to Poincaré Ball"""
        norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
        coeff = torch.tanh(torch.sqrt(self.curvature) * norm / 2)
        return coeff * v / (torch.sqrt(self.curvature) * norm)
    
    def logarithmic_map(self, x):
        """Map from Poincaré Ball to tangent space"""
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
        coeff = 2 / torch.sqrt(self.curvature) * torch.atanh(torch.sqrt(self.curvature) * norm)
        return coeff * x / norm
    
    def mixup_data(self, x, y, alpha=1.0):
        """
        Mixup augmentation: creates virtual training examples
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def forward(self, x, targets=None):
        # Apply Mixup if training and enabled
        if self.training and self.use_mixup and targets is not None:
            x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            
        # Encode to hyperbolic space
        x = self.encoder(x)
        x = self.exponential_map(x)
        
        # Apply hyperbolic convolutions with batch norm
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.h_conv1(x)))
        x = F.relu(self.bn2(self.h_conv2(x)))
        x = F.relu(self.bn3(self.h_conv3(x)))
        
        # Global average pooling
        x = x.mean(dim=2)
        
        # Apply attention
        x = x.unsqueeze(0)  # Add sequence dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)
        
        # Output
        output = self.output(x)
        
        # Return mixed targets if using mixup
        if self.training and self.use_mixup and targets is not None:
            return output, targets_a, targets_b, lam
        else:
            return output

# ============================================================================
# PART 4: ADVANCED TRAINING WITH EARLY STOPPING AND LEARNING RATE SCHEDULING
# ============================================================================

class AdvancedTrainer:
    """
    Advanced training pipeline with multiple techniques to prevent
    overfitting and underfitting
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function (Focal Loss for imbalanced data)
        self.criterion = FocalLoss(alpha=config.get('class_weights'), gamma=2.0)
        
        # Optimizer with L2 regularization (weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = self.setup_scheduler(config.get('scheduler', 'cosine'))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 20),
            verbose=True,
            delta=config.get('delta', 0.001)
        )
        
        # Gradient clipping value
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def setup_scheduler(self, scheduler_type):
        """
        Setup learning rate scheduler
        """
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch with gradient clipping and regularization
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (may include mixup)
            output = self.model(data, target)
            
            # Handle mixup loss
            if isinstance(output, tuple):
                output, targets_a, targets_b, lam = output
                loss = lam * self.criterion(output, targets_a) + \
                       (1 - lam) * self.criterion(output, targets_b)
            else:
                loss = self.criterion(output, target)
            
            # Add L1 regularization if specified
            if self.config.get('l1_lambda', 0) > 0:
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.config['l1_lambda'] * l1_norm
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validation with detailed metrics
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.squeeze().to(self.device)
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate detailed metrics
        report = classification_report(all_targets, all_preds, output_dict=True)
        
        return avg_loss, accuracy, report
    
    def train(self, train_loader, val_loader, epochs):
        """
        Complete training loop with early stopping
        """
        print("Starting Advanced Training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_report = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {lr:.6f}")
            print(f"  Per-class F1: Buy={val_report['2']['f1-score']:.3f}, "
                  f"Hold={val_report['1']['f1-score']:.3f}, "
                  f"Sell={val_report['0']['f1-score']:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': self.config
                }, 'best_model.pth')
                print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        return self.train_losses, self.val_losses, self.val_accuracies

# ============================================================================
# PART 5: EARLY STOPPING IMPLEMENTATION
# ============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience=20, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f})')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss

# ============================================================================
# PART 6: ENSEMBLE METHODS FOR ROBUSTNESS
# ============================================================================

class EnsembleModel:
    """
    Ensemble of multiple models to reduce overfitting and improve robustness
    """
    
    def __init__(self, n_models=5, base_config=None):
        self.n_models = n_models
        self.models = []
        self.config = base_config or {}
        
        # Create diverse models
        for i in range(n_models):
            # Vary architecture slightly
            config = self.config.copy()
            config['dropout_rate'] = 0.2 + i * 0.05  # Vary dropout
            config['embed_dim'] = 128 + i * 32  # Vary embedding dimension
            
            model = HyperbolicCNNWithRegularization(
                input_dim=config['input_dim'],
                embed_dim=config['embed_dim'],
                dropout_rate=config['dropout_rate']
            )
            self.models.append(model)
    
    def train_ensemble(self, train_data, val_data, epochs=100):
        """
        Train all models in ensemble with different random seeds
        """
        ensemble_results = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining Model {i+1}/{self.n_models}")
            
            # Set different random seed for each model
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            # Train with different subset of data (bagging)
            subset_indices = np.random.choice(
                len(train_data), 
                size=int(0.8 * len(train_data)), 
                replace=True
            )
            subset_train = train_data[subset_indices]
            
            trainer = AdvancedTrainer(model, self.config)
            losses = trainer.train(subset_train, val_data, epochs)
            ensemble_results.append(losses)
        
        return ensemble_results
    
    def predict_ensemble(self, data, method='voting'):
        """
        Make predictions using ensemble
        
        Args:
            method: 'voting' for hard voting, 'averaging' for soft voting
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(data)
                if method == 'voting':
                    pred = output.argmax(dim=1)
                    predictions.append(pred)
                else:  # averaging
                    predictions.append(F.softmax(output, dim=1))
        
        if method == 'voting':
            # Majority voting
            predictions = torch.stack(predictions)
            final_pred = torch.mode(predictions, dim=0)[0]
        else:
            # Average probabilities
            predictions = torch.stack(predictions)
            avg_probs = predictions.mean(dim=0)
            final_pred = avg_probs.argmax(dim=1)
        
        return final_pred

# ============================================================================
# PART 7: HYPERPARAMETER OPTIMIZATION
# ============================================================================

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Grid search for optimal hyperparameters to prevent over/underfitting
    """
    
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'embed_dim': [64, 128, 256],
        'l1_lambda': [0, 1e-5, 1e-4]
    }
    
    best_score = 0
    best_params = {}
    
    for lr in param_grid['learning_rate']:
        for dropout in param_grid['dropout_rate']:
            for wd in param_grid['weight_decay']:
                for batch_size in param_grid['batch_size']:
                    for embed_dim in param_grid['embed_dim']:
                        for l1 in param_grid['l1_lambda']:
                            config = {
                                'learning_rate': lr,
                                'dropout_rate': dropout,
                                'weight_decay': wd,
                                'batch_size': batch_size,
                                'embed_dim': embed_dim,
                                'l1_lambda': l1,
                                'input_dim': X_train.shape[1]
                            }
                            
                            # Train model
                            model = HyperbolicCNNWithRegularization(
                                input_dim=config['input_dim'],
                                embed_dim=config['embed_dim'],
                                dropout_rate=config['dropout_rate']
                            )
                            
                            # Quick training for hyperparameter search
                            trainer = AdvancedTrainer(model, config)
                            
                            # Create data loaders
                            train_dataset = BalancedTradingDataset(X_train, y_train)
                            val_dataset = BalancedTradingDataset(X_val, y_val)
                            
                            train_loader = DataLoader(
                                train_dataset, 
                                batch_size=config['batch_size'],
                                shuffle=True
                            )
                            val_loader = DataLoader(
                                val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False
                            )
                            
                            # Train for fewer epochs during search
                            _, _, val_accs = trainer.train(train_loader, val_loader, epochs=20)
                            
                            # Check best validation accuracy
                            max_val_acc = max(val_accs)
                            
                            if max_val_acc > best_score:
                                best_score = max_val_acc
                                best_params = config
                                print(f"New best params: {best_params}")
                                print(f"Validation Accuracy: {best_score:.2f}%")
    
    return best_params

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED MODEL TRAINING WITH BALANCING AND REGULARIZATION")
    print("="*70)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    # Simulate imbalanced trading data
    # More "Hold" signals, fewer "Buy" and "Sell" signals
    n_samples = 10000
    n_features = 50
    
    # Create imbalanced labels
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    
    # Create features (with some correlation to labels)
    X = np.random.randn(n_samples, n_features)
    for i in range(n_samples):
        if labels[i] == 2:  # Buy signal
            X[i, :5] += 1.5  # Positive bias in first 5 features
        elif labels[i] == 0:  # Sell signal
            X[i, :5] -= 1.5  # Negative bias in first 5 features
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train, y_train = X[:train_size], labels[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], labels[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], labels[train_size+val_size:]
    
    print(f"Dataset created:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    # Create balanced dataset
    print("\n" + "="*50)
    print("Creating Balanced Dataset...")
    
    balanced_train = BalancedTradingDataset(X_train, y_train, balance_method='smote')
    balanced_val = BalancedTradingDataset(X_val, y_val, balance_method='weighted')
    
    # Create data loaders
    train_loader = DataLoader(balanced_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(balanced_val, batch_size=64, shuffle=False)
    
    # Configure model with regularization
    config = {
        'input_dim': n_features,
        'embed_dim': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'l1_lambda': 1e-5,
        'class_weights': balanced_train.class_weights,
        'scheduler': 'cosine',
        'patience': 20,
        'grad_clip': 1.0
    }
    
    # Initialize model
    print("\n" + "="*50)
    print("Initializing Model with Regularization...")
    
    model = HyperbolicCNNWithRegularization(
        input_dim=config['input_dim'],
        embed_dim=config['embed_dim'],
        dropout_rate=config['dropout_rate'],
        use_batch_norm=True,
        use_layer_norm=True,
        use_spectral_norm=False,
        use_mixup=True
    )
    
    # Train model
    print("\n" + "="*50)
    print("Starting Training...")
    
    trainer = AdvancedTrainer(model, config)
    train_losses, val_losses, val_accuracies = trainer.train(
        train_loader, 
        val_loader, 
        epochs=50
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    
    # Check for overfitting/underfitting
    if train_losses[-1] < 0.1 and val_losses[-1] > 0.5:
        print("⚠️ WARNING: Model appears to be OVERFITTING")
        print("  Suggestions:")
        print("  - Increase dropout rate")
        print("  - Add more regularization (L1/L2)")
        print("  - Reduce model complexity")
        print("  - Get more training data")
    elif train_losses[-1] > 0.8:
        print("⚠️ WARNING: Model appears to be UNDERFITTING")
        print("  Suggestions:")
        print("  - Increase model complexity")
        print("  - Reduce regularization")
        print("  - Train for more epochs")
        print("  - Check data quality")
    else:
        print("✅ Model appears to be well-fitted!")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, 'final_model_with_regularization.pth')
    
    print("\n✅ Training complete! Model saved to 'final_model_with_regularization.pth'")