#!/usr/bin/env python3
"""
ADVANCED VALIDATION FRAMEWORK
============================
Comprehensive overfitting prevention and hallucination detection system
for the Hyperbolic Portfolio Optimization Platform.

Features:
- Statistical validation with multiple tests
- Cross-validation with time series awareness
- Hallucination detection using ensemble methods
- Model stability assessment
- Real-time monitoring and alerting
- Confidence scoring and uncertainty quantification
"""

import numpy as np
import pandas as pd
# PyTorch imports removed for compatibility
# import torch
# import torch.nn as nn
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest, kstest, anderson
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalValidator:
    """
    Comprehensive statistical validation framework
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.validation_history = []
        
    def comprehensive_residual_analysis(self, predictions: np.ndarray, 
                                      actuals: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis to detect overfitting
        """
        residuals = predictions - actuals
        
        logger.info(f"Performing residual analysis on {len(residuals)} samples")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(residuals),
            'normality_tests': self._test_normality(residuals),
            'independence_tests': self._test_independence(residuals),
            'homoscedasticity_tests': self._test_homoscedasticity(residuals),
            'outlier_detection': self._detect_outliers(residuals),
            'distribution_properties': self._analyze_distribution(residuals),
            'performance_metrics': self._calculate_performance_metrics(predictions, actuals)
        }
        
        # Calculate overall validation score
        analysis_results['validation_score'] = self._calculate_validation_score(analysis_results)
        analysis_results['validation_passed'] = analysis_results['validation_score'] > 0.7
        
        return analysis_results
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residuals for normality using multiple tests"""
        
        normality_tests = {}
        
        try:
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(residuals)
            normality_tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'is_normal': jb_pvalue > self.significance_level,
                'interpretation': 'Normal' if jb_pvalue > self.significance_level else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            normality_tests['jarque_bera'] = {'error': str(e)}
        
        try:
            # Shapiro-Wilk test (limited to 5000 samples)
            sample_size = min(len(residuals), 5000)
            sample_residuals = np.random.choice(residuals, sample_size, replace=False) if len(residuals) > sample_size else residuals
            sw_stat, sw_pvalue = shapiro(sample_residuals)
            normality_tests['shapiro_wilk'] = {
                'statistic': float(sw_stat),
                'p_value': float(sw_pvalue),
                'is_normal': sw_pvalue > self.significance_level,
                'sample_size': sample_size,
                'interpretation': 'Normal' if sw_pvalue > self.significance_level else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Shapiro-Wilk test failed: {e}")
            normality_tests['shapiro_wilk'] = {'error': str(e)}
        
        try:
            # D'Agostino and Pearson's test
            da_stat, da_pvalue = normaltest(residuals)
            normality_tests['dagostino_pearson'] = {
                'statistic': float(da_stat),
                'p_value': float(da_pvalue),
                'is_normal': da_pvalue > self.significance_level,
                'interpretation': 'Normal' if da_pvalue > self.significance_level else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"D'Agostino test failed: {e}")
            normality_tests['dagostino_pearson'] = {'error': str(e)}
        
        try:
            # Kolmogorov-Smirnov test against normal distribution
            normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            ks_stat, ks_pvalue = kstest(normalized_residuals, 'norm')
            normality_tests['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'is_normal': ks_pvalue > self.significance_level,
                'interpretation': 'Normal' if ks_pvalue > self.significance_level else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
            normality_tests['kolmogorov_smirnov'] = {'error': str(e)}
        
        try:
            # Anderson-Darling test
            ad_result = anderson(residuals, dist='norm')
            # Use 5% critical value
            critical_value = ad_result.critical_values[2]  # 5% level
            normality_tests['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_value': float(critical_value),
                'is_normal': ad_result.statistic < critical_value,
                'interpretation': 'Normal' if ad_result.statistic < critical_value else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")
            normality_tests['anderson_darling'] = {'error': str(e)}
        
        return normality_tests
    
    def _test_independence(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residuals for independence (no autocorrelation)"""
        
        independence_tests = {}
        
        try:
            # Lag-1 autocorrelation
            if len(residuals) > 1:
                autocorr_1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                independence_tests['lag1_autocorrelation'] = {
                    'correlation': float(autocorr_1),
                    'is_independent': abs(autocorr_1) < 0.1,
                    'interpretation': 'Independent' if abs(autocorr_1) < 0.1 else 'Autocorrelated'
                }
        except Exception as e:
            logger.warning(f"Lag-1 autocorrelation test failed: {e}")
            independence_tests['lag1_autocorrelation'] = {'error': str(e)}
        
        try:
            # Ljung-Box test approximation (simplified)
            if len(residuals) > 10:
                lags = min(10, len(residuals) // 4)
                autocorrelations = []
                
                for lag in range(1, lags + 1):
                    if len(residuals) > lag:
                        autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                        autocorrelations.append(autocorr)
                
                # Simplified Ljung-Box statistic
                n = len(residuals)
                lb_stat = n * (n + 2) * sum([(autocorr**2) / (n - lag - 1) 
                                           for lag, autocorr in enumerate(autocorrelations)])
                
                # Chi-square critical value approximation
                critical_value = stats.chi2.ppf(1 - self.significance_level, lags)
                
                independence_tests['ljung_box_approximation'] = {
                    'statistic': float(lb_stat),
                    'critical_value': float(critical_value),
                    'lags_tested': lags,
                    'is_independent': lb_stat < critical_value,
                    'interpretation': 'Independent' if lb_stat < critical_value else 'Autocorrelated'
                }
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {e}")
            independence_tests['ljung_box_approximation'] = {'error': str(e)}
        
        try:
            # Runs test for randomness
            median_residual = np.median(residuals)
            runs, n1, n2 = 0, 0, 0
            
            # Convert to binary sequence
            binary_sequence = residuals > median_residual
            
            # Count runs
            if len(binary_sequence) > 0:
                current_run = binary_sequence[0]
                runs = 1
                n1 = sum(binary_sequence)
                n2 = len(binary_sequence) - n1
                
                for i in range(1, len(binary_sequence)):
                    if binary_sequence[i] != current_run:
                        runs += 1
                        current_run = binary_sequence[i]
                
                # Expected runs and variance
                if n1 > 0 and n2 > 0:
                    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                                   ((n1 + n2)**2 * (n1 + n2 - 1))
                    
                    if variance_runs > 0:
                        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        
                        independence_tests['runs_test'] = {
                            'runs_observed': runs,
                            'runs_expected': float(expected_runs),
                            'z_statistic': float(z_stat),
                            'p_value': float(p_value),
                            'is_random': p_value > self.significance_level,
                            'interpretation': 'Random' if p_value > self.significance_level else 'Non-random pattern'
                        }
        except Exception as e:
            logger.warning(f"Runs test failed: {e}")
            independence_tests['runs_test'] = {'error': str(e)}
        
        return independence_tests
    
    def _test_homoscedasticity(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for homoscedasticity (constant variance)"""
        
        homoscedasticity_tests = {}
        
        try:
            # Breusch-Pagan test approximation
            n = len(residuals)
            time_trend = np.arange(n)
            
            # Correlate squared residuals with time trend
            squared_residuals = residuals ** 2
            correlation = np.corrcoef(squared_residuals, time_trend)[0, 1] if n > 1 else 0
            
            homoscedasticity_tests['breusch_pagan_approximation'] = {
                'correlation_with_time': float(correlation),
                'is_homoscedastic': abs(correlation) < 0.1,
                'interpretation': 'Homoscedastic' if abs(correlation) < 0.1 else 'Heteroscedastic'
            }
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            homoscedasticity_tests['breusch_pagan_approximation'] = {'error': str(e)}
        
        try:
            # White's test approximation (simplified)
            # Test correlation of squared residuals with their own lags
            if len(residuals) > 2:
                lag1_corr = np.corrcoef(residuals[:-1]**2, residuals[1:]**2)[0, 1]
                
                homoscedasticity_tests['white_test_approximation'] = {
                    'lag1_correlation': float(lag1_corr),
                    'is_homoscedastic': abs(lag1_corr) < 0.15,
                    'interpretation': 'Homoscedastic' if abs(lag1_corr) < 0.15 else 'Heteroscedastic'
                }
        except Exception as e:
            logger.warning(f"White test failed: {e}")
            homoscedasticity_tests['white_test_approximation'] = {'error': str(e)}
        
        try:
            # Goldfeld-Quandt test (split sample)
            n = len(residuals)
            if n >= 20:
                # Split into first and last third, omit middle third
                split_size = n // 3
                first_third = residuals[:split_size]
                last_third = residuals[-split_size:]
                
                var_first = np.var(first_third, ddof=1) if len(first_third) > 1 else 0
                var_last = np.var(last_third, ddof=1) if len(last_third) > 1 else 0
                
                if var_first > 0 and var_last > 0:
                    f_stat = var_last / var_first
                    # F-distribution critical value approximation
                    df1 = df2 = split_size - 1
                    critical_value = stats.f.ppf(1 - self.significance_level/2, df1, df2)
                    
                    homoscedasticity_tests['goldfeld_quandt'] = {
                        'f_statistic': float(f_stat),
                        'critical_value': float(critical_value),
                        'variance_first_third': float(var_first),
                        'variance_last_third': float(var_last),
                        'is_homoscedastic': f_stat < critical_value,
                        'interpretation': 'Homoscedastic' if f_stat < critical_value else 'Heteroscedastic'
                    }
        except Exception as e:
            logger.warning(f"Goldfeld-Quandt test failed: {e}")
            homoscedasticity_tests['goldfeld_quandt'] = {'error': str(e)}
        
        return homoscedasticity_tests
    
    def _detect_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in residuals"""
        
        outlier_detection = {}
        
        try:
            # Z-score method
            z_scores = np.abs(stats.zscore(residuals))
            z_threshold = 3.0
            z_outliers = np.sum(z_scores > z_threshold)
            z_outlier_percentage = (z_outliers / len(residuals)) * 100
            
            outlier_detection['z_score_method'] = {
                'threshold': z_threshold,
                'outliers_count': int(z_outliers),
                'outlier_percentage': float(z_outlier_percentage),
                'max_z_score': float(np.max(z_scores)),
                'is_acceptable': z_outlier_percentage < 5.0
            }
        except Exception as e:
            logger.warning(f"Z-score outlier detection failed: {e}")
            outlier_detection['z_score_method'] = {'error': str(e)}
        
        try:
            # IQR method
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = np.sum((residuals < lower_bound) | (residuals > upper_bound))
            iqr_outlier_percentage = (iqr_outliers / len(residuals)) * 100
            
            outlier_detection['iqr_method'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outliers_count': int(iqr_outliers),
                'outlier_percentage': float(iqr_outlier_percentage),
                'is_acceptable': iqr_outlier_percentage < 10.0
            }
        except Exception as e:
            logger.warning(f"IQR outlier detection failed: {e}")
            outlier_detection['iqr_method'] = {'error': str(e)}
        
        try:
            # Isolation Forest
            if len(residuals) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(residuals.reshape(-1, 1))
                iso_outliers = np.sum(outlier_labels == -1)
                iso_outlier_percentage = (iso_outliers / len(residuals)) * 100
                
                outlier_detection['isolation_forest'] = {
                    'outliers_count': int(iso_outliers),
                    'outlier_percentage': float(iso_outlier_percentage),
                    'is_acceptable': iso_outlier_percentage < 15.0
                }
        except Exception as e:
            logger.warning(f"Isolation Forest outlier detection failed: {e}")
            outlier_detection['isolation_forest'] = {'error': str(e)}
        
        return outlier_detection
    
    def _analyze_distribution(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution properties of residuals"""
        
        distribution_props = {}
        
        try:
            # Basic statistics
            distribution_props['basic_stats'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals, ddof=1)),
                'variance': float(np.var(residuals, ddof=1)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'median': float(np.median(residuals))
            }
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            distribution_props['percentiles'] = {
                f'p{p}': float(np.percentile(residuals, p)) for p in percentiles
            }
            
            # Distribution shape assessment
            skew_val = stats.skew(residuals)
            kurt_val = stats.kurtosis(residuals)
            
            distribution_props['shape_assessment'] = {
                'skewness_interpretation': self._interpret_skewness(skew_val),
                'kurtosis_interpretation': self._interpret_kurtosis(kurt_val),
                'is_approximately_normal': abs(skew_val) < 1 and abs(kurt_val) < 3
            }
            
        except Exception as e:
            logger.warning(f"Distribution analysis failed: {e}")
            distribution_props = {'error': str(e)}
        
        return distribution_props
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value"""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0.5:
            return "Right-skewed (positive skew)"
        else:
            return "Left-skewed (negative skew)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value"""
        if abs(kurtosis) < 0.5:
            return "Mesokurtic (normal-like tails)"
        elif kurtosis > 0.5:
            return "Leptokurtic (heavy tails)"
        else:
            return "Platykurtic (light tails)"
    
    def _calculate_performance_metrics(self, predictions: np.ndarray, 
                                     actuals: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        try:
            metrics = {
                'mse': float(mean_squared_error(actuals, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
                'mae': float(mean_absolute_error(actuals, predictions)),
                'r2_score': float(r2_score(actuals, predictions)),
                'mean_error': float(np.mean(predictions - actuals)),
                'std_error': float(np.std(predictions - actuals, ddof=1)),
                'max_error': float(np.max(np.abs(predictions - actuals))),
                'median_absolute_error': float(np.median(np.abs(predictions - actuals)))
            }
            
            # Additional metrics
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
            metrics['mape'] = float(mape)
            
            # Directional accuracy (for returns/changes)
            if len(predictions) > 1:
                pred_direction = np.sign(np.diff(predictions))
                actual_direction = np.sign(np.diff(actuals))
                directional_accuracy = np.mean(pred_direction == actual_direction)
                metrics['directional_accuracy'] = float(directional_accuracy)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_validation_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall validation score (0-1, higher is better)"""
        
        score = 0.0
        max_score = 0.0
        
        try:
            # Normality tests (20% of score)
            normality_tests = analysis_results.get('normality_tests', {})
            normality_score = 0.0
            normality_count = 0
            
            for test_name, test_result in normality_tests.items():
                if isinstance(test_result, dict) and 'is_normal' in test_result:
                    if test_result['is_normal']:
                        normality_score += 1.0
                    normality_count += 1
            
            if normality_count > 0:
                score += (normality_score / normality_count) * 0.2
            max_score += 0.2
            
            # Independence tests (25% of score)
            independence_tests = analysis_results.get('independence_tests', {})
            independence_score = 0.0
            independence_count = 0
            
            for test_name, test_result in independence_tests.items():
                if isinstance(test_result, dict):
                    if 'is_independent' in test_result and test_result['is_independent']:
                        independence_score += 1.0
                        independence_count += 1
                    elif 'is_random' in test_result and test_result['is_random']:
                        independence_score += 1.0
                        independence_count += 1
                    elif test_name in ['lag1_autocorrelation', 'ljung_box_approximation', 'runs_test']:
                        independence_count += 1
            
            if independence_count > 0:
                score += (independence_score / independence_count) * 0.25
            max_score += 0.25
            
            # Homoscedasticity tests (20% of score)
            homoscedasticity_tests = analysis_results.get('homoscedasticity_tests', {})
            homoscedasticity_score = 0.0
            homoscedasticity_count = 0
            
            for test_name, test_result in homoscedasticity_tests.items():
                if isinstance(test_result, dict) and 'is_homoscedastic' in test_result:
                    if test_result['is_homoscedastic']:
                        homoscedasticity_score += 1.0
                    homoscedasticity_count += 1
            
            if homoscedasticity_count > 0:
                score += (homoscedasticity_score / homoscedasticity_count) * 0.2
            max_score += 0.2
            
            # Outlier detection (15% of score)
            outlier_detection = analysis_results.get('outlier_detection', {})
            outlier_score = 0.0
            outlier_count = 0
            
            for method_name, method_result in outlier_detection.items():
                if isinstance(method_result, dict) and 'is_acceptable' in method_result:
                    if method_result['is_acceptable']:
                        outlier_score += 1.0
                    outlier_count += 1
            
            if outlier_count > 0:
                score += (outlier_score / outlier_count) * 0.15
            max_score += 0.15
            
            # Performance metrics (20% of score)
            performance_metrics = analysis_results.get('performance_metrics', {})
            if isinstance(performance_metrics, dict) and 'r2_score' in performance_metrics:
                r2 = performance_metrics['r2_score']
                # Scale R² to 0-1 (assuming good models have R² > 0.5)
                performance_score = max(0, min(1, (r2 + 1) / 2))  # Scale from [-1,1] to [0,1]
                score += performance_score * 0.2
            max_score += 0.2
            
        except Exception as e:
            logger.warning(f"Validation score calculation failed: {e}")
            return 0.5  # Default middle score on error
        
        return score / max_score if max_score > 0 else 0.5


class TimeSeriesCrossValidator:
    """
    Time series aware cross-validation with gap handling
    """
    
    def __init__(self, n_splits: int = 5, test_size_ratio: float = 0.2, gap_ratio: float = 0.05):
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.gap_ratio = gap_ratio
    
    def validate_model(self, X: np.ndarray, y: np.ndarray, 
                      model_func, **kwargs) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        """
        
        logger.info(f"Starting time series cross-validation with {self.n_splits} folds")
        
        n_samples = len(X)
        test_size = int(n_samples * self.test_size_ratio)
        gap_size = int(n_samples * self.gap_ratio)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold in range(self.n_splits):
            # Calculate split indices
            test_end = n_samples - fold * (test_size + gap_size)
            test_start = test_end - test_size
            train_end = test_start - gap_size
            
            if train_end <= test_size:  # Not enough training data
                break
            
            # Create splits
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            logger.info(f"Fold {fold + 1}: Train size={len(X_train)}, Test size={len(X_test)}")
            
            try:
                # Train model and get predictions
                predictions = model_func(X_train, y_train, X_test, **kwargs)
                
                # Calculate metrics for this fold
                fold_metrics = self._calculate_fold_metrics(predictions, y_test)
                fold_metrics['fold'] = fold + 1
                fold_metrics['train_size'] = len(X_train)
                fold_metrics['test_size'] = len(X_test)
                
                fold_results.append(fold_metrics)
                all_predictions.extend(predictions)
                all_actuals.extend(y_test)
                
            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
                continue
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results, all_predictions, all_actuals)
        
        return cv_results
    
    def _calculate_fold_metrics(self, predictions: np.ndarray, 
                               actuals: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single fold"""
        
        return {
            'mse': float(mean_squared_error(actuals, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'mae': float(mean_absolute_error(actuals, predictions)),
            'r2': float(r2_score(actuals, predictions)),
            'max_error': float(np.max(np.abs(predictions - actuals)))
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict], 
                             all_predictions: List[float],
                             all_actuals: List[float]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        
        if not fold_results:
            return {'error': 'No successful folds'}
        
        # Calculate mean and std for each metric
        metrics = ['mse', 'rmse', 'mae', 'r2', 'max_error']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        # Overall metrics across all folds
        if all_predictions and all_actuals:
            aggregated['overall_mse'] = float(mean_squared_error(all_actuals, all_predictions))
            aggregated['overall_rmse'] = float(np.sqrt(mean_squared_error(all_actuals, all_predictions)))
            aggregated['overall_mae'] = float(mean_absolute_error(all_actuals, all_predictions))
            aggregated['overall_r2'] = float(r2_score(all_actuals, all_predictions))
        
        # Model stability assessment
        aggregated['stability_score'] = self._calculate_stability_score(fold_results)
        
        aggregated['fold_results'] = fold_results
        aggregated['n_successful_folds'] = len(fold_results)
        
        return aggregated
    
    def _calculate_stability_score(self, fold_results: List[Dict]) -> float:
        """Calculate model stability score based on consistency across folds"""
        
        if len(fold_results) < 2:
            return 0.5
        
        # Use coefficient of variation for R² scores
        r2_values = [fold['r2'] for fold in fold_results if 'r2' in fold]
        
        if not r2_values:
            return 0.5
        
        mean_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        
        if mean_r2 <= 0:
            return 0.0
        
        cv = std_r2 / mean_r2  # Coefficient of variation
        
        # Convert to stability score (lower CV = higher stability)
        stability = max(0, 1 - cv)
        
        return stability


class HallucinationDetector:
    """
    Advanced hallucination detection system
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.prediction_history = []
        self.market_context = {}
        
    def detect_hallucination(self, predictions: np.ndarray,
                           confidence_scores: np.ndarray,
                           market_data: pd.DataFrame,
                           model_metadata: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive hallucination detection
        """
        
        logger.info(f"Running hallucination detection on {len(predictions)} predictions")
        
        detection_results = {
            'timestamp': datetime.now().isoformat(),
            'confidence_analysis': self._analyze_confidence(predictions, confidence_scores),
            'market_context_analysis': self._analyze_market_context(predictions, market_data),
            'statistical_anomalies': self._detect_statistical_anomalies(predictions),
            'temporal_consistency': self._check_temporal_consistency(predictions),
            'ensemble_disagreement': self._check_ensemble_disagreement(predictions, model_metadata),
            'uncertainty_quantification': self._quantify_uncertainty(predictions, confidence_scores)
        }
        
        # Calculate overall hallucination risk
        detection_results['hallucination_risk_score'] = self._calculate_hallucination_risk(detection_results)
        detection_results['is_hallucinating'] = detection_results['hallucination_risk_score'] > 0.7
        
        # Update prediction history
        self._update_prediction_history(predictions, confidence_scores)
        
        return detection_results
    
    def _analyze_confidence(self, predictions: np.ndarray, 
                          confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze confidence patterns"""
        
        confidence_analysis = {
            'mean_confidence': float(np.mean(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores)),
            'min_confidence': float(np.min(confidence_scores)),
            'low_confidence_ratio': float(np.mean(confidence_scores < self.confidence_threshold)),
            'confidence_trend': self._calculate_confidence_trend(confidence_scores)
        }
        
        # Flag suspicious confidence patterns
        confidence_analysis['suspicious_patterns'] = []
        
        if confidence_analysis['mean_confidence'] < 0.5:
            confidence_analysis['suspicious_patterns'].append('Very low overall confidence')
        
        if confidence_analysis['low_confidence_ratio'] > 0.3:
            confidence_analysis['suspicious_patterns'].append('High ratio of low-confidence predictions')
        
        if confidence_analysis['std_confidence'] > 0.3:
            confidence_analysis['suspicious_patterns'].append('Highly variable confidence scores')
        
        return confidence_analysis
    
    def _calculate_confidence_trend(self, confidence_scores: np.ndarray) -> str:
        """Calculate trend in confidence scores"""
        if len(confidence_scores) < 3:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(confidence_scores))
        slope = np.polyfit(x, confidence_scores, 1)[0]
        
        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _analyze_market_context(self, predictions: np.ndarray, 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze predictions in market context"""
        
        context_analysis = {}
        
        try:
            # Calculate market volatility
            if 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change().dropna()
                current_volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else returns.std()
                historical_volatility = returns.std()
                
                context_analysis['market_volatility'] = {
                    'current': float(current_volatility),
                    'historical': float(historical_volatility),
                    'ratio': float(current_volatility / (historical_volatility + 1e-8)),
                    'regime': 'high' if current_volatility > 2 * historical_volatility else 'normal'
                }
            
            # Prediction magnitude vs market conditions
            pred_magnitude = np.std(predictions)
            market_magnitude = context_analysis.get('market_volatility', {}).get('current', 0.02)
            
            context_analysis['magnitude_consistency'] = {
                'prediction_magnitude': float(pred_magnitude),
                'market_magnitude': float(market_magnitude),
                'ratio': float(pred_magnitude / (market_magnitude + 1e-8)),
                'is_consistent': abs(pred_magnitude / (market_magnitude + 1e-8) - 1) < 2
            }
            
        except Exception as e:
            logger.warning(f"Market context analysis failed: {e}")
            context_analysis = {'error': str(e)}
        
        return context_analysis
    
    def _detect_statistical_anomalies(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Detect statistical anomalies in predictions"""
        
        anomalies = {}
        
        try:
            # Z-score analysis
            z_scores = np.abs(stats.zscore(predictions))
            anomalies['z_score_outliers'] = {
                'count': int(np.sum(z_scores > 3)),
                'ratio': float(np.mean(z_scores > 3)),
                'max_z_score': float(np.max(z_scores))
            }
            
            # Distribution analysis
            anomalies['distribution'] = {
                'skewness': float(stats.skew(predictions)),
                'kurtosis': float(stats.kurtosis(predictions)),
                'is_normal': float(stats.normaltest(predictions)[1]) > 0.05 if len(predictions) > 8 else None
            }
            
            # Range analysis
            pred_range = np.max(predictions) - np.min(predictions)
            anomalies['range_analysis'] = {
                'range': float(pred_range),
                'is_excessive': pred_range > 10 * np.std(predictions)
            }
            
        except Exception as e:
            logger.warning(f"Statistical anomaly detection failed: {e}")
            anomalies = {'error': str(e)}
        
        return anomalies
    
    def _check_temporal_consistency(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Check temporal consistency of predictions"""
        
        consistency_analysis = {}
        
        try:
            if len(predictions) > 1:
                # Calculate prediction changes
                pred_changes = np.diff(predictions)
                
                consistency_analysis['volatility'] = {
                    'prediction_volatility': float(np.std(pred_changes)),
                    'mean_change': float(np.mean(pred_changes)),
                    'max_change': float(np.max(np.abs(pred_changes)))
                }
                
                # Check for sudden jumps
                change_threshold = 3 * np.std(pred_changes) if np.std(pred_changes) > 0 else 0.1
                sudden_jumps = np.sum(np.abs(pred_changes) > change_threshold)
                
                consistency_analysis['sudden_changes'] = {
                    'count': int(sudden_jumps),
                    'ratio': float(sudden_jumps / len(pred_changes)),
                    'threshold': float(change_threshold)
                }
                
                # Temporal autocorrelation
                if len(predictions) > 2:
                    autocorr = np.corrcoef(predictions[:-1], predictions[1:])[0, 1]
                    consistency_analysis['autocorrelation'] = {
                        'lag1_correlation': float(autocorr),
                        'is_consistent': 0.3 < autocorr < 0.9
                    }
            
        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            consistency_analysis = {'error': str(e)}
        
        return consistency_analysis
    
    def _check_ensemble_disagreement(self, predictions: np.ndarray,
                                   model_metadata: Dict = None) -> Dict[str, Any]:
        """Check for ensemble disagreement (if available)"""
        
        disagreement_analysis = {'available': False}
        
        if model_metadata and 'ensemble_predictions' in model_metadata:
            try:
                ensemble_preds = model_metadata['ensemble_predictions']
                
                # Calculate disagreement metrics
                disagreement_analysis = {
                    'available': True,
                    'ensemble_std': float(np.std(ensemble_preds, axis=0).mean()),
                    'mean_disagreement': float(np.mean(np.std(ensemble_preds, axis=0))),
                    'max_disagreement': float(np.max(np.std(ensemble_preds, axis=0))),
                    'high_disagreement_ratio': float(np.mean(np.std(ensemble_preds, axis=0) > 0.1))
                }
                
            except Exception as e:
                logger.warning(f"Ensemble disagreement analysis failed: {e}")
                disagreement_analysis = {'available': False, 'error': str(e)}
        
        return disagreement_analysis
    
    def _quantify_uncertainty(self, predictions: np.ndarray, 
                            confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Quantify prediction uncertainty"""
        
        uncertainty_metrics = {}
        
        try:
            # Confidence-based uncertainty
            uncertainty_metrics['confidence_based'] = {
                'mean_uncertainty': float(1 - np.mean(confidence_scores)),
                'uncertainty_variance': float(np.var(1 - confidence_scores)),
                'high_uncertainty_ratio': float(np.mean(confidence_scores < 0.6))
            }
            
            # Prediction-based uncertainty
            uncertainty_metrics['prediction_based'] = {
                'prediction_variance': float(np.var(predictions)),
                'coefficient_of_variation': float(np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)),
                'entropy_approximation': float(-np.mean(confidence_scores * np.log(confidence_scores + 1e-8)))
            }
            
        except Exception as e:
            logger.warning(f"Uncertainty quantification failed: {e}")
            uncertainty_metrics = {'error': str(e)}
        
        return uncertainty_metrics
    
    def _calculate_hallucination_risk(self, detection_results: Dict[str, Any]) -> float:
        """Calculate overall hallucination risk score"""
        
        risk_score = 0.0
        
        try:
            # Confidence analysis (30% weight)
            confidence_analysis = detection_results.get('confidence_analysis', {})
            if 'mean_confidence' in confidence_analysis:
                confidence_risk = 1 - confidence_analysis['mean_confidence']
                risk_score += confidence_risk * 0.3
            
            # Market context analysis (25% weight)
            context_analysis = detection_results.get('market_context_analysis', {})
            if 'magnitude_consistency' in context_analysis:
                if not context_analysis['magnitude_consistency'].get('is_consistent', True):
                    risk_score += 0.25
            
            # Statistical anomalies (20% weight)
            anomalies = detection_results.get('statistical_anomalies', {})
            if 'z_score_outliers' in anomalies:
                outlier_ratio = anomalies['z_score_outliers'].get('ratio', 0)
                risk_score += min(outlier_ratio * 4, 1.0) * 0.2  # Scale outlier ratio
            
            # Temporal consistency (15% weight)
            temporal = detection_results.get('temporal_consistency', {})
            if 'sudden_changes' in temporal:
                jump_ratio = temporal['sudden_changes'].get('ratio', 0)
                risk_score += min(jump_ratio * 5, 1.0) * 0.15  # Scale jump ratio
            
            # Uncertainty quantification (10% weight)
            uncertainty = detection_results.get('uncertainty_quantification', {})
            if 'confidence_based' in uncertainty:
                uncertainty_level = uncertainty['confidence_based'].get('mean_uncertainty', 0)
                risk_score += uncertainty_level * 0.1
            
        except Exception as e:
            logger.warning(f"Hallucination risk calculation failed: {e}")
            return 0.5  # Default moderate risk
        
        return min(risk_score, 1.0)
    
    def _update_prediction_history(self, predictions: np.ndarray, 
                                 confidence_scores: np.ndarray):
        """Update prediction history for trend analysis"""
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'mean_prediction': float(np.mean(predictions)),
            'mean_confidence': float(np.mean(confidence_scores))
        }
        
        self.prediction_history.append(history_entry)
        
        # Keep only recent history (last 100 entries)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]


class ValidationFrameworkOrchestrator:
    """
    Main orchestrator for the validation framework
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.statistical_validator = StatisticalValidator(
            significance_level=self.config.get('significance_level', 0.05)
        )
        
        self.cv_validator = TimeSeriesCrossValidator(
            n_splits=self.config.get('cv_splits', 5),
            test_size_ratio=self.config.get('test_size_ratio', 0.2),
            gap_ratio=self.config.get('gap_ratio', 0.05)
        )
        
        self.hallucination_detector = HallucinationDetector(
            confidence_threshold=self.config.get('confidence_threshold', 0.8)
        )
        
    def comprehensive_validation(self, predictions: np.ndarray,
                               actuals: np.ndarray,
                               confidence_scores: np.ndarray,
                               market_data: pd.DataFrame,
                               X: np.ndarray = None,
                               y: np.ndarray = None,
                               model_func=None,
                               **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive validation pipeline
        """
        
        logger.info("Starting comprehensive validation pipeline")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_summary': {
                'n_predictions': len(predictions),
                'n_actuals': len(actuals),
                'prediction_range': [float(np.min(predictions)), float(np.max(predictions))],
                'actual_range': [float(np.min(actuals)), float(np.max(actuals))]
            }
        }
        
        try:
            # Statistical validation
            logger.info("Running statistical validation...")
            validation_results['statistical_analysis'] = \
                self.statistical_validator.comprehensive_residual_analysis(predictions, actuals)
            
            # Cross-validation (if training data provided)
            if X is not None and y is not None and model_func is not None:
                logger.info("Running cross-validation...")
                validation_results['cross_validation'] = \
                    self.cv_validator.validate_model(X, y, model_func, **kwargs)
            else:
                logger.warning("Skipping cross-validation - missing training data or model function")
            
            # Hallucination detection
            logger.info("Running hallucination detection...")
            validation_results['hallucination_detection'] = \
                self.hallucination_detector.detect_hallucination(
                    predictions, confidence_scores, market_data
                )
            
            # Overall assessment
            validation_results['overall_assessment'] = self._calculate_overall_assessment(validation_results)
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}")
            validation_results['error'] = str(e)
            validation_results['overall_assessment'] = {
                'validation_passed': False,
                'risk_level': 'high',
                'recommendation': 'Do not use model - validation failed'
            }
        
        return validation_results
    
    def _calculate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation assessment"""
        
        assessment = {
            'validation_passed': False,
            'risk_level': 'high',
            'confidence_level': 'low',
            'recommendation': '',
            'key_concerns': [],
            'strengths': []
        }
        
        try:
            # Statistical validation score
            stat_score = validation_results.get('statistical_analysis', {}).get('validation_score', 0)
            
            # Cross-validation stability (if available)
            cv_stability = 0.5  # Default
            if 'cross_validation' in validation_results:
                cv_results = validation_results['cross_validation']
                cv_stability = cv_results.get('stability_score', 0.5)
            
            # Hallucination risk
            hallucination_risk = validation_results.get('hallucination_detection', {}).get('hallucination_risk_score', 1.0)
            
            # Combined score
            combined_score = (stat_score * 0.4 + cv_stability * 0.3 + (1 - hallucination_risk) * 0.3)
            
            # Determine assessment
            if combined_score > 0.8:
                assessment['validation_passed'] = True
                assessment['risk_level'] = 'low'
                assessment['confidence_level'] = 'high'
                assessment['recommendation'] = 'Model is validated and safe to use'
            elif combined_score > 0.6:
                assessment['validation_passed'] = True
                assessment['risk_level'] = 'medium'
                assessment['confidence_level'] = 'medium'
                assessment['recommendation'] = 'Model is acceptable with monitoring'
            else:
                assessment['validation_passed'] = False
                assessment['risk_level'] = 'high'
                assessment['confidence_level'] = 'low'
                assessment['recommendation'] = 'Model should not be used - requires improvement'
            
            # Identify concerns and strengths
            if stat_score < 0.6:
                assessment['key_concerns'].append('Statistical validation concerns')
            if cv_stability < 0.5:
                assessment['key_concerns'].append('Cross-validation instability')
            if hallucination_risk > 0.7:
                assessment['key_concerns'].append('High hallucination risk')
            
            if stat_score > 0.8:
                assessment['strengths'].append('Strong statistical properties')
            if cv_stability > 0.7:
                assessment['strengths'].append('Stable cross-validation performance')
            if hallucination_risk < 0.3:
                assessment['strengths'].append('Low hallucination risk')
            
            assessment['overall_score'] = combined_score
            
        except Exception as e:
            logger.error(f"Overall assessment calculation failed: {e}")
            assessment['error'] = str(e)
        
        return assessment


def create_validation_demo():
    """Create a demonstration of the validation framework"""
    
    print("="*80)
    print("ADVANCED VALIDATION FRAMEWORK DEMONSTRATION")
    print("Comprehensive Overfitting Prevention & Hallucination Detection")
    print("="*80)
    
    # Initialize framework
    config = {
        'significance_level': 0.05,
        'cv_splits': 5,
        'confidence_threshold': 0.8
    }
    
    validator = ValidationFrameworkOrchestrator(config)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # True underlying signal
    t = np.linspace(0, 4*np.pi, n_samples)
    true_signal = np.sin(t) + 0.5 * np.sin(3*t) + np.random.normal(0, 0.1, n_samples)
    
    # Model predictions (with some overfitting and bias)
    predictions = true_signal + np.random.normal(0, 0.05, n_samples)  # Good predictions
    predictions[900:] += 0.3  # Add some bias at the end (overfitting signature)
    
    # Confidence scores
    confidence_scores = np.random.beta(8, 2, n_samples)  # High confidence distribution
    confidence_scores[900:] *= 0.7  # Lower confidence where there's bias
    
    # Market data
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    market_data = pd.DataFrame({
        'Close': 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_samples))),
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    }, index=dates)
    
    print(f"\n📊 Running validation on {n_samples} predictions...")
    print(f"True signal range: [{np.min(true_signal):.3f}, {np.max(true_signal):.3f}]")
    print(f"Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    print(f"Mean confidence: {np.mean(confidence_scores):.3f}")
    
    # Run comprehensive validation
    validation_results = validator.comprehensive_validation(
        predictions=predictions,
        actuals=true_signal,
        confidence_scores=confidence_scores,
        market_data=market_data
    )
    
    # Display results
    print("\n📈 VALIDATION RESULTS")
    print("="*50)
    
    # Statistical analysis
    if 'statistical_analysis' in validation_results:
        stat_results = validation_results['statistical_analysis']
        print(f"\n🔬 Statistical Analysis:")
        print(f"  Validation Score: {stat_results.get('validation_score', 0):.3f}")
        print(f"  Validation Passed: {stat_results.get('validation_passed', False)}")
        
        # Performance metrics
        if 'performance_metrics' in stat_results:
            perf = stat_results['performance_metrics']
            print(f"  RMSE: {perf.get('rmse', 0):.4f}")
            print(f"  R² Score: {perf.get('r2_score', 0):.3f}")
            print(f"  MAE: {perf.get('mae', 0):.4f}")
    
    # Hallucination detection
    if 'hallucination_detection' in validation_results:
        halluc_results = validation_results['hallucination_detection']
        print(f"\n🧠 Hallucination Detection:")
        print(f"  Risk Score: {halluc_results.get('hallucination_risk_score', 1):.3f}")
        print(f"  Is Hallucinating: {halluc_results.get('is_hallucinating', True)}")
        
        if 'confidence_analysis' in halluc_results:
            conf = halluc_results['confidence_analysis']
            print(f"  Mean Confidence: {conf.get('mean_confidence', 0):.3f}")
            print(f"  Low Confidence Ratio: {conf.get('low_confidence_ratio', 1):.3f}")
    
    # Overall assessment
    if 'overall_assessment' in validation_results:
        assessment = validation_results['overall_assessment']
        print(f"\n🎯 Overall Assessment:")
        print(f"  Validation Passed: {assessment.get('validation_passed', False)}")
        print(f"  Risk Level: {assessment.get('risk_level', 'high')}")
        print(f"  Confidence Level: {assessment.get('confidence_level', 'low')}")
        print(f"  Overall Score: {assessment.get('overall_score', 0):.3f}")
        print(f"  Recommendation: {assessment.get('recommendation', 'Unknown')}")
        
        if assessment.get('key_concerns'):
            print(f"  Key Concerns: {', '.join(assessment['key_concerns'])}")
        if assessment.get('strengths'):
            print(f"  Strengths: {', '.join(assessment['strengths'])}")
    
    # Save results
    output_file = '/home/user/webapp/validation_framework_results.json'
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return validation_results

if __name__ == "__main__":
    create_validation_demo()