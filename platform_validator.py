#!/usr/bin/env python3
"""
REAL-TIME PLATFORM VALIDATION SYSTEM
====================================
Comprehensive validation framework that addresses critical statistical, 
mathematical, and engineering concerns identified in the platform.

CRITICAL ISSUES IDENTIFIED:
1. STATISTICAL: Extensive use of Math.random() for financial data simulation
2. MATHEMATICAL: No validation of portfolio optimization results
3. ENGINEERING: Missing error handling and data integrity checks
4. SECURITY: API keys and sensitive data handling
5. PERFORMANCE: Multiple setTimeout/setInterval without coordination
6. DATA QUALITY: No real-time data validation pipelines

This system implements real-time monitoring and validation across all components.
"""

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import traceback
from scipy import stats
import requests
from threading import Thread
import time
import hashlib
import sqlite3
import warnings
import os
import psutil
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PlatformValidator')

@dataclass
class ValidationResult:
    """Validation result structure"""
    component: str
    test_name: str
    status: str  # PASS, FAIL, WARNING, ERROR
    score: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    api_response_times: Dict[str, float]
    database_health: bool
    websocket_connections: int
    timestamp: datetime

class MathematicalValidator:
    """Validates mathematical operations and formulas"""
    
    @staticmethod
    def validate_portfolio_weights(weights: np.ndarray) -> ValidationResult:
        """Validate portfolio weight constraints"""
        try:
            # Check sum equals 1
            weight_sum = np.sum(weights)
            sum_valid = abs(weight_sum - 1.0) < 1e-6
            
            # Check no negative weights (long-only constraint)
            no_negatives = np.all(weights >= 0)
            
            # Check no single asset > 50% (concentration risk)
            max_weight = np.max(weights)
            concentration_ok = max_weight <= 0.5
            
            # Overall score
            score = sum([sum_valid, no_negatives, concentration_ok]) / 3
            
            status = "PASS" if score == 1.0 else "FAIL"
            severity = "CRITICAL" if score < 0.5 else "MEDIUM" if score < 1.0 else "LOW"
            
            return ValidationResult(
                component="Portfolio Optimization",
                test_name="Weight Constraints",
                status=status,
                score=score,
                message=f"Weight validation: Sum={weight_sum:.6f}, Max={max_weight:.3f}",
                details={
                    "weight_sum": weight_sum,
                    "sum_valid": sum_valid,
                    "no_negatives": no_negatives,
                    "max_weight": max_weight,
                    "concentration_ok": concentration_ok
                },
                timestamp=datetime.now(),
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                component="Portfolio Optimization",
                test_name="Weight Constraints",
                status="ERROR",
                score=0.0,
                message=f"Error validating weights: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )
    
    @staticmethod
    def validate_hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> ValidationResult:
        """Validate hyperbolic distance calculations"""
        try:
            # Check if points are in Poincaré ball (||x|| < 1, ||y|| < 1)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            
            x_valid = norm_x < 1.0
            y_valid = norm_y < 1.0
            
            if x_valid and y_valid:
                # Calculate hyperbolic distance
                diff = x - y
                norm_diff_sq = np.sum(diff * diff)
                denominator = (1 - norm_x**2) * (1 - norm_y**2)
                
                if denominator > 1e-10:  # Avoid division by zero
                    distance_arg = 1 + 2 * norm_diff_sq / denominator
                    distance = np.arccosh(max(1.0, distance_arg))
                    
                    # Validate distance properties
                    non_negative = distance >= 0
                    finite = np.isfinite(distance)
                    
                    score = sum([x_valid, y_valid, non_negative, finite]) / 4
                    status = "PASS" if score == 1.0 else "FAIL"
                else:
                    score = 0.0
                    status = "FAIL"
                    distance = float('inf')
            else:
                score = 0.0
                status = "FAIL"
                distance = float('nan')
            
            return ValidationResult(
                component="Hyperbolic Geometry",
                test_name="Distance Calculation", 
                status=status,
                score=score,
                message=f"Hyperbolic distance validation: {distance:.6f}",
                details={
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                    "x_valid": x_valid,
                    "y_valid": y_valid,
                    "distance": distance
                },
                timestamp=datetime.now(),
                severity="HIGH" if score < 0.8 else "LOW"
            )
            
        except Exception as e:
            return ValidationResult(
                component="Hyperbolic Geometry",
                test_name="Distance Calculation",
                status="ERROR", 
                score=0.0,
                message=f"Error in hyperbolic distance: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )

class StatisticalValidator:
    """Validates statistical assumptions and data quality"""
    
    @staticmethod
    def validate_return_distribution(returns: np.ndarray) -> ValidationResult:
        """Validate return distribution assumptions"""
        try:
            if len(returns) < 30:
                return ValidationResult(
                    component="Statistical Analysis",
                    test_name="Return Distribution",
                    status="WARNING",
                    score=0.5,
                    message="Insufficient data for reliable statistical tests",
                    details={"sample_size": len(returns)},
                    timestamp=datetime.now(),
                    severity="MEDIUM"
                )
            
            # Normality tests
            jarque_stat, jarque_p = stats.jarque_bera(returns)
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            
            # Stationarity test (ADF)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_stat, adf_p, _, _, _, _ = adfuller(returns)
                stationary = adf_p < 0.05
            except:
                adf_stat, adf_p = 0, 1
                stationary = False
            
            # Independence test (Ljung-Box)
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
                independent = lb_result['lb_pvalue'].iloc[-1] > 0.05
            except:
                independent = False
            
            # Outlier detection
            q1, q3 = np.percentile(returns, [25, 75])
            iqr = q3 - q1
            outlier_bounds = [q1 - 1.5*iqr, q3 + 1.5*iqr]
            outliers = np.sum((returns < outlier_bounds[0]) | (returns > outlier_bounds[1]))
            outlier_ratio = outliers / len(returns)
            
            # Calculate overall score
            normal_ok = jarque_p > 0.05 and shapiro_p > 0.05
            outliers_ok = outlier_ratio < 0.1
            
            score_components = [normal_ok, stationary, independent, outliers_ok]
            score = sum(score_components) / len(score_components)
            
            status = "PASS" if score >= 0.75 else "WARNING" if score >= 0.5 else "FAIL"
            severity = "LOW" if score >= 0.75 else "MEDIUM" if score >= 0.5 else "HIGH"
            
            return ValidationResult(
                component="Statistical Analysis",
                test_name="Return Distribution",
                status=status,
                score=score,
                message=f"Statistical tests: Normal={normal_ok}, Stationary={stationary}, Independent={independent}",
                details={
                    "jarque_bera": {"stat": jarque_stat, "p_value": jarque_p},
                    "shapiro_wilk": {"stat": shapiro_stat, "p_value": shapiro_p},
                    "adf_test": {"stat": adf_stat, "p_value": adf_p},
                    "outlier_ratio": outlier_ratio,
                    "sample_size": len(returns),
                    "stationary": stationary,
                    "independent": independent
                },
                timestamp=datetime.now(),
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                component="Statistical Analysis",
                test_name="Return Distribution",
                status="ERROR",
                score=0.0,
                message=f"Error in statistical validation: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )
    
    @staticmethod
    def detect_synthetic_data(data: np.ndarray) -> ValidationResult:
        """Detect if data appears to be synthetically generated (Math.random)"""
        try:
            # Test for uniform distribution (sign of Math.random usage)
            ks_stat, ks_p = stats.kstest(data, 'uniform', args=(data.min(), data.max() - data.min()))
            
            # Test for patterns typical of PRNG
            diff = np.diff(data)
            
            # Check for unrealistic uniformity in differences
            diff_std = np.std(diff)
            diff_range = np.ptp(diff)
            uniformity_ratio = diff_std / (diff_range / 12)  # For uniform: std = range/sqrt(12)
            
            # Check autocorrelation (should be low for good PRNG)
            if len(diff) > 1:
                autocorr = np.corrcoef(diff[:-1], diff[1:])[0, 1]
            else:
                autocorr = 0
            
            # Synthetic data indicators
            likely_uniform = ks_p > 0.1  # High p-value suggests uniform distribution
            suspicious_uniformity = abs(uniformity_ratio - 1) < 0.1
            low_autocorr = abs(autocorr) < 0.1
            
            synthetic_indicators = [likely_uniform, suspicious_uniformity, low_autocorr]
            synthetic_score = sum(synthetic_indicators) / len(synthetic_indicators)
            
            # Reverse score (lower is better for real data)
            score = 1.0 - synthetic_score
            
            if synthetic_score > 0.7:
                status = "FAIL"
                severity = "CRITICAL"
                message = "Data appears synthetically generated (Math.random pattern detected)"
            elif synthetic_score > 0.5:
                status = "WARNING" 
                severity = "HIGH"
                message = "Data shows signs of synthetic generation"
            else:
                status = "PASS"
                severity = "LOW"
                message = "Data appears genuine"
            
            return ValidationResult(
                component="Data Quality",
                test_name="Synthetic Data Detection",
                status=status,
                score=score,
                message=message,
                details={
                    "ks_test": {"stat": ks_stat, "p_value": ks_p},
                    "uniformity_ratio": uniformity_ratio,
                    "autocorrelation": autocorr,
                    "synthetic_score": synthetic_score,
                    "likely_uniform": likely_uniform,
                    "suspicious_uniformity": suspicious_uniformity
                },
                timestamp=datetime.now(),
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                component="Data Quality",
                test_name="Synthetic Data Detection",
                status="ERROR",
                score=0.0,
                message=f"Error detecting synthetic data: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )

class EngineeringValidator:
    """Validates engineering practices and system reliability"""
    
    @staticmethod
    def validate_api_response_times(base_url: str) -> ValidationResult:
        """Validate API response times and availability"""
        try:
            endpoints = [
                '/api/market/BTC-USD',
                '/api/signals/BTC-USD', 
                '/api/portfolio/metrics',
                '/api/model/performance'
            ]
            
            response_times = {}
            errors = []
            
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    response_time = time.time() - start_time
                    
                    response_times[endpoint] = response_time
                    
                    if response.status_code != 200:
                        errors.append(f"{endpoint}: HTTP {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    errors.append(f"{endpoint}: {str(e)}")
                    response_times[endpoint] = float('inf')
            
            # Calculate metrics
            valid_times = [t for t in response_times.values() if t != float('inf')]
            avg_response_time = np.mean(valid_times) if valid_times else float('inf')
            max_response_time = max(valid_times) if valid_times else float('inf')
            
            # Scoring
            fast_responses = sum(1 for t in valid_times if t < 1.0) / len(endpoints)
            available_endpoints = (len(endpoints) - len(errors)) / len(endpoints)
            
            score = (fast_responses + available_endpoints) / 2
            
            if score >= 0.8:
                status = "PASS"
                severity = "LOW"
            elif score >= 0.6:
                status = "WARNING"
                severity = "MEDIUM" 
            else:
                status = "FAIL"
                severity = "HIGH"
            
            return ValidationResult(
                component="API Performance",
                test_name="Response Time Validation",
                status=status,
                score=score,
                message=f"Avg response: {avg_response_time:.2f}s, Availability: {available_endpoints:.0%}",
                details={
                    "response_times": response_times,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "errors": errors,
                    "availability": available_endpoints
                },
                timestamp=datetime.now(),
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                component="API Performance",
                test_name="Response Time Validation",
                status="ERROR",
                score=0.0,
                message=f"Error validating API: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )
    
    @staticmethod
    def validate_memory_usage() -> ValidationResult:
        """Validate system memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_usage = memory.percent
            swap_usage = swap.percent
            
            # Scoring based on usage thresholds
            if memory_usage < 70:
                memory_score = 1.0
            elif memory_usage < 85:
                memory_score = 0.7
            elif memory_usage < 95:
                memory_score = 0.3
            else:
                memory_score = 0.0
            
            if swap_usage < 10:
                swap_score = 1.0
            elif swap_usage < 50:
                swap_score = 0.5
            else:
                swap_score = 0.0
            
            score = (memory_score + swap_score) / 2
            
            if score >= 0.8:
                status = "PASS"
                severity = "LOW"
            elif score >= 0.5:
                status = "WARNING"
                severity = "MEDIUM"
            else:
                status = "FAIL" 
                severity = "HIGH"
            
            return ValidationResult(
                component="System Resources",
                test_name="Memory Usage",
                status=status,
                score=score,
                message=f"Memory: {memory_usage:.1f}%, Swap: {swap_usage:.1f}%",
                details={
                    "memory_percent": memory_usage,
                    "memory_available": memory.available,
                    "memory_total": memory.total,
                    "swap_percent": swap_usage,
                    "swap_used": swap.used,
                    "swap_total": swap.total
                },
                timestamp=datetime.now(),
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                component="System Resources",
                test_name="Memory Usage",
                status="ERROR",
                score=0.0,
                message=f"Error checking memory: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                severity="CRITICAL"
            )

class RealTimeValidator:
    """Main real-time validation system"""
    
    def __init__(self, db_path: str = "validation.db"):
        self.db_path = db_path
        self.math_validator = MathematicalValidator()
        self.stat_validator = StatisticalValidator()
        self.eng_validator = EngineeringValidator()
        self.running = False
        self.websocket_clients = set()
        self.validation_results = []
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for validation results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT,
                test_name TEXT,
                status TEXT,
                score REAL,
                message TEXT,
                details TEXT,
                timestamp TEXT,
                severity TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_latency REAL,
                api_response_times TEXT,
                database_health INTEGER,
                websocket_connections INTEGER,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_result(self, result: ValidationResult):
        """Store validation result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO validation_results 
            (component, test_name, status, score, message, details, timestamp, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.component,
            result.test_name,
            result.status,
            result.score,
            result.message,
            json.dumps(result.details),
            result.timestamp.isoformat(),
            result.severity
        ))
        
        conn.commit()
        conn.close()
    
    async def broadcast_result(self, result: ValidationResult):
        """Broadcast validation result to WebSocket clients"""
        if self.websocket_clients:
            message = json.dumps({
                "type": "validation_result",
                "data": asdict(result),
                "timestamp": result.timestamp.isoformat()
            })
            
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected
    
    async def run_validation_cycle(self):
        """Run a complete validation cycle"""
        logger.info("Starting validation cycle")
        
        # Generate test data (this would be replaced with real market data)
        test_returns = np.random.normal(0.001, 0.02, 100)  # Realistic return simulation
        test_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        test_points = np.random.uniform(-0.8, 0.8, (2, 3))  # Points in Poincaré ball
        
        # Mathematical validation
        weight_result = self.math_validator.validate_portfolio_weights(test_weights)
        await self.broadcast_result(weight_result)
        self.store_result(weight_result)
        
        hyperbolic_result = self.math_validator.validate_hyperbolic_distance(
            test_points[0], test_points[1]
        )
        await self.broadcast_result(hyperbolic_result)
        self.store_result(hyperbolic_result)
        
        # Statistical validation
        return_result = self.stat_validator.validate_return_distribution(test_returns)
        await self.broadcast_result(return_result)
        self.store_result(return_result)
        
        synthetic_result = self.stat_validator.detect_synthetic_data(test_returns)
        await self.broadcast_result(synthetic_result)
        self.store_result(synthetic_result)
        
        # Engineering validation
        memory_result = self.eng_validator.validate_memory_usage()
        await self.broadcast_result(memory_result)
        self.store_result(memory_result)
        
        # API validation (if server is running)
        try:
            api_result = self.eng_validator.validate_api_response_times("http://localhost:8000")
            await self.broadcast_result(api_result)
            self.store_result(api_result)
        except Exception as e:
            logger.warning(f"API validation failed: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        logger.info(f"New WebSocket client connected: {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        
        try:
            # Send recent results to new client
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_results 
                ORDER BY timestamp DESC LIMIT 50
            """)
            results = cursor.fetchall()
            conn.close()
            
            for row in results:
                result_data = {
                    "component": row[1],
                    "test_name": row[2],
                    "status": row[3],
                    "score": row[4],
                    "message": row[5],
                    "details": json.loads(row[6]),
                    "timestamp": row[7],
                    "severity": row[8]
                }
                
                await websocket.send(json.dumps({
                    "type": "historical_result",
                    "data": result_data
                }))
            
            # Keep connection alive
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        finally:
            self.websocket_clients.discard(websocket)
    
    async def start_validation_server(self, host="localhost", port=9000):
        """Start the validation WebSocket server"""
        logger.info(f"Starting validation server on {host}:{port}")
        
        async def validation_loop():
            while self.running:
                try:
                    await self.run_validation_cycle()
                    await asyncio.sleep(30)  # Run every 30 seconds
                except Exception as e:
                    logger.error(f"Error in validation cycle: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        # Start validation loop
        self.running = True
        validation_task = asyncio.create_task(validation_loop())
        
        # Start WebSocket server
        start_server = websockets.serve(self.websocket_handler, host, port)
        
        logger.info("Real-time validation system started")
        await asyncio.gather(start_server, validation_task)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get current validation summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent results (last hour)
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        cursor.execute("""
            SELECT component, status, score, severity, COUNT(*) as count
            FROM validation_results 
            WHERE timestamp > ?
            GROUP BY component, status, severity
        """, (one_hour_ago,))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {
            "total_tests": sum(row[4] for row in results),
            "passed": sum(row[4] for row in results if row[1] == "PASS"),
            "failed": sum(row[4] for row in results if row[1] == "FAIL"),
            "warnings": sum(row[4] for row in results if row[1] == "WARNING"),
            "errors": sum(row[4] for row in results if row[1] == "ERROR"),
            "critical_issues": sum(row[4] for row in results if row[3] == "CRITICAL"),
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Group by component
        for row in results:
            component = row[0]
            if component not in summary["components"]:
                summary["components"][component] = {
                    "total": 0, "passed": 0, "failed": 0, "warnings": 0, "errors": 0
                }
            
            summary["components"][component]["total"] += row[4]
            if row[1] == "PASS":
                summary["components"][component]["passed"] += row[4]
            elif row[1] == "FAIL":
                summary["components"][component]["failed"] += row[4]
            elif row[1] == "WARNING":
                summary["components"][component]["warnings"] += row[4]
            elif row[1] == "ERROR":
                summary["components"][component]["errors"] += row[4]
        
        return summary

def main():
    """Main function to start the validation system"""
    print("🔍 Starting Real-Time Platform Validation System...")
    
    validator = RealTimeValidator()
    
    try:
        asyncio.run(validator.start_validation_server())
    except KeyboardInterrupt:
        print("\n✅ Validation system stopped")
        validator.running = False

if __name__ == "__main__":
    main()