/**
 * HFT Performance Monitor & Latency Analysis System
 * Real-time monitoring of all system components for ultra-low latency optimization
 */

class HFTPerformanceMonitor {
    constructor() {
        this.startTime = performance.now();
        this.isMonitoring = false;
        
        // Performance buckets for different components
        this.metrics = {
            marketDataIngestion: {
                latencies: [],
                throughput: [],
                errors: [],
                timestamps: []
            },
            arbitrageDetection: {
                latencies: [],
                throughput: [],
                opportunities: [],
                timestamps: []
            },
            orderExecution: {
                latencies: [],
                fillRates: [],
                slippage: [],
                timestamps: []
            },
            networkPerformance: {
                roundTripTimes: [],
                bandwidth: [],
                packetLoss: [],
                timestamps: []
            },
            systemResources: {
                cpuUsage: [],
                memoryUsage: [],
                gcTimes: [],
                timestamps: []
            }
        };
        
        // Performance targets (microseconds)
        this.targets = {
            marketDataIngestion: 50,    // 50μs max
            arbitrageDetection: 100,    // 100μs max
            orderExecution: 200,        // 200μs max
            networkRoundTrip: 500,      // 500μs max
            totalLatency: 1000          // 1ms total max
        };
        
        // Alert thresholds
        this.alertThresholds = {
            latencyP95: 1.5,    // 1.5x target
            latencyP99: 2.0,    // 2x target
            throughputDrop: 0.8, // 20% drop
            errorRate: 0.01     // 1% error rate
        };
        
        // Component instances for monitoring
        this.components = {
            hftEngine: null,
            wasmOptimizer: null,
            gpuEngine: null,
            networkOptimizer: null
        };
        
        this.initializeMonitoring();
    }
    
    initializeMonitoring() {
        console.log('🚀 Initializing HFT Performance Monitor...');
        
        // Setup performance observers
        this.setupPerformanceObservers();
        
        // Initialize real-time monitoring
        this.setupRealtimeMonitoring();
        
        // Setup alerting system
        this.setupAlertingSystem();
        
        console.log('✅ Performance monitoring initialized');
        console.log(`🎯 Latency targets: Market=${this.targets.marketDataIngestion}μs, Arbitrage=${this.targets.arbitrageDetection}μs, Order=${this.targets.orderExecution}μs`);
    }
    
    // Setup performance observers for automatic monitoring
    setupPerformanceObservers() {
        // Monitor long tasks (should be minimal in HFT)
        if ('PerformanceObserver' in window) {
            const longTaskObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach(entry => {
                    if (entry.duration > 1) { // Any task > 1ms is problematic
                        console.warn(`⚠️ Long task detected: ${entry.duration.toFixed(2)}ms`);
                        this.recordAlert('longTask', {
                            duration: entry.duration,
                            startTime: entry.startTime,
                            name: entry.name
                        });
                    }
                });
            });
            
            try {
                longTaskObserver.observe({ entryTypes: ['longtask'] });
            } catch (e) {
                console.warn('Long task observer not supported');
            }
            
            // Monitor measures and marks
            const measureObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach(entry => {
                    this.recordPerformanceMeasure(entry);
                });
            });
            
            try {
                measureObserver.observe({ entryTypes: ['measure'] });
            } catch (e) {
                console.warn('Measure observer not supported');
            }
        }
    }
    
    // Record performance measures
    recordPerformanceMeasure(entry) {
        const component = this.identifyComponent(entry.name);
        const latency = entry.duration * 1000; // Convert to microseconds
        
        if (component && this.metrics[component]) {
            this.metrics[component].latencies.push(latency);
            this.metrics[component].timestamps.push(performance.now());
            
            // Check against targets
            const target = this.targets[component.replace('Performance', '')];
            if (target && latency > target) {
                this.recordAlert('latencyExceeded', {
                    component,
                    latency,
                    target,
                    ratio: latency / target
                });
            }
        }
    }
    
    // Identify component from performance measure name
    identifyComponent(measureName) {
        const mapping = {
            'marketData': 'marketDataIngestion',
            'arbitrage': 'arbitrageDetection',
            'order': 'orderExecution',
            'network': 'networkPerformance'
        };
        
        for (const [key, component] of Object.entries(mapping)) {
            if (measureName.toLowerCase().includes(key)) {
                return component;
            }
        }
        
        return null;
    }
    
    // Setup real-time monitoring loops
    setupRealtimeMonitoring() {
        // High-frequency monitoring (every 10ms)
        this.highFreqInterval = setInterval(() => {
            this.collectSystemMetrics();
            this.analyzePerformanceTrends();
        }, 10);
        
        // Medium-frequency analysis (every 100ms)
        this.mediumFreqInterval = setInterval(() => {
            this.analyzeComponentPerformance();
            this.detectPerformanceAnomalies();
        }, 100);
        
        // Low-frequency reporting (every 1000ms)
        this.lowFreqInterval = setInterval(() => {
            this.generatePerformanceReport();
            this.cleanupOldMetrics();
        }, 1000);
    }
    
    // Collect system resource metrics
    collectSystemMetrics() {
        const now = performance.now();
        
        // Memory usage
        if (performance.memory) {
            const memoryUsage = {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                percentage: (performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize) * 100
            };
            
            this.metrics.systemResources.memoryUsage.push(memoryUsage);
            this.metrics.systemResources.timestamps.push(now);
            
            // Alert on high memory usage
            if (memoryUsage.percentage > 80) {
                this.recordAlert('highMemoryUsage', memoryUsage);
            }
        }
        
        // Estimate CPU usage (simplified)
        const cpuEstimate = this.estimateCPUUsage();
        this.metrics.systemResources.cpuUsage.push(cpuEstimate);
    }
    
    // Estimate CPU usage based on timing
    estimateCPUUsage() {
        const start = performance.now();
        
        // Perform a small computational task to estimate CPU load
        let sum = 0;
        for (let i = 0; i < 1000; i++) {
            sum += Math.random();
        }
        
        const end = performance.now();
        const executionTime = end - start;
        
        // Normalize to percentage (baseline: ~0.1ms for 1000 operations)
        const baselineTime = 0.1;
        const cpuUsage = Math.min(100, (executionTime / baselineTime) * 10);
        
        return cpuUsage;
    }
    
    // Analyze performance trends
    analyzePerformanceTrends() {
        Object.keys(this.metrics).forEach(component => {
            const latencies = this.metrics[component].latencies;
            
            if (latencies.length >= 10) {
                const recent = latencies.slice(-10);
                const older = latencies.slice(-20, -10);
                
                if (older.length > 0) {
                    const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
                    const olderAvg = older.reduce((sum, val) => sum + val, 0) / older.length;
                    
                    const trend = (recentAvg - olderAvg) / olderAvg;
                    
                    // Alert on significant performance degradation
                    if (trend > 0.2) { // 20% increase in latency
                        this.recordAlert('performanceDegradation', {
                            component,
                            trend: trend * 100,
                            recentAvg,
                            olderAvg
                        });
                    }
                }
            }
        });
    }
    
    // Analyze component performance
    analyzeComponentPerformance() {
        Object.entries(this.metrics).forEach(([component, metrics]) => {
            if (metrics.latencies.length === 0) return;
            
            const stats = this.calculateStatistics(metrics.latencies);
            const target = this.targets[component.replace('Performance', '')] || 1000;
            
            // Check P95 and P99 against thresholds
            if (stats.p95 > target * this.alertThresholds.latencyP95) {
                this.recordAlert('p95Exceeded', {
                    component,
                    p95: stats.p95,
                    target,
                    threshold: target * this.alertThresholds.latencyP95
                });
            }
            
            if (stats.p99 > target * this.alertThresholds.latencyP99) {
                this.recordAlert('p99Exceeded', {
                    component,
                    p99: stats.p99,
                    target,
                    threshold: target * this.alertThresholds.latencyP99
                });
            }
        });
    }
    
    // Detect performance anomalies
    detectPerformanceAnomalies() {
        // Detect sudden throughput drops
        Object.entries(this.metrics).forEach(([component, metrics]) => {
            if (metrics.throughput && metrics.throughput.length >= 5) {
                const recent = metrics.throughput.slice(-3);
                const baseline = metrics.throughput.slice(-10, -3);
                
                if (baseline.length > 0) {
                    const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
                    const baselineAvg = baseline.reduce((sum, val) => sum + val, 0) / baseline.length;
                    
                    if (recentAvg < baselineAvg * this.alertThresholds.throughputDrop) {
                        this.recordAlert('throughputDrop', {
                            component,
                            recentThroughput: recentAvg,
                            baselineThroughput: baselineAvg,
                            dropPercentage: ((baselineAvg - recentAvg) / baselineAvg * 100).toFixed(1)
                        });
                    }
                }
            }
        });
    }
    
    // Calculate comprehensive statistics
    calculateStatistics(data) {
        if (data.length === 0) {
            return { count: 0, avg: 0, min: 0, max: 0, p50: 0, p95: 0, p99: 0, std: 0 };
        }
        
        const sorted = [...data].sort((a, b) => a - b);
        const sum = sorted.reduce((acc, val) => acc + val, 0);
        const avg = sum / sorted.length;
        
        // Calculate standard deviation
        const variance = sorted.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / sorted.length;
        const std = Math.sqrt(variance);
        
        return {
            count: sorted.length,
            avg: avg,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            p50: sorted[Math.floor(sorted.length * 0.5)],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            p99: sorted[Math.floor(sorted.length * 0.99)],
            std: std
        };
    }
    
    // Setup alerting system
    setupAlertingSystem() {
        this.alerts = [];
        this.alertCallbacks = new Set();
        
        // Alert on browser visibility changes (affects performance)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.recordAlert('browserHidden', { timestamp: performance.now() });
            } else {
                this.recordAlert('browserVisible', { timestamp: performance.now() });
            }
        });
        
        console.log('✅ Alerting system initialized');
    }
    
    // Record performance alert
    recordAlert(type, data) {
        const alert = {
            type,
            timestamp: performance.now(),
            data,
            severity: this.getAlertSeverity(type, data)
        };
        
        this.alerts.push(alert);
        
        // Keep only last 100 alerts
        if (this.alerts.length > 100) {
            this.alerts = this.alerts.slice(-100);
        }
        
        // Log critical alerts immediately
        if (alert.severity === 'critical') {
            console.error(`🚨 CRITICAL: ${type}`, data);
        } else if (alert.severity === 'warning') {
            console.warn(`⚠️ WARNING: ${type}`, data);
        }
        
        // Notify callbacks
        this.alertCallbacks.forEach(callback => {
            try {
                callback(alert);
            } catch (error) {
                console.error('Alert callback error:', error);
            }
        });
    }
    
    // Determine alert severity
    getAlertSeverity(type, data) {
        const criticalAlerts = ['p99Exceeded', 'longTask', 'throughputDrop'];
        const warningAlerts = ['p95Exceeded', 'performanceDegradation', 'highMemoryUsage'];
        
        if (criticalAlerts.includes(type)) {
            return 'critical';
        } else if (warningAlerts.includes(type)) {
            return 'warning';
        }
        
        return 'info';
    }
    
    // Register component for monitoring
    registerComponent(name, instance) {
        this.components[name] = instance;
        console.log(`✅ Registered component for monitoring: ${name}`);
    }
    
    // Start monitoring specific metrics
    startMonitoring() {
        if (this.isMonitoring) {
            console.warn('⚠️ Monitoring already started');
            return;
        }
        
        this.isMonitoring = true;
        console.log('🟢 HFT Performance monitoring started');
        
        // Setup component-specific monitoring
        Object.entries(this.components).forEach(([name, component]) => {
            if (component && typeof component.getPerformanceMetrics === 'function') {
                this.monitorComponent(name, component);
            }
        });
    }
    
    // Monitor individual component
    monitorComponent(name, component) {
        const monitor = () => {
            try {
                const metrics = component.getPerformanceMetrics();
                this.integrateComponentMetrics(name, metrics);
            } catch (error) {
                console.error(`Error monitoring ${name}:`, error);
            }
        };
        
        // Monitor every 50ms
        setInterval(monitor, 50);
    }
    
    // Integrate metrics from components
    integrateComponentMetrics(componentName, metrics) {
        // Map component metrics to our tracking structure
        if (metrics.latency && metrics.latency.avg) {
            const targetMetric = this.mapComponentToMetric(componentName);
            if (targetMetric && this.metrics[targetMetric]) {
                this.metrics[targetMetric].latencies.push(metrics.latency.avg);
                this.metrics[targetMetric].timestamps.push(performance.now());
            }
        }
        
        if (metrics.throughput) {
            const targetMetric = this.mapComponentToMetric(componentName);
            if (targetMetric && this.metrics[targetMetric]) {
                this.metrics[targetMetric].throughput.push(metrics.throughput);
            }
        }
    }
    
    // Map component names to metric categories
    mapComponentToMetric(componentName) {
        const mapping = {
            'hftEngine': 'arbitrageDetection',
            'wasmOptimizer': 'arbitrageDetection',
            'gpuEngine': 'arbitrageDetection',
            'networkOptimizer': 'networkPerformance'
        };
        
        return mapping[componentName] || null;
    }
    
    // Generate comprehensive performance report
    generatePerformanceReport() {
        const report = {
            timestamp: new Date().toISOString(),
            uptime: performance.now() - this.startTime,
            summary: {},
            components: {},
            alerts: {
                total: this.alerts.length,
                critical: this.alerts.filter(a => a.severity === 'critical').length,
                warnings: this.alerts.filter(a => a.severity === 'warning').length,
                recent: this.alerts.slice(-10)
            },
            systemHealth: this.assessSystemHealth()
        };
        
        // Generate summary statistics for each component
        Object.entries(this.metrics).forEach(([component, metrics]) => {
            const stats = this.calculateStatistics(metrics.latencies);
            const target = this.targets[component.replace('Performance', '')] || 1000;
            
            report.components[component] = {
                latency: stats,
                target: target,
                healthScore: this.calculateHealthScore(stats, target),
                throughput: metrics.throughput.length > 0 ? 
                    this.calculateStatistics(metrics.throughput) : null
            };
            
            // Add to summary
            report.summary[component] = {
                avgLatency: stats.avg.toFixed(2) + 'μs',
                p95Latency: stats.p95.toFixed(2) + 'μs',
                status: stats.p95 > target * this.alertThresholds.latencyP95 ? 'DEGRADED' : 'OK'
            };
        });
        
        // Emit report event
        const event = new CustomEvent('performanceReport', { detail: report });
        document.dispatchEvent(event);
        
        return report;
    }
    
    // Calculate health score for component
    calculateHealthScore(stats, target) {
        if (stats.count === 0) return 100;
        
        // Score based on how close to target we are
        const latencyScore = Math.max(0, 100 - (stats.avg / target) * 50);
        const consistencyScore = Math.max(0, 100 - (stats.std / stats.avg) * 100);
        
        return Math.round((latencyScore + consistencyScore) / 2);
    }
    
    // Assess overall system health
    assessSystemHealth() {
        const componentHealthScores = Object.entries(this.metrics).map(([component, metrics]) => {
            const stats = this.calculateStatistics(metrics.latencies);
            const target = this.targets[component.replace('Performance', '')] || 1000;
            return this.calculateHealthScore(stats, target);
        });
        
        const avgHealth = componentHealthScores.length > 0 ?
            componentHealthScores.reduce((sum, score) => sum + score, 0) / componentHealthScores.length : 100;
        
        const criticalAlerts = this.alerts.filter(a => 
            a.severity === 'critical' && 
            performance.now() - a.timestamp < 60000 // Last minute
        ).length;
        
        let status = 'HEALTHY';
        if (avgHealth < 70 || criticalAlerts > 0) {
            status = 'DEGRADED';
        }
        if (avgHealth < 50 || criticalAlerts > 5) {
            status = 'CRITICAL';
        }
        
        return {
            status,
            score: Math.round(avgHealth),
            criticalAlerts,
            recommendation: this.getHealthRecommendation(status, avgHealth)
        };
    }
    
    // Get health-based recommendations
    getHealthRecommendation(status, score) {
        if (status === 'CRITICAL') {
            return 'Immediate optimization required. Check system resources and network connectivity.';
        } else if (status === 'DEGRADED') {
            return 'Performance degradation detected. Consider optimizing slow components.';
        } else if (score < 90) {
            return 'Good performance with room for optimization.';
        }
        return 'Excellent performance. System operating within targets.';
    }
    
    // Add alert callback
    onAlert(callback) {
        this.alertCallbacks.add(callback);
    }
    
    // Remove alert callback
    offAlert(callback) {
        this.alertCallbacks.delete(callback);
    }
    
    // Cleanup old metrics to prevent memory leaks
    cleanupOldMetrics() {
        const maxAge = 300000; // 5 minutes
        const now = performance.now();
        
        Object.values(this.metrics).forEach(componentMetrics => {
            if (componentMetrics.timestamps) {
                const cutoffIndex = componentMetrics.timestamps.findIndex(
                    timestamp => now - timestamp <= maxAge
                );
                
                if (cutoffIndex > 0) {
                    // Cleanup old data
                    Object.keys(componentMetrics).forEach(key => {
                        if (Array.isArray(componentMetrics[key])) {
                            componentMetrics[key] = componentMetrics[key].slice(cutoffIndex);
                        }
                    });
                }
            }
        });
    }
    
    // Get real-time dashboard data
    getDashboardData() {
        return {
            systemHealth: this.assessSystemHealth(),
            realtimeMetrics: this.getRealtimeMetrics(),
            alerts: this.alerts.slice(-5), // Last 5 alerts
            targets: this.targets,
            uptime: performance.now() - this.startTime
        };
    }
    
    // Get real-time metrics (last 10 data points)
    getRealtimeMetrics() {
        const realtime = {};
        
        Object.entries(this.metrics).forEach(([component, metrics]) => {
            realtime[component] = {
                latency: metrics.latencies.slice(-10),
                throughput: metrics.throughput ? metrics.throughput.slice(-10) : [],
                timestamps: metrics.timestamps.slice(-10)
            };
        });
        
        return realtime;
    }
    
    // Stop monitoring
    stopMonitoring() {
        this.isMonitoring = false;
        
        if (this.highFreqInterval) clearInterval(this.highFreqInterval);
        if (this.mediumFreqInterval) clearInterval(this.mediumFreqInterval);
        if (this.lowFreqInterval) clearInterval(this.lowFreqInterval);
        
        console.log('🔴 HFT Performance monitoring stopped');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HFTPerformanceMonitor;
}

console.log('📦 HFT Performance Monitor module loaded');