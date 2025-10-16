/**
 * GOMNA Real-Time Monitoring System
 * Advanced real-time data monitoring and alerting system
 */

class RealTimeMonitor {
    constructor(workflowEngine, analyticsDashboard) {
        this.workflowEngine = workflowEngine;
        this.analyticsDashboard = analyticsDashboard;
        this.monitoringState = {
            active: false,
            alerts: [],
            thresholds: {
                sensorFailure: 95, // Sensor availability percentage
                dataQuality: 90,   // Minimum data quality percentage
                processingLatency: 100, // Maximum latency in ms
                accuracy: 85,      // Minimum accuracy percentage
                throughput: 1000   // Minimum throughput per second
            },
            alertLevels: {
                info: '#2196F3',
                warning: '#FF9800',
                error: '#F44336',
                critical: '#B71C1C'
            }
        };
        this.websocket = null;
        this.monitoringInterval = null;
        this.historicalData = {
            sensors: [],
            performance: [],
            alerts: []
        };
        this.init();
    }

    init() {
        console.log('Initializing Real-Time Monitor...');
        this.setupMonitoring();
        this.setupAlertSystem();
        this.setupWebSocketConnection();
        this.startMonitoring();
        console.log('Real-Time Monitor initialized successfully');
    }

    setupMonitoring() {
        // Initialize monitoring configuration
        this.monitoringConfig = {
            updateInterval: 2000, // 2 seconds
            maxHistoryPoints: 200,
            alertTimeout: 30000, // 30 seconds
            criticalAlertTimeout: 5000 // 5 seconds for critical alerts
        };

        // Setup performance monitoring
        this.performanceMonitor = {
            startTime: Date.now(),
            metrics: {
                cpuUsage: 0,
                memoryUsage: 0,
                networkLatency: 0,
                diskIO: 0
            }
        };
    }

    setupAlertSystem() {
        // Create alert container if it doesn't exist
        if (!document.getElementById('alertContainer')) {
            const alertContainer = document.createElement('div');
            alertContainer.id = 'alertContainer';
            alertContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 2000;
                max-width: 400px;
                pointer-events: none;
            `;
            document.body.appendChild(alertContainer);
        }

        // Setup alert management
        this.alertManager = {
            activeAlerts: new Map(),
            alertQueue: [],
            maxVisibleAlerts: 5
        };
    }

    setupWebSocketConnection() {
        // Simulate WebSocket connection for real-time data
        // In a real implementation, this would connect to a backend service
        this.simulateWebSocketConnection();
    }

    simulateWebSocketConnection() {
        console.log('Simulating WebSocket connection for real-time data...');
        
        // Simulate incoming real-time data
        setInterval(() => {
            const mockData = this.generateMockRealTimeData();
            this.handleRealTimeData(mockData);
        }, 1000);
    }

    generateMockRealTimeData() {
        return {
            timestamp: new Date().toISOString(),
            sensors: {
                active: 1200 + Math.floor(Math.random() * 100),
                quality: 95 + Math.random() * 5,
                latency: 8 + Math.random() * 15,
                throughput: 2000 + Math.random() * 1000
            },
            performance: {
                cpu: 20 + Math.random() * 60,
                memory: 30 + Math.random() * 40,
                network: 10 + Math.random() * 40,
                disk: 5 + Math.random() * 20
            },
            workflow: {
                stage: ['sensor_init', 'feature_extraction', 'validation', 'decision_support'][Math.floor(Math.random() * 4)],
                progress: Math.random() * 100,
                status: ['running', 'completed', 'idle'][Math.floor(Math.random() * 3)]
            },
            anomalies: Math.random() > 0.9 ? this.generateAnomaly() : null
        };
    }

    generateAnomaly() {
        const anomalies = [
            { type: 'sensor_failure', severity: 'error', message: 'Sensor cluster offline' },
            { type: 'data_quality', severity: 'warning', message: 'Data quality degraded' },
            { type: 'high_latency', severity: 'warning', message: 'Processing latency elevated' },
            { type: 'accuracy_drop', severity: 'error', message: 'Model accuracy below threshold' },
            { type: 'system_overload', severity: 'critical', message: 'System resources critical' }
        ];
        
        return anomalies[Math.floor(Math.random() * anomalies.length)];
    }

    handleRealTimeData(data) {
        // Store historical data
        this.storeHistoricalData(data);
        
        // Check for threshold violations
        this.checkThresholds(data);
        
        // Update UI components
        this.updateRealTimeUI(data);
        
        // Handle anomalies
        if (data.anomalies) {
            this.handleAnomaly(data.anomalies);
        }
        
        // Notify observers
        this.notifyObservers('realTimeUpdate', data);
    }

    storeHistoricalData(data) {
        const timestamp = new Date(data.timestamp);
        
        // Store sensor data
        this.historicalData.sensors.push({
            timestamp: timestamp,
            ...data.sensors
        });
        
        // Store performance data
        this.historicalData.performance.push({
            timestamp: timestamp,
            ...data.performance
        });
        
        // Limit historical data size
        const maxPoints = this.monitoringConfig.maxHistoryPoints;
        if (this.historicalData.sensors.length > maxPoints) {
            this.historicalData.sensors = this.historicalData.sensors.slice(-maxPoints);
        }
        if (this.historicalData.performance.length > maxPoints) {
            this.historicalData.performance = this.historicalData.performance.slice(-maxPoints);
        }
    }

    checkThresholds(data) {
        const violations = [];
        
        // Check sensor thresholds
        if (data.sensors.quality < this.monitoringState.thresholds.dataQuality) {
            violations.push({
                type: 'data_quality',
                severity: 'warning',
                current: data.sensors.quality,
                threshold: this.monitoringState.thresholds.dataQuality,
                message: `Data quality (${data.sensors.quality.toFixed(1)}%) below threshold`
            });
        }
        
        if (data.sensors.latency > this.monitoringState.thresholds.processingLatency) {
            violations.push({
                type: 'high_latency',
                severity: 'warning',
                current: data.sensors.latency,
                threshold: this.monitoringState.thresholds.processingLatency,
                message: `Processing latency (${data.sensors.latency.toFixed(0)}ms) above threshold`
            });
        }
        
        if (data.sensors.throughput < this.monitoringState.thresholds.throughput) {
            violations.push({
                type: 'low_throughput',
                severity: 'warning',
                current: data.sensors.throughput,
                threshold: this.monitoringState.thresholds.throughput,
                message: `Throughput (${data.sensors.throughput.toFixed(0)}/s) below threshold`
            });
        }
        
        // Check performance thresholds
        if (data.performance.cpu > 90) {
            violations.push({
                type: 'high_cpu',
                severity: data.performance.cpu > 95 ? 'critical' : 'error',
                current: data.performance.cpu,
                threshold: 90,
                message: `CPU usage (${data.performance.cpu.toFixed(1)}%) critical`
            });
        }
        
        if (data.performance.memory > 85) {
            violations.push({
                type: 'high_memory',
                severity: data.performance.memory > 95 ? 'critical' : 'warning',
                current: data.performance.memory,
                threshold: 85,
                message: `Memory usage (${data.performance.memory.toFixed(1)}%) high`
            });
        }
        
        // Process violations
        violations.forEach(violation => {
            this.createAlert(violation);
        });
    }

    updateRealTimeUI(data) {
        // Update real-time metrics display
        const updateElement = (id, value, format = '') => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value + format;
                
                // Add visual feedback for changes
                element.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                }, 200);
            }
        };
        
        // Update sensor metrics
        updateElement('throughputMetric', (data.sensors.throughput / 1000).toFixed(1), 'K');
        updateElement('latencyMetric', data.sensors.latency.toFixed(0), 'ms');
        updateElement('accuracyMetric', data.sensors.quality.toFixed(1), '%');
        
        // Update system performance indicators
        this.updatePerformanceIndicators(data.performance);
        
        // Update workflow status
        this.updateWorkflowStatus(data.workflow);
        
        // Update last update time
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = 'just now';
        }
    }

    updatePerformanceIndicators(performance) {
        // Create or update performance indicators
        if (!document.getElementById('performanceIndicators')) {
            this.createPerformanceIndicators();
        }
        
        const indicators = {
            cpu: performance.cpu,
            memory: performance.memory,
            network: performance.network,
            disk: performance.disk
        };
        
        Object.entries(indicators).forEach(([type, value]) => {
            const indicator = document.getElementById(`${type}Indicator`);
            if (indicator) {
                const percentage = Math.min(100, Math.max(0, value));
                const bar = indicator.querySelector('.performance-bar-fill');
                if (bar) {
                    bar.style.width = percentage + '%';
                    bar.className = `performance-bar-fill ${this.getPerformanceLevel(percentage)}`;
                }
                
                const label = indicator.querySelector('.performance-label');
                if (label) {
                    label.textContent = `${type.toUpperCase()}: ${percentage.toFixed(1)}%`;
                }
            }
        });
    }

    createPerformanceIndicators() {
        const container = document.querySelector('.control-panel');
        if (!container) return;
        
        const indicatorsHTML = `
            <div id="performanceIndicators" style="margin-top: 1rem;">
                <h4 style="margin-bottom: 1rem; color: var(--primary-color);">System Performance</h4>
                <div class="performance-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    ${['cpu', 'memory', 'network', 'disk'].map(type => `
                        <div id="${type}Indicator" class="performance-indicator">
                            <div class="performance-label">${type.toUpperCase()}: 0%</div>
                            <div class="performance-bar">
                                <div class="performance-bar-fill normal" style="width: 0%; transition: all 0.3s ease;"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', indicatorsHTML);
        
        // Add CSS for performance bars
        const style = document.createElement('style');
        style.textContent = `
            .performance-indicator {
                background: var(--accent-color);
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            
            .performance-label {
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
                color: var(--text-dark);
            }
            
            .performance-bar {
                height: 8px;
                background: var(--border-color);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .performance-bar-fill.normal { background: var(--success-color); }
            .performance-bar-fill.warning { background: var(--warning-color); }
            .performance-bar-fill.error { background: var(--error-color); }
            .performance-bar-fill.critical { background: #B71C1C; }
        `;
        document.head.appendChild(style);
    }

    getPerformanceLevel(percentage) {
        if (percentage < 60) return 'normal';
        if (percentage < 80) return 'warning';
        if (percentage < 95) return 'error';
        return 'critical';
    }

    updateWorkflowStatus(workflow) {
        // Update workflow status indicators
        const workflowElement = document.querySelector('.flowchart-container');
        if (workflowElement) {
            const statusIndicator = workflowElement.querySelector('.workflow-status') || 
                this.createWorkflowStatusIndicator(workflowElement);
            
            statusIndicator.textContent = `Stage: ${workflow.stage} | Progress: ${workflow.progress.toFixed(1)}% | Status: ${workflow.status}`;
        }
    }

    createWorkflowStatusIndicator(container) {
        const indicator = document.createElement('div');
        indicator.className = 'workflow-status';
        indicator.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(139, 69, 19, 0.9);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            z-index: 100;
        `;
        container.style.position = 'relative';
        container.appendChild(indicator);
        return indicator;
    }

    handleAnomaly(anomaly) {
        console.warn('Anomaly detected:', anomaly);
        
        this.createAlert({
            type: anomaly.type,
            severity: anomaly.severity,
            message: anomaly.message,
            timestamp: new Date(),
            source: 'anomaly_detection'
        });
        
        // Store anomaly in historical data
        this.historicalData.alerts.push({
            timestamp: new Date(),
            ...anomaly
        });
    }

    createAlert(alertData) {
        const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Check if similar alert already exists
        const existingSimilar = Array.from(this.alertManager.activeAlerts.values())
            .find(alert => alert.type === alertData.type && alert.severity === alertData.severity);
        
        if (existingSimilar) {
            // Update existing alert instead of creating new one
            existingSimilar.count = (existingSimilar.count || 1) + 1;
            existingSimilar.lastSeen = new Date();
            this.updateAlertDisplay(existingSimilar);
            return;
        }
        
        const alert = {
            id: alertId,
            ...alertData,
            timestamp: alertData.timestamp || new Date(),
            count: 1,
            dismissed: false
        };
        
        this.alertManager.activeAlerts.set(alertId, alert);
        this.displayAlert(alert);
        
        // Auto-dismiss after timeout
        const timeout = alertData.severity === 'critical' ? 
            this.monitoringConfig.criticalAlertTimeout : 
            this.monitoringConfig.alertTimeout;
        
        setTimeout(() => {
            this.dismissAlert(alertId);
        }, timeout);
    }

    displayAlert(alert) {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;
        
        const alertElement = document.createElement('div');
        alertElement.id = alert.id;
        alertElement.className = `alert alert-${alert.severity}`;
        alertElement.style.cssText = `
            background: ${this.getAlertColor(alert.severity)};
            color: white;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            pointer-events: auto;
            cursor: pointer;
            animation: alertSlideIn 0.3s ease;
            position: relative;
        `;
        
        alertElement.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <strong>${this.getAlertIcon(alert.severity)} ${alert.severity.toUpperCase()}</strong>
                    <div>${alert.message}</div>
                    <small>${alert.timestamp.toLocaleTimeString()}${alert.count > 1 ? ` (${alert.count}x)` : ''}</small>
                </div>
                <button onclick="realTimeMonitor.dismissAlert('${alert.id}')" 
                        style="border: none; background: none; color: white; font-size: 1.2em; cursor: pointer;">&times;</button>
            </div>
        `;
        
        // Add click handler for alert details
        alertElement.addEventListener('click', () => {
            this.showAlertDetails(alert);
        });
        
        alertContainer.appendChild(alertElement);
        
        // Limit visible alerts
        const alerts = alertContainer.children;
        if (alerts.length > this.alertManager.maxVisibleAlerts) {
            alerts[0].remove();
        }
        
        // Add CSS animation if not exists
        if (!document.getElementById('alertAnimations')) {
            const style = document.createElement('style');
            style.id = 'alertAnimations';
            style.textContent = `
                @keyframes alertSlideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes alertSlideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
    }

    updateAlertDisplay(alert) {
        const alertElement = document.getElementById(alert.id);
        if (alertElement) {
            const countElement = alertElement.querySelector('small');
            if (countElement) {
                countElement.textContent = `${alert.timestamp.toLocaleTimeString()} (${alert.count}x)`;
            }
            
            // Add flash effect
            alertElement.style.background = '#fff';
            setTimeout(() => {
                alertElement.style.background = this.getAlertColor(alert.severity);
            }, 200);
        }
    }

    getAlertColor(severity) {
        const colors = {
            info: '#2196F3',
            warning: '#FF9800',
            error: '#F44336',
            critical: '#B71C1C'
        };
        return colors[severity] || colors.info;
    }

    getAlertIcon(severity) {
        const icons = {
            info: 'ℹ️',
            warning: '⚠️',
            error: '❌',
            critical: '🚨'
        };
        return icons[severity] || icons.info;
    }

    dismissAlert(alertId) {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            alertElement.style.animation = 'alertSlideOut 0.3s ease';
            setTimeout(() => {
                alertElement.remove();
            }, 300);
        }
        
        this.alertManager.activeAlerts.delete(alertId);
    }

    showAlertDetails(alert) {
        // Create alert details modal
        const modal = document.createElement('div');
        modal.className = 'alert-details-modal';
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: var(--shadow-hover);
            border: 1px solid var(--border-color);
            z-index: 2000;
            max-width: 500px;
            width: 90%;
        `;
        
        modal.innerHTML = `
            <h3 style="color: ${this.getAlertColor(alert.severity)}; margin-bottom: 1rem;">
                ${this.getAlertIcon(alert.severity)} Alert Details
            </h3>
            <div style="margin-bottom: 1rem;">
                <strong>Type:</strong> ${alert.type}<br>
                <strong>Severity:</strong> ${alert.severity}<br>
                <strong>Message:</strong> ${alert.message}<br>
                <strong>First Seen:</strong> ${alert.timestamp.toLocaleString()}<br>
                <strong>Occurrences:</strong> ${alert.count}
                ${alert.current ? `<br><strong>Current Value:</strong> ${alert.current}` : ''}
                ${alert.threshold ? `<br><strong>Threshold:</strong> ${alert.threshold}` : ''}
            </div>
            <div style="display: flex; gap: 1rem; justify-content: flex-end;">
                <button class="btn btn-secondary" onclick="this.closest('.alert-details-modal').remove(); document.querySelector('.modal-backdrop').remove();">Close</button>
                <button class="btn btn-primary" onclick="realTimeMonitor.resolveAlert('${alert.id}')">Mark Resolved</button>
            </div>
        `;
        
        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        backdrop.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1999;
        `;
        backdrop.onclick = () => {
            backdrop.remove();
            modal.remove();
        };
        
        document.body.appendChild(backdrop);
        document.body.appendChild(modal);
    }

    resolveAlert(alertId) {
        console.log(`Resolving alert: ${alertId}`);
        this.dismissAlert(alertId);
        
        // Close modal
        const modal = document.querySelector('.alert-details-modal');
        const backdrop = document.querySelector('.modal-backdrop');
        if (modal) modal.remove();
        if (backdrop) backdrop.remove();
    }

    startMonitoring() {
        if (this.monitoringState.active) return;
        
        this.monitoringState.active = true;
        console.log('Real-time monitoring started');
        
        // Update monitoring status in UI
        this.updateMonitoringStatus('active');
    }

    stopMonitoring() {
        if (!this.monitoringState.active) return;
        
        this.monitoringState.active = false;
        
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        
        console.log('Real-time monitoring stopped');
        
        // Update monitoring status in UI
        this.updateMonitoringStatus('inactive');
    }

    updateMonitoringStatus(status) {
        const statusElement = document.getElementById('monitoringStatus') || this.createMonitoringStatusElement();
        
        statusElement.textContent = status === 'active' ? '🟢 Monitoring Active' : '🔴 Monitoring Inactive';
        statusElement.style.color = status === 'active' ? 'var(--success-color)' : 'var(--error-color)';
    }

    createMonitoringStatusElement() {
        const header = document.querySelector('.header-content');
        if (!header) return document.createElement('div');
        
        const statusElement = document.createElement('div');
        statusElement.id = 'monitoringStatus';
        statusElement.style.cssText = `
            font-size: 0.9rem;
            font-weight: 600;
        `;
        
        header.appendChild(statusElement);
        return statusElement;
    }

    // Public API methods
    setThreshold(metric, value) {
        if (this.monitoringState.thresholds.hasOwnProperty(metric)) {
            this.monitoringState.thresholds[metric] = value;
            console.log(`Threshold updated: ${metric} = ${value}`);
        }
    }

    getAlerts() {
        return Array.from(this.alertManager.activeAlerts.values());
    }

    getHistoricalData(type, limit = 100) {
        if (!this.historicalData[type]) return [];
        return this.historicalData[type].slice(-limit);
    }

    exportMonitoringData() {
        const data = {
            timestamp: new Date().toISOString(),
            monitoringState: this.monitoringState,
            historicalData: this.historicalData,
            activeAlerts: Array.from(this.alertManager.activeAlerts.values()),
            configuration: this.monitoringConfig
        };
        
        const dataStr = JSON.stringify(data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `gomna-monitoring-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    notifyObservers(event, data) {
        // Notify workflow engine and analytics dashboard
        if (this.workflowEngine && this.workflowEngine.observers) {
            this.workflowEngine.notifyObservers(event, data);
        }
        
        if (this.analyticsDashboard && this.analyticsDashboard.handleWorkflowEvent) {
            this.analyticsDashboard.handleWorkflowEvent(event, data);
        }
    }

    // Cleanup
    destroy() {
        this.stopMonitoring();
        
        if (this.websocket) {
            this.websocket.close();
        }
        
        // Clear all alerts
        this.alertManager.activeAlerts.clear();
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            alertContainer.remove();
        }
        
        console.log('Real-Time Monitor destroyed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RealTimeMonitor;
} else {
    window.RealTimeMonitor = RealTimeMonitor;
}

// Initialize when other components are ready
if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            if (window.gomnaWorkflowEngine && window.analyticsDashboard) {
                window.realTimeMonitor = new RealTimeMonitor(
                    window.gomnaWorkflowEngine, 
                    window.analyticsDashboard
                );
                console.log('🔄 Real-Time Monitor integrated successfully');
            }
        }, 1500);
    });
}