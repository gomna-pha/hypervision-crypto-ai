/**
 * GOMNA Analytics Dashboard - Real-time Data Visualization & Monitoring
 * Integrates with the workflow engine to provide comprehensive analytics
 */

class AnalyticsDashboard {
    constructor(workflowEngine) {
        this.workflowEngine = workflowEngine;
        this.charts = {};
        this.realTimeData = {
            sensorMetrics: [],
            featureMetrics: [],
            validationMetrics: [],
            performanceMetrics: []
        };
        this.dashboardState = {
            initialized: false,
            activeView: 'overview',
            updateInterval: 5000,
            autoRefresh: true
        };
        this.colorScheme = {
            primary: '#8B4513',
            secondary: '#D2B48C',
            accent: '#F5F5DC',
            success: '#4CAF50',
            warning: '#FF9800',
            error: '#F44336',
            info: '#2196F3'
        };
        this.init();
    }

    init() {
        console.log('Initializing Analytics Dashboard...');
        this.setupEventListeners();
        this.initializeCharts();
        this.startRealTimeUpdates();
        this.bindWorkflowEvents();
        this.dashboardState.initialized = true;
        console.log('Analytics Dashboard initialized successfully');
    }

    setupEventListeners() {
        // Listen for dashboard interactions
        document.addEventListener('DOMContentLoaded', () => {
            this.setupDashboardControls();
            this.setupChartInteractions();
            this.setupExportControls();
        });
    }

    setupDashboardControls() {
        // Add dashboard control panel
        const controlPanel = document.getElementById('control-panel');
        if (controlPanel) {
            const analyticsControls = document.createElement('div');
            analyticsControls.className = 'analytics-controls';
            analyticsControls.innerHTML = `
                <h3>Analytics Controls</h3>
                <div class="control-group">
                    <button class="btn btn-secondary" onclick="analyticsDashboard.switchView('overview')">Overview</button>
                    <button class="btn btn-secondary" onclick="analyticsDashboard.switchView('sensors')">Sensors</button>
                    <button class="btn btn-secondary" onclick="analyticsDashboard.switchView('features')">Features</button>
                    <button class="btn btn-secondary" onclick="analyticsDashboard.switchView('validation')">Validation</button>
                    <button class="btn btn-secondary" onclick="analyticsDashboard.switchView('performance')">Performance</button>
                </div>
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="autoRefreshToggle" checked> Auto Refresh
                    </label>
                    <label>
                        Update Interval: 
                        <select id="updateInterval">
                            <option value="1000">1s</option>
                            <option value="5000" selected>5s</option>
                            <option value="10000">10s</option>
                            <option value="30000">30s</option>
                        </select>
                    </label>
                </div>
            `;
            controlPanel.appendChild(analyticsControls);

            // Setup control event listeners
            document.getElementById('autoRefreshToggle').addEventListener('change', (e) => {
                this.dashboardState.autoRefresh = e.target.checked;
                if (e.target.checked) {
                    this.startRealTimeUpdates();
                } else {
                    this.stopRealTimeUpdates();
                }
            });

            document.getElementById('updateInterval').addEventListener('change', (e) => {
                this.dashboardState.updateInterval = parseInt(e.target.value);
                if (this.dashboardState.autoRefresh) {
                    this.stopRealTimeUpdates();
                    this.startRealTimeUpdates();
                }
            });
        }
    }

    setupChartInteractions() {
        // Add chart interaction handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('chart-zoom-in')) {
                this.zoomChart(e.target.dataset.chartId, 'in');
            } else if (e.target.classList.contains('chart-zoom-out')) {
                this.zoomChart(e.target.dataset.chartId, 'out');
            } else if (e.target.classList.contains('chart-reset')) {
                this.resetChart(e.target.dataset.chartId);
            }
        });
    }

    setupExportControls() {
        const exportButton = document.createElement('button');
        exportButton.className = 'btn btn-info';
        exportButton.textContent = 'Export Analytics';
        exportButton.onclick = () => this.exportAnalyticsData();
        
        const controlButtons = document.querySelector('.control-buttons');
        if (controlButtons) {
            controlButtons.appendChild(exportButton);
        }
    }

    initializeCharts() {
        this.createOverviewChart();
        this.createSensorMetricsChart();
        this.createFeatureAnalysisChart();
        this.createValidationChart();
        this.createPerformanceChart();
        this.createWorkflowFlowChart();
    }

    createOverviewChart() {
        const chartContainer = document.getElementById('chartContainer');
        if (!chartContainer) return;

        chartContainer.innerHTML = `
            <div class="analytics-overview">
                <div class="overview-grid">
                    <div class="overview-chart" id="sensorOverview">
                        <h4>Sensor Network Status</h4>
                        <canvas id="sensorStatusChart" width="300" height="200"></canvas>
                    </div>
                    <div class="overview-chart" id="featureOverview">
                        <h4>Feature Processing</h4>
                        <canvas id="featureProcessingChart" width="300" height="200"></canvas>
                    </div>
                    <div class="overview-chart" id="validationOverview">
                        <h4>Validation Metrics</h4>
                        <canvas id="validationMetricsChart" width="300" height="200"></canvas>
                    </div>
                    <div class="overview-chart" id="performanceOverview">
                        <h4>System Performance</h4>
                        <canvas id="performanceMetricsChart" width="300" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;

        // Initialize overview charts
        this.charts.sensorStatus = this.createDonutChart('sensorStatusChart', {
            labels: ['Active', 'Inactive', 'Error'],
            data: [1200, 47, 12],
            colors: [this.colorScheme.success, this.colorScheme.warning, this.colorScheme.error]
        });

        this.charts.featureProcessing = this.createLineChart('featureProcessingChart', {
            labels: this.generateTimeLabels(10),
            datasets: [{
                label: 'Features Extracted',
                data: this.generateRandomData(10, 400, 500),
                borderColor: this.colorScheme.primary,
                backgroundColor: this.colorScheme.primary + '20'
            }]
        });
    }

    createSensorMetricsChart() {
        // Create detailed sensor metrics visualization
        if (this.dashboardState.activeView === 'sensors') {
            const sensorData = {
                activeSensors: this.generateTimeSeriesData(20, 1200, 1300),
                dataQuality: this.generateTimeSeriesData(20, 95, 100),
                latency: this.generateTimeSeriesData(20, 5, 15),
                throughput: this.generateTimeSeriesData(20, 2000, 3000)
            };

            this.renderSensorDashboard(sensorData);
        }
    }

    createFeatureAnalysisChart() {
        // Create feature analysis visualizations
        if (this.dashboardState.activeView === 'features') {
            const featureData = {
                extractionRate: this.generateTimeSeriesData(15, 450, 500),
                accuracy: this.generateTimeSeriesData(15, 92, 98),
                processingTime: this.generateTimeSeriesData(15, 100, 200),
                featureDistribution: this.generateCategoricalData()
            };

            this.renderFeaturesDashboard(featureData);
        }
    }

    createValidationChart() {
        // Create validation metrics visualization
        if (this.dashboardState.activeView === 'validation') {
            const validationData = {
                crossValidationScores: this.generateTimeSeriesData(10, 85, 95),
                confidenceScores: this.generateTimeSeriesData(15, 0.8, 0.95),
                statisticalTests: this.generateTestResults(),
                errorRates: this.generateTimeSeriesData(20, 2, 8)
            };

            this.renderValidationDashboard(validationData);
        }
    }

    createPerformanceChart() {
        // Create system performance visualization
        if (this.dashboardState.activeView === 'performance') {
            const performanceData = {
                cpuUsage: this.generateTimeSeriesData(30, 20, 80),
                memoryUsage: this.generateTimeSeriesData(30, 30, 70),
                networkIO: this.generateTimeSeriesData(30, 100, 1000),
                responseTime: this.generateTimeSeriesData(30, 10, 50)
            };

            this.renderPerformanceDashboard(performanceData);
        }
    }

    createWorkflowFlowChart() {
        // Enhanced workflow visualization with real-time status
        const flowchartCanvas = document.getElementById('flowchartCanvas');
        if (!flowchartCanvas) return;

        // Add workflow status indicators
        const statusOverlay = document.createElement('div');
        statusOverlay.className = 'workflow-status-overlay';
        statusOverlay.innerHTML = `
            <div class="status-legend">
                <div class="status-item">
                    <span class="status-indicator running"></span> Running
                </div>
                <div class="status-item">
                    <span class="status-indicator completed"></span> Completed
                </div>
                <div class="status-item">
                    <span class="status-indicator idle"></span> Idle
                </div>
                <div class="status-item">
                    <span class="status-indicator error"></span> Error
                </div>
            </div>
        `;
        
        flowchartCanvas.appendChild(statusOverlay);
        this.updateWorkflowStatus();
    }

    // Chart Creation Utilities
    createLineChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const ctx = canvas.getContext('2d');
        
        // Simple line chart implementation
        return {
            canvas: canvas,
            ctx: ctx,
            config: config,
            update: (newData) => {
                this.updateLineChart(ctx, canvas, newData);
            }
        };
    }

    createDonutChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const ctx = canvas.getContext('2d');
        
        // Simple donut chart implementation
        this.drawDonutChart(ctx, canvas, config);
        
        return {
            canvas: canvas,
            ctx: ctx,
            config: config,
            update: (newData) => {
                this.drawDonutChart(ctx, canvas, newData);
            }
        };
    }

    updateLineChart(ctx, canvas, data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const padding = 40;
        const chartWidth = canvas.width - (padding * 2);
        const chartHeight = canvas.height - (padding * 2);
        
        if (!data || !data.data || data.data.length === 0) return;
        
        const maxValue = Math.max(...data.data);
        const minValue = Math.min(...data.data);
        const valueRange = maxValue - minValue;
        
        ctx.strokeStyle = data.borderColor || this.colorScheme.primary;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < data.data.length; i++) {
            const x = padding + (i / (data.data.length - 1)) * chartWidth;
            const y = padding + chartHeight - ((data.data[i] - minValue) / valueRange) * chartHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Add labels and grid
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        // Y-axis labels
        for (let i = 0; i <= 5; i++) {
            const value = minValue + (valueRange * i / 5);
            const y = padding + chartHeight - (i / 5) * chartHeight;
            ctx.fillText(value.toFixed(1), 20, y + 4);
        }
    }

    drawDonutChart(ctx, canvas, data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const outerRadius = Math.min(centerX, centerY) - 20;
        const innerRadius = outerRadius * 0.6;
        
        const total = data.data.reduce((sum, value) => sum + value, 0);
        let currentAngle = -Math.PI / 2; // Start from top
        
        for (let i = 0; i < data.data.length; i++) {
            const sliceAngle = (data.data[i] / total) * 2 * Math.PI;
            
            ctx.fillStyle = data.colors[i] || this.colorScheme.primary;
            ctx.beginPath();
            ctx.arc(centerX, centerY, outerRadius, currentAngle, currentAngle + sliceAngle);
            ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
            ctx.closePath();
            ctx.fill();
            
            // Add labels
            const labelAngle = currentAngle + sliceAngle / 2;
            const labelRadius = (outerRadius + innerRadius) / 2;
            const labelX = centerX + Math.cos(labelAngle) * labelRadius;
            const labelY = centerY + Math.sin(labelAngle) * labelRadius;
            
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(data.data[i], labelX, labelY);
            
            currentAngle += sliceAngle;
        }
        
        // Add legend
        const legendX = 10;
        let legendY = 20;
        
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        
        for (let i = 0; i < data.labels.length; i++) {
            ctx.fillStyle = data.colors[i];
            ctx.fillRect(legendX, legendY, 12, 12);
            
            ctx.fillStyle = '#333';
            ctx.fillText(data.labels[i], legendX + 18, legendY + 10);
            
            legendY += 20;
        }
    }

    // Dashboard Rendering Methods
    renderSensorDashboard(data) {
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.innerHTML = `
            <div class="sensor-dashboard">
                <div class="sensor-metrics-grid">
                    <div class="metric-panel">
                        <h4>Active Sensors</h4>
                        <canvas id="activeSensorsChart" width="400" height="200"></canvas>
                    </div>
                    <div class="metric-panel">
                        <h4>Data Quality</h4>
                        <canvas id="dataQualityChart" width="400" height="200"></canvas>
                    </div>
                    <div class="metric-panel">
                        <h4>Network Latency</h4>
                        <canvas id="latencyChart" width="400" height="200"></canvas>
                    </div>
                    <div class="metric-panel">
                        <h4>Data Throughput</h4>
                        <canvas id="throughputChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;

        // Create sensor charts
        this.charts.activeSensors = this.createLineChart('activeSensorsChart', {
            data: data.activeSensors,
            borderColor: this.colorScheme.primary
        });
    }

    renderFeaturesDashboard(data) {
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.innerHTML = `
            <div class="features-dashboard">
                <div class="features-grid">
                    <div class="feature-panel">
                        <h4>Feature Extraction Rate</h4>
                        <canvas id="extractionRateChart" width="400" height="200"></canvas>
                    </div>
                    <div class="feature-panel">
                        <h4>Recognition Accuracy</h4>
                        <canvas id="accuracyChart" width="400" height="200"></canvas>
                    </div>
                    <div class="feature-panel">
                        <h4>Processing Time Distribution</h4>
                        <canvas id="processingTimeChart" width="400" height="200"></canvas>
                    </div>
                    <div class="feature-panel">
                        <h4>Feature Type Distribution</h4>
                        <canvas id="featureDistributionChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    renderValidationDashboard(data) {
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.innerHTML = `
            <div class="validation-dashboard">
                <div class="validation-grid">
                    <div class="validation-panel">
                        <h4>Cross-Validation Scores</h4>
                        <canvas id="cvScoresChart" width="400" height="200"></canvas>
                    </div>
                    <div class="validation-panel">
                        <h4>Confidence Levels</h4>
                        <canvas id="confidenceChart" width="400" height="200"></canvas>
                    </div>
                    <div class="validation-panel">
                        <h4>Statistical Test Results</h4>
                        <div class="test-results" id="testResults">
                            ${this.renderTestResults(data.statisticalTests)}
                        </div>
                    </div>
                    <div class="validation-panel">
                        <h4>Error Rate Trends</h4>
                        <canvas id="errorRatesChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    renderPerformanceDashboard(data) {
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.innerHTML = `
            <div class="performance-dashboard">
                <div class="performance-grid">
                    <div class="perf-panel">
                        <h4>CPU Usage (%)</h4>
                        <canvas id="cpuChart" width="400" height="200"></canvas>
                    </div>
                    <div class="perf-panel">
                        <h4>Memory Usage (%)</h4>
                        <canvas id="memoryChart" width="400" height="200"></canvas>
                    </div>
                    <div class="perf-panel">
                        <h4>Network I/O (MB/s)</h4>
                        <canvas id="networkChart" width="400" height="200"></canvas>
                    </div>
                    <div class="perf-panel">
                        <h4>Response Time (ms)</h4>
                        <canvas id="responseChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    // Data Generation Utilities
    generateTimeLabels(count) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60000); // 1 minute intervals
            labels.push(time.toLocaleTimeString());
        }
        return labels;
    }

    generateRandomData(count, min, max) {
        return Array.from({ length: count }, () => 
            Math.floor(Math.random() * (max - min + 1)) + min
        );
    }

    generateTimeSeriesData(points, min, max) {
        return Array.from({ length: points }, (_, i) => ({
            timestamp: new Date(Date.now() - (points - i) * 60000),
            value: min + Math.random() * (max - min)
        }));
    }

    generateCategoricalData() {
        return {
            'Temporal': 45,
            'Frequency': 38,
            'Statistical': 42,
            'Spectral': 35,
            'Wavelet': 28,
            'Correlation': 33
        };
    }

    generateTestResults() {
        return {
            'Shapiro-Wilk': { pValue: 0.023, significant: true, status: 'pass' },
            'Kolmogorov-Smirnov': { pValue: 0.156, significant: false, status: 'pass' },
            'Chi-Square': { pValue: 0.001, significant: true, status: 'warning' },
            'ANOVA': { pValue: 0.045, significant: true, status: 'pass' },
            'Granger Causality': { pValue: 0.002, significant: true, status: 'pass' }
        };
    }

    renderTestResults(tests) {
        return Object.entries(tests).map(([test, result]) => `
            <div class="test-result ${result.status}">
                <strong>${test}</strong>
                <span class="p-value">p=${result.pValue.toFixed(3)}</span>
                <span class="significance ${result.significant ? 'significant' : 'not-significant'}">
                    ${result.significant ? 'Significant' : 'Not Significant'}
                </span>
            </div>
        `).join('');
    }

    // Real-time Updates
    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            if (this.dashboardState.autoRefresh) {
                this.updateDashboardData();
            }
        }, this.dashboardState.updateInterval);
    }

    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    updateDashboardData() {
        // Update real-time data based on current view
        switch (this.dashboardState.activeView) {
            case 'overview':
                this.updateOverviewData();
                break;
            case 'sensors':
                this.updateSensorData();
                break;
            case 'features':
                this.updateFeatureData();
                break;
            case 'validation':
                this.updateValidationData();
                break;
            case 'performance':
                this.updatePerformanceData();
                break;
        }
        
        this.updateWorkflowStatus();
    }

    updateOverviewData() {
        // Update overview metrics
        if (this.charts.sensorStatus) {
            const newData = {
                labels: ['Active', 'Inactive', 'Error'],
                data: [
                    1200 + Math.floor(Math.random() * 100),
                    47 + Math.floor(Math.random() * 10),
                    12 + Math.floor(Math.random() * 5)
                ],
                colors: [this.colorScheme.success, this.colorScheme.warning, this.colorScheme.error]
            };
            this.charts.sensorStatus.update(newData);
        }
    }

    updateSensorData() {
        // Update sensor metrics in real-time
        console.log('Updating sensor data...');
    }

    updateFeatureData() {
        // Update feature metrics in real-time
        console.log('Updating feature data...');
    }

    updateValidationData() {
        // Update validation metrics in real-time
        console.log('Updating validation data...');
    }

    updatePerformanceData() {
        // Update system performance metrics in real-time
        console.log('Updating performance data...');
    }

    updateWorkflowStatus() {
        if (this.workflowEngine) {
            const status = this.workflowEngine.getWorkflowStatus();
            
            // Update workflow node status indicators
            const nodes = document.querySelectorAll('.flowchart-node');
            nodes.forEach((node, index) => {
                const statusClass = this.getNodeStatus(status.state, index);
                node.className = `flowchart-node ${statusClass}`;
            });
        }
    }

    getNodeStatus(workflowState, nodeIndex) {
        // Map node index to workflow state
        const stateKeys = Object.keys(workflowState);
        if (nodeIndex < stateKeys.length) {
            const state = workflowState[stateKeys[nodeIndex]];
            return state.status || 'idle';
        }
        return 'idle';
    }

    // Dashboard Navigation
    switchView(view) {
        this.dashboardState.activeView = view;
        console.log(`Switching to ${view} view`);
        
        // Update active button
        document.querySelectorAll('.analytics-controls .btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        event.target.classList.add('active');
        
        // Re-initialize charts for new view
        this.initializeCharts();
    }

    // Workflow Engine Integration
    bindWorkflowEvents() {
        if (this.workflowEngine) {
            this.workflowEngine.addObserver((event, data) => {
                this.handleWorkflowEvent(event, data);
            });
        }
    }

    handleWorkflowEvent(event, data) {
        switch (event) {
            case 'sensorInitialized':
                this.updateSensorMetrics(data);
                break;
            case 'uiUpdate':
                this.updateUIMetrics(data);
                break;
            case 'metricsUpdate':
                this.updateRealTimeMetrics(data);
                break;
            case 'workflowCompleted':
                this.handleWorkflowCompletion(data);
                break;
            case 'workflowError':
                this.handleWorkflowError(data);
                break;
        }
    }

    updateSensorMetrics(data) {
        this.realTimeData.sensorMetrics.push({
            timestamp: new Date(),
            ...data
        });
        
        // Keep only last 100 entries
        if (this.realTimeData.sensorMetrics.length > 100) {
            this.realTimeData.sensorMetrics.shift();
        }
    }

    updateUIMetrics(data) {
        const { module, data: moduleData } = data;
        this.realTimeData[`${module}Metrics`] = this.realTimeData[`${module}Metrics`] || [];
        this.realTimeData[`${module}Metrics`].push({
            timestamp: new Date(),
            ...moduleData
        });
    }

    updateRealTimeMetrics(data) {
        this.realTimeData.performanceMetrics.push(data);
        
        // Update UI elements
        if (document.getElementById('throughputMetric')) {
            document.getElementById('throughputMetric').textContent = (data.throughput / 1000).toFixed(1) + 'K';
        }
        if (document.getElementById('latencyMetric')) {
            document.getElementById('latencyMetric').textContent = data.latency + 'ms';
        }
        if (document.getElementById('accuracyMetric')) {
            document.getElementById('accuracyMetric').textContent = data.accuracy + '%';
        }
        if (document.getElementById('uptimeMetric')) {
            document.getElementById('uptimeMetric').textContent = data.uptime + '%';
        }
    }

    handleWorkflowCompletion(data) {
        console.log('Workflow completed:', data);
        
        // Show completion notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-success';
        notification.innerHTML = `
            <strong>Workflow Completed!</strong> All processes have finished successfully.
            <button onclick="this.parentElement.remove()" style="float: right; border: none; background: none; font-size: 1.2em;">&times;</button>
        `;
        
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 5000);
    }

    handleWorkflowError(error) {
        console.error('Workflow error:', error);
        
        // Show error notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-error';
        notification.innerHTML = `
            <strong>Workflow Error!</strong> ${error.message || 'An error occurred during workflow execution.'}
            <button onclick="this.parentElement.remove()" style="float: right; border: none; background: none; font-size: 1.2em;">&times;</button>
        `;
        
        document.body.appendChild(notification);
    }

    // Chart Interaction Methods
    zoomChart(chartId, direction) {
        console.log(`Zooming ${direction} on chart: ${chartId}`);
        // Implement chart zoom functionality
    }

    resetChart(chartId) {
        console.log(`Resetting chart: ${chartId}`);
        // Implement chart reset functionality
    }

    // Export Functionality
    exportAnalyticsData() {
        const analyticsData = {
            timestamp: new Date().toISOString(),
            dashboardState: this.dashboardState,
            realTimeData: this.realTimeData,
            workflowStatus: this.workflowEngine ? this.workflowEngine.getWorkflowStatus() : null,
            chartConfigurations: Object.keys(this.charts).map(key => ({
                chartId: key,
                config: this.charts[key].config
            }))
        };

        // Create and download JSON file
        const dataStr = JSON.stringify(analyticsData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `gomna-analytics-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);

        console.log('Analytics data exported successfully');
    }

    // Cleanup
    destroy() {
        this.stopRealTimeUpdates();
        this.charts = {};
        this.realTimeData = {};
        console.log('Analytics Dashboard destroyed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnalyticsDashboard;
} else {
    window.AnalyticsDashboard = AnalyticsDashboard;
}

// Initialize when workflow engine is available
if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        if (window.gomnaWorkflowEngine) {
            window.analyticsDashboard = new AnalyticsDashboard(window.gomnaWorkflowEngine);
        } else {
            // Wait for workflow engine to be ready
            setTimeout(() => {
                if (window.gomnaWorkflowEngine) {
                    window.analyticsDashboard = new AnalyticsDashboard(window.gomnaWorkflowEngine);
                }
            }, 1000);
        }
    });
}