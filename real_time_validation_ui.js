/**
 * REAL-TIME VALIDATION DASHBOARD UI
 * =================================
 * Frontend interface for the comprehensive platform validation system.
 * Addresses critical issues identified in statistical, mathematical, and engineering domains.
 * 
 * INTEGRATION POINTS:
 * - WebSocket connection to validation server
 * - Real-time alerts and notifications
 * - Integration with existing portfolio and trading systems
 * - Dashboard widgets for all platform components
 */

class RealTimeValidationUI {
    constructor() {
        this.websocket = null;
        this.validationResults = new Map();
        this.systemHealth = {};
        this.alertThresholds = {
            CRITICAL: 0.3,
            HIGH: 0.5,
            MEDIUM: 0.7,
            LOW: 0.9
        };
        this.charts = {};
        this.isConnected = false;
        
        this.initializeValidationDashboard();
        this.connectToValidationServer();
        this.setupEventListeners();
        this.startPeriodicUpdates();
    }
    
    initializeValidationDashboard() {
        console.log('🔍 Initializing Real-Time Validation Dashboard...');
        
        // Create validation container in main platform
        this.createValidationContainer();
        this.createValidationCharts();
        this.createAlertSystem();
    }
    
    createValidationContainer() {
        // Check if validation container already exists
        let container = document.getElementById('validation-dashboard');
        
        if (!container) {
            container = document.createElement('div');
            container.id = 'validation-dashboard';
            container.className = 'validation-dashboard fixed top-4 right-4 w-96 max-h-screen overflow-y-auto z-50';
            container.innerHTML = `
                <!-- Validation Dashboard Header -->
                <div class="validation-header glass-effect p-4 rounded-t-xl border-b border-gray-200">
                    <div class="flex justify-between items-center">
                        <div class="flex items-center space-x-2">
                            <div id="validation-status-indicator" class="w-3 h-3 bg-gray-400 rounded-full animate-pulse"></div>
                            <h3 class="font-semibold text-gray-800">Platform Validation</h3>
                        </div>
                        <div class="flex items-center space-x-2">
                            <button id="validation-toggle" class="text-gray-600 hover:text-gray-800 transition-colors">
                                <i data-lucide="minimize-2" class="w-4 h-4"></i>
                            </button>
                            <button id="validation-settings" class="text-gray-600 hover:text-gray-800 transition-colors">
                                <i data-lucide="settings" class="w-4 h-4"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Connection Status -->
                    <div class="mt-2 flex items-center space-x-2 text-sm">
                        <span class="text-gray-600">Status:</span>
                        <span id="validation-connection-status" class="text-red-600">Connecting...</span>
                    </div>
                </div>
                
                <!-- Validation Content -->
                <div id="validation-content" class="validation-content glass-effect rounded-b-xl">
                    
                    <!-- Critical Alerts -->
                    <div class="critical-alerts p-4 border-b border-gray-200">
                        <h4 class="font-medium text-gray-800 mb-2 flex items-center">
                            <i data-lucide="alert-triangle" class="w-4 h-4 mr-1 text-red-500"></i>
                            Critical Issues
                        </h4>
                        <div id="critical-alerts-list" class="space-y-1">
                            <!-- Dynamic alerts -->
                        </div>
                    </div>
                    
                    <!-- Validation Metrics -->
                    <div class="validation-metrics p-4 border-b border-gray-200">
                        <h4 class="font-medium text-gray-800 mb-3">Validation Metrics</h4>
                        
                        <div class="grid grid-cols-2 gap-3 mb-3">
                            <div class="metric-card p-2 rounded bg-gray-50 text-center">
                                <div class="text-lg font-bold text-green-600" id="tests-passed">0</div>
                                <div class="text-xs text-gray-600">Passed</div>
                            </div>
                            <div class="metric-card p-2 rounded bg-gray-50 text-center">
                                <div class="text-lg font-bold text-red-600" id="tests-failed">0</div>
                                <div class="text-xs text-gray-600">Failed</div>
                            </div>
                            <div class="metric-card p-2 rounded bg-gray-50 text-center">
                                <div class="text-lg font-bold text-yellow-600" id="tests-warnings">0</div>
                                <div class="text-xs text-gray-600">Warnings</div>
                            </div>
                            <div class="metric-card p-2 rounded bg-gray-50 text-center">
                                <div class="text-lg font-bold text-purple-600" id="tests-errors">0</div>
                                <div class="text-xs text-gray-600">Errors</div>
                            </div>
                        </div>
                        
                        <!-- Overall Score -->
                        <div class="overall-score p-2 rounded bg-gradient-to-r from-green-100 to-red-100">
                            <div class="flex justify-between items-center">
                                <span class="text-sm font-medium">Platform Health</span>
                                <span id="overall-health-score" class="text-lg font-bold">--</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div id="health-progress-bar" class="bg-green-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Component Status -->
                    <div class="component-status p-4 border-b border-gray-200">
                        <h4 class="font-medium text-gray-800 mb-3">Component Status</h4>
                        <div id="component-status-list" class="space-y-2">
                            <!-- Dynamic component status -->
                        </div>
                    </div>
                    
                    <!-- Recent Tests -->
                    <div class="recent-tests p-4">
                        <h4 class="font-medium text-gray-800 mb-3">Recent Validations</h4>
                        <div id="recent-tests-list" class="space-y-1 max-h-48 overflow-y-auto">
                            <!-- Dynamic test results -->
                        </div>
                    </div>
                    
                    <!-- Validation Charts -->
                    <div class="validation-charts p-4 border-t border-gray-200">
                        <h4 class="font-medium text-gray-800 mb-3">Performance Trends</h4>
                        
                        <!-- Score Over Time Chart -->
                        <div class="mb-4">
                            <canvas id="validation-score-chart" width="350" height="150"></canvas>
                        </div>
                        
                        <!-- Component Breakdown -->
                        <div class="mb-4">
                            <canvas id="component-breakdown-chart" width="350" height="150"></canvas>
                        </div>
                    </div>
                </div>
            `;
            
            // Insert into body
            document.body.appendChild(container);
            
            // Initialize Lucide icons
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        }
        
        // Setup toggle functionality
        this.setupValidationToggle();
    }
    
    setupValidationToggle() {
        const toggleBtn = document.getElementById('validation-toggle');
        const content = document.getElementById('validation-content');
        let isMinimized = false;
        
        if (toggleBtn && content) {
            toggleBtn.addEventListener('click', () => {
                isMinimized = !isMinimized;
                
                if (isMinimized) {
                    content.style.display = 'none';
                    toggleBtn.innerHTML = '<i data-lucide="maximize-2" class="w-4 h-4"></i>';
                } else {
                    content.style.display = 'block';
                    toggleBtn.innerHTML = '<i data-lucide="minimize-2" class="w-4 h-4"></i>';
                }
                
                // Re-initialize Lucide icons
                if (typeof lucide !== 'undefined') {
                    lucide.createIcons();
                }
            });
        }
    }
    
    createValidationCharts() {
        // Score over time chart
        const scoreCtx = document.getElementById('validation-score-chart');
        if (scoreCtx && typeof Chart !== 'undefined') {
            this.charts.scoreChart = new Chart(scoreCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Platform Health Score',
                        data: [],
                        borderColor: 'rgb(34, 197, 94)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                format: {
                                    style: 'percent'
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        // Component breakdown chart
        const componentCtx = document.getElementById('component-breakdown-chart');
        if (componentCtx && typeof Chart !== 'undefined') {
            this.charts.componentChart = new Chart(componentCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Pass', 'Fail', 'Warning', 'Error'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            'rgb(34, 197, 94)',   // Green
                            'rgb(239, 68, 68)',   // Red
                            'rgb(245, 158, 11)',  // Yellow
                            'rgb(147, 51, 234)'   // Purple
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    
    createAlertSystem() {
        // Create notification container if it doesn't exist
        if (!document.getElementById('validation-notifications')) {
            const notificationContainer = document.createElement('div');
            notificationContainer.id = 'validation-notifications';
            notificationContainer.className = 'fixed top-4 left-4 space-y-2 z-50';
            document.body.appendChild(notificationContainer);
        }
    }
    
    connectToValidationServer() {
        try {
            // Connect to validation WebSocket server
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const port = 9000; // Validation server port
            
            this.websocket = new WebSocket(`${protocol}//${host}:${port}`);
            
            this.websocket.onopen = () => {
                console.log('✅ Connected to validation server');
                this.isConnected = true;
                this.updateConnectionStatus('Connected', 'text-green-600');
                this.updateStatusIndicator('bg-green-500');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleValidationMessage(message);
                } catch (e) {
                    console.error('Error parsing validation message:', e);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('❌ Validation server connection closed');
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected', 'text-red-600');
                this.updateStatusIndicator('bg-red-500');
                
                // Attempt reconnection after 5 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.connectToValidationServer();
                    }
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('Validation WebSocket error:', error);
                this.updateConnectionStatus('Error', 'text-red-600');
                this.updateStatusIndicator('bg-red-500');
            };
            
        } catch (e) {
            console.error('Failed to connect to validation server:', e);
            this.updateConnectionStatus('Failed to Connect', 'text-red-600');
            this.updateStatusIndicator('bg-red-500');
            
            // Try to start a local fallback validation system
            this.startFallbackValidation();
        }
    }
    
    handleValidationMessage(message) {
        const { type, data, timestamp } = message;
        
        switch (type) {
            case 'validation_result':
                this.processValidationResult(data);
                break;
                
            case 'historical_result':
                this.processValidationResult(data);
                break;
                
            case 'system_health':
                this.updateSystemHealth(data);
                break;
                
            default:
                console.log('Unknown validation message type:', type);
        }
    }
    
    processValidationResult(result) {
        // Store result
        const key = `${result.component}:${result.test_name}`;
        this.validationResults.set(key, result);
        
        // Update UI
        this.updateValidationMetrics();
        this.updateComponentStatus();
        this.updateRecentTests(result);
        this.updateCharts();
        
        // Check for critical issues
        if (result.severity === 'CRITICAL' || result.status === 'FAIL') {
            this.showCriticalAlert(result);
        }
        
        // Update platform integration
        this.updatePlatformValidationStatus(result);
    }
    
    updateValidationMetrics() {
        const results = Array.from(this.validationResults.values());
        
        const passed = results.filter(r => r.status === 'PASS').length;
        const failed = results.filter(r => r.status === 'FAIL').length;
        const warnings = results.filter(r => r.status === 'WARNING').length;
        const errors = results.filter(r => r.status === 'ERROR').length;
        
        // Update counters
        document.getElementById('tests-passed').textContent = passed;
        document.getElementById('tests-failed').textContent = failed;
        document.getElementById('tests-warnings').textContent = warnings;
        document.getElementById('tests-errors').textContent = errors;
        
        // Calculate overall health score
        const total = results.length;
        if (total > 0) {
            const healthScore = (passed + warnings * 0.5) / total;
            const percentage = (healthScore * 100).toFixed(1);
            
            document.getElementById('overall-health-score').textContent = percentage + '%';
            
            const progressBar = document.getElementById('health-progress-bar');
            if (progressBar) {
                progressBar.style.width = percentage + '%';
                
                // Update color based on score
                if (healthScore >= 0.8) {
                    progressBar.className = 'bg-green-500 h-2 rounded-full transition-all duration-500';
                } else if (healthScore >= 0.6) {
                    progressBar.className = 'bg-yellow-500 h-2 rounded-full transition-all duration-500';
                } else {
                    progressBar.className = 'bg-red-500 h-2 rounded-full transition-all duration-500';
                }
            }
        }
    }
    
    updateComponentStatus() {
        const componentList = document.getElementById('component-status-list');
        if (!componentList) return;
        
        // Group results by component
        const componentResults = new Map();
        
        Array.from(this.validationResults.values()).forEach(result => {
            if (!componentResults.has(result.component)) {
                componentResults.set(result.component, {
                    passed: 0,
                    failed: 0,
                    warnings: 0,
                    errors: 0,
                    total: 0
                });
            }
            
            const stats = componentResults.get(result.component);
            stats.total++;
            
            switch (result.status) {
                case 'PASS':
                    stats.passed++;
                    break;
                case 'FAIL':
                    stats.failed++;
                    break;
                case 'WARNING':
                    stats.warnings++;
                    break;
                case 'ERROR':
                    stats.errors++;
                    break;
            }
        });
        
        // Update component list
        componentList.innerHTML = '';
        
        componentResults.forEach((stats, component) => {
            const healthScore = (stats.passed + stats.warnings * 0.5) / stats.total;
            let statusColor, statusIcon;
            
            if (stats.errors > 0) {
                statusColor = 'text-purple-600';
                statusIcon = 'alert-circle';
            } else if (stats.failed > 0) {
                statusColor = 'text-red-600';
                statusIcon = 'x-circle';
            } else if (stats.warnings > 0) {
                statusColor = 'text-yellow-600';
                statusIcon = 'alert-triangle';
            } else {
                statusColor = 'text-green-600';
                statusIcon = 'check-circle';
            }
            
            const componentDiv = document.createElement('div');
            componentDiv.className = 'component-item flex justify-between items-center p-2 rounded bg-gray-50';
            componentDiv.innerHTML = `
                <div class="flex items-center space-x-2">
                    <i data-lucide="${statusIcon}" class="w-4 h-4 ${statusColor}"></i>
                    <span class="text-sm font-medium text-gray-800">${component}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="text-xs text-gray-600">${(healthScore * 100).toFixed(0)}%</span>
                    <span class="text-xs text-gray-500">${stats.total}</span>
                </div>
            `;
            
            componentList.appendChild(componentDiv);
        });
        
        // Re-initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    updateRecentTests(result) {
        const testsList = document.getElementById('recent-tests-list');
        if (!testsList) return;
        
        // Create test item
        const testItem = document.createElement('div');
        testItem.className = 'test-item flex justify-between items-center p-2 rounded bg-gray-50 text-xs';
        
        let statusColor;
        switch (result.status) {
            case 'PASS':
                statusColor = 'text-green-600';
                break;
            case 'FAIL':
                statusColor = 'text-red-600';
                break;
            case 'WARNING':
                statusColor = 'text-yellow-600';
                break;
            case 'ERROR':
                statusColor = 'text-purple-600';
                break;
            default:
                statusColor = 'text-gray-600';
        }
        
        testItem.innerHTML = `
            <div>
                <div class="font-medium text-gray-800">${result.test_name}</div>
                <div class="text-gray-600">${result.component}</div>
            </div>
            <div class="text-right">
                <div class="${statusColor} font-medium">${result.status}</div>
                <div class="text-gray-500">${(result.score * 100).toFixed(0)}%</div>
            </div>
        `;
        
        // Add to top of list
        testsList.insertBefore(testItem, testsList.firstChild);
        
        // Keep only last 20 items
        while (testsList.children.length > 20) {
            testsList.removeChild(testsList.lastChild);
        }
    }
    
    updateCharts() {
        // Update score chart
        if (this.charts.scoreChart) {
            const now = new Date().toLocaleTimeString();
            const results = Array.from(this.validationResults.values());
            const totalScore = results.reduce((sum, r) => sum + r.score, 0) / results.length || 0;
            
            this.charts.scoreChart.data.labels.push(now);
            this.charts.scoreChart.data.datasets[0].data.push(totalScore);
            
            // Keep only last 20 points
            if (this.charts.scoreChart.data.labels.length > 20) {
                this.charts.scoreChart.data.labels.shift();
                this.charts.scoreChart.data.datasets[0].data.shift();
            }
            
            this.charts.scoreChart.update();
        }
        
        // Update component breakdown chart
        if (this.charts.componentChart) {
            const results = Array.from(this.validationResults.values());
            
            const passed = results.filter(r => r.status === 'PASS').length;
            const failed = results.filter(r => r.status === 'FAIL').length;
            const warnings = results.filter(r => r.status === 'WARNING').length;
            const errors = results.filter(r => r.status === 'ERROR').length;
            
            this.charts.componentChart.data.datasets[0].data = [passed, failed, warnings, errors];
            this.charts.componentChart.update();
        }
    }
    
    showCriticalAlert(result) {
        const alertsList = document.getElementById('critical-alerts-list');
        if (!alertsList) return;
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item p-2 rounded border-l-4 ${
            result.severity === 'CRITICAL' ? 'border-red-500 bg-red-50' : 'border-yellow-500 bg-yellow-50'
        }`;
        
        alertDiv.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <div class="font-medium text-sm text-gray-800">${result.test_name}</div>
                    <div class="text-xs text-gray-600">${result.message}</div>
                    <div class="text-xs text-gray-500">${result.component}</div>
                </div>
                <button class="alert-dismiss text-gray-400 hover:text-gray-600">
                    <i data-lucide="x" class="w-3 h-3"></i>
                </button>
            </div>
        `;
        
        // Add dismiss functionality
        alertDiv.querySelector('.alert-dismiss').addEventListener('click', () => {
            alertDiv.remove();
        });
        
        // Add to top of alerts list
        alertsList.insertBefore(alertDiv, alertsList.firstChild);
        
        // Keep only last 10 alerts
        while (alertsList.children.length > 10) {
            alertsList.removeChild(alertsList.lastChild);
        }
        
        // Re-initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        
        // Also show desktop notification for critical issues
        if (result.severity === 'CRITICAL' && 'Notification' in window) {
            if (Notification.permission === 'granted') {
                new Notification('Critical Platform Issue', {
                    body: `${result.test_name}: ${result.message}`,
                    icon: '/favicon.ico'
                });
            } else if (Notification.permission !== 'denied') {
                Notification.requestPermission().then(permission => {
                    if (permission === 'granted') {
                        new Notification('Critical Platform Issue', {
                            body: `${result.test_name}: ${result.message}`,
                            icon: '/favicon.ico'
                        });
                    }
                });
            }
        }
    }
    
    updateConnectionStatus(status, colorClass) {
        const statusElement = document.getElementById('validation-connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = colorClass;
        }
    }
    
    updateStatusIndicator(colorClass) {
        const indicator = document.getElementById('validation-status-indicator');
        if (indicator) {
            indicator.className = `w-3 h-3 ${colorClass} rounded-full animate-pulse`;
        }
    }
    
    updatePlatformValidationStatus(result) {
        // Integration with existing platform components
        
        // Update portfolio tab if it's a portfolio-related validation
        if (result.component.toLowerCase().includes('portfolio')) {
            this.updatePortfolioValidationStatus(result);
        }
        
        // Update trading API status
        if (result.component.toLowerCase().includes('api')) {
            this.updateAPIValidationStatus(result);
        }
        
        // Update mathematical validation status
        if (result.component.toLowerCase().includes('hyperbolic') || 
            result.component.toLowerCase().includes('mathematical')) {
            this.updateMathValidationStatus(result);
        }
    }
    
    updatePortfolioValidationStatus(result) {
        // Update portfolio dashboard with validation status
        const portfolioContainer = document.getElementById('hyperbolic-portfolio-container');
        if (portfolioContainer) {
            const validationBadge = portfolioContainer.querySelector('.validation-badge') || 
                                   this.createValidationBadge();
            
            if (!portfolioContainer.querySelector('.validation-badge')) {
                const header = portfolioContainer.querySelector('.portfolio-header');
                if (header) {
                    header.appendChild(validationBadge);
                }
            }
            
            // Update badge based on result
            this.updateValidationBadge(validationBadge, result);
        }
    }
    
    createValidationBadge() {
        const badge = document.createElement('div');
        badge.className = 'validation-badge flex items-center space-x-1 px-2 py-1 rounded-full text-xs';
        badge.innerHTML = `
            <i data-lucide="shield-check" class="w-3 h-3"></i>
            <span class="validation-badge-text">Validated</span>
        `;
        return badge;
    }
    
    updateValidationBadge(badge, result) {
        const badgeText = badge.querySelector('.validation-badge-text');
        
        if (result.status === 'PASS') {
            badge.className = 'validation-badge flex items-center space-x-1 px-2 py-1 rounded-full text-xs bg-green-100 text-green-800';
            badgeText.textContent = 'Validated ✓';
        } else if (result.status === 'WARNING') {
            badge.className = 'validation-badge flex items-center space-x-1 px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800';
            badgeText.textContent = 'Warning ⚠';
        } else {
            badge.className = 'validation-badge flex items-center space-x-1 px-2 py-1 rounded-full text-xs bg-red-100 text-red-800';
            badgeText.textContent = 'Invalid ✗';
        }
        
        // Re-initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    setupEventListeners() {
        // Listen for portfolio optimization events
        document.addEventListener('portfolioOptimized', (event) => {
            const { weights, returns } = event.detail;
            this.validatePortfolioData(weights, returns);
        });
        
        // Listen for API calls
        document.addEventListener('apiCall', (event) => {
            const { endpoint, response, timing } = event.detail;
            this.validateAPIResponse(endpoint, response, timing);
        });
        
        // Listen for mathematical operations
        document.addEventListener('mathematicalOperation', (event) => {
            const { operation, inputs, result } = event.detail;
            this.validateMathematicalOperation(operation, inputs, result);
        });
    }
    
    validatePortfolioData(weights, returns) {
        // Client-side validation for immediate feedback
        if (weights && weights.length > 0) {
            const weightsArray = Array.isArray(weights) ? weights : Object.values(weights);
            const weightSum = weightsArray.reduce((sum, w) => sum + w, 0);
            
            if (Math.abs(weightSum - 1.0) > 0.01) {
                this.showClientSideAlert('Portfolio weights do not sum to 1.0', 'WARNING');
            }
            
            const maxWeight = Math.max(...weightsArray);
            if (maxWeight > 0.5) {
                this.showClientSideAlert('High concentration risk detected', 'WARNING');
            }
        }
    }
    
    showClientSideAlert(message, severity) {
        const alertsContainer = document.getElementById('critical-alerts-list');
        if (alertsContainer) {
            const alert = document.createElement('div');
            alert.className = `alert-item p-2 rounded border-l-4 ${
                severity === 'CRITICAL' ? 'border-red-500 bg-red-50' : 'border-yellow-500 bg-yellow-50'
            }`;
            
            alert.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <div class="font-medium text-sm text-gray-800">Client-Side Validation</div>
                        <div class="text-xs text-gray-600">${message}</div>
                    </div>
                    <button class="alert-dismiss text-gray-400 hover:text-gray-600">
                        <i data-lucide="x" class="w-3 h-3"></i>
                    </button>
                </div>
            `;
            
            alert.querySelector('.alert-dismiss').addEventListener('click', () => {
                alert.remove();
            });
            
            alertsContainer.insertBefore(alert, alertsContainer.firstChild);
            
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        }
    }
    
    startFallbackValidation() {
        // Fallback client-side validation when server is unavailable
        console.log('🔄 Starting fallback validation system...');
        
        setInterval(() => {
            this.runClientSideValidation();
        }, 60000); // Every minute
    }
    
    runClientSideValidation() {
        // Basic client-side health checks
        const performance = window.performance;
        const timing = performance.timing;
        
        // Check page load time
        const loadTime = timing.loadEventEnd - timing.navigationStart;
        if (loadTime > 5000) {
            this.showClientSideAlert('Slow page load detected', 'WARNING');
        }
        
        // Check memory usage (if available)
        if (performance.memory) {
            const memoryUsage = performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize;
            if (memoryUsage > 0.8) {
                this.showClientSideAlert('High memory usage detected', 'WARNING');
            }
        }
        
        // Check for JavaScript errors
        if (window.jsErrorCount && window.jsErrorCount > 5) {
            this.showClientSideAlert('Multiple JavaScript errors detected', 'CRITICAL');
        }
        
        this.updateConnectionStatus('Fallback Mode', 'text-yellow-600');
    }
    
    startPeriodicUpdates() {
        // Update timestamps and refresh data periodically
        setInterval(() => {
            this.updateTimestamps();
        }, 30000); // Every 30 seconds
    }
    
    updateTimestamps() {
        // Update relative timestamps in the UI
        const testItems = document.querySelectorAll('.test-item');
        testItems.forEach(item => {
            // Add timestamp updates if needed
        });
    }
}

// Global error tracking for validation
window.jsErrorCount = 0;
window.addEventListener('error', () => {
    window.jsErrorCount++;
});

// Initialize validation UI when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔍 Initializing Real-Time Validation UI...');
    window.realTimeValidationUI = new RealTimeValidationUI();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RealTimeValidationUI;
}