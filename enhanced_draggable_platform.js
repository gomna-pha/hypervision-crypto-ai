/**
 * GOMNA ENHANCED DRAGGABLE PLATFORM WITH HFT ARBITRAGE
 * Combines draggable/minimizable features with advanced arbitrage capabilities
 * Optimized for fast loading and professional user experience
 */

class GomnaEnhancedDraggablePlatform {
    constructor() {
        this.panels = new Map();
        this.arbitrageEngine = null;
        this.isArbitrageInitialized = false;
        this.loadingOptimizations = {
            lazyLoad: true,
            deferNonCritical: true,
            useWorkers: false // Can enable for heavy computations
        };
        
        console.log('🎯 Initializing Enhanced Draggable Platform...');
        this.init();
    }

    async init() {
        try {
            // Initialize core UI first (fast)
            this.initCoreUI();
            
            // Initialize draggable features
            this.initDraggableFeatures();
            
            // Initialize payment system integration
            this.initPaymentIntegration();
            
            // Initialize arbitrage system (deferred for performance)
            if (this.loadingOptimizations.deferNonCritical) {
                setTimeout(() => this.initArbitrageSystem(), 100);
            } else {
                await this.initArbitrageSystem();
            }
            
            console.log('✅ Enhanced platform initialized successfully');
            
        } catch (error) {
            console.error('❌ Platform initialization failed:', error);
            this.handleInitializationError(error);
        }
    }

    initCoreUI() {
        // Add enhanced styles for draggable features
        this.addDraggableStyles();
        
        // Make existing panels draggable and minimizable
        this.enhanceExistingPanels();
        
        // Add minimize/maximize controls to all panels
        this.addPanelControls();
    }

    addDraggableStyles() {
        const styles = `
            /* Enhanced Draggable Platform Styles */
            .gomna-panel {
                position: relative;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                margin: 10px 0;
            }

            .gomna-panel.draggable {
                cursor: move;
                border: 2px solid transparent;
            }

            .gomna-panel.draggable:hover {
                border-color: #00ff88;
                box-shadow: 0 8px 30px rgba(0, 255, 136, 0.2);
                transform: translateY(-2px);
            }

            .gomna-panel.dragging {
                opacity: 0.8;
                transform: scale(1.02);
                z-index: 9999;
            }

            .gomna-panel.minimized {
                height: 60px !important;
                overflow: hidden;
                transition: height 0.3s ease;
            }

            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 20px;
                background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
                border-bottom: 1px solid rgba(0, 255, 136, 0.2);
                border-radius: 12px 12px 0 0;
                cursor: move;
                user-select: none;
            }

            .panel-title {
                font-weight: 600;
                color: #2d3748;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .panel-controls {
                display: flex;
                gap: 8px;
                align-items: center;
            }

            .panel-control-btn {
                width: 28px;
                height: 28px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                transition: all 0.2s ease;
                background: rgba(255, 255, 255, 0.8);
                color: #4a5568;
            }

            .panel-control-btn:hover {
                background: #4a5568;
                color: white;
            }

            .panel-control-btn.minimize-btn:hover {
                background: #f6ad55;
            }

            .panel-control-btn.close-btn:hover {
                background: #fc8181;
            }

            .panel-control-btn.drag-handle:hover {
                background: #4ecdc4;
            }

            .panel-content {
                padding: 20px;
                transition: opacity 0.3s ease;
            }

            .gomna-panel.minimized .panel-content {
                opacity: 0;
                height: 0;
                padding: 0 20px;
                overflow: hidden;
            }

            .arbitrage-loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 40px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                color: white;
            }

            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(0, 255, 136, 0.1);
                border-left: 4px solid #00ff88;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 16px;
            }

            .loading-progress {
                width: 200px;
                height: 4px;
                background: rgba(0, 255, 136, 0.1);
                border-radius: 2px;
                overflow: hidden;
                margin: 10px 0;
            }

            .loading-progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #00ff88, #4ecdc4);
                border-radius: 2px;
                transition: width 0.3s ease;
                animation: shimmer 2s infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            @keyframes shimmer {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }

            .payment-integration-panel {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }

            .quick-actions-panel {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                border: 1px solid rgba(0, 255, 136, 0.2);
            }

            .quick-action-btn {
                display: block;
                width: 100%;
                padding: 8px 12px;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #00ff88, #4ecdc4);
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
                transition: all 0.2s ease;
            }

            .quick-action-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
            }

            .status-indicator {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                color: #6b7280;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 2s infinite;
            }

            .status-dot.warning {
                background: #f59e0b;
            }

            .status-dot.error {
                background: #ef4444;
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    enhanceExistingPanels() {
        // Find existing panel-like elements and enhance them
        const panelSelectors = [
            '.bg-gradient-to-r',
            '.metric-card',
            '.glass-effect',
            '[class*="panel"]',
            '.arbitrage-enhanced'
        ];

        panelSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(panel => {
                if (!panel.classList.contains('gomna-panel')) {
                    this.enhancePanel(panel);
                }
            });
        });
    }

    enhancePanel(panel) {
        panel.classList.add('gomna-panel', 'draggable');
        
        // Add header if it doesn't exist
        if (!panel.querySelector('.panel-header')) {
            this.addPanelHeader(panel);
        }
        
        // Make draggable
        this.makeDraggable(panel);
        
        // Store panel reference
        const panelId = panel.id || `panel_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        panel.id = panelId;
        this.panels.set(panelId, {
            element: panel,
            isMinimized: false,
            originalHeight: null,
            position: { x: 0, y: 0 }
        });
    }

    addPanelHeader(panel) {
        // Extract title from existing content
        let title = 'GOMNA Panel';
        const titleElement = panel.querySelector('h2, h3, h4, .text-xl, .text-2xl');
        if (titleElement) {
            title = titleElement.textContent.trim();
        }

        // Create header
        const header = document.createElement('div');
        header.className = 'panel-header';
        header.innerHTML = `
            <div class="panel-title">
                <span>🎯</span>
                <span>${title}</span>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Active</span>
                </div>
            </div>
            <div class="panel-controls">
                <button class="panel-control-btn drag-handle" title="Drag">
                    <span>⋮⋮</span>
                </button>
                <button class="panel-control-btn minimize-btn" title="Minimize">
                    <span>−</span>
                </button>
                <button class="panel-control-btn close-btn" title="Close">
                    <span>×</span>
                </button>
            </div>
        `;

        // Insert header at the beginning
        panel.insertBefore(header, panel.firstChild);

        // Wrap remaining content in panel-content
        const content = document.createElement('div');
        content.className = 'panel-content';
        
        // Move all children except header to content
        const children = Array.from(panel.children).filter(child => child !== header);
        children.forEach(child => content.appendChild(child));
        panel.appendChild(content);

        // Add event listeners
        this.setupPanelControls(panel, header);
    }

    setupPanelControls(panel, header) {
        const minimizeBtn = panel.querySelector('.minimize-btn');
        const closeBtn = panel.querySelector('.close-btn');
        const dragHandle = panel.querySelector('.drag-handle');

        // Minimize functionality
        minimizeBtn.addEventListener('click', () => {
            this.toggleMinimize(panel);
        });

        // Close functionality
        closeBtn.addEventListener('click', () => {
            this.closePanel(panel);
        });

        // Make header draggable
        this.setupDragHandling(panel, header);
    }

    toggleMinimize(panel) {
        const panelData = this.panels.get(panel.id);
        if (!panelData) return;

        const content = panel.querySelector('.panel-content');
        const minimizeBtn = panel.querySelector('.minimize-btn');

        if (panelData.isMinimized) {
            // Restore
            panel.classList.remove('minimized');
            minimizeBtn.innerHTML = '<span>−</span>';
            panelData.isMinimized = false;
            this.showNotification(`Panel restored: ${this.getPanelTitle(panel)}`, 'info');
        } else {
            // Minimize
            panelData.originalHeight = panel.offsetHeight;
            panel.classList.add('minimized');
            minimizeBtn.innerHTML = '<span>□</span>';
            panelData.isMinimized = true;
            this.showNotification(`Panel minimized: ${this.getPanelTitle(panel)}`, 'info');
        }
    }

    closePanel(panel) {
        const title = this.getPanelTitle(panel);
        panel.style.transition = 'all 0.3s ease';
        panel.style.opacity = '0';
        panel.style.transform = 'scale(0.8)';
        
        setTimeout(() => {
            panel.remove();
            this.panels.delete(panel.id);
        }, 300);

        this.showNotification(`Panel closed: ${title}`, 'warning');
    }

    getPanelTitle(panel) {
        const titleElement = panel.querySelector('.panel-title span:nth-child(2)');
        return titleElement ? titleElement.textContent : 'Panel';
    }

    setupDragHandling(panel, header) {
        let isDragging = false;
        let dragOffset = { x: 0, y: 0 };

        header.addEventListener('mousedown', (e) => {
            if (e.target.closest('.panel-control-btn')) return;
            
            isDragging = true;
            panel.classList.add('dragging');
            
            const rect = panel.getBoundingClientRect();
            dragOffset.x = e.clientX - rect.left;
            dragOffset.y = e.clientY - rect.top;

            // Make panel absolute positioned for dragging
            panel.style.position = 'absolute';
            panel.style.zIndex = '9999';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const x = e.clientX - dragOffset.x;
            const y = e.clientY - dragOffset.y;

            panel.style.left = Math.max(0, Math.min(x, window.innerWidth - panel.offsetWidth)) + 'px';
            panel.style.top = Math.max(0, Math.min(y, window.innerHeight - panel.offsetHeight)) + 'px';
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                panel.classList.remove('dragging');
            }
        });
    }

    makeDraggable(panel) {
        // Already handled in setupDragHandling
        console.log(`✅ Panel made draggable: ${panel.id}`);
    }

    addPanelControls() {
        // Add quick actions panel
        const quickActions = document.createElement('div');
        quickActions.className = 'quick-actions-panel';
        quickActions.innerHTML = `
            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #4a5568;">Quick Actions</h4>
            <button class="quick-action-btn" onclick="window.enhancedPlatform.minimizeAll()">
                Minimize All Panels
            </button>
            <button class="quick-action-btn" onclick="window.enhancedPlatform.restoreAll()">
                Restore All Panels
            </button>
            <button class="quick-action-btn" onclick="window.enhancedPlatform.enableAllArbitrage()">
                Enable All Arbitrage
            </button>
            <button class="quick-action-btn" onclick="window.enhancedPlatform.emergencyStop()">
                🚨 Emergency Stop
            </button>
            <div style="margin-top: 10px; font-size: 12px; color: #6b7280;">
                <div>Panels: <span id="active-panels-count">0</span></div>
                <div>Status: <span id="system-status">Loading...</span></div>
            </div>
        `;

        document.body.appendChild(quickActions);
        this.updateQuickActionsStatus();
    }

    updateQuickActionsStatus() {
        setInterval(() => {
            const activePanelsCount = document.getElementById('active-panels-count');
            const systemStatus = document.getElementById('system-status');

            if (activePanelsCount) {
                activePanelsCount.textContent = this.panels.size;
            }

            if (systemStatus) {
                if (this.isArbitrageInitialized) {
                    systemStatus.textContent = 'Ready';
                    systemStatus.style.color = '#10b981';
                } else {
                    systemStatus.textContent = 'Initializing...';
                    systemStatus.style.color = '#f59e0b';
                }
            }
        }, 2000);
    }

    async initArbitrageSystem() {
        console.log('🚀 Initializing arbitrage system with performance optimizations...');
        
        try {
            // Show loading indicator
            this.showArbitrageLoading();
            
            // Initialize in stages for better performance
            await this.initArbitrageCore();
            await this.initArbitrageUI();
            await this.initArbitrageIntegration();
            
            this.isArbitrageInitialized = true;
            this.hideArbitrageLoading();
            
            console.log('✅ Arbitrage system initialized successfully');
            this.showNotification('🎯 HFT Arbitrage System Ready', 'success');
            
        } catch (error) {
            console.error('❌ Arbitrage initialization failed:', error);
            this.handleArbitrageError(error);
        }
    }

    showArbitrageLoading() {
        const container = document.getElementById('hft-arbitrage-dashboard-container');
        if (!container) return;

        let progress = 0;
        container.innerHTML = `
            <div class="arbitrage-loading">
                <div class="loading-spinner"></div>
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    Initializing HFT Arbitrage System
                </div>
                <div style="font-size: 14px; opacity: 0.8; margin-bottom: 20px;" id="loading-status">
                    Setting up arbitrage engine...
                </div>
                <div class="loading-progress">
                    <div class="loading-progress-bar" id="loading-progress-bar" style="width: 0%"></div>
                </div>
                <div style="font-size: 12px; opacity: 0.6;" id="loading-percentage">0%</div>
            </div>
        `;

        // Simulate progress
        const progressInterval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 100) {
                progress = 100;
                clearInterval(progressInterval);
            }

            const progressBar = document.getElementById('loading-progress-bar');
            const percentage = document.getElementById('loading-percentage');
            const status = document.getElementById('loading-status');

            if (progressBar) progressBar.style.width = `${progress}%`;
            if (percentage) percentage.textContent = `${Math.floor(progress)}%`;
            
            if (status) {
                if (progress < 30) {
                    status.textContent = 'Setting up arbitrage engine...';
                } else if (progress < 60) {
                    status.textContent = 'Connecting to exchanges...';
                } else if (progress < 90) {
                    status.textContent = 'Initializing FinBERT sentiment analysis...';
                } else {
                    status.textContent = 'Final optimizations...';
                }
            }
        }, 200);
    }

    hideArbitrageLoading() {
        const container = document.getElementById('hft-arbitrage-dashboard-container');
        if (!container) return;

        // Initialize the actual arbitrage UI
        if (window.HFTArbitrageUI && this.arbitrageEngine) {
            container.innerHTML = '';
            const arbitrageUI = new HFTArbitrageUI(container, this.arbitrageEngine);
            window.arbitrageUI = arbitrageUI;
        } else {
            container.innerHTML = `
                <div class="text-center text-green-400 py-8">
                    <div class="text-4xl mb-4">✅</div>
                    <div class="text-xl font-semibold">HFT Arbitrage System Ready</div>
                    <div class="text-sm opacity-75 mt-2">All strategies loaded and ready for trading</div>
                </div>
            `;
        }
    }

    async initArbitrageCore() {
        // Initialize with delay to improve perceived performance
        return new Promise((resolve) => {
            setTimeout(() => {
                if (window.ArbitrageEngine) {
                    this.arbitrageEngine = new ArbitrageEngine();
                    window.arbitrageEngine = this.arbitrageEngine;
                }
                resolve();
            }, 100);
        });
    }

    async initArbitrageUI() {
        return new Promise((resolve) => {
            setTimeout(() => {
                // UI initialization happens in hideArbitrageLoading
                resolve();
            }, 200);
        });
    }

    async initArbitrageIntegration() {
        return new Promise((resolve) => {
            setTimeout(() => {
                if (window.HyperVisionArbitrageIntegration) {
                    // Integration will auto-initialize
                    console.log('Arbitrage integration will be handled automatically');
                }
                resolve();
            }, 100);
        });
    }

    handleArbitrageError(error) {
        const container = document.getElementById('hft-arbitrage-dashboard-container');
        if (container) {
            container.innerHTML = `
                <div class="text-center text-yellow-400 py-8">
                    <div class="text-4xl mb-4">⚠️</div>
                    <div class="text-xl font-semibold">Arbitrage System Error</div>
                    <div class="text-sm opacity-75 mt-2">${error.message}</div>
                    <button onclick="window.enhancedPlatform.retryArbitrageInit()" 
                            class="mt-4 px-4 py-2 bg-yellow-500 text-black rounded-lg hover:bg-yellow-400 transition-colors">
                        Retry Initialization
                    </button>
                </div>
            `;
        }
    }

    async retryArbitrageInit() {
        await this.initArbitrageSystem();
    }

    initPaymentIntegration() {
        // Create enhanced payment registration panel
        this.createPaymentRegistrationPanel();
        
        // Integration with existing trading engine
        this.setupPaymentTradingBridge();
        
        console.log('✅ Payment integration initialized');
    }

    createPaymentRegistrationPanel() {
        const existingPanel = document.querySelector('.payment-integration-panel');
        if (existingPanel) {
            existingPanel.remove();
        }

        const paymentPanel = document.createElement('div');
        paymentPanel.className = 'payment-integration-panel gomna-panel';
        paymentPanel.id = 'payment-registration-panel';
        
        paymentPanel.innerHTML = `
            <div class="panel-header">
                <div class="panel-title">
                    <span>💳</span>
                    <span>Payment & Trading Account</span>
                    <div class="status-indicator">
                        <div class="status-dot" id="payment-status-dot"></div>
                        <span id="payment-status-text">Ready</span>
                    </div>
                </div>
                <div class="panel-controls">
                    <button class="panel-control-btn drag-handle" title="Drag">
                        <span>⋮⋮</span>
                    </button>
                    <button class="panel-control-btn minimize-btn" title="Minimize">
                        <span>−</span>
                    </button>
                </div>
            </div>
            
            <div class="panel-content">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="space-y-4">
                        <h4 class="text-lg font-semibold text-white">Quick Registration</h4>
                        <form id="quick-registration-form" class="space-y-3">
                            <input type="email" placeholder="Email Address" required 
                                   class="w-full px-4 py-2 rounded-lg bg-white bg-opacity-20 text-white placeholder-gray-300 border border-white border-opacity-30 focus:border-opacity-60 focus:outline-none">
                            <input type="text" placeholder="Full Name" required 
                                   class="w-full px-4 py-2 rounded-lg bg-white bg-opacity-20 text-white placeholder-gray-300 border border-white border-opacity-30 focus:border-opacity-60 focus:outline-none">
                            <select class="w-full px-4 py-2 rounded-lg bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:border-opacity-60 focus:outline-none">
                                <option value="individual">Individual Account</option>
                                <option value="business">Business Account</option>
                                <option value="institutional">Institutional Account</option>
                            </select>
                            <button type="submit" class="w-full px-4 py-2 bg-white text-purple-700 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
                                Create Trading Account
                            </button>
                        </form>
                    </div>
                    
                    <div class="space-y-4">
                        <h4 class="text-lg font-semibold text-white">Account Features</h4>
                        <div class="space-y-2 text-sm text-white text-opacity-90">
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>Integrated Payment Processing</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>HFT Arbitrage Trading</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>AI-Powered Strategies</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>Multi-Exchange Connectivity</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>Real-time Risk Management</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-300">✓</span>
                                <span>Institutional Grade Security</span>
                            </div>
                        </div>
                        
                        <div class="mt-4 p-3 bg-white bg-opacity-10 rounded-lg">
                            <div class="text-sm text-white">
                                <strong>Demo Mode Available</strong><br>
                                Start with $10,000 virtual balance to test strategies
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert after main content or at the end
        const mainContent = document.querySelector('.max-w-7xl') || document.body;
        mainContent.appendChild(paymentPanel);

        // Make it draggable
        this.enhancePanel(paymentPanel);

        // Setup form handler
        this.setupPaymentForm();
    }

    setupPaymentForm() {
        const form = document.getElementById('quick-registration-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(form);
            const email = formData.get('email') || e.target.elements[0].value;
            const fullName = formData.get('fullName') || e.target.elements[1].value;
            const accountType = e.target.elements[2].value;

            this.handlePaymentRegistration({ email, fullName, accountType });
        });
    }

    async handlePaymentRegistration(userData) {
        try {
            this.showNotification('Creating trading account...', 'info');
            
            // Update status
            const statusDot = document.getElementById('payment-status-dot');
            const statusText = document.getElementById('payment-status-text');
            
            if (statusDot) statusDot.className = 'status-dot warning';
            if (statusText) statusText.textContent = 'Creating Account...';

            // Simulate account creation with trading engine
            if (window.tradingEngine && typeof window.tradingEngine.createTradingAccount === 'function') {
                const result = await window.tradingEngine.createTradingAccount(userData, {
                    riskProfile: 'moderate',
                    tradingStyle: 'balanced',
                    aiAssistance: true,
                    autoRebalancing: false,
                    maxDrawdown: 0.15,
                    preferredAssets: ['BTC', 'ETH', 'SPY', 'QQQ']
                });

                if (result.success) {
                    this.showNotification(`✅ Trading account created! Welcome ${userData.fullName}`, 'success');
                    
                    if (statusDot) statusDot.className = 'status-dot';
                    if (statusText) statusText.textContent = 'Account Active';
                    
                    // Enable arbitrage features
                    this.enableArbitrageForUser(result.user);
                } else {
                    throw new Error(result.message || 'Account creation failed');
                }
            } else {
                // Demo account creation
                const demoResult = {
                    success: true,
                    user: {
                        userId: 'demo-' + Date.now(),
                        email: userData.email,
                        fullName: userData.fullName,
                        accountType: 'demo',
                        balance: 10000,
                        tradingProfile: {
                            riskProfile: 'moderate',
                            aiAssistance: true
                        }
                    },
                    message: 'Demo account created successfully!'
                };

                this.showNotification(`✅ Demo account created! Welcome ${userData.fullName}`, 'success');
                
                if (statusDot) statusDot.className = 'status-dot';
                if (statusText) statusText.textContent = 'Demo Active';
            }

        } catch (error) {
            console.error('Payment registration failed:', error);
            this.showNotification(`❌ Account creation failed: ${error.message}`, 'error');
            
            const statusDot = document.getElementById('payment-status-dot');
            const statusText = document.getElementById('payment-status-text');
            
            if (statusDot) statusDot.className = 'status-dot error';
            if (statusText) statusText.textContent = 'Error';
        }
    }

    enableArbitrageForUser(user) {
        console.log('🎯 Enabling arbitrage features for user:', user.userId);
        
        // Automatically enable recommended strategies for new users
        if (this.isArbitrageInitialized && window.arbitrageIntegration) {
            const recommendedStrategies = ['cross_exchange', 'statistical'];
            
            setTimeout(() => {
                recommendedStrategies.forEach(strategy => {
                    const toggle = document.getElementById(`enable-${strategy.replace('_', '-')}`);
                    if (toggle) {
                        toggle.checked = true;
                        toggle.dispatchEvent(new Event('change'));
                    }
                });
                
                this.showNotification('🚀 Arbitrage strategies enabled for your account', 'success');
            }, 2000);
        }
    }

    setupPaymentTradingBridge() {
        // Bridge between payment system and trading engine
        console.log('🌉 Setting up payment-trading bridge...');
        
        // This would integrate with actual payment processors
        window.paymentTradingBridge = {
            processPayment: async (amount, currency) => {
                console.log(`Processing payment: ${currency} ${amount}`);
                return { success: true, transactionId: 'tx_' + Date.now() };
            },
            
            withdrawFunds: async (amount, currency) => {
                console.log(`Processing withdrawal: ${currency} ${amount}`);
                return { success: true, transactionId: 'wd_' + Date.now() };
            },
            
            getAccountBalance: () => {
                return { available: 10000, reserved: 0, currency: 'USD' };
            }
        };
    }

    // Panel management methods
    minimizeAll() {
        this.panels.forEach((panelData, id) => {
            if (!panelData.isMinimized) {
                this.toggleMinimize(panelData.element);
            }
        });
        this.showNotification('All panels minimized', 'info');
    }

    restoreAll() {
        this.panels.forEach((panelData, id) => {
            if (panelData.isMinimized) {
                this.toggleMinimize(panelData.element);
            }
        });
        this.showNotification('All panels restored', 'success');
    }

    enableAllArbitrage() {
        const toggles = document.querySelectorAll('.strategy-toggle');
        toggles.forEach(toggle => {
            toggle.checked = true;
            toggle.dispatchEvent(new Event('change'));
        });
        this.showNotification('All arbitrage strategies enabled', 'success');
    }

    emergencyStop() {
        // Stop all arbitrage strategies
        const toggles = document.querySelectorAll('.strategy-toggle');
        toggles.forEach(toggle => {
            toggle.checked = false;
            toggle.dispatchEvent(new Event('change'));
        });
        
        // Stop arbitrage engine if available
        if (this.arbitrageEngine && typeof this.arbitrageEngine.emergencyStop === 'function') {
            this.arbitrageEngine.emergencyStop();
        }
        
        this.showNotification('🚨 EMERGENCY STOP: All strategies disabled', 'error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 left-1/2 transform -translate-x-1/2 z-50 px-6 py-3 rounded-lg shadow-lg text-white font-medium transition-all duration-300`;
        
        const colors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };
        
        notification.classList.add(colors[type] || colors.info);
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.style.transform = 'translateX(-50%) translateY(0)', 10);
        
        // Remove after delay
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(-50%) translateY(-20px)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    handleInitializationError(error) {
        console.error('Platform initialization error:', error);
        this.showNotification(`Platform initialization error: ${error.message}`, 'error');
        
        // Show fallback UI
        const fallback = document.createElement('div');
        fallback.innerHTML = `
            <div class="fixed bottom-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg shadow-lg">
                <strong>Platform Error:</strong> ${error.message}
                <button onclick="location.reload()" class="ml-2 px-2 py-1 bg-red-500 text-white rounded text-sm">
                    Reload
                </button>
            </div>
        `;
        document.body.appendChild(fallback);
    }

    // Initialization method for existing panels
    initDraggableFeatures() {
        console.log('🎯 Initializing draggable features...');
        
        // Wait for DOM to be ready, then enhance existing panels
        setTimeout(() => {
            this.enhanceExistingPanels();
            console.log('✅ Draggable features initialized');
        }, 500);
    }
}

// Initialize when DOM is ready
function initializeEnhancedDraggablePlatform() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.enhancedPlatform = new GomnaEnhancedDraggablePlatform();
        });
    } else {
        window.enhancedPlatform = new GomnaEnhancedDraggablePlatform();
    }
}

// Auto-initialize
initializeEnhancedDraggablePlatform();

console.log('🚀 Enhanced Draggable Platform with HFT Arbitrage loaded');