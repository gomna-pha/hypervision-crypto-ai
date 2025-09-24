/**
 * GOMNA CLEAN PLATFORM INITIALIZATION
 * Single source of truth for platform initialization
 * Prevents conflicts and ensures smooth operation
 */

class CleanPlatformManager {
    constructor() {
        this.isInitialized = false;
        this.components = {
            draggable: false,
            arbitrage: false,
            payment: false,
            marketplace: false,
            investor: false,
            liveData: false,
            ui: false
        };
        this.panels = new Map();
        
        // Prevent multiple initializations
        if (window.cleanPlatform) {
            console.warn('🚫 Platform already initialized, skipping duplicate');
            return window.cleanPlatform;
        }
        
        window.cleanPlatform = this;
        this.init();
    }

    async init() {
        console.log('🧹 Initializing Clean GOMNA Platform...');
        
        try {
            // Step 1: Clean up any existing mess
            this.cleanup();
            
            // Step 2: Initialize core UI (fast)
            this.initCoreUI();
            
            // Step 3: Initialize features progressively
            await this.initFeatures();
            
            // Step 4: Mark as complete
            this.isInitialized = true;
            console.log('✅ Clean Platform initialized successfully');
            
        } catch (error) {
            console.error('❌ Platform initialization failed:', error);
        }
    }

    cleanup() {
        console.log('🧹 Cleaning up existing platform mess...');
        
        // Remove duplicate quick action panels
        document.querySelectorAll('.quick-actions-panel').forEach((panel, index) => {
            if (index > 0) panel.remove(); // Keep only first one
        });
        
        // Remove duplicate payment panels
        document.querySelectorAll('.payment-integration-panel').forEach((panel, index) => {
            if (index > 0) panel.remove();
        });
        
        // Reset any broken draggable elements
        document.querySelectorAll('.gomna-panel.dragging').forEach(panel => {
            panel.classList.remove('dragging');
            panel.style.position = '';
            panel.style.zIndex = '';
        });
        
        // Clear any interval spam
        for (let i = 1; i < 99999; i++) {
            window.clearInterval(i);
            window.clearTimeout(i);
        }
        
        console.log('✅ Cleanup complete');
    }

    initCoreUI() {
        console.log('🎨 Initializing core UI...');
        
        // Add clean styles
        this.addCleanStyles();
        
        // Initialize tab system (if not already working)
        this.ensureTabsWork();
        
        // Add loading states
        this.addLoadingStates();
        
        this.components.ui = true;
        console.log('✅ Core UI initialized');
    }

    addCleanStyles() {
        // Only add if not already added
        if (document.getElementById('clean-platform-styles')) return;
        
        const styles = `
            /* Clean Platform Styles */
            .clean-panel {
                position: relative;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                margin: 15px 0;
                transition: all 0.3s ease;
                border: 1px solid #e2e8f0;
            }

            .clean-panel:hover {
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            }

            .clean-panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 20px;
                border-bottom: 1px solid #e2e8f0;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px 12px 0 0;
                cursor: move;
                user-select: none;
            }

            .clean-panel-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                color: #1a202c;
            }

            .clean-panel-controls {
                display: flex;
                gap: 4px;
            }

            .clean-control-btn {
                width: 24px;
                height: 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                background: rgba(0, 0, 0, 0.1);
                color: #4a5568;
                transition: all 0.2s ease;
            }

            .clean-control-btn:hover {
                background: #4a5568;
                color: white;
            }

            .clean-panel-content {
                padding: 20px;
            }

            .clean-panel.minimized .clean-panel-content {
                display: none;
            }

            .clean-loading {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 40px;
                color: #64748b;
            }

            .clean-spinner {
                width: 24px;
                height: 24px;
                border: 2px solid #e2e8f0;
                border-left: 2px solid #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 12px;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .clean-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 8px;
                padding: 12px 16px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                border-left: 4px solid #10b981;
                z-index: 10000;
                max-width: 350px;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            .arbitrage-clean {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                border-radius: 12px;
                padding: 20px;
            }

            .strategy-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
                margin-top: 20px;
            }

            .strategy-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 16px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .strategy-toggle {
                position: relative;
                display: inline-block;
                width: 44px;
                height: 24px;
            }

            .strategy-toggle input {
                opacity: 0;
                width: 0;
                height: 0;
            }

            .strategy-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #64748b;
                transition: 0.3s;
                border-radius: 24px;
            }

            .strategy-slider:before {
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: 0.3s;
                border-radius: 50%;
            }

            input:checked + .strategy-slider {
                background-color: #10b981;
            }

            input:checked + .strategy-slider:before {
                transform: translateX(20px);
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.id = 'clean-platform-styles';
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    ensureTabsWork() {
        // Make sure tab switching works properly
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(btn => {
            // Remove existing listeners to prevent duplicates
            btn.replaceWith(btn.cloneNode(true));
        });
        
        // Re-add listeners to cloned buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = btn.dataset.tab;
                
                // Hide all tabs
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                
                // Show selected tab
                const targetTab = document.getElementById(`${tabName}-tab`);
                if (targetTab) {
                    targetTab.classList.remove('hidden');
                }
                
                // Update button styles
                tabButtons.forEach(b => {
                    b.classList.remove('bg-white', 'text-blue-600', 'border-b-2', 'border-blue-600');
                    b.classList.add('text-gray-600');
                });
                
                btn.classList.add('bg-white', 'text-blue-600', 'border-b-2', 'border-blue-600');
                btn.classList.remove('text-gray-600');
                
                // Initialize specific features for arbitrage tab
                if (tabName === 'arbitrage') {
                    this.initArbitrageFeatures();
                }
                
                // Initialize marketplace when switching to marketplace tab
                if (tabName === 'marketplace') {
                    this.initMarketplaceFeatures();
                }
            });
        });
    }

    addLoadingStates() {
        // Add loading to arbitrage container
        const arbitrageContainer = document.getElementById('hft-arbitrage-dashboard-container');
        if (arbitrageContainer) {
            arbitrageContainer.innerHTML = `
                <div class="clean-loading">
                    <div class="clean-spinner"></div>
                    <span>Loading HFT Arbitrage System...</span>
                </div>
            `;
        }
    }

    async initFeatures() {
        console.log('🚀 Initializing features...');
        
        // Initialize draggable features
        setTimeout(() => this.initDraggableFeatures(), 500);
        
        // Initialize payment system
        setTimeout(() => this.initPaymentSystem(), 1000);
        
        // Initialize arbitrage system
        setTimeout(() => this.initArbitrageSystem(), 1500);
        
        // Initialize marketplace system
        setTimeout(() => this.initMarketplaceSystem(), 2000);
        
        // Initialize investor account system
        setTimeout(() => this.initInvestorSystem(), 2500);
        
        // Initialize live data API system
        setTimeout(() => this.initLiveDataSystem(), 3000);
    }

    initDraggableFeatures() {
        if (this.components.draggable) return;
        
        console.log('🎯 Initializing draggable features...');
        
        // Find panels that should be draggable
        const panelSelectors = [
            '.bg-gradient-to-r',
            '.metric-card',
            '.glass-effect'
        ];
        
        let panelCount = 0;
        panelSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(panel => {
                if (!panel.classList.contains('clean-panel') && panelCount < 10) { // Limit to prevent spam
                    this.makePanelDraggable(panel);
                    panelCount++;
                }
            });
        });
        
        this.components.draggable = true;
        console.log(`✅ Made ${panelCount} panels draggable`);
    }

    makePanelDraggable(panel) {
        panel.classList.add('clean-panel');
        
        // Add header if it doesn't exist
        const existingHeader = panel.querySelector('.clean-panel-header, .panel-header');
        if (!existingHeader) {
            const title = this.extractPanelTitle(panel);
            
            const header = document.createElement('div');
            header.className = 'clean-panel-header';
            header.innerHTML = `
                <div class="clean-panel-title">
                    <span>📊</span>
                    <span>${title}</span>
                </div>
                <div class="clean-panel-controls">
                    <button class="clean-control-btn minimize-btn" title="Minimize">
                        <span>−</span>
                    </button>
                </div>
            `;
            
            // Wrap existing content
            const content = document.createElement('div');
            content.className = 'clean-panel-content';
            while (panel.firstChild) {
                content.appendChild(panel.firstChild);
            }
            
            panel.appendChild(header);
            panel.appendChild(content);
            
            // Add minimize functionality
            const minimizeBtn = panel.querySelector('.minimize-btn');
            minimizeBtn.addEventListener('click', () => {
                panel.classList.toggle('minimized');
                minimizeBtn.innerHTML = panel.classList.contains('minimized') ? 
                    '<span>□</span>' : '<span>−</span>';
            });
            
            // Make draggable
            let isDragging = false;
            let dragOffset = { x: 0, y: 0 };
            
            header.addEventListener('mousedown', (e) => {
                if (e.target.closest('.clean-control-btn')) return;
                
                isDragging = true;
                const rect = panel.getBoundingClientRect();
                dragOffset.x = e.clientX - rect.left;
                dragOffset.y = e.clientY - rect.top;
                
                panel.style.position = 'absolute';
                panel.style.zIndex = '1000';
                panel.style.width = rect.width + 'px';
            });
            
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const x = e.clientX - dragOffset.x;
                const y = e.clientY - dragOffset.y;
                
                panel.style.left = Math.max(0, x) + 'px';
                panel.style.top = Math.max(0, y) + 'px';
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = false;
            });
        }
        
        // Store panel reference
        const panelId = panel.id || `clean_panel_${Date.now()}`;
        panel.id = panelId;
        this.panels.set(panelId, panel);
    }

    extractPanelTitle(panel) {
        const titleSelectors = [
            'h1', 'h2', 'h3', 'h4',
            '.text-xl', '.text-2xl', '.text-3xl',
            '.font-bold'
        ];
        
        for (const selector of titleSelectors) {
            const titleEl = panel.querySelector(selector);
            if (titleEl && titleEl.textContent.trim()) {
                return titleEl.textContent.trim().substring(0, 30);
            }
        }
        
        return 'Panel';
    }

    initPaymentSystem() {
        if (this.components.payment) return;
        
        console.log('💳 Initializing payment system...');
        
        // Remove existing payment panels to prevent duplicates
        document.querySelectorAll('.payment-integration-panel').forEach(panel => {
            panel.remove();
        });
        
        // Create clean payment panel
        const paymentPanel = document.createElement('div');
        paymentPanel.className = 'clean-panel';
        paymentPanel.innerHTML = `
            <div class="clean-panel-header">
                <div class="clean-panel-title">
                    <span>💳</span>
                    <span>Quick Account Registration</span>
                </div>
                <div class="clean-panel-controls">
                    <button class="clean-control-btn minimize-btn" title="Minimize">
                        <span>−</span>
                    </button>
                </div>
            </div>
            <div class="clean-panel-content">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <form id="clean-registration-form" class="space-y-3">
                            <input type="email" placeholder="Email Address" required class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                            <input type="text" placeholder="Full Name" required class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                            <select class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                                <option value="demo">Demo Account ($10,000 virtual)</option>
                                <option value="individual">Individual Account</option>
                                <option value="business">Business Account</option>
                            </select>
                            <button type="submit" class="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                                Create Account
                            </button>
                        </form>
                    </div>
                    <div class="text-sm text-gray-600 space-y-2">
                        <h4 class="font-semibold text-gray-800">Account Features:</h4>
                        <div class="space-y-1">
                            <div>✓ HFT Arbitrage Trading</div>
                            <div>✓ AI-Powered Strategies</div>
                            <div>✓ Multi-Exchange Access</div>
                            <div>✓ Real-time Risk Management</div>
                            <div>✓ Professional Dashboard</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Insert into page
        const mainContent = document.querySelector('#dashboard-tab') || document.querySelector('.max-w-7xl');
        if (mainContent) {
            mainContent.appendChild(paymentPanel);
        }
        
        // Make it draggable
        this.makePanelDraggable(paymentPanel);
        
        // Setup form
        const form = document.getElementById('clean-registration-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            this.handleRegistration(formData);
        });
        
        this.components.payment = true;
        console.log('✅ Payment system initialized');
    }

    handleRegistration(formData) {
        const email = formData.get('email') || document.querySelector('#clean-registration-form input[type="email"]').value;
        const name = formData.get('name') || document.querySelector('#clean-registration-form input[type="text"]').value;
        
        this.showNotification(`Account created for ${name}! Welcome to GOMNA Trading.`);
        
        // Enable some arbitrage strategies for demo
        setTimeout(() => {
            const arbitrageToggles = document.querySelectorAll('.strategy-toggle input');
            arbitrageToggles.forEach((toggle, index) => {
                if (index < 2) { // Enable first 2 strategies
                    toggle.checked = true;
                }
            });
        }, 2000);
    }

    initArbitrageSystem() {
        if (this.components.arbitrage) return;
        
        console.log('🎯 Initializing arbitrage system...');
        
        const arbitrageContainer = document.getElementById('hft-arbitrage-dashboard-container');
        if (!arbitrageContainer) return;
        
        arbitrageContainer.innerHTML = `
            <div class="arbitrage-clean">
                <div class="text-center mb-6">
                    <h3 class="text-2xl font-bold mb-2">🚀 HFT Arbitrage Strategies</h3>
                    <p class="opacity-90">Professional arbitrage trading with AI optimization</p>
                </div>
                
                <div class="strategy-grid">
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Cross-Exchange Arbitrage</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="cross-exchange-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">Exploit price differences across exchanges</p>
                        <div class="text-xs opacity-60 mt-2">Min Spread: 0.02% | Max Latency: 100ms</div>
                    </div>
                    
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Statistical Arbitrage</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="statistical-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">Pairs trading with mean reversion</p>
                        <div class="text-xs opacity-60 mt-2">Z-Score: 2.0 | Lookback: 252 days</div>
                    </div>
                    
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">News-Based (FinBERT)</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="news-based-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">AI sentiment analysis trading</p>
                        <div class="text-xs opacity-60 mt-2">Reaction Time: 50ms | Confidence: 80%</div>
                    </div>
                    
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Index Arbitrage</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="index-arbitrage-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">ETF vs basket trading</p>
                        <div class="text-xs opacity-60 mt-2">ETFs: SPY, QQQ | Tracking: 0.001</div>
                    </div>
                    
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Latency Arbitrage</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="latency-arbitrage-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">Ultra-low latency advantage</p>
                        <div class="text-xs opacity-60 mt-2">Max Latency: 1ms | IOC/FOK Orders</div>
                    </div>
                    
                    <div class="strategy-card">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Volatility Arbitrage</h4>
                            <label class="strategy-toggle">
                                <input type="checkbox" id="volatility-arbitrage-toggle">
                                <span class="strategy-slider"></span>
                            </label>
                        </div>
                        <p class="text-sm opacity-75">Implied vs realized vol</p>
                        <div class="text-xs opacity-60 mt-2">Threshold: 0.05 | Window: 30 days</div>
                    </div>
                </div>
                
                <div class="flex gap-4 justify-center mt-6">
                    <button id="enable-all-clean" class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                        Enable All Strategies
                    </button>
                    <button id="disable-all-clean" class="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors">
                        Disable All Strategies
                    </button>
                    <button id="emergency-stop-clean" class="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
                        🚨 Emergency Stop
                    </button>
                </div>
                
                <div class="mt-6 p-4 bg-white bg-opacity-10 rounded-lg">
                    <div class="text-center">
                        <div class="text-sm opacity-75">System Status</div>
                        <div class="text-xl font-bold text-green-400">✅ Ready</div>
                        <div class="text-xs opacity-60 mt-1">All systems operational</div>
                    </div>
                </div>
            </div>
        `;
        
        // Setup strategy toggles
        document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const strategy = e.target.id.replace('-toggle', '');
                const status = e.target.checked ? 'enabled' : 'disabled';
                this.showNotification(`${strategy} strategy ${status}`);
            });
        });
        
        // Setup control buttons
        document.getElementById('enable-all-clean').addEventListener('click', () => {
            document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
                toggle.checked = true;
            });
            this.showNotification('All arbitrage strategies enabled');
        });
        
        document.getElementById('disable-all-clean').addEventListener('click', () => {
            document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
                toggle.checked = false;
            });
            this.showNotification('All arbitrage strategies disabled');
        });
        
        document.getElementById('emergency-stop-clean').addEventListener('click', () => {
            document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
                toggle.checked = false;
            });
            this.showNotification('🚨 EMERGENCY STOP: All strategies disabled', 'error');
        });
        
        this.components.arbitrage = true;
        console.log('✅ Arbitrage system initialized');
    }

    initArbitrageFeatures() {
        // Initialize arbitrage features when tab is opened
        if (!this.components.arbitrage) {
            this.initArbitrageSystem();
        }
    }

    initMarketplaceSystem() {
        if (this.components.marketplace) return;
        
        console.log('🏪 Initializing Algorithmic Marketplace System...');
        
        try {
            // Load marketplace UI if available
            if (typeof AlgorithmicMarketplaceUI !== 'undefined') {
                if (!window.marketplaceUI) {
                    window.marketplaceUI = new AlgorithmicMarketplaceUI();
                }
            } else {
                console.log('⏳ AlgorithmicMarketplaceUI not yet loaded, will retry...');
                setTimeout(() => this.initMarketplaceSystem(), 500);
                return;
            }
            
            this.components.marketplace = true;
            console.log('✅ Marketplace system initialized');
            
        } catch (error) {
            console.error('❌ Failed to initialize marketplace system:', error);
        }
    }

    initMarketplaceFeatures() {
        // Initialize marketplace features when tab is opened
        if (!this.components.marketplace) {
            this.initMarketplaceSystem();
        }
        
        // Ensure marketplace UI is displayed
        const marketplaceContainer = document.getElementById('marketplace-container');
        if (marketplaceContainer && window.marketplaceUI) {
            // Check if marketplace UI is already loaded
            if (marketplaceContainer.querySelector('.algorithmic-marketplace')) {
                // Already loaded, just show it
                return;
            }
            
            // Replace loading screen with actual marketplace
            marketplaceContainer.innerHTML = window.marketplaceUI.getMarketplaceHTML();
            
            // Initialize real-time updates for this tab
            window.marketplaceUI.showMarketplaceTab();
        }
    }

    initInvestorSystem() {
        if (this.components.investor) return;
        
        console.log('👤 Initializing Investor Account System...');
        
        try {
            // Load investor system if available
            if (typeof InvestorAccountSystem !== 'undefined') {
                if (!window.investorAccounts) {
                    window.investorAccounts = new InvestorAccountSystem();
                }
                
                // Connect to marketplace if available
                if (window.marketplaceUI) {
                    window.marketplaceUI.investorSystem = window.investorAccounts;
                }
            } else {
                console.log('⏳ InvestorAccountSystem not yet loaded, will retry...');
                setTimeout(() => this.initInvestorSystem(), 500);
                return;
            }
            
            this.components.investor = true;
            console.log('✅ Investor system initialized');
            
        } catch (error) {
            console.error('❌ Failed to initialize investor system:', error);
        }
    }

    initLiveDataSystem() {
        if (this.components.liveData) return;
        
        console.log('📡 Initializing Live Data API System...');
        
        try {
            // Load live data system if available
            if (typeof LiveDataAPISystem !== 'undefined') {
                if (!window.liveDataAPI) {
                    window.liveDataAPI = new LiveDataAPISystem();
                }
                
                // Connect to marketplace if available
                if (window.marketplaceUI) {
                    window.marketplaceUI.liveDataAPI = window.liveDataAPI;
                }
            } else {
                console.log('⏳ LiveDataAPISystem not yet loaded, will retry...');
                setTimeout(() => this.initLiveDataSystem(), 500);
                return;
            }
            
            this.components.liveData = true;
            console.log('✅ Live data system initialized');
            
        } catch (error) {
            console.error('❌ Failed to initialize live data system:', error);
        }
    }

    showNotification(message, type = 'success') {
        // Remove existing notifications
        document.querySelectorAll('.clean-notification').forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = 'clean-notification';
        notification.textContent = message;
        
        if (type === 'error') {
            notification.style.borderLeftColor = '#ef4444';
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Public methods for external control
    enableAllStrategies() {
        document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
            toggle.checked = true;
        });
        this.showNotification('All strategies enabled');
    }

    disableAllStrategies() {
        document.querySelectorAll('.strategy-toggle input').forEach(toggle => {
            toggle.checked = false;
        });
        this.showNotification('All strategies disabled');
    }

    minimizeAllPanels() {
        this.panels.forEach(panel => {
            panel.classList.add('minimized');
            const btn = panel.querySelector('.minimize-btn');
            if (btn) btn.innerHTML = '<span>□</span>';
        });
        this.showNotification('All panels minimized');
    }

    restoreAllPanels() {
        this.panels.forEach(panel => {
            panel.classList.remove('minimized');
            const btn = panel.querySelector('.minimize-btn');
            if (btn) btn.innerHTML = '<span>−</span>';
        });
        this.showNotification('All panels restored');
    }
}

// Initialize the clean platform
function initCleanPlatform() {
    // Only initialize once
    if (window.cleanPlatform) {
        console.log('🔄 Clean platform already exists');
        return;
    }
    
    console.log('🧹 Starting clean platform initialization...');
    new CleanPlatformManager();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCleanPlatform);
} else {
    // DOM already loaded
    setTimeout(initCleanPlatform, 100);
}

// Global access
window.initCleanPlatform = initCleanPlatform;

console.log('🧹 Clean Platform Manager loaded');