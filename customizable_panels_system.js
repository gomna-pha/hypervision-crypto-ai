/**
 * Cocoa Trading AI - Drag-and-Drop Customizable Panels System
 * Professional dashboard customization with drag-and-drop panels
 * 
 * Features:
 * - Drag-and-drop panel rearrangement
 * - Resizable panels and grid system
 * - Panel templates and presets
 * - Save/load custom layouts
 */

class CustomizablePanelsSystem {
    constructor() {
        this.panels = new Map();
        this.gridSystem = null;
        this.draggedPanel = null;
        this.layouts = {};
        this.currentLayout = 'default';
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        try {
            console.log('🎛️ Initializing Customizable Panels System...');
            
            await this.loadSavedLayouts();
            this.initializeGridSystem();
            this.createControlPanel();
            this.registerDefaultPanels();
            this.setupDragAndDrop();
            this.loadLayout(this.currentLayout);
            
            this.isInitialized = true;
            console.log('✅ Customizable Panels System initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing customizable panels system:', error);
        }
    }

    async loadSavedLayouts() {
        // Load saved layouts from localStorage
        const savedLayouts = localStorage.getItem('cocoaTradingAI_layouts');
        if (savedLayouts) {
            this.layouts = JSON.parse(savedLayouts);
        } else {
            this.layouts = {
                default: this.getDefaultLayout(),
                trading: this.getTradingLayout(),
                analytics: this.getAnalyticsLayout()
            };
        }
        
        console.log('✅ Layouts loaded');
    }

    getDefaultLayout() {
        return {
            name: 'Default Layout',
            description: 'Balanced view with all essential panels',
            panels: [
                { id: 'market-overview', x: 0, y: 0, w: 12, h: 4 },
                { id: 'trading-pairs', x: 0, y: 4, w: 6, h: 6 },
                { id: 'arbitrage-opportunities', x: 6, y: 4, w: 6, h: 6 },
                { id: 'portfolio-metrics', x: 0, y: 10, w: 8, h: 6 },
                { id: 'news-feed', x: 8, y: 10, w: 4, h: 6 },
                { id: 'account-tier', x: 0, y: 16, w: 6, h: 4 },
                { id: 'regulatory-credentials', x: 6, y: 16, w: 6, h: 4 }
            ]
        };
    }

    getTradingLayout() {
        return {
            name: 'Trading Focus',
            description: 'Optimized for active trading',
            panels: [
                { id: 'trading-pairs', x: 0, y: 0, w: 8, h: 8 },
                { id: 'arbitrage-opportunities', x: 8, y: 0, w: 4, h: 8 },
                { id: 'market-overview', x: 0, y: 8, w: 6, h: 4 },
                { id: 'portfolio-metrics', x: 6, y: 8, w: 6, h: 4 },
                { id: 'news-feed', x: 0, y: 12, w: 12, h: 4 }
            ]
        };
    }

    getAnalyticsLayout() {
        return {
            name: 'Analytics Dashboard',
            description: 'Focus on data and performance metrics',
            panels: [
                { id: 'portfolio-metrics', x: 0, y: 0, w: 8, h: 8 },
                { id: 'market-overview', x: 8, y: 0, w: 4, h: 4 },
                { id: 'regulatory-credentials', x: 8, y: 4, w: 4, h: 4 },
                { id: 'trading-pairs', x: 0, y: 8, w: 6, h: 6 },
                { id: 'arbitrage-opportunities', x: 6, y: 8, w: 6, h: 6 }
            ]
        };
    }

    initializeGridSystem() {
        // Create main grid container
        const gridContainer = document.createElement('div');
        gridContainer.id = 'customizable-dashboard-grid';
        gridContainer.className = 'dashboard-grid-system';
        
        // Add grid system styles
        const gridStyles = document.createElement('style');
        gridStyles.textContent = `
            .dashboard-grid-system {
                display: grid;
                grid-template-columns: repeat(12, 1fr);
                grid-template-rows: repeat(auto-fit, 100px);
                gap: 15px;
                padding: 20px;
                min-height: 100vh;
                background: rgba(26, 26, 26, 0.3);
                position: relative;
            }

            .grid-panel {
                background: rgba(26, 26, 26, 0.95);
                border: 2px solid rgba(212, 165, 116, 0.2);
                border-radius: 12px;
                backdrop-filter: blur(15px);
                box-shadow: 0 8px 32px rgba(139, 69, 19, 0.2);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                cursor: move;
            }

            .grid-panel:hover {
                border-color: rgba(212, 165, 116, 0.4);
                box-shadow: 0 12px 40px rgba(139, 69, 19, 0.3);
                transform: translateY(-2px);
            }

            .grid-panel.dragging {
                opacity: 0.7;
                transform: rotate(3deg) scale(1.02);
                z-index: 1000;
                cursor: grabbing;
            }

            .grid-panel.drop-target {
                border-color: var(--cocoa-accent);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
            }

            .panel-header {
                background: linear-gradient(135deg, var(--cocoa-primary) 0%, var(--cocoa-secondary) 100%);
                color: var(--cocoa-text);
                padding: 12px 16px;
                border-radius: 12px 12px 0 0;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: move;
            }

            .panel-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 1rem;
            }

            .panel-controls {
                display: flex;
                gap: 5px;
                align-items: center;
            }

            .panel-control-btn {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                transition: all 0.2s ease;
            }

            .panel-control-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: scale(1.1);
            }

            .panel-content {
                padding: 16px;
                height: calc(100% - 50px);
                overflow-y: auto;
                color: var(--cocoa-text);
            }

            .panel-content::-webkit-scrollbar {
                width: 6px;
            }

            .panel-content::-webkit-scrollbar-track {
                background: rgba(139, 69, 19, 0.1);
                border-radius: 3px;
            }

            .panel-content::-webkit-scrollbar-thumb {
                background: rgba(139, 69, 19, 0.3);
                border-radius: 3px;
            }

            .panel-content::-webkit-scrollbar-thumb:hover {
                background: rgba(139, 69, 19, 0.5);
            }

            /* Grid positioning classes */
            .grid-w-1 { grid-column: span 1; }
            .grid-w-2 { grid-column: span 2; }
            .grid-w-3 { grid-column: span 3; }
            .grid-w-4 { grid-column: span 4; }
            .grid-w-5 { grid-column: span 5; }
            .grid-w-6 { grid-column: span 6; }
            .grid-w-7 { grid-column: span 7; }
            .grid-w-8 { grid-column: span 8; }
            .grid-w-9 { grid-column: span 9; }
            .grid-w-10 { grid-column: span 10; }
            .grid-w-11 { grid-column: span 11; }
            .grid-w-12 { grid-column: span 12; }

            .grid-h-1 { grid-row: span 1; }
            .grid-h-2 { grid-row: span 2; }
            .grid-h-3 { grid-row: span 3; }
            .grid-h-4 { grid-row: span 4; }
            .grid-h-5 { grid-row: span 5; }
            .grid-h-6 { grid-row: span 6; }
            .grid-h-7 { grid-row: span 7; }
            .grid-h-8 { grid-row: span 8; }

            /* Resize handles */
            .resize-handle {
                position: absolute;
                background: var(--cocoa-accent);
                opacity: 0;
                transition: opacity 0.2s ease;
            }

            .grid-panel:hover .resize-handle {
                opacity: 0.6;
            }

            .resize-handle:hover {
                opacity: 1 !important;
            }

            .resize-se {
                bottom: 0;
                right: 0;
                width: 15px;
                height: 15px;
                cursor: se-resize;
                border-radius: 12px 0 12px 0;
            }

            .resize-s {
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 30px;
                height: 6px;
                cursor: s-resize;
                border-radius: 3px 3px 0 0;
            }

            .resize-e {
                right: 0;
                top: 50%;
                transform: translateY(-50%);
                width: 6px;
                height: 30px;
                cursor: e-resize;
                border-radius: 3px 0 0 3px;
            }

            /* Control Panel Styles */
            .customization-control-panel {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                background: rgba(26, 26, 26, 0.95);
                border: 2px solid rgba(139, 69, 19, 0.3);
                border-radius: 16px;
                backdrop-filter: blur(15px);
                box-shadow: 0 12px 40px rgba(139, 69, 19, 0.4);
                min-width: 280px;
                transition: all 0.3s ease;
            }

            .customization-control-panel.collapsed {
                width: 60px;
                overflow: hidden;
            }

            .control-panel-header {
                background: linear-gradient(135deg, var(--cocoa-primary), var(--cocoa-secondary));
                color: white;
                padding: 15px 20px;
                border-radius: 16px 16px 0 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: pointer;
            }

            .control-panel-title {
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .control-panel-toggle {
                background: none;
                border: none;
                color: white;
                font-size: 18px;
                cursor: pointer;
                padding: 5px;
                border-radius: 4px;
                transition: all 0.2s ease;
            }

            .control-panel-toggle:hover {
                background: rgba(255, 255, 255, 0.1);
            }

            .control-panel-content {
                padding: 20px;
                max-height: 80vh;
                overflow-y: auto;
            }

            .layout-selector {
                margin-bottom: 20px;
            }

            .layout-option {
                background: rgba(139, 69, 19, 0.1);
                border: 1px solid rgba(139, 69, 19, 0.2);
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .layout-option:hover {
                border-color: var(--cocoa-accent);
                background: rgba(139, 69, 19, 0.15);
            }

            .layout-option.active {
                border-color: var(--cocoa-accent);
                background: rgba(255, 215, 0, 0.1);
                box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
            }

            .layout-name {
                font-weight: 600;
                color: var(--cocoa-text);
                margin-bottom: 4px;
            }

            .layout-description {
                font-size: 0.85rem;
                color: var(--cocoa-text);
                opacity: 0.7;
            }

            .panel-library {
                margin-top: 20px;
            }

            .available-panel {
                background: rgba(139, 69, 19, 0.05);
                border: 1px solid rgba(139, 69, 19, 0.1);
                border-radius: 6px;
                padding: 10px 12px;
                margin: 6px 0;
                cursor: grab;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .available-panel:hover {
                border-color: var(--cocoa-secondary);
                background: rgba(139, 69, 19, 0.1);
            }

            .available-panel:active {
                cursor: grabbing;
            }

            .control-section {
                margin: 20px 0;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(139, 69, 19, 0.2);
            }

            .control-section:last-child {
                border-bottom: none;
            }

            .section-title {
                color: var(--cocoa-secondary);
                font-weight: 600;
                font-size: 0.9rem;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .control-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin-top: 15px;
            }

            .control-btn {
                background: transparent;
                border: 1px solid var(--cocoa-secondary);
                color: var(--cocoa-secondary);
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }

            .control-btn:hover {
                background: var(--cocoa-secondary);
                color: var(--cocoa-bg);
            }

            .control-btn.primary {
                background: linear-gradient(135deg, var(--cocoa-primary), var(--cocoa-secondary));
                border-color: transparent;
                color: white;
            }

            .control-btn.primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .dashboard-grid-system {
                    grid-template-columns: 1fr;
                    padding: 10px;
                }
                
                .customization-control-panel {
                    position: relative;
                    right: auto;
                    top: auto;
                    margin: 20px;
                    width: calc(100% - 40px);
                }
                
                .grid-panel {
                    grid-column: 1 !important;
                }
            }
        `;
        
        document.head.appendChild(gridStyles);
        
        // Replace existing content with grid system
        const targetContainer = document.querySelector('.container, .main-content, body');
        if (targetContainer) {
            // Store existing content
            const existingPanels = targetContainer.querySelectorAll('.cocoa-panel');
            
            // Clear container and add grid
            targetContainer.innerHTML = '';
            targetContainer.appendChild(gridContainer);
            
            this.gridSystem = gridContainer;
        }

        console.log('✅ Grid system initialized');
    }

    createControlPanel() {
        const controlPanel = document.createElement('div');
        controlPanel.id = 'customization-control-panel';
        controlPanel.className = 'customization-control-panel';
        
        controlPanel.innerHTML = `
            <div class="control-panel-header" onclick="window.customizablePanelsSystem.toggleControlPanel()">
                <div class="control-panel-title">
                    <span>🎛️</span>
                    <span>Dashboard Customization</span>
                </div>
                <button class="control-panel-toggle">−</button>
            </div>
            <div class="control-panel-content">
                <div class="control-section">
                    <div class="section-title">Layout Presets</div>
                    <div class="layout-selector">
                        ${Object.entries(this.layouts).map(([key, layout]) => `
                            <div class="layout-option ${key === this.currentLayout ? 'active' : ''}" 
                                 onclick="window.customizablePanelsSystem.loadLayout('${key}')">
                                <div class="layout-name">${layout.name}</div>
                                <div class="layout-description">${layout.description}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="control-section">
                    <div class="section-title">Panel Library</div>
                    <div class="panel-library">
                        ${this.getAvailablePanels().map(panel => `
                            <div class="available-panel" draggable="true" data-panel-type="${panel.id}">
                                <span>${panel.icon}</span>
                                <span>${panel.name}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="control-section">
                    <div class="section-title">Layout Actions</div>
                    <div class="control-buttons">
                        <button class="control-btn" onclick="window.customizablePanelsSystem.saveCurrentLayout()">
                            💾 Save Layout
                        </button>
                        <button class="control-btn" onclick="window.customizablePanelsSystem.resetLayout()">
                            🔄 Reset
                        </button>
                        <button class="control-btn primary" onclick="window.customizablePanelsSystem.exportLayout()">
                            📤 Export
                        </button>
                        <button class="control-btn" onclick="window.customizablePanelsSystem.importLayout()">
                            📥 Import
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(controlPanel);
        
        console.log('✅ Control panel created');
    }

    getAvailablePanels() {
        return [
            { id: 'market-overview', name: 'Market Overview', icon: '🌍' },
            { id: 'trading-pairs', name: 'Trading Pairs', icon: '💱' },
            { id: 'arbitrage-opportunities', name: 'Arbitrage Opportunities', icon: '⚡' },
            { id: 'portfolio-metrics', name: 'Portfolio Metrics', icon: '📈' },
            { id: 'news-feed', name: 'News Feed', icon: '📰' },
            { id: 'account-tier', name: 'Account Tier', icon: '🏦' },
            { id: 'regulatory-credentials', name: 'Regulatory Credentials', icon: '🛡️' },
            { id: 'ai-analysis', name: 'AI Analysis', icon: '🧠' },
            { id: 'risk-management', name: 'Risk Management', icon: '⚠️' },
            { id: 'trading-signals', name: 'Trading Signals', icon: '🎯' }
        ];
    }

    registerDefaultPanels() {
        // Register all available panel types
        this.panels.set('market-overview', {
            title: '🌍 Market Overview',
            content: this.createMarketOverviewContent(),
            defaultSize: { w: 12, h: 4 }
        });

        this.panels.set('trading-pairs', {
            title: '💱 Trading Pairs',
            content: this.createTradingPairsContent(),
            defaultSize: { w: 6, h: 6 }
        });

        this.panels.set('arbitrage-opportunities', {
            title: '⚡ Arbitrage Opportunities',
            content: this.createArbitrageContent(),
            defaultSize: { w: 6, h: 6 }
        });

        this.panels.set('portfolio-metrics', {
            title: '📈 Portfolio Metrics',
            content: this.createPortfolioMetricsContent(),
            defaultSize: { w: 8, h: 6 }
        });

        this.panels.set('news-feed', {
            title: '📰 Market News',
            content: this.createNewsFeedContent(),
            defaultSize: { w: 4, h: 6 }
        });

        this.panels.set('account-tier', {
            title: '🏦 Account Tier',
            content: this.createAccountTierContent(),
            defaultSize: { w: 6, h: 4 }
        });

        this.panels.set('regulatory-credentials', {
            title: '🛡️ Regulatory Credentials',
            content: this.createRegulatoryContent(),
            defaultSize: { w: 6, h: 4 }
        });

        console.log('✅ Default panels registered');
    }

    createMarketOverviewContent() {
        return `
            <div class="market-overview-mini">
                <div class="mini-stats-grid">
                    <div class="mini-stat">
                        <div class="mini-value">$67,250</div>
                        <div class="mini-label">BTC/USDT</div>
                        <div class="mini-change positive">+2.34%</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-value">2.45T</div>
                        <div class="mini-label">Market Cap</div>
                        <div class="mini-change positive">+2.1%</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-value">54.2%</div>
                        <div class="mini-label">BTC Dominance</div>
                        <div class="mini-change positive">+0.3%</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-value">72</div>
                        <div class="mini-label">Fear & Greed</div>
                        <div class="mini-change positive">+5</div>
                    </div>
                </div>
            </div>
        `;
    }

    createTradingPairsContent() {
        return `
            <div class="trading-pairs-mini">
                <div class="pair-item">
                    <span class="pair-symbol">BTC/USDT</span>
                    <span class="pair-price">$67,250</span>
                    <span class="pair-change positive">+2.34%</span>
                </div>
                <div class="pair-item">
                    <span class="pair-symbol">ETH/USDT</span>
                    <span class="pair-price">$2,640</span>
                    <span class="pair-change positive">+1.87%</span>
                </div>
                <div class="pair-item">
                    <span class="pair-symbol">BNB/USDT</span>
                    <span class="pair-price">$585</span>
                    <span class="pair-change negative">-0.45%</span>
                </div>
                <div class="pair-item">
                    <span class="pair-symbol">ADA/USDT</span>
                    <span class="pair-price">$0.385</span>
                    <span class="pair-change positive">+3.21%</span>
                </div>
            </div>
        `;
    }

    createArbitrageContent() {
        return `
            <div class="arbitrage-mini">
                <div class="arb-item">
                    <div class="arb-pair">BTC/USDT</div>
                    <div class="arb-profit">+$325.50</div>
                    <div class="arb-spread">0.15% spread</div>
                </div>
                <div class="arb-item">
                    <div class="arb-pair">ETH/USDT</div>
                    <div class="arb-profit">+$145.20</div>
                    <div class="arb-spread">0.08% spread</div>
                </div>
                <div class="arb-item">
                    <div class="arb-pair">BNB/USDT</div>
                    <div class="arb-profit">+$89.75</div>
                    <div class="arb-spread">0.22% spread</div>
                </div>
            </div>
        `;
    }

    createPortfolioMetricsContent() {
        return `
            <div class="portfolio-metrics-mini">
                <div class="metrics-grid-mini">
                    <div class="metric-mini">
                        <div class="metric-value">$1,250,000</div>
                        <div class="metric-label">Portfolio Value</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-value">+25.0%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-value">2.45</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-mini">
                        <div class="metric-value">78%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                </div>
            </div>
        `;
    }

    createNewsFeedContent() {
        return `
            <div class="news-feed-mini">
                <div class="news-item-mini">
                    <div class="news-time">2 min ago</div>
                    <div class="news-headline">Bitcoin ETF sees $200M inflows</div>
                </div>
                <div class="news-item-mini">
                    <div class="news-time">5 min ago</div>
                    <div class="news-headline">Ethereum upgrade reduces fees</div>
                </div>
                <div class="news-item-mini">
                    <div class="news-time">8 min ago</div>
                    <div class="news-headline">New DeFi trading pairs announced</div>
                </div>
            </div>
        `;
    }

    createAccountTierContent() {
        return `
            <div class="account-tier-mini">
                <div class="current-tier">
                    <div class="tier-badge">⚡ Professional Account</div>
                    <div class="tier-limits">
                        <div>Monthly trades: 45/1000</div>
                        <div>Leverage: 1:5 available</div>
                    </div>
                </div>
                <button class="upgrade-btn-mini">Upgrade to Institutional</button>
            </div>
        `;
    }

    createRegulatoryContent() {
        return `
            <div class="regulatory-mini">
                <div class="credential-mini">
                    <span class="cred-icon">🏛️</span>
                    <span class="cred-name">SEC Registered</span>
                    <span class="cred-status">✓</span>
                </div>
                <div class="credential-mini">
                    <span class="cred-icon">🏦</span>
                    <span class="cred-name">FINRA Member</span>
                    <span class="cred-status">✓</span>
                </div>
                <div class="credential-mini">
                    <span class="cred-icon">🛡️</span>
                    <span class="cred-name">SIPC Protected</span>
                    <span class="cred-status">✓</span>
                </div>
            </div>
        `;
    }

    setupDragAndDrop() {
        // Set up drag and drop functionality
        this.gridSystem.addEventListener('dragover', this.handleDragOver.bind(this));
        this.gridSystem.addEventListener('drop', this.handleDrop.bind(this));
        
        // Set up panel dragging from library
        document.addEventListener('dragstart', this.handleDragStart.bind(this));
        document.addEventListener('dragend', this.handleDragEnd.bind(this));

        console.log('✅ Drag and drop setup complete');
    }

    handleDragStart(e) {
        if (e.target.classList.contains('available-panel')) {
            this.draggedPanel = {
                type: e.target.dataset.panelType,
                source: 'library'
            };
            e.target.style.opacity = '0.5';
        } else if (e.target.closest('.grid-panel')) {
            const panel = e.target.closest('.grid-panel');
            this.draggedPanel = {
                element: panel,
                source: 'grid'
            };
            panel.classList.add('dragging');
        }
    }

    handleDragEnd(e) {
        if (e.target.classList.contains('available-panel')) {
            e.target.style.opacity = '1';
        } else if (e.target.closest('.grid-panel')) {
            e.target.closest('.grid-panel').classList.remove('dragging');
        }
        this.draggedPanel = null;
        
        // Remove all drop target indicators
        document.querySelectorAll('.drop-target').forEach(el => {
            el.classList.remove('drop-target');
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        
        // Add drop target indicator
        const targetPanel = e.target.closest('.grid-panel');
        if (targetPanel && this.draggedPanel) {
            targetPanel.classList.add('drop-target');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        
        if (!this.draggedPanel) return;
        
        if (this.draggedPanel.source === 'library') {
            // Add new panel from library
            this.addPanelToGrid(this.draggedPanel.type, {
                x: 0,
                y: this.getNextAvailableRow(),
                w: this.panels.get(this.draggedPanel.type)?.defaultSize.w || 6,
                h: this.panels.get(this.draggedPanel.type)?.defaultSize.h || 4
            });
        } else if (this.draggedPanel.source === 'grid') {
            // Reposition existing panel
            const targetPanel = e.target.closest('.grid-panel');
            if (targetPanel && targetPanel !== this.draggedPanel.element) {
                // Swap positions
                this.swapPanelPositions(this.draggedPanel.element, targetPanel);
            }
        }
        
        this.saveCurrentLayout();
    }

    addPanelToGrid(panelType, position) {
        const panelConfig = this.panels.get(panelType);
        if (!panelConfig) return;

        const panelElement = document.createElement('div');
        panelElement.className = `grid-panel grid-w-${position.w} grid-h-${position.h}`;
        panelElement.draggable = true;
        panelElement.dataset.panelType = panelType;
        
        panelElement.innerHTML = `
            <div class="panel-header">
                <div class="panel-title">${panelConfig.title}</div>
                <div class="panel-controls">
                    <button class="panel-control-btn" onclick="window.customizablePanelsSystem.resizePanel(this, 'smaller')" title="Make Smaller">−</button>
                    <button class="panel-control-btn" onclick="window.customizablePanelsSystem.resizePanel(this, 'larger')" title="Make Larger">+</button>
                    <button class="panel-control-btn" onclick="window.customizablePanelsSystem.removePanel(this)" title="Remove Panel">×</button>
                </div>
            </div>
            <div class="panel-content">
                ${panelConfig.content}
            </div>
            <div class="resize-handle resize-se"></div>
            <div class="resize-handle resize-s"></div>
            <div class="resize-handle resize-e"></div>
        `;
        
        this.gridSystem.appendChild(panelElement);
        
        // Add entrance animation
        setTimeout(() => {
            panelElement.style.animation = 'cocoaFadeIn 0.5s ease-out';
        }, 50);

        console.log(`✅ Added panel: ${panelType}`);
    }

    loadLayout(layoutName) {
        const layout = this.layouts[layoutName];
        if (!layout) return;

        // Clear current grid
        this.gridSystem.innerHTML = '';
        
        // Add panels from layout
        layout.panels.forEach(panelConfig => {
            this.addPanelToGrid(panelConfig.id, panelConfig);
        });
        
        // Update current layout
        this.currentLayout = layoutName;
        
        // Update UI
        document.querySelectorAll('.layout-option').forEach(option => {
            option.classList.remove('active');
        });
        document.querySelector(`[onclick*="${layoutName}"]`).classList.add('active');

        console.log(`✅ Loaded layout: ${layoutName}`);
    }

    saveCurrentLayout() {
        const currentPanels = Array.from(this.gridSystem.children).map(panel => {
            const rect = panel.getBoundingClientRect();
            const gridRect = this.gridSystem.getBoundingClientRect();
            
            return {
                id: panel.dataset.panelType,
                x: Math.round((rect.left - gridRect.left) / (gridRect.width / 12)),
                y: Math.round((rect.top - gridRect.top) / 100),
                w: parseInt(panel.className.match(/grid-w-(\d+)/)?.[1] || '6'),
                h: parseInt(panel.className.match(/grid-h-(\d+)/)?.[1] || '4')
            };
        });

        // Save to current layout
        this.layouts[this.currentLayout].panels = currentPanels;
        
        // Save to localStorage
        localStorage.setItem('cocoaTradingAI_layouts', JSON.stringify(this.layouts));

        console.log('💾 Layout saved');
    }

    resetLayout() {
        if (confirm('Are you sure you want to reset the current layout to defaults?')) {
            // Reset to default configuration
            this.layouts[this.currentLayout] = this.getDefaultLayout();
            this.loadLayout(this.currentLayout);
            this.saveCurrentLayout();
            
            console.log('🔄 Layout reset to default');
        }
    }

    exportLayout() {
        const layoutData = {
            name: `Custom Layout ${new Date().toISOString().split('T')[0]}`,
            description: 'Exported custom layout',
            panels: this.layouts[this.currentLayout].panels,
            exportDate: new Date().toISOString()
        };
        
        const dataStr = JSON.stringify(layoutData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `cocoa_trading_layout_${new Date().toISOString().split('T')[0]}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        console.log('📤 Layout exported');
    }

    importLayout() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const layoutData = JSON.parse(e.target.result);
                        const layoutId = `imported_${Date.now()}`;
                        
                        this.layouts[layoutId] = {
                            name: layoutData.name || 'Imported Layout',
                            description: layoutData.description || 'Imported custom layout',
                            panels: layoutData.panels
                        };
                        
                        this.saveCurrentLayout();
                        this.loadLayout(layoutId);
                        
                        // Update control panel
                        this.updateControlPanel();
                        
                        console.log('📥 Layout imported successfully');
                    } catch (error) {
                        alert('Error importing layout: Invalid file format');
                        console.error('Import error:', error);
                    }
                };
                reader.readAsText(file);
            }
        };
        input.click();
    }

    toggleControlPanel() {
        const panel = document.getElementById('customization-control-panel');
        const toggle = panel.querySelector('.control-panel-toggle');
        
        panel.classList.toggle('collapsed');
        toggle.textContent = panel.classList.contains('collapsed') ? '+' : '−';
    }

    resizePanel(button, direction) {
        const panel = button.closest('.grid-panel');
        const currentW = parseInt(panel.className.match(/grid-w-(\d+)/)?.[1] || '6');
        const currentH = parseInt(panel.className.match(/grid-h-(\d+)/)?.[1] || '4');
        
        let newW = currentW;
        let newH = currentH;
        
        if (direction === 'larger') {
            newW = Math.min(12, currentW + 2);
            newH = Math.min(8, currentH + 1);
        } else {
            newW = Math.max(2, currentW - 2);
            newH = Math.max(2, currentH - 1);
        }
        
        // Update classes
        panel.className = panel.className.replace(/grid-w-\d+/, `grid-w-${newW}`);
        panel.className = panel.className.replace(/grid-h-\d+/, `grid-h-${newH}`);
        
        this.saveCurrentLayout();
    }

    removePanel(button) {
        if (confirm('Remove this panel from the dashboard?')) {
            const panel = button.closest('.grid-panel');
            panel.style.animation = 'cocoaFadeOut 0.3s ease-in';
            setTimeout(() => {
                panel.remove();
                this.saveCurrentLayout();
            }, 300);
        }
    }

    swapPanelPositions(panel1, panel2) {
        // Get positions
        const temp = panel1.nextSibling;
        panel1.parentNode.insertBefore(panel1, panel2.nextSibling);
        panel2.parentNode.insertBefore(panel2, temp);
        
        this.saveCurrentLayout();
    }

    getNextAvailableRow() {
        const panels = Array.from(this.gridSystem.children);
        if (panels.length === 0) return 0;
        
        // Find the maximum bottom position
        let maxBottom = 0;
        panels.forEach(panel => {
            const rect = panel.getBoundingClientRect();
            const gridRect = this.gridSystem.getBoundingClientRect();
            const bottom = Math.round((rect.bottom - gridRect.top) / 100);
            maxBottom = Math.max(maxBottom, bottom);
        });
        
        return maxBottom;
    }

    updateControlPanel() {
        // Update the layout selector in control panel
        const layoutSelector = document.querySelector('.layout-selector');
        if (layoutSelector) {
            layoutSelector.innerHTML = Object.entries(this.layouts).map(([key, layout]) => `
                <div class="layout-option ${key === this.currentLayout ? 'active' : ''}" 
                     onclick="window.customizablePanelsSystem.loadLayout('${key}')">
                    <div class="layout-name">${layout.name}</div>
                    <div class="layout-description">${layout.description}</div>
                </div>
            `).join('');
        }
    }

    // Public API methods
    getCurrentLayout() {
        return this.currentLayout;
    }

    getLayoutConfig(layoutName = null) {
        return this.layouts[layoutName || this.currentLayout];
    }

    addCustomPanel(panelType, config) {
        this.panels.set(panelType, config);
        console.log(`✅ Custom panel registered: ${panelType}`);
    }

    // Cleanup method
    destroy() {
        // Remove event listeners and cleanup
        console.log('🧹 Customizable Panels System cleaned up');
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.customizablePanelsSystem = new CustomizablePanelsSystem();
    });
} else {
    window.customizablePanelsSystem = new CustomizablePanelsSystem();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CustomizablePanelsSystem;
}

console.log('🎛️ Customizable Panels System loaded and ready for dashboard customization');