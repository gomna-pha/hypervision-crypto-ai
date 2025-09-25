/**
 * DRAGGABLE TRADING DASHBOARD FOR GOMNA AI
 * Interactive panels with drag, resize, minimize functionality
 * Based on Hyperbolic CNN Quantitative Trading Platform design
 */

class DraggablePanel {
    constructor(id, title, content, options = {}) {
        this.id = id;
        this.title = title;
        this.content = content;
        this.options = {
            x: options.x || Math.random() * 300 + 100,
            y: options.y || Math.random() * 200 + 100,
            width: options.width || 350,
            height: options.height || 400,
            minWidth: options.minWidth || 250,
            minHeight: options.minHeight || 150,
            resizable: options.resizable !== false,
            draggable: options.draggable !== false,
            minimizable: options.minimizable !== false,
            closeable: options.closeable !== false,
            zIndex: options.zIndex || 1000
        };
        
        this.isMinimized = false;
        this.isDragging = false;
        this.isResizing = false;
        this.dragOffset = { x: 0, y: 0 };
        
        this.createElement();
        this.attachEventListeners();
    }

    createElement() {
        this.element = document.createElement('div');
        this.element.className = 'draggable-panel';
        this.element.id = `panel-${this.id}`;
        this.element.style.cssText = `
            position: absolute;
            left: ${this.options.x}px;
            top: ${this.options.y}px;
            width: ${this.options.width}px;
            height: ${this.options.height}px;
            z-index: ${this.options.zIndex};
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(139, 115, 85, 0.3);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            transition: box-shadow 0.3s ease;
            overflow: hidden;
        `;

        this.element.innerHTML = `
            <div class="panel-header" style="
                background: linear-gradient(135deg, #8b7355 0%, #6d5d48 100%);
                color: white;
                padding: 12px 16px;
                cursor: move;
                display: flex;
                justify-content: space-between;
                align-items: center;
                user-select: none;
                border-radius: 11px 11px 0 0;
            ">
                <div class="panel-title" style="
                    font-weight: 600;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                ">
                    <div class="panel-icon" style="
                        width: 6px;
                        height: 6px;
                        background: #fdf6e3;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                    "></div>
                    ${this.title}
                </div>
                <div class="panel-controls" style="
                    display: flex;
                    gap: 8px;
                ">
                    ${this.options.minimizable ? `
                        <button class="panel-minimize" style="
                            background: rgba(255, 255, 255, 0.2);
                            border: none;
                            color: white;
                            width: 24px;
                            height: 24px;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            transition: background 0.2s ease;
                        " title="Minimize">−</button>
                    ` : ''}
                    ${this.options.closeable ? `
                        <button class="panel-close" style="
                            background: rgba(255, 255, 255, 0.2);
                            border: none;
                            color: white;
                            width: 24px;
                            height: 24px;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            transition: background 0.2s ease;
                        " title="Close">×</button>
                    ` : ''}
                </div>
            </div>
            <div class="panel-content" style="
                padding: 16px;
                height: calc(100% - 48px);
                overflow-y: auto;
                background: #fefbf3;
                font-size: 13px;
                line-height: 1.4;
            ">
                ${this.content}
            </div>
            ${this.options.resizable ? `
                <div class="panel-resize-handle" style="
                    position: absolute;
                    bottom: 0;
                    right: 0;
                    width: 16px;
                    height: 16px;
                    cursor: nw-resize;
                    background: linear-gradient(-45deg, transparent 40%, #8b7355 40%, #8b7355 60%, transparent 60%);
                "></div>
            ` : ''}
        `;

        document.body.appendChild(this.element);
    }

    attachEventListeners() {
        const header = this.element.querySelector('.panel-header');
        const minimizeBtn = this.element.querySelector('.panel-minimize');
        const closeBtn = this.element.querySelector('.panel-close');
        const resizeHandle = this.element.querySelector('.panel-resize-handle');

        // Dragging functionality
        if (this.options.draggable) {
            header.addEventListener('mousedown', (e) => this.startDrag(e));
        }

        // Minimize functionality
        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleMinimize();
            });
        }

        // Close functionality
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.close();
            });
        }

        // Resize functionality
        if (resizeHandle) {
            resizeHandle.addEventListener('mousedown', (e) => this.startResize(e));
        }

        // Bring to front on click
        this.element.addEventListener('mousedown', () => this.bringToFront());

        // Global mouse events
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mouseup', () => this.stopDrag());
    }

    startDrag(e) {
        if (this.isMinimized) return;
        
        this.isDragging = true;
        this.dragOffset.x = e.clientX - this.element.offsetLeft;
        this.dragOffset.y = e.clientY - this.element.offsetTop;
        this.element.style.cursor = 'grabbing';
        
        e.preventDefault();
    }

    startResize(e) {
        this.isResizing = true;
        this.dragOffset.x = e.clientX;
        this.dragOffset.y = e.clientY;
        this.dragOffset.width = this.element.offsetWidth;
        this.dragOffset.height = this.element.offsetHeight;
        
        e.preventDefault();
        e.stopPropagation();
    }

    onMouseMove(e) {
        if (this.isDragging) {
            const newX = Math.max(0, Math.min(window.innerWidth - this.element.offsetWidth, e.clientX - this.dragOffset.x));
            const newY = Math.max(0, Math.min(window.innerHeight - this.element.offsetHeight, e.clientY - this.dragOffset.y));
            
            this.element.style.left = newX + 'px';
            this.element.style.top = newY + 'px';
        } else if (this.isResizing) {
            const deltaX = e.clientX - this.dragOffset.x;
            const deltaY = e.clientY - this.dragOffset.y;
            
            const newWidth = Math.max(this.options.minWidth, this.dragOffset.width + deltaX);
            const newHeight = Math.max(this.options.minHeight, this.dragOffset.height + deltaY);
            
            this.element.style.width = newWidth + 'px';
            this.element.style.height = newHeight + 'px';
        }
    }

    stopDrag() {
        this.isDragging = false;
        this.isResizing = false;
        this.element.style.cursor = '';
    }

    toggleMinimize() {
        this.isMinimized = !this.isMinimized;
        const content = this.element.querySelector('.panel-content');
        const resizeHandle = this.element.querySelector('.panel-resize-handle');
        const minimizeBtn = this.element.querySelector('.panel-minimize');
        
        if (this.isMinimized) {
            this.originalHeight = this.element.offsetHeight;
            this.element.style.height = '48px';
            content.style.display = 'none';
            if (resizeHandle) resizeHandle.style.display = 'none';
            minimizeBtn.innerHTML = '+';
            minimizeBtn.title = 'Restore';
        } else {
            this.element.style.height = (this.originalHeight || this.options.height) + 'px';
            content.style.display = 'block';
            if (resizeHandle) resizeHandle.style.display = 'block';
            minimizeBtn.innerHTML = '−';
            minimizeBtn.title = 'Minimize';
        }
    }

    close() {
        this.element.remove();
        // Notify dashboard manager
        if (window.dashboardManager) {
            window.dashboardManager.removePanel(this.id);
        }
    }

    bringToFront() {
        if (window.dashboardManager) {
            window.dashboardManager.bringPanelToFront(this);
        }
    }

    updateContent(newContent) {
        const contentEl = this.element.querySelector('.panel-content');
        contentEl.innerHTML = newContent;
    }
}

class DashboardManager {
    constructor() {
        this.panels = new Map();
        this.nextZIndex = 1000;
        this.setupStyles();
        this.createDashboard();
    }

    setupStyles() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .panel-controls button:hover {
                background: rgba(255, 255, 255, 0.3) !important;
            }
            
            .draggable-panel:hover {
                box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15) !important;
            }
            
            .dashboard-toolbar {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                background: rgba(139, 115, 85, 0.9);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            }
            
            .toolbar-button {
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                padding: 8px 12px;
                margin: 2px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s ease;
            }
            
            .toolbar-button:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-1px);
            }
            
            .panel-menu {
                position: absolute;
                top: 100%;
                right: 0;
                background: white;
                border-radius: 8px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                min-width: 200px;
                padding: 8px;
                margin-top: 8px;
                display: none;
            }
            
            .panel-menu-item {
                padding: 8px 12px;
                cursor: pointer;
                border-radius: 4px;
                font-size: 13px;
                color: #333;
                transition: background 0.2s ease;
            }
            
            .panel-menu-item:hover {
                background: #f5e6d3;
            }
        `;
        document.head.appendChild(style);
    }

    createDashboard() {
        this.createToolbar();
        this.createDefaultPanels();
    }

    createToolbar() {
        const toolbar = document.createElement('div');
        toolbar.className = 'dashboard-toolbar';
        toolbar.innerHTML = `
            <button class="toolbar-button" onclick="dashboardManager.togglePanelMenu()">
                📊 Add Panel
            </button>
            <button class="toolbar-button" onclick="dashboardManager.arrangeWindows()">
                🔧 Arrange
            </button>
            <button class="toolbar-button" onclick="dashboardManager.minimizeAll()">
                📥 Minimize All
            </button>
            <div class="panel-menu" id="panel-menu">
                <div class="panel-menu-item" onclick="dashboardManager.createMarketDataPanel()">
                    📈 Live Market Data
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createActivePositionsPanel()">
                    💼 Active Positions
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createArbitragePanel()">
                    ⚡ Arbitrage Opportunities
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createPerformancePanel()">
                    📊 Performance Metrics
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createTradingPanel()">
                    🎯 Trading Controls
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createNewsPanel()">
                    📰 Market News
                </div>
                <div class="panel-menu-item" onclick="dashboardManager.createAlgorithmsPanel()">
                    🧠 AI Algorithms
                </div>
            </div>
        `;
        document.body.appendChild(toolbar);
    }

    createDefaultPanels() {
        // Create initial panels matching the PDF design
        this.createMarketDataPanel();
        this.createActivePositionsPanel();
        this.createArbitragePanel();
        this.createPerformancePanel();
    }

    createMarketDataPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    📡 Live Market Data
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 12px; border-radius: 8px; color: white;">
                        <div style="font-size: 11px; opacity: 0.8;">BTC/USD</div>
                        <div style="font-size: 18px; font-weight: 700;">$43,234</div>
                        <div style="font-size: 11px; color: #fef3c7;">+2.34% ↗</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 12px; border-radius: 8px; color: white;">
                        <div style="font-size: 11px; opacity: 0.8;">ETH/USD</div>
                        <div style="font-size: 18px; font-weight: 700;">$2,845</div>
                        <div style="font-size: 11px; color: #ede9fe;">+1.87% ↗</div>
                    </div>
                </div>
                <div style="background: #f9f9f9; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Cross-Exchange Spreads</div>
                    <div style="font-size: 11px; margin-bottom: 4px;">BTC: Binance vs Coinbase <span style="color: #059669; font-weight: 600;">0.23%</span></div>
                    <div style="font-size: 11px;">ETH: Kraken vs KuCoin <span style="color: #059669; font-weight: 600;">0.18%</span></div>
                </div>
                <div style="margin-top: 12px; text-align: center;">
                    <div style="font-size: 11px; color: #666;">Last Update: ${new Date().toLocaleTimeString()}</div>
                    <div style="width: 100%; height: 2px; background: linear-gradient(90deg, #8b7355 0%, #f59e0b 100%); border-radius: 1px; margin-top: 4px;"></div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('market-data', '📈 Live Market Data', content, {
            x: 50,
            y: 100,
            width: 320,
            height: 280
        });

        this.addPanel(panel);

        // Auto-update market data
        setInterval(() => {
            if (this.panels.has('market-data')) {
                const btcPrice = 43000 + Math.random() * 1000;
                const ethPrice = 2800 + Math.random() * 100;
                const btcChange = (Math.random() - 0.5) * 5;
                const ethChange = (Math.random() - 0.5) * 5;
                
                const updatedContent = content
                    .replace('$43,234', `$${Math.floor(btcPrice).toLocaleString()}`)
                    .replace('$2,845', `$${Math.floor(ethPrice).toLocaleString()}`)
                    .replace('+2.34%', `${btcChange >= 0 ? '+' : ''}${btcChange.toFixed(2)}%`)
                    .replace('+1.87%', `${ethChange >= 0 ? '+' : ''}${ethChange.toFixed(2)}%`)
                    .replace(new Date().toLocaleTimeString(), new Date().toLocaleTimeString());
                
                panel.updateContent(updatedContent);
            }
        }, 3000);
    }

    createActivePositionsPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    💼 Active Positions
                </h4>
                <div style="space-y: 12px;">
                    <div style="background: #f0f9ff; padding: 12px; border-radius: 8px; border-left: 4px solid #0ea5e9; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 600; font-size: 13px;">BTC 2.5 units</span>
                            <span style="color: #059669; font-weight: 600; font-size: 12px;">+12.4%</span>
                        </div>
                        <div style="font-size: 18px; font-weight: 700; color: #1e293b;">$168,581</div>
                        <div style="font-size: 11px; color: #64748b;">Entry: $62,450 • Current: $67,234</div>
                    </div>
                    
                    <div style="background: #f7fee7; padding: 12px; border-radius: 8px; border-left: 4px solid #84cc16; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 600; font-size: 13px;">ETH 45.2 units</span>
                            <span style="color: #059669; font-weight: 600; font-size: 12px;">+8.7%</span>
                        </div>
                        <div style="font-size: 18px; font-weight: 700; color: #1e293b;">$159,252</div>
                        <div style="font-size: 11px; color: #64748b;">Entry: $2,620 • Current: $2,845</div>
                    </div>
                    
                    <div style="background: #fffbeb; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 600; font-size: 13px;">SOL 1,250 units</span>
                            <span style="color: #059669; font-weight: 600; font-size: 12px;">+15.3%</span>
                        </div>
                        <div style="font-size: 18px; font-weight: 700; color: #1e293b;">$178,125</div>
                        <div style="font-size: 11px; color: #64748b;">Entry: $120 • Current: $142.50</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px;">
                        <span>Total P&L:</span>
                        <span style="color: #059669; font-weight: 600;">+$48,732 (12.1%)</span>
                    </div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('active-positions', '💼 Active Positions', content, {
            x: 400,
            y: 100,
            width: 340,
            height: 380
        });

        this.addPanel(panel);
    }

    createArbitragePanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    ⚡ Live Arbitrage Opportunities
                </h4>
                <div style="margin-bottom: 12px;">
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 12px; border-radius: 8px; color: white; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 12px; opacity: 0.9;">Cross-Exchange • BTC/USD</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">ACTIVE</span>
                        </div>
                        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">0.52% Profit</div>
                        <div style="font-size: 11px; opacity: 0.8;">Binance → Coinbase • $2,347 potential</div>
                        <button style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; margin-top: 8px; cursor: pointer;">Execute Trade</button>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 12px; border-radius: 8px; color: white; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 12px; opacity: 0.9;">Triangular • ETH-BTC-USD</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">HIGH CONF</span>
                        </div>
                        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">0.34% Profit</div>
                        <div style="font-size: 11px; opacity: 0.8;">Multi-hop • $1,832 potential</div>
                        <button style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; margin-top: 8px; cursor: pointer;">Execute Trade</button>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 12px; border-radius: 8px; color: white;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 12px; opacity: 0.9;">Statistical • Mean Reversion</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">ML SIGNAL</span>
                        </div>
                        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">0.28% Profit</div>
                        <div style="font-size: 11px; opacity: 0.8;">AI Predicted • $1,456 potential</div>
                        <button style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; margin-top: 8px; cursor: pointer;">Execute Trade</button>
                    </div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('arbitrage-opportunities', '⚡ Arbitrage Opportunities', content, {
            x: 750,
            y: 100,
            width: 360,
            height: 420
        });

        this.addPanel(panel);
    }

    createPerformancePanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    📊 Performance Metrics
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="text-center; padding: 12px; background: #f8fafc; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: 700; color: #059669;">+15.45%</div>
                        <div style="font-size: 11px; color: #64748b;">Alpha</div>
                    </div>
                    <div style="text-center; padding: 12px; background: #f8fafc; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: 700; color: #1e293b;">2.34</div>
                        <div style="font-size: 11px; color: #64748b;">Sharpe Ratio</div>
                    </div>
                    <div style="text-center; padding: 12px; background: #f8fafc; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: 700; color: #059669;">73.8%</div>
                        <div style="font-size: 11px; color: #64748b;">Win Rate</div>
                    </div>
                    <div style="text-center; padding: 12px; background: #f8fafc; border-radius: 8px;">
                        <div style="font-size: 20px; font-weight: 700; color: #1e293b;">91.2%</div>
                        <div style="font-size: 11px; color: #64748b;">AI Accuracy</div>
                    </div>
                </div>
                
                <div style="background: #f9f9f9; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Hyperbolic CNN Status</div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                        <span style="font-size: 11px; color: #059669; font-weight: 600;">Optimal Performance</span>
                    </div>
                    <div style="font-size: 10px; color: #666; margin-top: 4px;">Neural network operating at 94.2% efficiency</div>
                </div>
                
                <div style="margin-top: 12px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Recent Trades</div>
                    <div style="font-size: 11px; margin-bottom: 4px;">✓ BTC Cross-Exchange: +$2,347 (42s ago)</div>
                    <div style="font-size: 11px; margin-bottom: 4px;">✓ ETH Triangular: +$1,832 (1m ago)</div>
                    <div style="font-size: 11px;">✓ SOL Statistical: +$1,456 (3m ago)</div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('performance-metrics', '📊 Performance Metrics', content, {
            x: 50,
            y: 400,
            width: 320,
            height: 350
        });

        this.addPanel(panel);
    }

    createPositionsPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    💼 Active Positions
                </h4>
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 8px;">
                        <div>
                            <div style="font-weight: 600; color: #16a34a;">BTC/USD</div>
                            <div style="font-size: 11px; color: #666;">Long • Size: 0.25 BTC</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 600; color: #16a34a;">+$2,347</div>
                            <div style="font-size: 11px; color: #666;">+3.45%</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 8px;">
                        <div>
                            <div style="font-weight: 600; color: #16a34a;">ETH/USD</div>
                            <div style="font-size: 11px; color: #666;">Long • Size: 2.1 ETH</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 600; color: #16a34a;">+$1,832</div>
                            <div style="font-size: 11px; color: #666;">+2.87%</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #fef2f2; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 8px;">
                        <div>
                            <div style="font-weight: 600; color: #dc2626;">SOL/USD</div>
                            <div style="font-size: 11px; color: #666;">Short • Size: 15 SOL</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 600; color: #dc2626;">-$432</div>
                            <div style="font-size: 11px; color: #666;">-1.23%</div>
                        </div>
                    </div>
                </div>
                
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Portfolio Summary</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 11px;">Total Value:</span>
                        <span style="font-size: 11px; font-weight: 600;">$847,563</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 11px;">Unrealized P&L:</span>
                        <span style="font-size: 11px; font-weight: 600; color: #16a34a;">+$3,747</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 11px;">Day Change:</span>
                        <span style="font-size: 11px; font-weight: 600; color: #16a34a;">+2.14%</span>
                    </div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('active-positions', '💼 Active Positions', content, {
            x: 500,
            y: 50,
            width: 320,
            height: 400
        });

        this.addPanel(panel);
    }

    createTradingPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    🎯 Trading Controls
                </h4>
                <div style="margin-bottom: 16px;">
                    <button style="width: 100%; background: linear-gradient(135deg, #059669 0%, #047857 100%); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; margin-bottom: 8px; cursor: pointer;">
                        🚀 Enable Auto Trading
                    </button>
                    <button style="width: 100%; background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; margin-bottom: 16px; cursor: pointer;">
                        🛑 Emergency Stop
                    </button>
                </div>
                
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Risk Parameters</div>
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 11px; color: #666;">Max Position Size: 10%</label>
                        <div style="background: #e5e7eb; height: 4px; border-radius: 2px; margin-top: 4px;">
                            <div style="background: #059669; width: 60%; height: 100%; border-radius: 2px;"></div>
                        </div>
                    </div>
                    <div>
                        <label style="font-size: 11px; color: #666;">Stop Loss: 3%</label>
                        <div style="background: #e5e7eb; height: 4px; border-radius: 2px; margin-top: 4px;">
                            <div style="background: #f59e0b; width: 30%; height: 100%; border-radius: 2px;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="background: #f0f9ff; padding: 12px; border-radius: 8px; border: 1px solid #bae6fd;">
                    <div style="font-size: 12px; color: #0369a1; font-weight: 600; margin-bottom: 4px;">AI Agent Status</div>
                    <div style="font-size: 11px; color: #0369a1;">Scanning 47 opportunities across 5 exchanges</div>
                    <div style="font-size: 11px; color: #0369a1;">Next execution in: 23 seconds</div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('trading-controls', '🎯 Trading Controls', content, {
            x: 400,
            y: 500,
            width: 300,
            height: 320
        });

        this.addPanel(panel);
    }

    createNewsPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    📰 Market News & Sentiment
                </h4>
                <div style="margin-bottom: 12px;">
                    <div style="background: #f0f9ff; padding: 10px; border-radius: 6px; border-left: 4px solid #3b82f6; margin-bottom: 8px;">
                        <div style="font-size: 12px; font-weight: 600; color: #1e40af; margin-bottom: 4px;">Bitcoin ETF Approval</div>
                        <div style="font-size: 11px; color: #64748b;">SEC approves spot Bitcoin ETFs, driving institutional demand</div>
                        <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">2 minutes ago • Bullish sentiment</div>
                    </div>
                    
                    <div style="background: #f0fdf4; padding: 10px; border-radius: 6px; border-left: 4px solid #22c55e; margin-bottom: 8px;">
                        <div style="font-size: 12px; font-weight: 600; color: #15803d; margin-bottom: 4px;">Ethereum Upgrade Complete</div>
                        <div style="font-size: 11px; color: #64748b;">Successful network upgrade reduces gas fees by 40%</div>
                        <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">15 minutes ago • Very bullish</div>
                    </div>
                    
                    <div style="background: #fffbeb; padding: 10px; border-radius: 6px; border-left: 4px solid #f59e0b;">
                        <div style="font-size: 12px; font-weight: 600; color: #d97706; margin-bottom: 4px;">Fed Interest Rates</div>
                        <div style="font-size: 11px; color: #64748b;">Central bank signals potential rate cuts in Q2</div>
                        <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">1 hour ago • Neutral</div>
                    </div>
                </div>
                
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">AI Sentiment Analysis</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 11px;">Overall Market</span>
                        <span style="font-size: 11px; color: #059669; font-weight: 600;">73% Bullish</span>
                    </div>
                    <div style="background: #e5e7eb; height: 4px; border-radius: 2px;">
                        <div style="background: #059669; width: 73%; height: 100%; border-radius: 2px;"></div>
                    </div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('market-news', '📰 Market News', content, {
            x: 750,
            y: 540,
            width: 340,
            height: 380
        });

        this.addPanel(panel);
    }

    createAlgorithmsPanel() {
        const content = `
            <div style="padding: 0;">
                <h4 style="margin: 0 0 16px 0; color: #8b7355; font-size: 16px; font-weight: 600;">
                    🧠 AI Algorithm Status
                </h4>
                <div style="margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 12px; border-radius: 8px; color: white; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 13px; font-weight: 600;">Hyperbolic CNN</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">ACTIVE</span>
                        </div>
                        <div style="font-size: 11px; opacity: 0.9; margin-bottom: 8px;">Operating in Poincaré Ball Model</div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span>Accuracy: 94.2%</span>
                            <span>Latency: 12ms</span>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 12px; border-radius: 8px; color: white; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 13px; font-weight: 600;">FinBERT Sentiment</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">RUNNING</span>
                        </div>
                        <div style="font-size: 11px; opacity: 0.9; margin-bottom: 8px;">Processing social & news sentiment</div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span>Confidence: 87%</span>
                            <span>Sources: 1,247</span>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 12px; border-radius: 8px; color: white;">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 13px; font-weight: 600;">Risk Management AI</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 10px;">OPTIMAL</span>
                        </div>
                        <div style="font-size: 11px; opacity: 0.9; margin-bottom: 8px;">Dynamic position sizing & hedging</div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span>Risk Score: 0.23</span>
                            <span>VaR: $8,432</span>
                        </div>
                    </div>
                </div>
                
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">System Performance</div>
                    <div style="font-size: 11px; margin-bottom: 4px;">CPU Usage: 23%</div>
                    <div style="font-size: 11px; margin-bottom: 4px;">GPU Utilization: 67%</div>
                    <div style="font-size: 11px;">Memory: 4.2GB / 16GB</div>
                </div>
            </div>
        `;

        const panel = new DraggablePanel('ai-algorithms', '🧠 AI Algorithms', content, {
            x: 1120,
            y: 100,
            width: 320,
            height: 450
        });

        this.addPanel(panel);
    }

    addPanel(panel) {
        this.panels.set(panel.id, panel);
        panel.options.zIndex = this.nextZIndex++;
        if (panel.element && panel.element.style) {
            panel.element.style.zIndex = panel.options.zIndex;
        }
        
        // Append panel to container if it exists
        const container = document.getElementById('draggable-dashboard-container') || document.body;
        if (container && panel.element) {
            container.appendChild(panel.element);
        }
    }
    
    // Method to create panels by type - matches the method calls from index.html
    createPanelByType(type, title, options = {}) {
        switch (type) {
            case 'market-data':
                this.createMarketDataPanel();
                break;
            case 'positions':
                this.createPositionsPanel();
                break;
            case 'arbitrage':
                this.createArbitragePanel();
                break;
            case 'performance':
                this.createPerformancePanel();
                break;
            case 'sentiment':
                this.createAIAlgorithmsPanel();
                break;
            default:
                console.warn(`Unknown panel type: ${type}`);
        }
    }

    removePanel(panelId) {
        const panel = this.panels.get(panelId);
        if (panel && panel.element && panel.element.parentNode) {
            panel.element.parentNode.removeChild(panel.element);
        }
        this.panels.delete(panelId);
    }

    bringPanelToFront(panel) {
        panel.options.zIndex = this.nextZIndex++;
        if (panel.element && panel.element.style) {
            panel.element.style.zIndex = panel.options.zIndex;
        }
    }

    togglePanelMenu() {
        const menu = document.getElementById('panel-menu');
        menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    }

    arrangeWindows() {
        const panelArray = Array.from(this.panels.values());
        const cols = Math.ceil(Math.sqrt(panelArray.length));
        const panelWidth = Math.min(350, (window.innerWidth - 100) / cols);
        const panelHeight = Math.min(400, (window.innerHeight - 150) / Math.ceil(panelArray.length / cols));

        panelArray.forEach((panel, index) => {
            const row = Math.floor(index / cols);
            const col = index % cols;
            
            panel.element.style.left = (50 + col * (panelWidth + 20)) + 'px';
            panel.element.style.top = (80 + row * (panelHeight + 20)) + 'px';
            panel.element.style.width = panelWidth + 'px';
            panel.element.style.height = panelHeight + 'px';
        });
    }

    minimizeAll() {
        this.panels.forEach(panel => {
            if (!panel.isMinimized) {
                panel.toggleMinimize();
            }
        });
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        window.dashboardManager = new DashboardManager();
        console.log('🎛️ Draggable Trading Dashboard initialized successfully');
    }, 3000);
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DraggablePanel, DashboardManager };
}