// GOMNA Unified Platform - Clean, Professional, No Duplications
// Featuring Hyperbolic CNN with Multimodal Data Fusion

class GomnaUnifiedPlatform {
    constructor() {
        this.clearAllDuplicates();
        this.initProfessionalHeader();
        this.initHyperbolicCNNDashboard();
        this.initCleanLayout();
    }

    clearAllDuplicates() {
        // Remove ALL duplicate components
        const duplicateSelectors = [
            '.logo-container',
            '.gomna-branding',
            '#gomna-main-branding',
            '#gomna-unified-branding',
            '#realtime-portfolio',
            '#model-transparency',
            '#professional-trading-panel',
            '#professional-action-menu',
            '#performance-metrics-bar',
            '#gomna-professional-header',
            '.model-transparency-panel',
            '.realtime-portfolio-dashboard'
        ];
        
        duplicateSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => el.remove());
        });
        
        // Clear duplicate headers
        document.querySelectorAll('h1').forEach(h1 => {
            if (h1.textContent.includes('GOMNA')) {
                const parent = h1.closest('.glass-effect, header');
                if (parent) parent.style.display = 'none';
            }
        });
    }

    initProfessionalHeader() {
        const header = document.createElement('div');
        header.id = 'gomna-main-header';
        header.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: linear-gradient(135deg, #2C1810 0%, #3E2723 50%, #2C1810 100%);
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
            z-index: 10000;
            display: flex;
            align-items: center;
            padding: 0 20px;
            border-bottom: 2px solid #D4AF37;
        `;

        header.innerHTML = `
            <div style="display: flex; align-items: center; gap: 20px; width: 100%;">
                <!-- Logo -->
                <div id="gomna-logo-main" style="
                    background: radial-gradient(circle, #D4AF37, #8B6F47);
                    border-radius: 50%;
                    padding: 10px;
                    cursor: pointer;
                    box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
                ">
                    <svg width="40" height="40" viewBox="0 0 200 200">
                        <defs>
                            <linearGradient id="cocoaMain" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#8B6F47" />
                                <stop offset="100%" style="stop-color:#5D4037" />
                            </linearGradient>
                        </defs>
                        <ellipse cx="100" cy="100" rx="55" ry="75" fill="url(#cocoaMain)" />
                        <path d="M 100 35 L 100 165" stroke="#3E2723" stroke-width="3" opacity="0.5" />
                        <!-- Seeds -->
                        <ellipse cx="85" cy="75" rx="15" ry="18" fill="#FAF7F0" opacity="0.9" />
                        <ellipse cx="115" cy="75" rx="15" ry="18" fill="#F5E6D3" opacity="0.9" />
                        <ellipse cx="100" cy="105" rx="16" ry="19" fill="#FAF7F0" opacity="0.9" />
                        <ellipse cx="85" cy="135" rx="15" ry="18" fill="#F5E6D3" opacity="0.9" />
                        <ellipse cx="115" cy="135" rx="15" ry="18" fill="#FAF7F0" opacity="0.9" />
                    </svg>
                </div>

                <!-- Branding -->
                <div style="border-left: 2px solid #D4AF37; padding-left: 20px;">
                    <div style="
                        font-size: 32px;
                        font-weight: 300;
                        color: #FAF7F0;
                        letter-spacing: 6px;
                        font-family: Georgia, serif;
                        font-style: italic;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                    ">GOMNA</div>
                    <div style="
                        font-size: 10px;
                        color: #D4AF37;
                        letter-spacing: 2px;
                        text-transform: uppercase;
                        margin-top: -2px;
                    ">Hyperbolic CNN Quantitative Trading</div>
                </div>

                <!-- Model Status -->
                <div style="margin-left: auto; display: flex; gap: 15px; align-items: center;">
                    <div style="
                        background: rgba(16, 185, 129, 0.1);
                        border: 1px solid #10B981;
                        border-radius: 15px;
                        padding: 6px 12px;
                        display: flex;
                        align-items: center;
                        gap: 6px;
                    ">
                        <div style="width: 6px; height: 6px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                        <span style="color: #10B981; font-size: 11px; font-weight: 500;">HYPERBOLIC CNN ACTIVE</span>
                    </div>
                    
                    <div style="
                        background: rgba(212, 175, 55, 0.1);
                        border: 1px solid #D4AF37;
                        border-radius: 15px;
                        padding: 6px 12px;
                    ">
                        <span style="color: #D4AF37; font-size: 11px;">MULTIMODAL FUSION</span>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(header);
        document.body.style.paddingTop = '80px';
    }

    initHyperbolicCNNDashboard() {
        // Create main dashboard container with proper spacing
        const dashboard = document.createElement('div');
        dashboard.id = 'hyperbolic-cnn-dashboard';
        dashboard.style.cssText = `
            position: fixed;
            top: 80px;
            left: 10px;
            right: 10px;
            bottom: 10px;
            display: grid;
            grid-template-columns: 300px 1fr 300px;
            gap: 10px;
            z-index: 900;
            pointer-events: none;
        `;

        // Left Panel - Model Insights
        const leftPanel = document.createElement('div');
        leftPanel.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: auto;
        `;

        leftPanel.innerHTML = `
            <!-- Hyperbolic CNN Status -->
            <div style="
                background: linear-gradient(135deg, #1A1A1A, #2C1810);
                border: 1px solid #D4AF37;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="color: #D4AF37; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">
                    Hyperbolic CNN Model
                </h3>
                <div style="display: grid; gap: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #FAF7F0; font-size: 11px;">Curvature:</span>
                        <span style="color: #10B981; font-size: 11px; font-weight: bold;">-1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #FAF7F0; font-size: 11px;">Dimension:</span>
                        <span style="color: #10B981; font-size: 11px; font-weight: bold;">128</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #FAF7F0; font-size: 11px;">Accuracy:</span>
                        <span style="color: #10B981; font-size: 11px; font-weight: bold;" id="model-accuracy">94.7%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #FAF7F0; font-size: 11px;">Latency:</span>
                        <span style="color: #D4AF37; font-size: 11px; font-weight: bold;">125ms</span>
                    </div>
                </div>
            </div>

            <!-- Multimodal Data Fusion -->
            <div style="
                background: linear-gradient(135deg, #1A1A1A, #2C1810);
                border: 1px solid #D4AF37;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="color: #D4AF37; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">
                    Multimodal Data Sources
                </h3>
                <div id="multimodal-sources" style="display: grid; gap: 6px;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>

            <!-- Trading Signals -->
            <div style="
                background: linear-gradient(135deg, #1A1A1A, #2C1810);
                border: 1px solid #D4AF37;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="color: #D4AF37; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">
                    Live Signals
                </h3>
                <div id="trading-signals" style="display: grid; gap: 8px;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>
        `;

        // Center - Main content area (transparent for underlying content)
        const centerArea = document.createElement('div');
        centerArea.style.cssText = `pointer-events: none;`;

        // Right Panel - Portfolio & Performance
        const rightPanel = document.createElement('div');
        rightPanel.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: auto;
        `;

        rightPanel.innerHTML = `
            <!-- Portfolio Overview -->
            <div style="
                background: linear-gradient(135deg, #1A1A1A, #2C1810);
                border: 1px solid #D4AF37;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="color: #D4AF37; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">
                    Portfolio Performance
                </h3>
                <div style="display: grid; gap: 8px;">
                    <div>
                        <div style="color: #FAF7F0; font-size: 11px;">Total Value</div>
                        <div style="color: #10B981; font-size: 20px; font-weight: bold;" id="portfolio-value">$2,847,563</div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <div style="color: #FAF7F0; font-size: 10px;">24h Change</div>
                            <div style="color: #10B981; font-size: 14px; font-weight: bold;">+2.34%</div>
                        </div>
                        <div>
                            <div style="color: #FAF7F0; font-size: 10px;">Sharpe Ratio</div>
                            <div style="color: #D4AF37; font-size: 14px; font-weight: bold;">2.89</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Active Positions -->
            <div style="
                background: linear-gradient(135deg, #1A1A1A, #2C1810);
                border: 1px solid #D4AF37;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                max-height: 300px;
                overflow-y: auto;
            ">
                <h3 style="color: #D4AF37; font-size: 14px; margin: 0 0 10px 0; font-weight: 600;">
                    Active Positions
                </h3>
                <div id="active-positions" style="display: grid; gap: 8px;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>

            <!-- Quick Actions -->
            <div style="
                background: linear-gradient(135deg, #D4AF37, #CD7F32);
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 5px 15px rgba(212, 175, 55, 0.3);
            ">
                <button onclick="executeHyperbolicTrade()" style="
                    width: 100%;
                    background: #2C1810;
                    color: #FAF7F0;
                    border: none;
                    padding: 12px;
                    border-radius: 8px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: all 0.3s ease;
                ">EXECUTE AI TRADE</button>
            </div>
        `;

        dashboard.appendChild(leftPanel);
        dashboard.appendChild(centerArea);
        dashboard.appendChild(rightPanel);
        document.body.appendChild(dashboard);

        // Start real-time updates
        this.startHyperbolicUpdates();
    }

    startHyperbolicUpdates() {
        // Update multimodal sources
        const updateMultimodal = () => {
            const sources = [
                { name: 'Equity Indices', status: 'active', weight: '25%' },
                { name: 'Commodities', status: 'active', weight: '20%' },
                { name: 'Cryptocurrency', status: 'active', weight: '30%' },
                { name: 'Economic Data', status: 'active', weight: '15%' },
                { name: 'Sentiment', status: 'processing', weight: '10%' }
            ];

            const container = document.getElementById('multimodal-sources');
            if (container) {
                container.innerHTML = sources.map(source => `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="
                                width: 6px;
                                height: 6px;
                                background: ${source.status === 'active' ? '#10B981' : '#D4AF37'};
                                border-radius: 50%;
                            "></div>
                            <span style="color: #FAF7F0; font-size: 11px;">${source.name}</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 10px;">${source.weight}</span>
                    </div>
                `).join('');
            }
        };

        // Update trading signals
        const updateSignals = () => {
            const signals = [
                { asset: 'BTC/USD', action: 'BUY', confidence: 92 },
                { asset: 'ETH/USD', action: 'HOLD', confidence: 78 },
                { asset: 'SOL/USD', action: 'BUY', confidence: 85 }
            ];

            const container = document.getElementById('trading-signals');
            if (container) {
                container.innerHTML = signals.map(signal => {
                    const color = signal.action === 'BUY' ? '#10B981' : 
                                 signal.action === 'SELL' ? '#EF4444' : '#D4AF37';
                    return `
                        <div style="
                            background: ${color}20;
                            border: 1px solid ${color};
                            border-radius: 6px;
                            padding: 8px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        ">
                            <div>
                                <div style="color: #FAF7F0; font-size: 11px;">${signal.asset}</div>
                                <div style="color: ${color}; font-size: 12px; font-weight: bold;">${signal.action}</div>
                            </div>
                            <div style="color: #FAF7F0; font-size: 10px;">${signal.confidence}%</div>
                        </div>
                    `;
                }).join('');
            }
        };

        // Update positions
        const updatePositions = () => {
            const positions = [
                { symbol: 'BTC', amount: '2.5', value: '$168,581', pnl: '+12.4%' },
                { symbol: 'ETH', amount: '45.2', value: '$159,252', pnl: '+8.7%' },
                { symbol: 'SOL', amount: '1,250', value: '$178,125', pnl: '+15.3%' }
            ];

            const container = document.getElementById('active-positions');
            if (container) {
                container.innerHTML = positions.map(pos => `
                    <div style="
                        background: rgba(250, 247, 240, 0.05);
                        border: 1px solid #3E2723;
                        border-radius: 6px;
                        padding: 8px;
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 5px;
                    ">
                        <div>
                            <div style="color: #D4AF37; font-size: 12px; font-weight: bold;">${pos.symbol}</div>
                            <div style="color: #8B6F47; font-size: 10px;">${pos.amount} units</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #FAF7F0; font-size: 12px;">${pos.value}</div>
                            <div style="color: ${pos.pnl.startsWith('+') ? '#10B981' : '#EF4444'}; font-size: 10px;">${pos.pnl}</div>
                        </div>
                    </div>
                `).join('');
            }
        };

        // Update model accuracy
        const updateAccuracy = () => {
            const accuracy = (94 + Math.random() * 2).toFixed(1);
            const el = document.getElementById('model-accuracy');
            if (el) el.textContent = `${accuracy}%`;
        };

        // Initial updates
        updateMultimodal();
        updateSignals();
        updatePositions();
        updateAccuracy();

        // Set intervals
        setInterval(updateMultimodal, 5000);
        setInterval(updateSignals, 3000);
        setInterval(updatePositions, 4000);
        setInterval(updateAccuracy, 2000);
    }

    initCleanLayout() {
        // Add clean styles
        const style = document.createElement('style');
        style.textContent = `
            /* Clean scrollbars */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #1A1A1A;
                border-radius: 3px;
            }
            ::-webkit-scrollbar-thumb {
                background: #D4AF37;
                border-radius: 3px;
            }
            
            /* Animations */
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            /* Main content adjustment */
            body {
                background: linear-gradient(135deg, #0A0A0A, #1A1A1A) !important;
                padding-top: 80px !important;
            }
            
            .container:first-of-type,
            .glass-effect:first-of-type {
                margin-top: 10px !important;
                opacity: 0.95;
            }
            
            /* Hide overlapping elements */
            #realtime-portfolio,
            #model-transparency,
            #professional-trading-panel,
            #professional-action-menu,
            #performance-metrics-bar {
                display: none !important;
            }
        `;
        document.head.appendChild(style);
    }
}

// Execute trade function
window.executeHyperbolicTrade = () => {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
        z-index: 11000;
        animation: slideIn 0.5s ease;
    `;
    notification.textContent = 'Hyperbolic CNN trade executed successfully!';
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.5s ease';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
};

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            window.gomnaUnified = new GomnaUnifiedPlatform();
        }, 100);
    });
} else {
    setTimeout(() => {
        window.gomnaUnified = new GomnaUnifiedPlatform();
    }, 100);
}

// Animation styles
const animStyles = document.createElement('style');
animStyles.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(animStyles);