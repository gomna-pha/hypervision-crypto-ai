// GOMNA Enhanced Platform - Fixed Positioning & Real-Time Updates
// This script fixes the logo positioning and adds real-time portfolio & model transparency

class GomnaEnhancedPlatform {
    constructor() {
        this.portfolioUpdateInterval = null;
        this.modelUpdateInterval = null;
        this.fixLogoPositioning();
        this.initializeRealTimePortfolio();
        this.initializeModelTransparency();
        this.enhanceUIElements();
    }

    fixLogoPositioning() {
        // Remove any existing logo containers to prevent duplicates
        const existingLogos = document.querySelectorAll('.logo-container');
        existingLogos.forEach(logo => logo.remove());

        // Find the main header section
        const headerSection = document.querySelector('.glass-effect') || 
                             document.querySelector('.bg-white.rounded-xl') ||
                             document.querySelector('header');

        if (headerSection) {
            // Update the header layout to properly accommodate the logo
            const headerContent = headerSection.querySelector('.flex.items-center.justify-between');
            if (headerContent) {
                // Restructure header for proper spacing
                headerContent.style.position = 'relative';
                headerContent.style.paddingLeft = '80px'; // Make room for logo
            }
        }

        // Create new properly positioned logo container
        const logoContainer = document.createElement('div');
        logoContainer.className = 'gomna-logo-container';
        logoContainer.style.cssText = `
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: #FAF7F0;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(107, 68, 35, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            cursor: pointer;
            transition: all 0.3s ease;
        `;

        // Smaller, more compact logo
        const logoSVG = `
            <svg width="44" height="44" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="podGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#8B6F47" />
                        <stop offset="100%" style="stop-color:#5D4037" />
                    </linearGradient>
                    <filter id="shadow">
                        <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
                    </filter>
                </defs>
                <!-- Simplified cocoa pod -->
                <ellipse cx="100" cy="100" rx="50" ry="75" fill="url(#podGrad)" filter="url(#shadow)" transform="rotate(12 100 100)" />
                <!-- Center line -->
                <path d="M 100 40 L 100 160" stroke="#6B4423" stroke-width="3" opacity="0.5" />
                <!-- Cocoa seeds -->
                <circle cx="85" cy="80" r="14" fill="#FAF7F0" opacity="0.9" />
                <circle cx="115" cy="80" r="14" fill="#F5E6D3" opacity="0.9" />
                <circle cx="100" cy="110" r="15" fill="#FAF7F0" opacity="0.9" />
                <circle cx="85" cy="140" r="14" fill="#F5E6D3" opacity="0.9" />
                <circle cx="115" cy="140" r="14" fill="#FAF7F0" opacity="0.9" />
                <!-- Seed details -->
                <circle cx="85" cy="80" r="4" fill="#8B6F47" opacity="0.7" />
                <circle cx="115" cy="80" r="4" fill="#8B6F47" opacity="0.7" />
                <circle cx="100" cy="110" r="4" fill="#8B6F47" opacity="0.7" />
                <circle cx="85" cy="140" r="4" fill="#8B6F47" opacity="0.7" />
                <circle cx="115" cy="140" r="4" fill="#8B6F47" opacity="0.7" />
            </svg>
        `;

        logoContainer.innerHTML = logoSVG;
        
        // Add hover effects
        logoContainer.onmouseenter = () => {
            logoContainer.style.transform = 'scale(1.1) rotate(5deg)';
            logoContainer.style.boxShadow = '0 6px 16px rgba(107, 68, 35, 0.3)';
        };
        logoContainer.onmouseleave = () => {
            logoContainer.style.transform = 'scale(1) rotate(0deg)';
            logoContainer.style.boxShadow = '0 4px 12px rgba(107, 68, 35, 0.2)';
        };

        // Add click handler for logo selection
        logoContainer.onclick = () => {
            window.open('cocoa_logos.html', '_blank');
        };

        // Add tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'logo-tooltip';
        tooltip.textContent = 'Click to change logo';
        tooltip.style.cssText = `
            position: absolute;
            bottom: -30px;
            left: 50%;
            transform: translateX(-50%);
            background: #3E2723;
            color: #FAF7F0;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        `;
        logoContainer.appendChild(tooltip);

        logoContainer.onmouseenter = function() {
            this.style.transform = 'scale(1.1) rotate(5deg)';
            tooltip.style.opacity = '1';
        };
        logoContainer.onmouseleave = function() {
            this.style.transform = 'scale(1) rotate(0deg)';
            tooltip.style.opacity = '0';
        };

        // Add to page
        document.body.appendChild(logoContainer);

        // Add GOMNA branding next to the logo (separate element)
        const brandingContainer = document.createElement('div');
        brandingContainer.className = 'gomna-branding';
        brandingContainer.style.cssText = `
            position: absolute;
            top: 25px;
            left: 90px;
            z-index: 999;
        `;
        brandingContainer.innerHTML = `
            <div style="display: flex; flex-direction: column;">
                <span style="font-size: 32px; font-weight: bold; color: #3E2723; letter-spacing: 4px; font-family: 'Georgia', serif; font-style: italic;">GOMNA</span>
                <span style="font-size: 12px; color: #6B4423; letter-spacing: 0.5px; margin-top: 2px; font-family: 'Georgia', serif;">Wall Street Grade Quantitative Trading Platform</span>
            </div>
        `;
        document.body.appendChild(brandingContainer);
    }

    initializeRealTimePortfolio() {
        // Create real-time portfolio dashboard
        const portfolioDashboard = document.createElement('div');
        portfolioDashboard.id = 'realtime-portfolio';
        portfolioDashboard.className = 'realtime-portfolio-dashboard';
        portfolioDashboard.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            background: linear-gradient(135deg, #FAF7F0 0%, #F5E6D3 100%);
            border: 2px solid #8B6F47;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(107, 68, 35, 0.2);
            z-index: 900;
            transition: all 0.3s ease;
        `;

        portfolioDashboard.innerHTML = `
            <div class="portfolio-header" style="margin-bottom: 15px; border-bottom: 2px solid #8B6F47; padding-bottom: 10px;">
                <h3 style="font-size: 18px; font-weight: bold; color: #3E2723; margin: 0;">
                    Real-Time Portfolio
                </h3>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 5px;">
                    <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                    <span style="font-size: 12px; color: #6B4423;">LIVE</span>
                    <span id="portfolio-time" style="font-size: 11px; color: #8B6F47; margin-left: auto;"></span>
                </div>
            </div>
            <div class="portfolio-metrics">
                <div class="metric-row" style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 11px; color: #8B6F47; text-transform: uppercase;">Total Value</div>
                        <div id="portfolio-total" style="font-size: 24px; font-weight: bold; color: #3E2723;">$0.00</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 11px; color: #8B6F47; text-transform: uppercase;">24h Change</div>
                        <div id="portfolio-change" style="font-size: 20px; font-weight: bold;">+0.00%</div>
                    </div>
                </div>
                <div class="holdings-list" id="holdings-list" style="max-height: 200px; overflow-y: auto;">
                    <!-- Holdings will be dynamically added here -->
                </div>
                <div class="performance-chart" style="margin-top: 15px;">
                    <canvas id="mini-performance-chart" width="310" height="100"></canvas>
                </div>
            </div>
        `;

        document.body.appendChild(portfolioDashboard);

        // Start real-time updates
        this.startPortfolioUpdates();
    }

    startPortfolioUpdates() {
        const updatePortfolio = () => {
            const totalElement = document.getElementById('portfolio-total');
            const changeElement = document.getElementById('portfolio-change');
            const timeElement = document.getElementById('portfolio-time');
            const holdingsList = document.getElementById('holdings-list');

            if (totalElement && changeElement && timeElement) {
                // Simulate real-time data (replace with actual API calls)
                const baseValue = 2847563;
                const variation = (Math.random() - 0.5) * 10000;
                const totalValue = baseValue + variation;
                const changePercent = ((Math.random() - 0.45) * 5).toFixed(2);

                totalElement.textContent = `$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                
                changeElement.textContent = `${changePercent > 0 ? '+' : ''}${changePercent}%`;
                changeElement.style.color = changePercent > 0 ? '#10B981' : '#EF4444';

                timeElement.textContent = new Date().toLocaleTimeString();

                // Update holdings
                if (holdingsList && Math.random() > 0.7) {
                    const holdings = [
                        { symbol: 'BTC', name: 'Bitcoin', value: 145230.50, change: 2.3 },
                        { symbol: 'ETH', name: 'Ethereum', value: 89450.00, change: -1.2 },
                        { symbol: 'SOL', name: 'Solana', value: 34200.00, change: 5.7 },
                        { symbol: 'AVAX', name: 'Avalanche', value: 28900.00, change: 3.1 }
                    ];

                    holdingsList.innerHTML = holdings.map(holding => `
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #E8DCC7;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="width: 32px; height: 32px; background: #8B6F47; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #FAF7F0; font-weight: bold; font-size: 10px;">
                                    ${holding.symbol}
                                </div>
                                <div>
                                    <div style="font-size: 13px; font-weight: 600; color: #3E2723;">${holding.name}</div>
                                    <div style="font-size: 11px; color: #8B6F47;">$${holding.value.toLocaleString()}</div>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 13px; font-weight: 600; color: ${holding.change > 0 ? '#10B981' : '#EF4444'};">
                                    ${holding.change > 0 ? '+' : ''}${holding.change}%
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            }
        };

        // Update immediately and then every 2 seconds
        updatePortfolio();
        this.portfolioUpdateInterval = setInterval(updatePortfolio, 2000);
    }

    initializeModelTransparency() {
        // Create model transparency panel
        const modelPanel = document.createElement('div');
        modelPanel.id = 'model-transparency';
        modelPanel.className = 'model-transparency-panel';
        modelPanel.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 350px;
            background: linear-gradient(135deg, #FAF7F0 0%, #F5E6D3 100%);
            border: 2px solid #8B6F47;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(107, 68, 35, 0.2);
            z-index: 900;
            transition: all 0.3s ease;
        `;

        modelPanel.innerHTML = `
            <div class="model-header" style="margin-bottom: 15px; border-bottom: 2px solid #8B6F47; padding-bottom: 10px;">
                <h3 style="font-size: 18px; font-weight: bold; color: #3E2723; margin: 0;">
                    Model Transparency
                </h3>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 5px;">
                    <div style="width: 8px; height: 8px; background: #D4AF37; border-radius: 50%; animation: pulse 2s infinite;"></div>
                    <span style="font-size: 12px; color: #6B4423;">AI ACTIVE</span>
                </div>
            </div>
            <div class="model-metrics">
                <div class="accuracy-meter" style="margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #8B6F47; margin-bottom: 5px;">Model Accuracy</div>
                    <div style="background: #E8DCC7; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div id="accuracy-bar" style="background: linear-gradient(90deg, #8B6F47, #D4AF37); height: 100%; width: 0%; transition: width 1s ease;">
                            <span style="color: white; font-size: 11px; padding: 2px 8px; display: block; text-align: right;">0%</span>
                        </div>
                    </div>
                </div>
                <div class="model-signals" id="model-signals">
                    <!-- Signals will be added dynamically -->
                </div>
                <div class="feature-importance" style="margin-top: 15px;">
                    <div style="font-size: 12px; color: #8B6F47; margin-bottom: 8px;">Top Features</div>
                    <div id="feature-list">
                        <!-- Features will be added dynamically -->
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modelPanel);

        // Start model transparency updates
        this.startModelUpdates();
    }

    startModelUpdates() {
        const updateModel = () => {
            const accuracyBar = document.getElementById('accuracy-bar');
            const signalsDiv = document.getElementById('model-signals');
            const featureList = document.getElementById('feature-list');

            if (accuracyBar) {
                const accuracy = 85 + Math.random() * 10;
                accuracyBar.style.width = `${accuracy}%`;
                accuracyBar.querySelector('span').textContent = `${accuracy.toFixed(1)}%`;
            }

            if (signalsDiv) {
                const signals = [
                    { type: 'BUY', asset: 'BTC/USD', confidence: 92, reason: 'Bullish divergence detected' },
                    { type: 'HOLD', asset: 'ETH/USD', confidence: 78, reason: 'Consolidation phase' },
                    { type: 'SELL', asset: 'XRP/USD', confidence: 85, reason: 'Overbought conditions' }
                ];

                const randomSignal = signals[Math.floor(Math.random() * signals.length)];
                const signalColor = randomSignal.type === 'BUY' ? '#10B981' : 
                                   randomSignal.type === 'SELL' ? '#EF4444' : '#D4AF37';

                signalsDiv.innerHTML = `
                    <div style="background: ${signalColor}20; border: 1px solid ${signalColor}; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: bold; color: ${signalColor}; font-size: 14px;">${randomSignal.type}</span>
                            <span style="font-size: 12px; color: #3E2723;">${randomSignal.asset}</span>
                        </div>
                        <div style="font-size: 11px; color: #6B4423; margin-top: 5px;">${randomSignal.reason}</div>
                        <div style="font-size: 10px; color: #8B6F47; margin-top: 3px;">Confidence: ${randomSignal.confidence}%</div>
                    </div>
                `;
            }

            if (featureList) {
                const features = [
                    { name: 'RSI (14)', importance: 95 },
                    { name: 'Volume Profile', importance: 88 },
                    { name: 'MACD Signal', importance: 82 },
                    { name: 'Sentiment Score', importance: 76 }
                ];

                featureList.innerHTML = features.map(feature => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0;">
                        <span style="font-size: 11px; color: #3E2723;">${feature.name}</span>
                        <div style="width: 100px; background: #E8DCC7; border-radius: 3px; height: 8px; overflow: hidden;">
                            <div style="background: #8B6F47; height: 100%; width: ${feature.importance}%;"></div>
                        </div>
                    </div>
                `).join('');
            }
        };

        // Update immediately and then every 3 seconds
        updateModel();
        this.modelUpdateInterval = setInterval(updateModel, 3000);
    }

    enhanceUIElements() {
        // Add minimize/maximize buttons to panels
        const panels = [
            document.getElementById('realtime-portfolio'),
            document.getElementById('model-transparency')
        ];

        panels.forEach((panel, index) => {
            if (panel) {
                const minimizeBtn = document.createElement('button');
                minimizeBtn.style.cssText = `
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    width: 20px;
                    height: 20px;
                    background: #8B6F47;
                    color: #FAF7F0;
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    font-size: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                `;
                minimizeBtn.innerHTML = '−';
                
                let isMinimized = false;
                const originalHeight = panel.style.height;
                
                minimizeBtn.onclick = () => {
                    if (isMinimized) {
                        panel.style.height = originalHeight || 'auto';
                        minimizeBtn.innerHTML = '−';
                        panel.querySelectorAll('.portfolio-metrics, .model-metrics').forEach(el => {
                            el.style.display = 'block';
                        });
                    } else {
                        panel.style.height = '60px';
                        minimizeBtn.innerHTML = '+';
                        panel.querySelectorAll('.portfolio-metrics, .model-metrics').forEach(el => {
                            el.style.display = 'none';
                        });
                    }
                    isMinimized = !isMinimized;
                };
                
                panel.appendChild(minimizeBtn);
            }
        });

        // Add animation styles
        const animationStyles = document.createElement('style');
        animationStyles.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .realtime-portfolio-dashboard:hover,
            .model-transparency-panel:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 32px rgba(107, 68, 35, 0.3) !important;
            }
            
            /* Custom scrollbar for panels */
            #holdings-list::-webkit-scrollbar,
            #model-signals::-webkit-scrollbar {
                width: 6px;
            }
            
            #holdings-list::-webkit-scrollbar-track,
            #model-signals::-webkit-scrollbar-track {
                background: #E8DCC7;
                border-radius: 3px;
            }
            
            #holdings-list::-webkit-scrollbar-thumb,
            #model-signals::-webkit-scrollbar-thumb {
                background: #8B6F47;
                border-radius: 3px;
            }
            
            /* Ensure main content doesn't overlap with logo */
            body > .container,
            body > div > .container,
            main {
                padding-top: 100px !important;
            }
            
            /* Fix header spacing */
            .glass-effect,
            header {
                margin-top: 20px;
                margin-left: 100px;
            }
        `;
        document.head.appendChild(animationStyles);
    }

    // Cleanup method
    destroy() {
        if (this.portfolioUpdateInterval) {
            clearInterval(this.portfolioUpdateInterval);
        }
        if (this.modelUpdateInterval) {
            clearInterval(this.modelUpdateInterval);
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.gomnaEnhanced = new GomnaEnhancedPlatform();
    });
} else {
    window.gomnaEnhanced = new GomnaEnhancedPlatform();
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GomnaEnhancedPlatform;
}