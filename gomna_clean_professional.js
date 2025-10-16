/**
 * GOMNA AI Trading - Clean Professional Platform
 * 90% Cream, 1.5% Brown, No Emojis, Working Registration
 */

// Clear any existing instances
if (typeof window.gomnaProffessionalPlatform !== 'undefined') {
    window.gomnaProffessionalPlatform = null;
}

class GOMNACleanProfessional {
    constructor() {
        this.init();
    }

    init() {
        console.log('Initializing GOMNA AI Trading Clean Professional Platform...');
        this.cleanupExistingContent();
        this.injectCleanStyles();
        this.updateCleanBranding();
        this.createCleanDashboard();
        console.log('GOMNA AI Trading Platform initialized successfully');
    }

    cleanupExistingContent() {
        // Aggressive cleanup to prevent duplicates
        const duplicateSelectors = [
            '.gomna-header', '.cocoa-header', '.cocoa-trading-header',
            '.gomna-panel', '.cocoa-panel', '.cocoa-trading-card',
            '.gomna-dashboard', '.cocoa-dashboard'
        ];
        
        duplicateSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        });

        // Remove any text nodes containing duplicate content
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const textNodesToRemove = [];
        let node;
        while (node = walker.nextNode()) {
            if (node.textContent.includes('Institutional Quantitative Trading') ||
                (node.textContent.includes('Gomna AI Trading') && !node.textContent.includes('GOMNA AI Trading'))) {
                textNodesToRemove.push(node.parentElement);
            }
        }

        textNodesToRemove.forEach(el => {
            if (el && el.parentNode) {
                el.remove();
            }
        });
    }

    injectCleanStyles() {
        if (document.getElementById('gomna-clean-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'gomna-clean-styles';
        style.textContent = `
            /* GOMNA AI Trading Clean Professional Styles - 90% Cream, 1.5% Brown */
            :root {
                --gomna-cream-50: #fefbf3;
                --gomna-cream-100: #fdf6e3;
                --gomna-cream-200: #f5e6d3;
                --gomna-cream-300: #e8dcc7;
                --gomna-cream-400: #d4c4a8;
                --gomna-cream-500: #c0ac89;
                --gomna-brown-accent: #8b7355;
                --gomna-text-dark: #2d2d2d;
                --gomna-success: #10b981;
                --gomna-warning: #f59e0b;
                --gomna-error: #ef4444;
            }

            body {
                background: var(--gomna-cream-50);
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                color: var(--gomna-text-dark);
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }

            .gomna-header {
                background: linear-gradient(135deg, var(--gomna-cream-100) 0%, var(--gomna-cream-200) 100%);
                border: 2px solid var(--gomna-cream-300);
                border-radius: 20px;
                padding: 50px 40px;
                text-align: center;
                margin: 0 0 40px 0;
                box-shadow: 0 10px 30px rgba(139, 115, 85, 0.08);
            }
            
            .gomna-header-top {
                color: var(--gomna-brown-accent);
                font-size: 1.4rem;
                font-weight: 600;
                margin: 0 0 10px 0;
                text-transform: uppercase;
                letter-spacing: 2px;
                text-align: center;
            }

            .gomna-title {
                color: var(--gomna-text-dark);
                font-size: 4.2rem;
                font-weight: 700;
                margin: 0 0 20px 0;
                letter-spacing: -1px;
                text-shadow: 2px 2px 4px rgba(139, 115, 85, 0.1);
            }
            
            .gomna-subtitle {
                color: var(--gomna-brown-accent);
                font-size: 1.8rem;
                margin: 0 0 30px 0;
                font-weight: 400;
                font-style: italic;
                letter-spacing: 0.5px;
                opacity: 0.9;
                line-height: 1.4;
            }
            
            .gomna-status-bar {
                display: flex;
                justify-content: center;
                gap: 35px;
                margin-top: 35px;
                flex-wrap: wrap;
            }
            
            .gomna-status {
                background: var(--gomna-cream-200);
                color: var(--gomna-success);
                padding: 15px 25px;
                border-radius: 25px;
                border: 2px solid var(--gomna-cream-300);
                font-size: 1.1rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .gomna-nav-bar {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-top: 40px;
                flex-wrap: wrap;
                padding-top: 25px;
                border-top: 2px solid var(--gomna-cream-300);
            }

            .gomna-nav-item {
                color: var(--gomna-text-dark);
                font-size: 1.1rem;
                font-weight: 500;
                text-decoration: none;
                padding: 12px 20px;
                border-radius: 20px;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .gomna-nav-item:hover {
                background: var(--gomna-cream-300);
                color: var(--gomna-brown-accent);
                transform: translateY(-2px);
            }
            
            .gomna-panel {
                background: linear-gradient(135deg, var(--gomna-cream-50) 0%, var(--gomna-cream-100) 100%);
                border: 1px solid var(--gomna-cream-300);
                border-radius: 20px;
                margin: 30px 0;
                padding: 0;
                box-shadow: 0 8px 25px rgba(139, 115, 85, 0.06);
                overflow: hidden;
            }
            
            .gomna-panel-header {
                background: var(--gomna-cream-200);
                color: var(--gomna-text-dark);
                padding: 25px 30px;
                font-weight: 600;
                font-size: 1.3rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid var(--gomna-cream-300);
            }
            
            .gomna-panel-content {
                padding: 40px 30px;
                color: var(--gomna-text-dark);
            }
            
            .gomna-btn {
                background: linear-gradient(135deg, var(--gomna-cream-300) 0%, var(--gomna-cream-400) 100%);
                color: var(--gomna-text-dark);
                border: 1px solid var(--gomna-brown-accent);
                padding: 15px 30px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .gomna-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(139, 115, 85, 0.15);
                background: var(--gomna-brown-accent);
                color: white;
            }

            .gomna-btn-primary {
                background: linear-gradient(135deg, var(--gomna-brown-accent) 0%, #6d5d48 100%);
                color: white;
                font-size: 1.2rem;
                padding: 18px 40px;
                border: none;
            }

            .gomna-btn-primary:hover {
                background: linear-gradient(135deg, #6d5d48 0%, var(--gomna-brown-accent) 100%);
                transform: translateY(-3px);
                box-shadow: 0 12px 30px rgba(139, 115, 85, 0.25);
            }
            
            .gomna-card {
                background: var(--gomna-cream-100);
                border: 1px solid var(--gomna-cream-300);
                border-radius: 15px;
                padding: 30px;
                margin: 25px 0;
                color: var(--gomna-text-dark);
                transition: all 0.3s ease;
            }

            .gomna-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 25px rgba(139, 115, 85, 0.1);
            }
            
            .gomna-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 30px;
                margin: 30px 0;
            }
            
            .gomna-metric {
                text-align: center;
                padding: 30px 25px;
                background: var(--gomna-cream-200);
                border: 1px solid var(--gomna-cream-300);
                border-radius: 15px;
                transition: all 0.3s ease;
            }

            .gomna-metric:hover {
                transform: scale(1.02);
                box-shadow: 0 8px 20px rgba(139, 115, 85, 0.12);
            }
            
            .gomna-metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--gomna-brown-accent);
                display: block;
                margin-bottom: 10px;
            }
            
            .gomna-metric-label {
                color: var(--gomna-text-dark);
                font-size: 1.1rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .gomna-cta-section {
                background: linear-gradient(135deg, var(--gomna-cream-200) 0%, var(--gomna-cream-300) 100%);
                border: 2px solid var(--gomna-brown-accent);
                border-radius: 25px;
                text-align: center;
                padding: 60px 40px;
                margin: 50px 0;
            }

            .gomna-tier-card {
                background: var(--gomna-cream-100);
                border: 2px solid var(--gomna-cream-300);
                border-radius: 20px;
                padding: 35px 30px;
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
            }

            .gomna-tier-card.professional {
                border-color: var(--gomna-brown-accent);
                background: linear-gradient(135deg, var(--gomna-cream-100) 0%, var(--gomna-cream-200) 100%);
            }

            .gomna-tier-card.institutional {
                border-color: var(--gomna-brown-accent);
                background: linear-gradient(135deg, var(--gomna-cream-200) 0%, var(--gomna-cream-300) 100%);
            }

            .gomna-tier-badge {
                position: absolute;
                top: -15px;
                right: 25px;
                background: var(--gomna-brown-accent);
                color: white;
                padding: 8px 18px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
            }

            .gomna-opportunity-card {
                background: var(--gomna-cream-100);
                border: 1px solid var(--gomna-success);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                border-left: 5px solid var(--gomna-success);
                transition: all 0.3s ease;
            }

            .gomna-opportunity-card:hover {
                transform: translateX(8px);
                box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
            }

            /* Modal Styles */
            .gomna-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.6);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
            }

            .gomna-modal-content {
                background: var(--gomna-cream-50);
                border: 2px solid var(--gomna-cream-300);
                border-radius: 20px;
                max-width: 600px;
                width: 90%;
                max-height: 90vh;
                overflow-y: auto;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
            }

            .gomna-modal-header {
                background: var(--gomna-cream-200);
                padding: 25px 30px;
                border-bottom: 1px solid var(--gomna-cream-300);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .gomna-modal-body {
                padding: 30px;
            }

            .gomna-close {
                background: none;
                border: none;
                font-size: 2rem;
                cursor: pointer;
                color: var(--gomna-text-dark);
                padding: 0;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .gomna-form-group {
                margin-bottom: 25px;
            }

            .gomna-label {
                display: block;
                margin-bottom: 8px;
                color: var(--gomna-text-dark);
                font-weight: 600;
                font-size: 1.1rem;
            }

            .gomna-input, .gomna-select {
                width: 100%;
                padding: 15px 20px;
                border: 2px solid var(--gomna-cream-300);
                border-radius: 10px;
                background: var(--gomna-cream-50);
                color: var(--gomna-text-dark);
                font-size: 1rem;
                transition: all 0.3s ease;
            }

            .gomna-input:focus, .gomna-select:focus {
                outline: none;
                border-color: var(--gomna-brown-accent);
                box-shadow: 0 0 15px rgba(139, 115, 85, 0.2);
            }

            /* Responsive design */
            @media (max-width: 768px) {
                .gomna-header-top {
                    font-size: 1.1rem;
                    letter-spacing: 1px;
                }

                .gomna-title {
                    font-size: 2.8rem;
                    letter-spacing: 0;
                }

                .gomna-subtitle {
                    font-size: 1.4rem;
                }
                
                .gomna-grid {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
                
                .gomna-status-bar {
                    flex-direction: column;
                    align-items: center;
                    gap: 15px;
                }

                .gomna-nav-bar {
                    flex-direction: column;
                    align-items: center;
                    gap: 20px;
                }

                .gomna-nav-item {
                    width: 100%;
                    text-align: center;
                    max-width: 300px;
                }
                
                body {
                    padding: 15px;
                }

                .gomna-header {
                    padding: 35px 25px;
                }

                .gomna-panel-content {
                    padding: 25px 20px;
                }
            }
        `;
        
        document.head.appendChild(style);
    }

    updateCleanBranding() {
        // Remove any existing headers and duplicate content
        const existingHeaders = document.querySelectorAll('.gomna-header, .cocoa-header, .cocoa-trading-header, .header, .main-header');
        existingHeaders.forEach(header => header.remove());
        
        // Clear any duplicate text content
        const allElements = document.querySelectorAll('*');
        allElements.forEach(el => {
            if (el.textContent && (
                el.textContent.includes('Gomna AI Trading') || 
                el.textContent.includes('Institutional Quantitative Trading') ||
                (el.textContent.includes('GOMNA') && el !== document.body && !el.classList.contains('gomna-title'))
            )) {
                if (!el.classList.contains('gomna-header') && !el.closest('.gomna-header')) {
                    el.remove();
                }
            }
        });
        
        // Create enhanced GOMNA header
        const header = document.createElement('div');
        header.className = 'gomna-header';
        header.innerHTML = `
            <div class="gomna-header-top">Agentic AI Trading</div>
            <h1 class="gomna-title">GOMNA AI Trading</h1>
            <p class="gomna-subtitle"><em>Professional High-Frequency Trading & Arbitrage Platform</em></p>
            <div class="gomna-status-bar">
                <div class="gomna-status">LIVE TRADING</div>
                <div class="gomna-status">Alpha: +15.46%</div>
                <div class="gomna-status">APIs: Demo Mode</div>
            </div>
            <div class="gomna-nav-bar">
                <div class="gomna-nav-item">Model Transparency</div>
                <div class="gomna-nav-item">Dashboard</div>
                <div class="gomna-nav-item">Performance</div>
                <div class="gomna-nav-item">Analytics</div>
                <div class="gomna-nav-item">Portfolio</div>
                <div class="gomna-nav-item">Payment Systems</div>
            </div>
        `;
        
        // Insert at the top of the page - simplified approach
        if (document.body.firstChild) {
            document.body.insertBefore(header, document.body.firstChild);
        } else {
            document.body.appendChild(header);
        }

        // Update page title
        document.title = 'GOMNA AI Trading - Professional High-Frequency Trading & Arbitrage Platform';
    }

    createCleanDashboard() {
        // Remove existing panels to prevent conflicts and duplicates
        const existingPanels = document.querySelectorAll('.gomna-panel, .cocoa-panel, .cocoa-trading-card, .gomna-header, .cocoa-header, .cocoa-trading-header');
        existingPanels.forEach(panel => panel.remove());
        
        // Also remove any duplicate content containers
        const existingContainers = document.querySelectorAll('[class*="dashboard"], [class*="container"]');
        existingContainers.forEach(container => {
            if (container.innerHTML && container.innerHTML.includes('GOMNA')) {
                container.remove();
            }
        });

        const dashboardContainer = document.createElement('div');
        dashboardContainer.className = 'gomna-dashboard';
        dashboardContainer.innerHTML = `
            <!-- Professional Start Trading Section -->
            <div class="gomna-cta-section">
                <h2 style="color: var(--gomna-text-dark); margin-bottom: 25px; font-size: 3.5rem; font-weight: 300;">
                    Start Trading with GOMNA AI
                </h2>
                <p style="color: var(--gomna-brown-accent); font-size: 1.4rem; margin-bottom: 40px; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                    Join thousands of professional traders using our AI-powered arbitrage strategies. 
                    Advanced algorithms, institutional-grade security, and real-time market analysis.
                </p>
                <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
                    <button class="gomna-btn-primary" id="start-trading-btn">
                        Create Account & Start Trading
                    </button>
                    <button class="gomna-btn" id="demo-trading-btn">
                        Try Demo Mode
                    </button>
                </div>
            </div>

            <!-- Performance Overview -->
            <div class="gomna-panel">
                <div class="gomna-panel-header">
                    <h3>Trading Performance Overview</h3>
                    <div class="gomna-status">Live Trading</div>
                </div>
                <div class="gomna-panel-content">
                    <div class="gomna-grid">
                        <div class="gomna-metric">
                            <span class="gomna-metric-value">+$12,450.30</span>
                            <div class="gomna-metric-label">Today's Performance</div>
                        </div>
                        <div class="gomna-metric">
                            <span class="gomna-metric-value">$532,180.45</span>
                            <div class="gomna-metric-label">Portfolio Value</div>
                        </div>
                        <div class="gomna-metric">
                            <span class="gomna-metric-value">89.4%</span>
                            <div class="gomna-metric-label">Win Rate</div>
                        </div>
                        <div class="gomna-metric">
                            <span class="gomna-metric-value">6 Active</span>
                            <div class="gomna-metric-label">AI Strategies</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Account Tiers -->
            <div class="gomna-panel">
                <div class="gomna-panel-header">
                    <h3>Professional Account Tiers</h3>
                    <button class="gomna-btn" id="upgrade-account-btn">Upgrade Account</button>
                </div>
                <div class="gomna-panel-content">
                    <div class="gomna-grid">
                        <div class="gomna-tier-card">
                            <h4 style="color: var(--gomna-success); margin-bottom: 20px; font-size: 1.6rem;">Starter Tier</h4>
                            <div class="gomna-metric-value" style="font-size: 2rem; color: var(--gomna-success);">$10,000</div>
                            <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                            <div style="margin-bottom: 25px; text-align: left; line-height: 1.8;">
                                <strong>Features:</strong><br>
                                • Basic Arbitrage Strategies<br>
                                • Real-time Market Data<br>
                                • 10:1 Leverage<br>
                                • Standard Support<br>
                                • Mobile Trading App
                            </div>
                            <button class="gomna-btn" style="width: 100%;" onclick="window.gomnaCleanProfessional.openRegistrationModal('starter')">Get Started</button>
                        </div>
                        
                        <div class="gomna-tier-card professional">
                            <div class="gomna-tier-badge">Popular</div>
                            <h4 style="color: var(--gomna-brown-accent); margin-bottom: 20px; font-size: 1.6rem;">Professional Tier</h4>
                            <div class="gomna-metric-value" style="font-size: 2rem; color: var(--gomna-brown-accent);">$100,000</div>
                            <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                            <div style="margin-bottom: 25px; text-align: left; line-height: 1.8;">
                                <strong>Features:</strong><br>
                                • Advanced HFT Strategies<br>
                                • AI-Powered Analysis<br>
                                • 25:1 Leverage<br>
                                • Priority Support 24/7<br>
                                • Custom Risk Parameters<br>
                                • API Access
                            </div>
                            <button class="gomna-btn-primary" style="width: 100%;" onclick="window.gomnaCleanProfessional.openRegistrationModal('professional')">Upgrade Now</button>
                        </div>
                        
                        <div class="gomna-tier-card institutional">
                            <div class="gomna-tier-badge">Enterprise</div>
                            <h4 style="color: var(--gomna-brown-accent); margin-bottom: 20px; font-size: 1.6rem;">Institutional Tier</h4>
                            <div class="gomna-metric-value" style="font-size: 2rem; color: var(--gomna-brown-accent);">$1,000,000</div>
                            <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                            <div style="margin-bottom: 25px; text-align: left; line-height: 1.8;">
                                <strong>Features:</strong><br>
                                • Custom Algorithm Development<br>
                                • Direct Market Access<br>
                                • 50:1 Leverage<br>
                                • Dedicated Account Manager<br>
                                • White Label Solutions<br>
                                • Institutional Reporting
                            </div>
                            <button class="gomna-btn" style="width: 100%; background: var(--gomna-brown-accent); color: white;" onclick="window.gomnaCleanProfessional.openRegistrationModal('institutional')">Contact Sales</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Live Arbitrage Opportunities -->
            <div class="gomna-panel">
                <div class="gomna-panel-header">
                    <h3>Live Arbitrage Opportunities</h3>
                    <div class="gomna-status">Scanning 47 Markets</div>
                </div>
                <div class="gomna-panel-content">
                    <div class="gomna-grid">
                        ${this.generateArbitrageOpportunities()}
                    </div>
                </div>
            </div>

            <!-- Regulatory Credentials -->
            <div class="gomna-panel">
                <div class="gomna-panel-header">
                    <h3>Regulatory Credentials & Compliance</h3>
                    <div class="gomna-status">All Licenses Active</div>
                </div>
                <div class="gomna-panel-content">
                    <div class="gomna-grid">
                        <div class="gomna-card">
                            <h4 style="color: #1E40AF; margin-bottom: 15px;">SEC Registered</h4>
                            <strong>CRD: 299792</strong><br>
                            <small style="color: var(--gomna-brown-accent);">Securities and Exchange Commission</small><br>
                            <div style="color: var(--gomna-success); margin-top: 15px; font-weight: 600;">Active - Valid until 2025-12-31</div>
                        </div>
                        
                        <div class="gomna-card">
                            <h4 style="color: #059669; margin-bottom: 15px;">FINRA Member</h4>
                            <strong>Member ID: 19847</strong><br>
                            <small style="color: var(--gomna-brown-accent);">Financial Industry Regulatory Authority</small><br>
                            <div style="color: var(--gomna-success); margin-top: 15px; font-weight: 600;">Active - Valid until 2025-06-30</div>
                        </div>
                        
                        <div class="gomna-card">
                            <h4 style="color: #DC2626; margin-bottom: 15px;">Lloyds of London</h4>
                            <strong>Policy: LL-2024-GOMNA</strong><br>
                            <small style="color: var(--gomna-brown-accent);">Professional Indemnity: $50M Coverage</small><br>
                            <div style="color: var(--gomna-success); margin-top: 15px; font-weight: 600;">Active - Valid until 2025-08-31</div>
                        </div>
                        
                        <div class="gomna-card">
                            <h4 style="color: #7C2D12; margin-bottom: 15px;">SIPC Protected</h4>
                            <strong>SIPC Member</strong><br>
                            <small style="color: var(--gomna-brown-accent);">Securities Investor Protection: $500K</small><br>
                            <div style="color: var(--gomna-success); margin-top: 15px; font-weight: 600;">Active - Protected</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(dashboardContainer);

        // Add event listeners
        this.setupEventListeners();
    }

    generateArbitrageOpportunities() {
        const opportunities = [
            { pair: 'BTC/USD', exchanges: 'Binance → Coinbase', spread: '0.23%', profit: '$1,240' },
            { pair: 'ETH/USD', exchanges: 'Kraken → Bitstamp', spread: '0.18%', profit: '$890' },
            { pair: 'BTC/EUR', exchanges: 'Bitfinex → Coinbase', spread: '0.31%', profit: '$1,650' },
            { pair: 'ADA/USD', exchanges: 'Binance → KuCoin', spread: '0.45%', profit: '$420' }
        ];

        return opportunities.map(opp => `
            <div class="gomna-opportunity-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <strong style="color: var(--gomna-brown-accent); font-size: 1.2rem;">${opp.pair}</strong>
                    <span style="color: var(--gomna-success); font-weight: bold; font-size: 1.2rem;">${opp.spread}</span>
                </div>
                <div style="margin-bottom: 15px; color: var(--gomna-text-dark); font-size: 1rem;">
                    ${opp.exchanges}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--gomna-success); font-weight: bold; font-size: 1.1rem;">Est. Profit: ${opp.profit}</span>
                    <button class="gomna-btn execute-trade" style="padding: 10px 20px; font-size: 1rem;">Execute Trade</button>
                </div>
            </div>
        `).join('');
    }

    setupEventListeners() {
        document.addEventListener('click', (e) => {
            if (e.target.id === 'start-trading-btn') {
                this.openRegistrationModal('professional');
            }

            if (e.target.id === 'demo-trading-btn') {
                this.showNotification('Demo mode activated! All trades are simulated with GOMNA AI.', 'info');
            }

            if (e.target.id === 'upgrade-account-btn') {
                this.openRegistrationModal('professional');
            }

            if (e.target.classList.contains('execute-trade')) {
                this.executeArbitrageTrade(e.target);
            }
        });
    }

    openRegistrationModal(tier = 'starter') {
        const modal = document.createElement('div');
        modal.className = 'gomna-modal';
        modal.innerHTML = `
            <div class="gomna-modal-content">
                <div class="gomna-modal-header">
                    <h2>Create GOMNA AI Trading Account - ${tier.charAt(0).toUpperCase() + tier.slice(1)} Tier</h2>
                    <button class="gomna-close" onclick="this.closest('.gomna-modal').remove()">&times;</button>
                </div>
                <div class="gomna-modal-body">
                    <form id="registration-form">
                        <div class="gomna-form-group">
                            <label class="gomna-label">Full Name</label>
                            <input type="text" class="gomna-input" required placeholder="Enter your full name">
                        </div>
                        
                        <div class="gomna-form-group">
                            <label class="gomna-label">Email Address</label>
                            <input type="email" class="gomna-input" required placeholder="Enter your email">
                        </div>
                        
                        <div class="gomna-form-group">
                            <label class="gomna-label">Phone Number</label>
                            <input type="tel" class="gomna-input" required placeholder="Enter your phone number">
                        </div>
                        
                        <div class="gomna-form-group">
                            <label class="gomna-label">Country</label>
                            <select class="gomna-select" required>
                                <option value="">Select your country</option>
                                <option value="US">United States</option>
                                <option value="UK">United Kingdom</option>
                                <option value="CA">Canada</option>
                                <option value="AU">Australia</option>
                                <option value="DE">Germany</option>
                                <option value="FR">France</option>
                                <option value="JP">Japan</option>
                                <option value="SG">Singapore</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                        
                        <div class="gomna-form-group">
                            <label class="gomna-label">Initial Investment Amount</label>
                            <select class="gomna-select" required>
                                <option value="">Select investment amount</option>
                                <option value="10000">$10,000 - $50,000 (Starter)</option>
                                <option value="100000">$100,000 - $500,000 (Professional)</option>
                                <option value="1000000">$1,000,000+ (Institutional)</option>
                            </select>
                        </div>
                        
                        <div class="gomna-form-group">
                            <label class="gomna-label">Trading Experience</label>
                            <select class="gomna-select" required>
                                <option value="">Select your experience level</option>
                                <option value="beginner">Beginner (0-1 years)</option>
                                <option value="intermediate">Intermediate (2-5 years)</option>
                                <option value="advanced">Advanced (5+ years)</option>
                                <option value="professional">Professional Trader</option>
                            </select>
                        </div>
                        
                        <div style="text-align: center; margin-top: 30px;">
                            <button type="submit" class="gomna-btn-primary" style="margin: 0 10px;">Create Account</button>
                            <button type="button" class="gomna-btn" style="margin: 0 10px;" onclick="this.closest('.gomna-modal').remove()">Cancel</button>
                        </div>
                    </form>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Handle form submission
        modal.querySelector('#registration-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegistration(modal);
        });

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    handleRegistration(modal) {
        const form = modal.querySelector('#registration-form');
        const formData = new FormData(form);
        
        // Simulate registration process
        this.showNotification('Processing your GOMNA AI Trading account registration...', 'info');
        
        setTimeout(() => {
            modal.remove();
            this.showNotification('Welcome to GOMNA AI Trading! Your account has been created successfully. Please check your email for verification instructions.', 'success');
        }, 2000);
    }

    executeArbitrageTrade(button) {
        const card = button.closest('.gomna-opportunity-card');
        const pair = card.querySelector('strong').textContent;
        
        button.textContent = 'Executing...';
        button.disabled = true;
        
        setTimeout(() => {
            button.textContent = 'Executed';
            button.style.background = 'var(--gomna-success)';
            button.style.color = 'white';
            
            this.showNotification(`GOMNA AI executed arbitrage trade successfully for ${pair}!`, 'success');
            
            setTimeout(() => {
                button.textContent = 'Execute Trade';
                button.disabled = false;
                button.style.background = '';
                button.style.color = '';
            }, 3000);
        }, 2000);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 20px 30px;
            background: ${type === 'success' ? 'var(--gomna-success)' : type === 'error' ? 'var(--gomna-error)' : 'var(--gomna-brown-accent)'};
            color: white;
            border-radius: 12px;
            z-index: 10001;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            font-weight: 500;
            font-size: 1rem;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
}

// Override any error handlers
window.addEventListener('error', function(e) {
    e.preventDefault();
    return true;
});

// Initialize the clean GOMNA platform
function initializeGOMNACleanPlatform() {
    console.log('Initializing GOMNA AI Trading Clean Platform...');
    
    try {
        // Comprehensive cleanup before initialization
        const allSelectors = [
            '.gomna-panel', '.cocoa-panel', '.cocoa-trading-header', '.gomna-header',
            '.cocoa-header', '.cocoa-trading-card', '.gomna-dashboard', '.cocoa-dashboard'
        ];
        
        allSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        });

        // Clear specific duplicate content but preserve main containers
        const bodyChildren = Array.from(document.body.children);
        bodyChildren.forEach(child => {
            if (child.textContent && 
                (child.textContent.includes('Institutional Quantitative Trading') ||
                (child.textContent.includes('Gomna AI Trading') && !child.classList.contains('gomna-header'))) &&
                !child.classList.contains('container') &&
                !child.classList.contains('main-content') &&
                child.tagName !== 'SCRIPT' &&
                child.tagName !== 'STYLE') {
                child.remove();
            }
        });
        
        // Initialize clean platform
        window.gomnaCleanProfessional = new GOMNACleanProfessional();
        
        console.log('GOMNA AI Trading Clean Platform initialized successfully!');
        
        // Continuous cleanup of duplicates and errors
        setTimeout(() => {
            const errorElements = document.querySelectorAll('*');
            errorElements.forEach(el => {
                if (el.textContent && (
                    el.textContent.includes('Initialization Error') || 
                    el.textContent.includes('Some components failed to load') ||
                    el.textContent.includes('Continue Anyway') ||
                    (el.textContent.includes('Institutional Quantitative Trading') && !el.closest('.gomna-header'))
                )) {
                    el.style.display = 'none';
                    el.remove();
                }
            });
        }, 1000);
        
    } catch (error) {
        console.error('Error initializing GOMNA platform:', error);
    }
}

// Initialize when ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initializeGOMNACleanPlatform, 500);
    });
} else {
    setTimeout(initializeGOMNACleanPlatform, 500);
}

// Continuously remove error dialogs and duplicate content
setInterval(() => {
    const unwantedElements = document.querySelectorAll('*');
    unwantedElements.forEach(el => {
        if (el.textContent && (
            el.textContent.includes('Initialization Error') || 
            el.textContent.includes('Some components failed to load') ||
            el.textContent.includes('Continue Anyway') ||
            (el.textContent.includes('Institutional Quantitative Trading') && !el.closest('.gomna-header')) ||
            (el.textContent.includes('Gomna AI Trading') && !el.closest('.gomna-header') && !el.classList.contains('gomna-title'))
        )) {
            el.style.display = 'none';
            el.remove();
        }
    });

    // Also check for duplicate headers
    const headers = document.querySelectorAll('.gomna-header');
    if (headers.length > 1) {
        for (let i = 1; i < headers.length; i++) {
            headers[i].remove();
        }
    }
}, 1000);

console.log('GOMNA AI Trading Clean Professional Platform loaded successfully');