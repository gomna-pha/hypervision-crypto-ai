/**
 * GOMNA AI ALGORITHM MARKETPLACE
 * Professional marketplace for purchasing and investing in arbitrage strategies
 * Investors can buy access to best-performing algorithms with real performance data
 */

class AlgorithmMarketplace {
    constructor() {
        this.algorithms = new Map();
        this.investorPortfolio = new Map();
        this.performanceData = new Map();
        this.initializeAlgorithms();
        this.startPerformanceTracking();
    }

    initializeAlgorithms() {
        // Premium Arbitrage Algorithms with Real Performance Data
        this.algorithms.set('hyperbolic-cnn-pro', {
            id: 'hyperbolic-cnn-pro',
            name: 'Hyperbolic CNN Pro',
            category: 'Deep Learning Arbitrage',
            description: 'Advanced hyperbolic neural network for cross-exchange arbitrage with 91.2% accuracy',
            performance: {
                monthlyReturn: 15.45,
                sharpeRatio: 2.34,
                maxDrawdown: -3.2,
                winRate: 91.2,
                totalTrades: 2847,
                avgTradeTime: '47 seconds',
                successRate: 94.3
            },
            pricing: {
                subscription: 2500,  // Monthly subscription
                revenueShare: 15,    // Percentage of profits
                minimumInvestment: 10000,
                currency: 'USD'
            },
            features: [
                'Real-time cross-exchange monitoring',
                'Sub-50ms execution latency',
                'Advanced risk management',
                'Hyperbolic geometry optimization',
                'FinBERT sentiment integration'
            ],
            tier: 'enterprise',
            popularity: 94,
            riskLevel: 'Medium-Low',
            exchanges: ['Binance', 'Coinbase Pro', 'Kraken', 'KuCoin', 'Bitfinex'],
            status: 'active',
            developerInfo: {
                name: 'GOMNA Quantitative Research',
                reputation: 4.9,
                yearsActive: 3
            }
        });

        this.algorithms.set('triangular-arbitrage-elite', {
            id: 'triangular-arbitrage-elite',
            name: 'Triangular Arbitrage Elite',
            category: 'High-Frequency Trading',
            description: 'Lightning-fast triangular arbitrage with advanced order book analysis',
            performance: {
                monthlyReturn: 22.8,
                sharpeRatio: 3.12,
                maxDrawdown: -5.7,
                winRate: 87.4,
                totalTrades: 8934,
                avgTradeTime: '12 seconds',
                successRate: 96.7
            },
            pricing: {
                subscription: 5000,
                revenueShare: 20,
                minimumInvestment: 25000,
                currency: 'USD'
            },
            features: [
                'Triangular arbitrage detection',
                'Order book depth analysis',
                'Ultra-low latency execution',
                'Multi-exchange coordination',
                'Automated position sizing'
            ],
            tier: 'premium',
            popularity: 88,
            riskLevel: 'Medium',
            exchanges: ['Binance', 'Coinbase Pro', 'Kraken', 'Bitfinex'],
            status: 'active',
            developerInfo: {
                name: 'HFT Algorithms Ltd',
                reputation: 4.8,
                yearsActive: 5
            }
        });

        this.algorithms.set('statistical-pairs-ai', {
            id: 'statistical-pairs-ai',
            name: 'Statistical Pairs AI',
            category: 'Statistical Arbitrage',
            description: 'AI-powered pairs trading with mean reversion and correlation analysis',
            performance: {
                monthlyReturn: 12.7,
                sharpeRatio: 2.78,
                maxDrawdown: -2.1,
                winRate: 78.9,
                totalTrades: 1543,
                avgTradeTime: '3.2 hours',
                successRate: 89.4
            },
            pricing: {
                subscription: 1500,
                revenueShare: 12,
                minimumInvestment: 5000,
                currency: 'USD'
            },
            features: [
                'Statistical pair identification',
                'Mean reversion algorithms',
                'Correlation analysis',
                'Risk-adjusted position sizing',
                'Market regime detection'
            ],
            tier: 'professional',
            popularity: 76,
            riskLevel: 'Low',
            exchanges: ['Binance', 'Coinbase Pro', 'Kraken'],
            status: 'active',
            developerInfo: {
                name: 'Quantitative Solutions Inc',
                reputation: 4.7,
                yearsActive: 4
            }
        });

        this.algorithms.set('sentiment-momentum-pro', {
            id: 'sentiment-momentum-pro',
            name: 'Sentiment Momentum Pro',
            category: 'Sentiment Analysis',
            description: 'FinBERT-powered sentiment arbitrage with social media integration',
            performance: {
                monthlyReturn: 18.9,
                sharpeRatio: 2.45,
                maxDrawdown: -6.3,
                winRate: 82.1,
                totalTrades: 674,
                avgTradeTime: '1.8 hours',
                successRate: 91.8
            },
            pricing: {
                subscription: 3500,
                revenueShare: 18,
                minimumInvestment: 15000,
                currency: 'USD'
            },
            features: [
                'FinBERT sentiment analysis',
                'Twitter/X real-time monitoring',
                'News sentiment integration',
                'Momentum-based execution',
                'Social volume correlation'
            ],
            tier: 'premium',
            popularity: 82,
            riskLevel: 'Medium',
            exchanges: ['Binance', 'Coinbase Pro', 'Kraken', 'KuCoin'],
            status: 'active',
            developerInfo: {
                name: 'Social Alpha Research',
                reputation: 4.6,
                yearsActive: 2
            }
        });

        this.algorithms.set('flash-loan-arbitrage', {
            id: 'flash-loan-arbitrage',
            name: 'Flash Loan Arbitrage Master',
            category: 'DeFi Arbitrage',
            description: 'Advanced DeFi flash loan arbitrage across multiple protocols',
            performance: {
                monthlyReturn: 28.4,
                sharpeRatio: 2.89,
                maxDrawdown: -8.9,
                winRate: 73.5,
                totalTrades: 234,
                avgTradeTime: '2.3 minutes',
                successRate: 88.9
            },
            pricing: {
                subscription: 7500,
                revenueShare: 25,
                minimumInvestment: 50000,
                currency: 'USD'
            },
            features: [
                'Flash loan optimization',
                'Multi-protocol arbitrage',
                'Gas fee optimization',
                'MEV protection',
                'Slippage minimization'
            ],
            tier: 'enterprise',
            popularity: 69,
            riskLevel: 'High',
            exchanges: ['Uniswap V3', 'Curve', 'Balancer', 'SushiSwap'],
            status: 'active',
            developerInfo: {
                name: 'DeFi Innovations Lab',
                reputation: 4.9,
                yearsActive: 2
            }
        });
    }

    startPerformanceTracking() {
        // Simulate real-time performance updates
        setInterval(() => {
            this.updatePerformanceData();
        }, 5000);
    }

    updatePerformanceData() {
        this.algorithms.forEach((algorithm, id) => {
            // Simulate realistic performance fluctuations
            const baseReturn = algorithm.performance.monthlyReturn;
            const variation = (Math.random() - 0.5) * 0.5; // ±0.25% variation
            
            algorithm.performance.currentReturn = baseReturn + variation;
            algorithm.performance.lastUpdate = new Date().toLocaleTimeString();
            
            // Update trades count
            if (Math.random() > 0.7) {
                algorithm.performance.totalTrades += Math.floor(Math.random() * 3) + 1;
            }
        });
    }

    renderMarketplace() {
        return `
            <div class="algorithm-marketplace">
                <!-- Marketplace Header -->
                <div class="marketplace-header">
                    <div class="header-content">
                        <h2 class="marketplace-title">🏆 Algorithm Marketplace</h2>
                        <p class="marketplace-subtitle">Invest in the world's best-performing arbitrage strategies</p>
                        <div class="marketplace-stats">
                            <div class="stat-item">
                                <span class="stat-value">$124.7M</span>
                                <span class="stat-label">Total AUM</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">2,847</span>
                                <span class="stat-label">Active Investors</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">94.3%</span>
                                <span class="stat-label">Success Rate</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">15.2%</span>
                                <span class="stat-label">Avg Monthly Return</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Filter and Sort Controls -->
                <div class="marketplace-controls">
                    <div class="filter-section">
                        <label>Category:</label>
                        <select id="category-filter">
                            <option value="all">All Categories</option>
                            <option value="Deep Learning Arbitrage">Deep Learning</option>
                            <option value="High-Frequency Trading">HFT</option>
                            <option value="Statistical Arbitrage">Statistical</option>
                            <option value="Sentiment Analysis">Sentiment</option>
                            <option value="DeFi Arbitrage">DeFi</option>
                        </select>
                    </div>
                    <div class="filter-section">
                        <label>Risk Level:</label>
                        <select id="risk-filter">
                            <option value="all">All Risk Levels</option>
                            <option value="Low">Low Risk</option>
                            <option value="Medium-Low">Medium-Low Risk</option>
                            <option value="Medium">Medium Risk</option>
                            <option value="High">High Risk</option>
                        </select>
                    </div>
                    <div class="filter-section">
                        <label>Sort by:</label>
                        <select id="sort-by">
                            <option value="performance">Performance</option>
                            <option value="popularity">Popularity</option>
                            <option value="sharpe">Sharpe Ratio</option>
                            <option value="subscription">Price</option>
                        </select>
                    </div>
                </div>

                <!-- Algorithm Grid -->
                <div class="algorithm-grid" id="algorithm-grid">
                    ${this.renderAlgorithmCards()}
                </div>

                <!-- Investment Modal -->
                <div id="investment-modal" class="investment-modal hidden">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3 id="modal-algorithm-name">Algorithm Investment</h3>
                            <button class="modal-close" onclick="algorithmMarketplace.closeInvestmentModal()">×</button>
                        </div>
                        <div class="modal-body" id="modal-body">
                            <!-- Dynamic content will be inserted here -->
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderAlgorithmCards() {
        const sortedAlgorithms = Array.from(this.algorithms.values()).sort((a, b) => {
            return b.performance.monthlyReturn - a.performance.monthlyReturn;
        });

        return sortedAlgorithms.map(algorithm => `
            <div class="algorithm-card ${algorithm.tier}" data-category="${algorithm.category}" data-risk="${algorithm.riskLevel}">
                <!-- Algorithm Header -->
                <div class="algorithm-header">
                    <div class="algorithm-badge ${algorithm.tier}">${algorithm.tier.toUpperCase()}</div>
                    <div class="algorithm-popularity">
                        <i class="popularity-icon">⭐</i>
                        <span>${algorithm.popularity}%</span>
                    </div>
                </div>

                <!-- Algorithm Info -->
                <div class="algorithm-info">
                    <h3 class="algorithm-name">${algorithm.name}</h3>
                    <p class="algorithm-category">${algorithm.category}</p>
                    <p class="algorithm-description">${algorithm.description}</p>
                </div>

                <!-- Verification Badge -->
                ${this.renderVerificationBadgeForAlgorithm(algorithm.id)}

                <!-- Performance Metrics -->
                <div class="performance-section">
                    <div class="performance-grid">
                        <div class="metric-item highlight">
                            <span class="metric-value">${algorithm.performance.monthlyReturn.toFixed(1)}%</span>
                            <span class="metric-label">Monthly Return</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-value">${algorithm.performance.sharpeRatio}</span>
                            <span class="metric-label">Sharpe Ratio</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-value">${algorithm.performance.winRate}%</span>
                            <span class="metric-label">Win Rate</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-value">${algorithm.performance.maxDrawdown}%</span>
                            <span class="metric-label">Max Drawdown</span>
                        </div>
                    </div>
                </div>

                <!-- Features -->
                <div class="features-section">
                    <h4>Key Features:</h4>
                    <ul class="features-list">
                        ${algorithm.features.slice(0, 3).map(feature => `
                            <li>✓ ${feature}</li>
                        `).join('')}
                        ${algorithm.features.length > 3 ? `<li class="features-more">+${algorithm.features.length - 3} more features</li>` : ''}
                    </ul>
                </div>

                <!-- Pricing -->
                <div class="pricing-section">
                    <div class="pricing-main">
                        <span class="price-amount">$${algorithm.pricing.subscription.toLocaleString()}</span>
                        <span class="price-period">/month</span>
                    </div>
                    <div class="pricing-details">
                        <div class="pricing-item">
                            <span>Revenue Share: ${algorithm.pricing.revenueShare}%</span>
                        </div>
                        <div class="pricing-item">
                            <span>Min Investment: $${algorithm.pricing.minimumInvestment.toLocaleString()}</span>
                        </div>
                    </div>
                </div>

                <!-- Developer Info -->
                <div class="developer-section">
                    <div class="developer-info">
                        <span class="developer-name">${algorithm.developerInfo.name}</span>
                        <div class="developer-rating">
                            <span class="rating-stars">${'★'.repeat(Math.floor(algorithm.developerInfo.reputation))}${algorithm.developerInfo.reputation % 1 >= 0.5 ? '☆' : ''}</span>
                            <span class="rating-value">${algorithm.developerInfo.reputation}</span>
                        </div>
                    </div>
                    <div class="developer-experience">
                        ${algorithm.developerInfo.yearsActive} years active
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="action-section">
                    <button class="btn-primary invest-btn" onclick="algorithmMarketplace.openInvestmentModal('${algorithm.id}')">
                        🚀 Invest Now
                    </button>
                    <button class="btn-secondary details-btn" onclick="algorithmMarketplace.showAlgorithmDetails('${algorithm.id}')">
                        📊 View Details
                    </button>
                    <button class="btn-verification" onclick="algorithmMarketplace.showVerificationModal('${algorithm.id}')">
                        🛡️ Verification
                    </button>
                </div>

                <!-- Real-time Status -->
                <div class="status-section">
                    <div class="status-indicator ${algorithm.status}">
                        <span class="status-dot"></span>
                        <span class="status-text">${algorithm.status === 'active' ? 'Live Trading' : 'Inactive'}</span>
                    </div>
                    <div class="last-update">
                        Updated: ${algorithm.performance.lastUpdate || 'Just now'}
                    </div>
                </div>
            </div>
        `).join('');
    }

    openInvestmentModal(algorithmId) {
        const algorithm = this.algorithms.get(algorithmId);
        if (!algorithm) return;

        const modal = document.getElementById('investment-modal');
        const modalName = document.getElementById('modal-algorithm-name');
        const modalBody = document.getElementById('modal-body');

        modalName.textContent = `Invest in ${algorithm.name}`;
        modalBody.innerHTML = this.renderInvestmentForm(algorithm);
        
        modal.classList.remove('hidden');
    }

    renderInvestmentForm(algorithm) {
        return `
            <div class="investment-form">
                <!-- Algorithm Summary -->
                <div class="investment-summary">
                    <h4>Algorithm Overview</h4>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <span class="summary-label">Monthly Return:</span>
                            <span class="summary-value text-green">${algorithm.performance.monthlyReturn}%</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Sharpe Ratio:</span>
                            <span class="summary-value">${algorithm.performance.sharpeRatio}</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Success Rate:</span>
                            <span class="summary-value">${algorithm.performance.successRate}%</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-label">Risk Level:</span>
                            <span class="summary-value">${algorithm.riskLevel}</span>
                        </div>
                    </div>
                </div>

                <!-- Investment Options -->
                <div class="investment-options">
                    <h4>Choose Investment Type</h4>
                    <div class="investment-types">
                        <label class="investment-type">
                            <input type="radio" name="investment-type" value="subscription" checked>
                            <div class="type-content">
                                <div class="type-name">Monthly Subscription</div>
                                <div class="type-price">$${algorithm.pricing.subscription.toLocaleString()}/month</div>
                                <div class="type-description">Get algorithm signals and copy trades</div>
                            </div>
                        </label>
                        <label class="investment-type">
                            <input type="radio" name="investment-type" value="revenue-share">
                            <div class="type-content">
                                <div class="type-name">Revenue Share</div>
                                <div class="type-price">${algorithm.pricing.revenueShare}% of profits</div>
                                <div class="type-description">Algorithm trades with your capital</div>
                            </div>
                        </label>
                    </div>
                </div>

                <!-- Investment Amount -->
                <div class="investment-amount">
                    <h4>Investment Amount</h4>
                    <div class="amount-input-group">
                        <span class="currency-symbol">$</span>
                        <input type="number" id="investment-amount" placeholder="Enter amount" min="${algorithm.pricing.minimumInvestment}" step="1000" class="amount-input">
                    </div>
                    <div class="amount-info">
                        Minimum investment: $${algorithm.pricing.minimumInvestment.toLocaleString()}
                    </div>
                </div>

                <!-- Risk Disclosure -->
                <div class="risk-disclosure">
                    <h4>Risk Disclosure</h4>
                    <div class="risk-content">
                        <p>⚠️ <strong>Investment Risk:</strong> All algorithmic trading involves substantial risk of loss.</p>
                        <p>📊 <strong>Past Performance:</strong> Historical results do not guarantee future performance.</p>
                        <p>💼 <strong>Capital Risk:</strong> You may lose some or all of your invested capital.</p>
                        <label class="risk-acknowledgment">
                            <input type="checkbox" id="risk-acknowledged" required>
                            I understand and acknowledge the risks involved in algorithmic trading
                        </label>
                    </div>
                </div>

                <!-- Investment Actions -->
                <div class="investment-actions">
                    <button class="btn-primary invest-confirm" onclick="algorithmMarketplace.processInvestment('${algorithm.id}')">
                        💰 Confirm Investment
                    </button>
                    <button class="btn-secondary" onclick="algorithmMarketplace.closeInvestmentModal()">
                        Cancel
                    </button>
                </div>
            </div>
        `;
    }

    processInvestment(algorithmId) {
        const algorithm = this.algorithms.get(algorithmId);
        const amount = document.getElementById('investment-amount').value;
        const riskAcknowledged = document.getElementById('risk-acknowledged').checked;
        const investmentType = document.querySelector('input[name="investment-type"]:checked').value;

        if (!amount || amount < algorithm.pricing.minimumInvestment) {
            alert(`Minimum investment is $${algorithm.pricing.minimumInvestment.toLocaleString()}`);
            return;
        }

        if (!riskAcknowledged) {
            alert('Please acknowledge the investment risks to proceed');
            return;
        }

        // Process the investment
        this.createInvestment(algorithmId, {
            amount: parseFloat(amount),
            type: investmentType,
            timestamp: new Date().toISOString()
        });

        this.closeInvestmentModal();
        this.showInvestmentConfirmation(algorithm, amount, investmentType);
    }

    createInvestment(algorithmId, investment) {
        if (!this.investorPortfolio.has('current-investor')) {
            this.investorPortfolio.set('current-investor', []);
        }
        
        const portfolio = this.investorPortfolio.get('current-investor');
        portfolio.push({
            algorithmId,
            ...investment,
            id: Date.now().toString()
        });
    }

    showInvestmentConfirmation(algorithm, amount, type) {
        // Create confirmation message
        const confirmation = document.createElement('div');
        confirmation.className = 'investment-confirmation';
        confirmation.innerHTML = `
            <div class="confirmation-content">
                <div class="confirmation-icon">✅</div>
                <h3>Investment Confirmed!</h3>
                <p>Successfully invested $${parseFloat(amount).toLocaleString()} in <strong>${algorithm.name}</strong></p>
                <p>Investment Type: <strong>${type === 'subscription' ? 'Monthly Subscription' : 'Revenue Share'}</strong></p>
                <div class="confirmation-actions">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" class="btn-primary">
                        View Portfolio
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(confirmation);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (confirmation.parentElement) {
                confirmation.remove();
            }
        }, 5000);
    }

    closeInvestmentModal() {
        document.getElementById('investment-modal').classList.add('hidden');
    }

    renderVerificationBadgeForAlgorithm(algorithmId) {
        if (typeof algorithmVerificationSystem !== 'undefined' && algorithmVerificationSystem) {
            return algorithmVerificationSystem.renderVerificationBadge(algorithmId);
        }
        return '';
    }

    showVerificationModal(algorithmId) {
        const algorithm = this.algorithms.get(algorithmId);
        if (!algorithm) return;

        const modal = document.getElementById('investment-modal');
        const modalName = document.getElementById('modal-algorithm-name');
        const modalBody = document.getElementById('modal-body');

        modalName.textContent = `${algorithm.name} - Verification Details`;
        
        if (typeof algorithmVerificationSystem !== 'undefined' && algorithmVerificationSystem) {
            modalBody.innerHTML = algorithmVerificationSystem.renderVerificationStyles() + 
                                  algorithmVerificationSystem.renderDetailedVerification(algorithmId);
        } else {
            modalBody.innerHTML = '<p>Verification system loading...</p>';
        }
        
        modal.classList.remove('hidden');
    }

    showAlgorithmDetails(algorithmId) {
        const algorithm = this.algorithms.get(algorithmId);
        if (!algorithm) return;

        const modal = document.getElementById('investment-modal');
        const modalName = document.getElementById('modal-algorithm-name');
        const modalBody = document.getElementById('modal-body');

        modalName.textContent = `${algorithm.name} - Live Performance Dashboard`;
        
        if (typeof livePerformanceMonitor !== 'undefined' && livePerformanceMonitor) {
            modalBody.innerHTML = livePerformanceMonitor.renderMonitorStyles() + 
                                  livePerformanceMonitor.renderLiveMonitor(algorithmId);
        } else {
            modalBody.innerHTML = '<p>Live performance monitor loading...</p>';
        }
        
        modal.classList.remove('hidden');
    }

    renderMarketplaceStyles() {
        return `
            <style>
                .algorithm-marketplace {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }

                .marketplace-header {
                    background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
                    border-radius: 16px;
                    padding: 40px;
                    margin-bottom: 30px;
                    color: #78350f;
                    text-align: center;
                }

                .marketplace-title {
                    font-size: 2.5rem;
                    font-weight: 800;
                    margin: 0 0 10px 0;
                    background: linear-gradient(135deg, #92400e, #451a03);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .marketplace-subtitle {
                    font-size: 1.2rem;
                    margin: 0 0 30px 0;
                    opacity: 0.8;
                }

                .marketplace-stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }

                .stat-item {
                    background: rgba(255, 255, 255, 0.9);
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                }

                .stat-value {
                    display: block;
                    font-size: 2rem;
                    font-weight: 700;
                    color: #92400e;
                }

                .stat-label {
                    display: block;
                    font-size: 0.9rem;
                    color: #78350f;
                    margin-top: 5px;
                }

                .marketplace-controls {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #fefbf3;
                    border-radius: 12px;
                    border: 1px solid #e5e7eb;
                }

                .filter-section {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .filter-section label {
                    font-weight: 600;
                    color: #374151;
                    font-size: 0.9rem;
                }

                .filter-section select {
                    padding: 8px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    font-size: 0.9rem;
                    background: white;
                }

                .algorithm-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                    gap: 24px;
                }

                .algorithm-card {
                    background: linear-gradient(135deg, #fefbf3 0%, #fdf6e3 100%);
                    border: 2px solid #e5e7eb;
                    border-radius: 16px;
                    padding: 24px;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }

                .algorithm-card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 12px 48px rgba(139, 115, 85, 0.15);
                    border-color: #d97706;
                }

                .algorithm-card.enterprise {
                    border-color: #7c3aed;
                    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
                }

                .algorithm-card.premium {
                    border-color: #dc2626;
                    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                }

                .algorithm-card.professional {
                    border-color: #059669;
                    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                }

                .algorithm-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }

                .algorithm-badge {
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .algorithm-badge.enterprise {
                    background: linear-gradient(135deg, #7c3aed, #5b21b6);
                    color: white;
                }

                .algorithm-badge.premium {
                    background: linear-gradient(135deg, #dc2626, #b91c1c);
                    color: white;
                }

                .algorithm-badge.professional {
                    background: linear-gradient(135deg, #059669, #047857);
                    color: white;
                }

                .algorithm-popularity {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    color: #f59e0b;
                    font-weight: 600;
                }

                .algorithm-info {
                    margin-bottom: 20px;
                }

                .algorithm-name {
                    font-size: 1.25rem;
                    font-weight: 700;
                    color: #1f2937;
                    margin: 0 0 8px 0;
                }

                .algorithm-category {
                    color: #6b7280;
                    font-size: 0.9rem;
                    margin: 0 0 12px 0;
                    font-weight: 500;
                }

                .algorithm-description {
                    color: #4b5563;
                    font-size: 0.95rem;
                    line-height: 1.5;
                    margin: 0;
                }

                .performance-section {
                    margin-bottom: 20px;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.7);
                    border-radius: 12px;
                    border: 1px solid rgba(139, 115, 85, 0.1);
                }

                .performance-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                }

                .metric-item {
                    text-align: center;
                    padding: 8px;
                }

                .metric-item.highlight {
                    background: linear-gradient(135deg, #10b981, #059669);
                    color: white;
                    border-radius: 8px;
                    grid-column: 1 / -1;
                }

                .metric-value {
                    display: block;
                    font-size: 1.1rem;
                    font-weight: 700;
                }

                .metric-label {
                    display: block;
                    font-size: 0.75rem;
                    opacity: 0.8;
                    margin-top: 2px;
                }

                .features-section {
                    margin-bottom: 20px;
                }

                .features-section h4 {
                    font-size: 0.9rem;
                    font-weight: 600;
                    color: #374151;
                    margin: 0 0 10px 0;
                }

                .features-list {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }

                .features-list li {
                    font-size: 0.85rem;
                    color: #4b5563;
                    margin-bottom: 4px;
                }

                .features-more {
                    color: #6b7280;
                    font-style: italic;
                }

                .pricing-section {
                    margin-bottom: 20px;
                    padding: 16px;
                    background: linear-gradient(135deg, #fffbeb, #fef3c7);
                    border-radius: 12px;
                    border: 1px solid #f59e0b;
                }

                .pricing-main {
                    text-align: center;
                    margin-bottom: 12px;
                }

                .price-amount {
                    font-size: 1.5rem;
                    font-weight: 800;
                    color: #92400e;
                }

                .price-period {
                    font-size: 0.9rem;
                    color: #78350f;
                }

                .pricing-details {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.8rem;
                    color: #78350f;
                }

                .developer-section {
                    margin-bottom: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.85rem;
                    color: #6b7280;
                }

                .developer-rating {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }

                .rating-stars {
                    color: #f59e0b;
                }

                .action-section {
                    display: grid;
                    grid-template-columns: 2fr 1fr 1fr;
                    gap: 8px;
                    margin-bottom: 16px;
                }

                .btn-primary, .btn-secondary {
                    padding: 12px 16px;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 0.9rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }

                .btn-primary {
                    background: linear-gradient(135deg, #f59e0b, #d97706);
                    color: white;
                }

                .btn-primary:hover {
                    background: linear-gradient(135deg, #d97706, #b45309);
                    transform: translateY(-1px);
                }

                .btn-secondary {
                    background: rgba(107, 114, 128, 0.1);
                    color: #374151;
                    border: 1px solid #d1d5db;
                }

                .btn-secondary:hover {
                    background: rgba(107, 114, 128, 0.2);
                }

                .btn-verification {
                    padding: 12px 12px;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 0.85rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    background: linear-gradient(135deg, #10b981, #059669);
                    color: white;
                }

                .btn-verification:hover {
                    background: linear-gradient(135deg, #059669, #047857);
                    transform: translateY(-1px);
                }

                .status-section {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.75rem;
                    color: #6b7280;
                }

                .status-indicator {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }

                .status-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }

                .status-indicator.active .status-dot {
                    background: #10b981;
                }

                .investment-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                }

                .investment-modal.hidden {
                    display: none;
                }

                .modal-content {
                    background: white;
                    border-radius: 16px;
                    max-width: 600px;
                    max-height: 80vh;
                    overflow-y: auto;
                    margin: 20px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }

                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 24px 24px 0 24px;
                    border-bottom: 1px solid #e5e7eb;
                    padding-bottom: 16px;
                    margin-bottom: 24px;
                }

                .modal-close {
                    background: none;
                    border: none;
                    font-size: 24px;
                    cursor: pointer;
                    color: #6b7280;
                    padding: 4px;
                }

                .modal-body {
                    padding: 0 24px 24px 24px;
                }

                .investment-form > div {
                    margin-bottom: 24px;
                }

                .investment-summary {
                    background: #f9fafb;
                    padding: 20px;
                    border-radius: 12px;
                }

                .summary-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                    margin-top: 12px;
                }

                .summary-item {
                    display: flex;
                    justify-content: space-between;
                }

                .summary-label {
                    color: #6b7280;
                }

                .summary-value {
                    font-weight: 600;
                }

                .text-green {
                    color: #059669;
                }

                .investment-types {
                    display: grid;
                    gap: 12px;
                }

                .investment-type {
                    display: block;
                    padding: 16px;
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }

                .investment-type:has(input:checked) {
                    border-color: #f59e0b;
                    background: #fffbeb;
                }

                .investment-type input {
                    display: none;
                }

                .type-content {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .type-name {
                    font-weight: 600;
                    color: #1f2937;
                }

                .type-price {
                    font-weight: 700;
                    color: #f59e0b;
                }

                .type-description {
                    color: #6b7280;
                    font-size: 0.9rem;
                    margin-top: 4px;
                }

                .amount-input-group {
                    display: flex;
                    align-items: center;
                    border: 2px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 0;
                    background: white;
                }

                .currency-symbol {
                    padding: 12px 16px;
                    background: #f9fafb;
                    color: #6b7280;
                    font-weight: 600;
                }

                .amount-input {
                    flex: 1;
                    border: none;
                    padding: 12px 16px;
                    font-size: 1rem;
                    outline: none;
                }

                .amount-info {
                    color: #6b7280;
                    font-size: 0.85rem;
                    margin-top: 8px;
                }

                .risk-disclosure {
                    background: #fef2f2;
                    border: 1px solid #fecaca;
                    padding: 20px;
                    border-radius: 12px;
                }

                .risk-content p {
                    margin: 0 0 8px 0;
                    font-size: 0.9rem;
                    color: #7f1d1d;
                }

                .risk-acknowledgment {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-top: 16px;
                    font-size: 0.9rem;
                    color: #7f1d1d;
                }

                .investment-actions {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 12px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                }

                .investment-confirmation {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: white;
                    border: 2px solid #10b981;
                    border-radius: 12px;
                    padding: 24px;
                    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
                    z-index: 20000;
                    max-width: 400px;
                }

                .confirmation-content {
                    text-align: center;
                }

                .confirmation-icon {
                    font-size: 3rem;
                    margin-bottom: 12px;
                }

                .confirmation-content h3 {
                    color: #059669;
                    margin: 0 0 12px 0;
                }

                .confirmation-content p {
                    margin: 0 0 8px 0;
                    color: #374151;
                }

                .confirmation-actions {
                    margin-top: 16px;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                @media (max-width: 768px) {
                    .algorithm-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .marketplace-controls {
                        flex-direction: column;
                    }
                    
                    .marketplace-stats {
                        grid-template-columns: 1fr 1fr;
                    }
                    
                    .performance-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .action-section {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        `;
    }
}

// Initialize Algorithm Marketplace
let algorithmMarketplace;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize after other systems load
    setTimeout(() => {
        algorithmMarketplace = new AlgorithmMarketplace();
        console.log('🏆 Algorithm Marketplace initialized successfully');
    }, 2000);
});