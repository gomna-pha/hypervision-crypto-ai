/**
 * Cocoa Trading AI - Tiered Account Structure System
 * Implements professional account tiers (Starter/Professional/Institutional)
 * 
 * Features:
 * - Multi-tier account management (Starter, Professional, Institutional)
 * - Dynamic feature access and limits based on tier
 * - Professional upgrade pathways and benefits
 * - Real-time tier status and usage monitoring
 */

class TieredAccountSystem {
    constructor() {
        this.currentTier = 'starter';
        this.accountData = null;
        this.tierConfigs = this.initializeTierConfigs();
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        try {
            console.log('🏦 Initializing Tiered Account System...');
            
            await this.loadAccountData();
            this.createTierInterface();
            this.setupTierMonitoring();
            this.initializeTierBenefits();
            
            this.isInitialized = true;
            console.log('✅ Tiered Account System initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing tiered account system:', error);
        }
    }

    initializeTierConfigs() {
        return {
            starter: {
                name: 'Starter Account',
                price: '$0/month',
                deposit: '$1,000 minimum',
                maxTrades: 100,
                leverage: '1:2',
                features: [
                    'Basic arbitrage strategies',
                    'Real-time market data',
                    'Standard execution speed',
                    'Email support',
                    'Basic analytics dashboard',
                    'Mobile app access'
                ],
                restrictions: [
                    'Limited to 100 trades/month',
                    'Standard execution priority',
                    'Basic risk management tools'
                ],
                color: '#10B981',
                icon: '🚀',
                recommended: false
            },
            professional: {
                name: 'Professional Account',
                price: '$299/month',
                deposit: '$25,000 minimum',
                maxTrades: 1000,
                leverage: '1:5',
                features: [
                    'Advanced arbitrage strategies',
                    'Priority execution speed',
                    'Custom algorithm parameters',
                    'Advanced risk management',
                    'Professional analytics suite',
                    'Phone + chat support',
                    'API access',
                    'Custom backtesting',
                    'Performance reporting',
                    'Tax optimization tools'
                ],
                restrictions: [
                    'Limited to 1,000 trades/month',
                    'Priority execution queue'
                ],
                color: '#3B82F6',
                icon: '⚡',
                recommended: true
            },
            institutional: {
                name: 'Institutional Account',
                price: 'Custom Pricing',
                deposit: '$500,000 minimum',
                maxTrades: 'Unlimited',
                leverage: '1:10',
                features: [
                    'Full HFT arbitrage suite',
                    'Ultra-low latency execution',
                    'Colocation services',
                    'Custom algorithm development',
                    'Dedicated relationship manager',
                    '24/7 premium support',
                    'Full API suite',
                    'Custom reporting',
                    'Regulatory compliance support',
                    'White-label solutions',
                    'Direct market access',
                    'Prime brokerage integration'
                ],
                restrictions: [
                    'None - unlimited access'
                ],
                color: '#8B5CF6',
                icon: '🏛️',
                recommended: false
            }
        };
    }

    async loadAccountData() {
        // Simulate loading account data (in production, this would be an API call)
        return new Promise((resolve) => {
            setTimeout(() => {
                this.accountData = {
                    userId: 'user_12345',
                    currentTier: localStorage.getItem('userTier') || 'starter',
                    accountBalance: 15000,
                    monthlyTrades: 45,
                    joinDate: new Date(2024, 0, 15),
                    kycStatus: 'verified',
                    features: this.tierConfigs[this.currentTier].features
                };
                this.currentTier = this.accountData.currentTier;
                resolve(this.accountData);
            }, 500);
        });
    }

    createTierInterface() {
        // Create tier selection and upgrade interface
        const tierContainer = document.createElement('div');
        tierContainer.id = 'tier-account-system';
        tierContainer.className = 'cocoa-panel cocoa-fade-in';
        
        tierContainer.innerHTML = `
            <div class="cocoa-panel-header">
                <h3>🏦 Account Tier Management</h3>
                <div class="tier-current-status">
                    Current: <span class="tier-badge tier-${this.currentTier}">${this.tierConfigs[this.currentTier].icon} ${this.tierConfigs[this.currentTier].name}</span>
                </div>
            </div>
            <div class="cocoa-panel-content">
                <div class="tier-overview">
                    ${this.createTierOverview()}
                </div>
                <div class="tier-comparison">
                    ${this.createTierComparison()}
                </div>
                <div class="tier-upgrade-section">
                    ${this.createUpgradeSection()}
                </div>
            </div>
        `;

        // Add tier-specific styles
        const tierStyles = document.createElement('style');
        tierStyles.textContent = `
            .tier-account-system {
                margin: 20px 0;
            }

            .tier-current-status {
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 600;
            }

            .tier-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }

            .tier-starter {
                background: rgba(16, 185, 129, 0.2);
                color: #10B981;
                border: 1px solid #10B981;
            }

            .tier-professional {
                background: rgba(59, 130, 246, 0.2);
                color: #3B82F6;
                border: 1px solid #3B82F6;
            }

            .tier-institutional {
                background: rgba(139, 92, 246, 0.2);
                color: #8B5CF6;
                border: 1px solid #8B5CF6;
            }

            .tier-comparison-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }

            .tier-card {
                background: rgba(26, 26, 26, 0.9);
                border: 2px solid transparent;
                border-radius: 16px;
                padding: 24px;
                position: relative;
                transition: all 0.3s ease;
            }

            .tier-card.current {
                border-color: var(--cocoa-accent);
                box-shadow: 0 8px 32px rgba(255, 215, 0, 0.2);
            }

            .tier-card.recommended::before {
                content: "MOST POPULAR";
                position: absolute;
                top: -10px;
                left: 50%;
                transform: translateX(-50%);
                background: linear-gradient(135deg, var(--cocoa-accent), #FFA500);
                color: var(--cocoa-bg);
                padding: 4px 16px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.5px;
            }

            .tier-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(139, 69, 19, 0.3);
            }

            .tier-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            }

            .tier-name {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--cocoa-text);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .tier-price {
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--cocoa-accent);
            }

            .tier-deposit {
                color: var(--cocoa-secondary);
                font-size: 0.9rem;
                margin-bottom: 20px;
            }

            .tier-features {
                list-style: none;
                padding: 0;
                margin-bottom: 20px;
            }

            .tier-features li {
                padding: 8px 0;
                color: var(--cocoa-text);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .tier-features li::before {
                content: "✓";
                color: var(--cocoa-success);
                font-weight: bold;
                font-size: 1.1rem;
            }

            .tier-restrictions {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid rgba(212, 165, 116, 0.2);
            }

            .tier-restrictions li {
                color: var(--cocoa-warning);
            }

            .tier-restrictions li::before {
                content: "!";
                color: var(--cocoa-warning);
            }

            .tier-action-btn {
                width: 100%;
                padding: 14px;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 1.1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .tier-action-current {
                background: rgba(16, 185, 129, 0.2);
                color: var(--cocoa-success);
                border: 2px solid var(--cocoa-success);
                cursor: default;
            }

            .tier-action-upgrade {
                background: linear-gradient(135deg, var(--cocoa-primary), var(--cocoa-secondary));
                color: white;
                box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
            }

            .tier-action-upgrade:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
            }

            .usage-stats {
                background: rgba(139, 69, 19, 0.1);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }

            .usage-stat {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 10px 0;
            }

            .usage-bar {
                width: 100px;
                height: 8px;
                background: rgba(139, 69, 19, 0.2);
                border-radius: 4px;
                overflow: hidden;
            }

            .usage-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--cocoa-success), var(--cocoa-accent));
                transition: width 0.3s ease;
            }

            @media (max-width: 768px) {
                .tier-comparison-grid {
                    grid-template-columns: 1fr;
                }
                
                .tier-header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 10px;
                }
            }
        `;
        
        document.head.appendChild(tierStyles);

        // Insert into the main content area
        const targetContainer = document.querySelector('.container, .main-content') || document.body;
        const marketplaceTab = document.querySelector('#marketplace-tab') || targetContainer.querySelector('[data-tab="marketplace"]');
        
        if (marketplaceTab) {
            marketplaceTab.appendChild(tierContainer);
        } else {
            targetContainer.appendChild(tierContainer);
        }

        console.log('✅ Tier interface created');
    }

    createTierOverview() {
        const currentConfig = this.tierConfigs[this.currentTier];
        const usagePercent = (this.accountData.monthlyTrades / currentConfig.maxTrades) * 100;
        
        return `
            <div class="tier-overview-section">
                <h4 class="cocoa-heading-3">Current Account Status</h4>
                <div class="usage-stats">
                    <div class="usage-stat">
                        <span>Monthly Trades Used</span>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span>${this.accountData.monthlyTrades} / ${currentConfig.maxTrades}</span>
                            <div class="usage-bar">
                                <div class="usage-fill" style="width: ${Math.min(usagePercent, 100)}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="usage-stat">
                        <span>Account Balance</span>
                        <span style="color: var(--cocoa-success); font-weight: 600;">$${this.accountData.accountBalance.toLocaleString()}</span>
                    </div>
                    <div class="usage-stat">
                        <span>Current Leverage</span>
                        <span style="color: var(--cocoa-accent); font-weight: 600;">${currentConfig.leverage}</span>
                    </div>
                    <div class="usage-stat">
                        <span>KYC Status</span>
                        <span style="color: var(--cocoa-success); font-weight: 600;">✓ Verified</span>
                    </div>
                </div>
            </div>
        `;
    }

    createTierComparison() {
        return `
            <h4 class="cocoa-heading-3">Choose Your Account Tier</h4>
            <div class="tier-comparison-grid">
                ${Object.keys(this.tierConfigs).map(tier => this.createTierCard(tier)).join('')}
            </div>
        `;
    }

    createTierCard(tierKey) {
        const config = this.tierConfigs[tierKey];
        const isCurrent = tierKey === this.currentTier;
        const isRecommended = config.recommended;
        
        return `
            <div class="tier-card ${isCurrent ? 'current' : ''} ${isRecommended ? 'recommended' : ''}" data-tier="${tierKey}">
                <div class="tier-header">
                    <div class="tier-name">
                        <span style="font-size: 1.8rem;">${config.icon}</span>
                        ${config.name}
                    </div>
                    <div class="tier-price">${config.price}</div>
                </div>
                <div class="tier-deposit">Minimum Deposit: ${config.deposit}</div>
                <ul class="tier-features">
                    ${config.features.map(feature => `<li>${feature}</li>`).join('')}
                </ul>
                ${config.restrictions.length > 0 ? `
                    <ul class="tier-features tier-restrictions">
                        ${config.restrictions.map(restriction => `<li>${restriction}</li>`).join('')}
                    </ul>
                ` : ''}
                <button class="tier-action-btn ${isCurrent ? 'tier-action-current' : 'tier-action-upgrade'}"
                        onclick="window.tieredAccountSystem.${isCurrent ? 'showCurrentTierInfo' : 'upgradeToTier'}('${tierKey}')">
                    ${isCurrent ? 'Current Plan' : 'Upgrade Now'}
                </button>
            </div>
        `;
    }

    createUpgradeSection() {
        if (this.currentTier === 'institutional') {
            return `
                <div class="upgrade-section">
                    <h4 class="cocoa-heading-3">🏛️ You're on our highest tier!</h4>
                    <p>Enjoy unlimited access to all Cocoa Trading AI features and premium support.</p>
                </div>
            `;
        }

        const nextTier = this.currentTier === 'starter' ? 'professional' : 'institutional';
        const nextConfig = this.tierConfigs[nextTier];

        return `
            <div class="upgrade-section">
                <h4 class="cocoa-heading-3">🚀 Ready to Upgrade?</h4>
                <div class="upgrade-benefits">
                    <p>Upgrade to <strong>${nextConfig.name}</strong> and unlock:</p>
                    <ul class="tier-features">
                        ${nextConfig.features.slice(-3).map(feature => `<li>${feature}</li>`).join('')}
                    </ul>
                    <button class="cocoa-btn-primary" onclick="window.tieredAccountSystem.upgradeToTier('${nextTier}')" style="margin-top: 15px;">
                        Upgrade to ${nextConfig.name} - ${nextConfig.price}
                    </button>
                </div>
            </div>
        `;
    }

    setupTierMonitoring() {
        // Monitor usage and send alerts when approaching limits
        setInterval(() => {
            this.checkUsageLimits();
        }, 30000); // Check every 30 seconds

        console.log('✅ Tier monitoring setup complete');
    }

    checkUsageLimits() {
        const currentConfig = this.tierConfigs[this.currentTier];
        const usagePercent = (this.accountData.monthlyTrades / currentConfig.maxTrades) * 100;

        if (usagePercent >= 90) {
            this.showUpgradeNotification('You\'re approaching your monthly trade limit. Consider upgrading to continue trading.');
        } else if (usagePercent >= 75) {
            this.showUpgradeNotification('You\'ve used 75% of your monthly trades. Time to consider an upgrade?');
        }
    }

    showUpgradeNotification(message) {
        // Create upgrade notification
        const notification = document.createElement('div');
        notification.className = 'tier-upgrade-notification cocoa-fade-in';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--cocoa-primary), var(--cocoa-secondary));
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(139, 69, 19, 0.4);
            z-index: 10000;
            max-width: 350px;
            font-weight: 500;
        `;
        notification.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <div style="font-weight: 600; margin-bottom: 5px;">📈 Upgrade Available</div>
                    <div>${message}</div>
                </div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; color: white; font-size: 18px; cursor: pointer;">×</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 10000);
    }

    initializeTierBenefits() {
        // Apply tier-specific benefits and restrictions
        this.applyTierRestrictions();
        this.enableTierFeatures();
        
        console.log('✅ Tier benefits initialized for:', this.currentTier);
    }

    applyTierRestrictions() {
        const currentConfig = this.tierConfigs[this.currentTier];
        
        // Apply trade limits
        if (this.accountData.monthlyTrades >= currentConfig.maxTrades && currentConfig.maxTrades !== 'Unlimited') {
            this.disableTradingButtons();
        }
        
        // Apply leverage limits
        document.querySelectorAll('[data-leverage]').forEach(element => {
            const leverage = parseFloat(element.dataset.leverage.replace('1:', ''));
            const maxLeverage = parseFloat(currentConfig.leverage.replace('1:', ''));
            
            if (leverage > maxLeverage) {
                element.disabled = true;
                element.title = `Requires ${this.getNextTier()} account or higher`;
            }
        });
    }

    enableTierFeatures() {
        const currentConfig = this.tierConfigs[this.currentTier];
        
        // Enable/disable features based on tier
        if (!currentConfig.features.includes('API access')) {
            document.querySelectorAll('[data-feature="api"]').forEach(element => {
                element.style.opacity = '0.5';
                element.disabled = true;
            });
        }
        
        if (!currentConfig.features.includes('Custom backtesting')) {
            document.querySelectorAll('[data-feature="backtesting"]').forEach(element => {
                element.style.opacity = '0.5';
                element.disabled = true;
            });
        }
    }

    disableTradingButtons() {
        document.querySelectorAll('[data-action="trade"], .trade-button').forEach(button => {
            button.disabled = true;
            button.textContent = 'Upgrade Required';
            button.onclick = () => this.showUpgradeModal();
        });
    }

    upgradeToTier(targetTier) {
        const targetConfig = this.tierConfigs[targetTier];
        
        // Show upgrade modal
        this.showUpgradeModal(targetTier);
    }

    showUpgradeModal(targetTier = null) {
        const tier = targetTier || this.getNextTier();
        const config = this.tierConfigs[tier];
        
        const modal = document.createElement('div');
        modal.className = 'tier-upgrade-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        modal.innerHTML = `
            <div class="modal-content" style="
                background: var(--cocoa-bg);
                border: 2px solid var(--cocoa-secondary);
                border-radius: 16px;
                padding: 30px;
                max-width: 500px;
                width: 90%;
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 20px;">${config.icon}</div>
                <h3 class="cocoa-heading-2">Upgrade to ${config.name}</h3>
                <p style="color: var(--cocoa-text); margin: 20px 0;">
                    Unlock advanced trading features and higher limits with ${config.name}.
                </p>
                <div style="background: rgba(139, 69, 19, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
                    <div style="color: var(--cocoa-accent); font-size: 1.5rem; font-weight: 700;">${config.price}</div>
                    <div style="color: var(--cocoa-secondary); margin-top: 5px;">${config.deposit}</div>
                </div>
                <div style="display: flex; gap: 15px; justify-content: center;">
                    <button class="cocoa-btn-secondary" onclick="this.closest('.tier-upgrade-modal').remove()">
                        Maybe Later
                    </button>
                    <button class="cocoa-btn-primary" onclick="window.tieredAccountSystem.processUpgrade('${tier}')">
                        Upgrade Now
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    processUpgrade(targetTier) {
        // Simulate upgrade process (in production, this would handle payment)
        console.log(`Processing upgrade to ${targetTier}...`);
        
        // Update local state
        this.currentTier = targetTier;
        this.accountData.currentTier = targetTier;
        localStorage.setItem('userTier', targetTier);
        
        // Close modal
        document.querySelector('.tier-upgrade-modal')?.remove();
        
        // Show success message
        this.showUpgradeSuccess(targetTier);
        
        // Refresh interface
        setTimeout(() => {
            location.reload();
        }, 3000);
    }

    showUpgradeSuccess(tier) {
        const config = this.tierConfigs[tier];
        const successModal = document.createElement('div');
        successModal.className = 'tier-success-modal';
        successModal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        successModal.innerHTML = `
            <div class="modal-content" style="
                background: var(--cocoa-bg);
                border: 2px solid var(--cocoa-success);
                border-radius: 16px;
                padding: 30px;
                max-width: 400px;
                width: 90%;
                text-align: center;
            ">
                <div style="font-size: 4rem; color: var(--cocoa-success); margin-bottom: 20px;">✅</div>
                <h3 class="cocoa-heading-2" style="color: var(--cocoa-success);">Upgrade Successful!</h3>
                <p style="color: var(--cocoa-text); margin: 20px 0;">
                    Welcome to ${config.name}! You now have access to advanced trading features.
                </p>
                <div style="color: var(--cocoa-secondary); font-size: 0.9rem;">
                    Page will refresh automatically in 3 seconds...
                </div>
            </div>
        `;
        
        document.body.appendChild(successModal);
        
        setTimeout(() => {
            successModal.remove();
        }, 3000);
    }

    showCurrentTierInfo(tier) {
        console.log('Showing current tier info for:', tier);
        // Could show detailed tier benefits modal
    }

    getNextTier() {
        switch (this.currentTier) {
            case 'starter': return 'professional';
            case 'professional': return 'institutional';
            default: return 'institutional';
        }
    }

    // Public API methods
    getCurrentTier() {
        return this.currentTier;
    }

    getTierConfig(tier = null) {
        return this.tierConfigs[tier || this.currentTier];
    }

    hasFeature(feature) {
        return this.tierConfigs[this.currentTier].features.includes(feature);
    }

    canTrade() {
        const config = this.tierConfigs[this.currentTier];
        return config.maxTrades === 'Unlimited' || this.accountData.monthlyTrades < config.maxTrades;
    }

    getUsageStats() {
        const config = this.tierConfigs[this.currentTier];
        return {
            tradesUsed: this.accountData.monthlyTrades,
            tradesLimit: config.maxTrades,
            usagePercent: config.maxTrades === 'Unlimited' ? 0 : (this.accountData.monthlyTrades / config.maxTrades) * 100,
            leverage: config.leverage,
            tier: this.currentTier
        };
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.tieredAccountSystem = new TieredAccountSystem();
    });
} else {
    window.tieredAccountSystem = new TieredAccountSystem();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TieredAccountSystem;
}

console.log('🏦 Tiered Account System loaded and ready for professional account management');