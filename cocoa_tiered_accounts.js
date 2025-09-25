/**
 * Cocoa Trading AI - Tiered Account Structure System
 * Implements Starter/Professional/Institutional account tiers
 * Based on the professional interface shown in user's Cocoa Trading AI image
 */

class CocoaTieredAccounts {
    constructor() {
        this.accountTiers = {
            starter: {
                name: 'Starter',
                minDeposit: 10000,
                maxLeverage: '10:1',
                features: [
                    'Basic Arbitrage Strategies',
                    'Real-time Market Data',
                    'Standard Support',
                    'Basic Analytics',
                    'Mobile Access'
                ],
                tradingLimits: {
                    dailyVolume: 100000,
                    maxPositions: 5,
                    apiCalls: 1000
                },
                fees: {
                    tradingFee: 0.1, // 0.1%
                    withdrawalFee: 25
                },
                color: '#10B981' // Green
            },
            professional: {
                name: 'Professional',
                minDeposit: 100000,
                maxLeverage: '25:1',
                features: [
                    'Advanced HFT Strategies',
                    'AI-Powered Sentiment Analysis',
                    'Priority Support 24/7',
                    'Advanced Analytics & Reporting',
                    'API Access',
                    'Custom Risk Parameters',
                    'FinBERT Integration',
                    'Multi-Exchange Arbitrage'
                ],
                tradingLimits: {
                    dailyVolume: 1000000,
                    maxPositions: 25,
                    apiCalls: 10000
                },
                fees: {
                    tradingFee: 0.05, // 0.05%
                    withdrawalFee: 10
                },
                color: '#8B5CF6' // Purple
            },
            institutional: {
                name: 'Institutional',
                minDeposit: 1000000,
                maxLeverage: '50:1',
                features: [
                    'All Professional Features',
                    'Dedicated Account Manager',
                    'Custom Algorithm Development',
                    'Direct Market Access',
                    'Institutional Reporting',
                    'Regulatory Compliance Tools',
                    'Colocation Services',
                    'White Label Solutions',
                    'Custom Integration Support'
                ],
                tradingLimits: {
                    dailyVolume: 'Unlimited',
                    maxPositions: 'Unlimited',
                    apiCalls: 'Unlimited'
                },
                fees: {
                    tradingFee: 0.025, // 0.025%
                    withdrawalFee: 0
                },
                color: '#F59E0B' // Gold
            }
        };

        this.currentUser = {
            tier: 'starter',
            balance: 50000,
            verificationLevel: 'verified',
            accountFeatures: []
        };

        this.init();
    }

    init() {
        console.log('🏦 Initializing Cocoa Trading AI Tiered Account System...');
        this.createAccountInterface();
        this.setupTierUpgradeSystem();
        this.updateUIBasedOnTier();
        console.log('✅ Tiered Account System initialized');
    }

    createAccountInterface() {
        const accountContainer = document.createElement('div');
        accountContainer.id = 'cocoa-account-interface';
        accountContainer.innerHTML = `
            <div class="cocoa-panel" style="margin: 20px 0;">
                <div class="cocoa-panel-header">
                    <h3>🏦 Account Management - ${this.accountTiers[this.currentUser.tier].name} Tier</h3>
                    <div class="cocoa-status-indicator cocoa-status-live">
                        ${this.currentUser.verificationLevel === 'verified' ? 'Verified' : 'Pending Verification'}
                    </div>
                </div>
                <div class="cocoa-panel-content">
                    ${this.createAccountDashboard()}
                    ${this.createTierComparison()}
                </div>
            </div>
        `;

        // Insert into the main container
        const mainContainer = document.querySelector('.container, .main-content, #app') || document.body;
        const existingAccountInterface = document.querySelector('#cocoa-account-interface');
        if (existingAccountInterface) {
            existingAccountInterface.replaceWith(accountContainer);
        } else {
            mainContainer.appendChild(accountContainer);
        }
    }

    createAccountDashboard() {
        const currentTier = this.accountTiers[this.currentUser.tier];
        return `
            <div class="account-dashboard" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <!-- Account Balance -->
                <div class="cocoa-trading-card">
                    <h4 style="color: var(--cocoa-accent); margin-bottom: 10px;">💰 Account Balance</h4>
                    <div class="cocoa-metric-value">$${this.currentUser.balance.toLocaleString()}</div>
                    <div class="cocoa-metric-label">Available for Trading</div>
                </div>

                <!-- Current Tier -->
                <div class="cocoa-trading-card" style="border-color: ${currentTier.color};">
                    <h4 style="color: ${currentTier.color}; margin-bottom: 10px;">🎖️ Current Tier</h4>
                    <div class="cocoa-metric-value" style="color: ${currentTier.color};">${currentTier.name}</div>
                    <div class="cocoa-metric-label">Min Deposit: $${currentTier.minDeposit.toLocaleString()}</div>
                </div>

                <!-- Leverage -->
                <div class="cocoa-trading-card">
                    <h4 style="color: var(--cocoa-secondary); margin-bottom: 10px;">📊 Max Leverage</h4>
                    <div class="cocoa-metric-value">${currentTier.maxLeverage}</div>
                    <div class="cocoa-metric-label">Trading Leverage</div>
                </div>

                <!-- Daily Limit -->
                <div class="cocoa-trading-card">
                    <h4 style="color: var(--cocoa-success); margin-bottom: 10px;">📈 Daily Volume</h4>
                    <div class="cocoa-metric-value">$${typeof currentTier.tradingLimits.dailyVolume === 'number' ? currentTier.tradingLimits.dailyVolume.toLocaleString() : currentTier.tradingLimits.dailyVolume}</div>
                    <div class="cocoa-metric-label">Daily Trading Limit</div>
                </div>
            </div>

            <!-- Account Features -->
            <div style="margin-bottom: 30px;">
                <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">✨ Your Account Features</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    ${currentTier.features.map(feature => `
                        <div style="display: flex; align-items: center; padding: 10px; background: rgba(139, 69, 19, 0.1); border-radius: 8px; border-left: 3px solid ${currentTier.color};">
                            <span style="color: ${currentTier.color}; margin-right: 10px;">✅</span>
                            <span style="color: var(--cocoa-text);">${feature}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    createTierComparison() {
        return `
            <div class="tier-comparison">
                <h4 style="color: var(--cocoa-accent); margin-bottom: 20px; text-align: center;">🚀 Upgrade Your Trading Experience</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px;">
                    ${Object.entries(this.accountTiers).map(([key, tier]) => this.createTierCard(key, tier)).join('')}
                </div>
            </div>
        `;
    }

    createTierCard(tierKey, tier) {
        const isCurrentTier = tierKey === this.currentUser.tier;
        const canUpgrade = this.currentUser.balance >= tier.minDeposit;
        
        return `
            <div class="cocoa-trading-card ${isCurrentTier ? 'current-tier' : ''}" style="
                border: 2px solid ${isCurrentTier ? tier.color : 'rgba(212, 165, 116, 0.2)'};
                background: ${isCurrentTier ? `linear-gradient(135deg, ${tier.color}15, ${tier.color}05)` : ''};
                position: relative;
            ">
                ${isCurrentTier ? `
                    <div style="position: absolute; top: -10px; right: -10px; background: ${tier.color}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                        CURRENT
                    </div>
                ` : ''}
                
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: ${tier.color}; margin-bottom: 5px;">${tier.name}</h3>
                    <div style="font-size: 1.2rem; font-weight: bold; color: var(--cocoa-text);">
                        Min: $${tier.minDeposit.toLocaleString()}
                    </div>
                    <div style="color: var(--cocoa-secondary);">Max Leverage: ${tier.maxLeverage}</div>
                </div>

                <div style="margin-bottom: 20px;">
                    <h5 style="color: var(--cocoa-accent); margin-bottom: 10px;">Features:</h5>
                    ${tier.features.slice(0, 4).map(feature => `
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <span style="color: ${tier.color}; margin-right: 8px;">•</span>
                            <span style="font-size: 0.9rem; color: var(--cocoa-text);">${feature}</span>
                        </div>
                    `).join('')}
                    ${tier.features.length > 4 ? `<div style="color: var(--cocoa-secondary); font-size: 0.9rem; margin-top: 5px;">+${tier.features.length - 4} more features</div>` : ''}
                </div>

                <div style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: var(--cocoa-text);">Trading Fee:</span>
                        <span style="color: ${tier.color}; font-weight: bold;">${tier.fees.tradingFee}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--cocoa-text);">Daily Volume:</span>
                        <span style="color: ${tier.color}; font-weight: bold;">${typeof tier.tradingLimits.dailyVolume === 'number' ? '$' + tier.tradingLimits.dailyVolume.toLocaleString() : tier.tradingLimits.dailyVolume}</span>
                    </div>
                </div>

                <div style="text-align: center;">
                    ${isCurrentTier ? `
                        <button class="cocoa-btn-secondary" style="width: 100%;" disabled>
                            Current Plan
                        </button>
                    ` : `
                        <button class="cocoa-btn-primary upgrade-tier-btn" 
                                data-tier="${tierKey}"
                                style="width: 100%; ${!canUpgrade ? 'opacity: 0.6; cursor: not-allowed;' : ''}"
                                ${!canUpgrade ? 'disabled' : ''}>
                            ${canUpgrade ? 'Upgrade to ' + tier.name : 'Insufficient Balance'}
                        </button>
                    `}
                </div>
            </div>
        `;
    }

    setupTierUpgradeSystem() {
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('upgrade-tier-btn')) {
                const targetTier = e.target.getAttribute('data-tier');
                this.showUpgradeModal(targetTier);
            }
        });
    }

    showUpgradeModal(targetTier) {
        const tier = this.accountTiers[targetTier];
        
        // Create modal
        const modal = document.createElement('div');
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
            <div class="cocoa-panel" style="max-width: 500px; margin: 20px;">
                <div class="cocoa-panel-header">
                    <h3>🚀 Upgrade to ${tier.name} Tier</h3>
                    <button class="close-modal" style="background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">&times;</button>
                </div>
                <div class="cocoa-panel-content">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 2rem; color: ${tier.color}; font-weight: bold;">
                            $${tier.minDeposit.toLocaleString()}
                        </div>
                        <div style="color: var(--cocoa-secondary);">Minimum Deposit Required</div>
                    </div>

                    <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">Upgrade Benefits:</h4>
                    <ul style="list-style: none; padding: 0;">
                        ${tier.features.map(feature => `
                            <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="color: ${tier.color}; margin-right: 10px;">✅</span>
                                <span style="color: var(--cocoa-text);">${feature}</span>
                            </li>
                        `).join('')}
                    </ul>

                    <div style="background: rgba(139, 69, 19, 0.1); padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h5 style="color: var(--cocoa-accent); margin-bottom: 10px;">Enhanced Limits:</h5>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Max Leverage:</span>
                            <span style="color: ${tier.color}; font-weight: bold;">${tier.maxLeverage}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Daily Volume:</span>
                            <span style="color: ${tier.color}; font-weight: bold;">${typeof tier.tradingLimits.dailyVolume === 'number' ? '$' + tier.tradingLimits.dailyVolume.toLocaleString() : tier.tradingLimits.dailyVolume}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Trading Fee:</span>
                            <span style="color: ${tier.color}; font-weight: bold;">${tier.fees.tradingFee}%</span>
                        </div>
                    </div>

                    <div style="display: flex; gap: 15px; margin-top: 25px;">
                        <button class="cocoa-btn-secondary close-modal" style="flex: 1;">
                            Cancel
                        </button>
                        <button class="cocoa-btn-primary confirm-upgrade" data-tier="${targetTier}" style="flex: 1;">
                            Confirm Upgrade
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Modal event handlers
        modal.addEventListener('click', (e) => {
            if (e.target.classList.contains('close-modal') || e.target === modal) {
                document.body.removeChild(modal);
            }
            if (e.target.classList.contains('confirm-upgrade')) {
                this.upgradeTier(targetTier);
                document.body.removeChild(modal);
            }
        });
    }

    upgradeTier(targetTier) {
        const previousTier = this.currentUser.tier;
        this.currentUser.tier = targetTier;
        
        // Show success message
        this.showNotification(`🎉 Successfully upgraded to ${this.accountTiers[targetTier].name} tier!`, 'success');
        
        // Refresh the interface
        this.createAccountInterface();
        this.updateUIBasedOnTier();
        
        console.log(`✅ User upgraded from ${previousTier} to ${targetTier}`);
    }

    updateUIBasedOnTier() {
        const currentTier = this.accountTiers[this.currentUser.tier];
        
        // Update any tier-specific UI elements
        const tierBadges = document.querySelectorAll('.tier-badge');
        tierBadges.forEach(badge => {
            badge.textContent = currentTier.name;
            badge.style.background = currentTier.color;
        });

        // Enable/disable features based on tier
        this.toggleFeaturesBasedOnTier();
    }

    toggleFeaturesBasedOnTier() {
        const currentTier = this.accountTiers[this.currentUser.tier];
        
        // Enable/disable advanced features
        const advancedFeatures = document.querySelectorAll('[data-tier-requirement]');
        advancedFeatures.forEach(element => {
            const requirement = element.getAttribute('data-tier-requirement');
            const tierLevels = ['starter', 'professional', 'institutional'];
            const currentLevel = tierLevels.indexOf(this.currentUser.tier);
            const requiredLevel = tierLevels.indexOf(requirement);
            
            if (currentLevel >= requiredLevel) {
                element.style.opacity = '1';
                element.style.pointerEvents = 'auto';
                element.removeAttribute('disabled');
            } else {
                element.style.opacity = '0.5';
                element.style.pointerEvents = 'none';
                element.setAttribute('disabled', 'true');
            }
        });
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `cocoa-notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'success' ? 'var(--cocoa-success)' : type === 'error' ? 'var(--cocoa-error)' : 'var(--cocoa-primary)'};
            color: white;
            border-radius: 8px;
            z-index: 10001;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            animation: slideInRight 0.3s ease-out;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // Public methods for external access
    getCurrentTier() {
        return {
            tier: this.currentUser.tier,
            tierData: this.accountTiers[this.currentUser.tier],
            balance: this.currentUser.balance
        };
    }

    setUserBalance(newBalance) {
        this.currentUser.balance = newBalance;
        this.createAccountInterface();
    }

    hasFeatureAccess(feature) {
        const currentTier = this.accountTiers[this.currentUser.tier];
        return currentTier.features.includes(feature);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.cocoaTieredAccounts = new CocoaTieredAccounts();
    });
} else {
    window.cocoaTieredAccounts = new CocoaTieredAccounts();
}

// Add notification animations
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(notificationStyles);

console.log('🏦 Cocoa Trading AI Tiered Account System loaded successfully');