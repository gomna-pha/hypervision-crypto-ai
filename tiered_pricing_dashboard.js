/**
 * TIERED PRICING DASHBOARD FOR GOMNA AI TRADING
 * Professional investment tiers with payment methods integration
 */

class TieredPricingDashboard {
    constructor() {
        this.pricingTiers = [
            {
                id: 'starter',
                name: 'STARTER',
                tier: 'TIER 1',
                minDeposit: 1000,
                dailyLimit: '100K',
                leverage: '2:1',
                features: 'Basic AI signals',
                popular: false,
                color: 'blue',
                benefits: [
                    'Real-time market data',
                    'Basic arbitrage alerts',
                    'Email support',
                    'Mobile app access',
                    'Standard execution speed'
                ],
                restrictions: [
                    'Limited to 10 trades/day',
                    'Basic risk management',
                    'Standard spreads'
                ]
            },
            {
                id: 'professional',
                name: 'PROFESSIONAL',
                tier: 'TIER 2',
                minDeposit: 10000,
                dailyLimit: '1M',
                leverage: '10:1',
                features: 'Advanced AI suite',
                popular: true,
                color: 'amber',
                benefits: [
                    'Advanced ML algorithms',
                    'HFT arbitrage strategies',
                    'Priority support 24/7',
                    'Custom risk parameters',
                    'Sub-50ms execution',
                    'Portfolio analytics',
                    'API access'
                ],
                restrictions: [
                    'Up to 100 trades/day',
                    'Enhanced risk controls',
                    'Preferential spreads'
                ]
            },
            {
                id: 'institutional',
                name: 'INSTITUTIONAL',
                tier: 'TIER 3',
                minDeposit: 25000,
                dailyLimit: 'Unlimited',
                leverage: '50:1',
                features: 'White-glove service',
                popular: false,
                color: 'purple',
                benefits: [
                    'Dedicated account manager',
                    'Custom algorithm development',
                    'Direct market access',
                    'Institutional spreads',
                    'Co-location services',
                    'Custom reporting',
                    'Regulatory compliance support',
                    'Multi-venue execution'
                ],
                restrictions: [
                    'Unlimited trading',
                    'Maximum risk flexibility',
                    'Institutional-grade infrastructure'
                ]
            }
        ];

        this.paymentMethods = [
            {
                category: 'Traditional Banking',
                methods: [
                    { name: 'Credit/Debit Cards', icon: '💳', fee: '2.9%', time: 'Instant', limits: '$50K/day' },
                    { name: 'Bank Wire Transfer', icon: '🏦', fee: '$25', time: '1-3 business days', limits: '$10M/day' },
                    { name: 'ACH Transfer', icon: '🔄', fee: 'Free', time: '3-5 business days', limits: '$500K/day' }
                ]
            },
            {
                category: 'Cryptocurrency',
                methods: [
                    { name: 'Bitcoin (BTC)', icon: '₿', fee: 'Network fee only', time: '30-60 minutes', limits: 'No limit' },
                    { name: 'Ethereum (ETH)', icon: '♦', fee: 'Gas fee only', time: '5-15 minutes', limits: 'No limit' },
                    { name: 'USDC/USDT Stablecoins', icon: '💰', fee: 'Minimal', time: '5-15 minutes', limits: 'No limit' }
                ]
            },
            {
                category: 'Alternative Methods',
                methods: [
                    { name: 'PayPal Business', icon: '📱', fee: '3.5%', time: 'Instant', time: 'Instant', limits: '$100K/day' },
                    { name: 'Apple Pay/Google Pay', icon: '📲', fee: '2.9%', time: 'Instant', limits: '$25K/day' },
                    { name: 'International Wire', icon: '🌍', fee: '$45', time: '3-7 business days', limits: '$25M/day' }
                ]
            }
        ];

        this.initializeDashboard();
    }

    initializeDashboard() {
        this.createPricingSection();
        this.createPaymentMethodsSection();
        this.setupEventListeners();
    }

    createPricingSection() {
        const pricingHTML = `
            <div id="pricing-dashboard" class="bg-gradient-to-br from-cream-50 to-cream-100 p-8 rounded-xl shadow-2xl mb-8">
                <div class="text-center mb-10">
                    <h2 class="text-4xl font-bold text-gray-900 mb-4">Choose Your Trading Tier</h2>
                    <p class="text-xl text-gray-600">Professional algorithmic trading access for every investment level</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-10">
                    ${this.pricingTiers.map(tier => this.renderPricingTier(tier)).join('')}
                </div>
                
                <div class="text-center">
                    <div class="bg-white p-6 rounded-xl shadow-lg inline-block">
                        <h3 class="text-lg font-bold text-gray-900 mb-4">🔐 Secure & Regulated Platform</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
                            <div class="flex items-center gap-2">
                                <span class="text-green-600">✓</span>
                                <span>SEC Registered</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-600">✓</span>
                                <span>FINRA Member</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-600">✓</span>
                                <span>$250M Insurance</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-green-600">✓</span>
                                <span>SOC 2 Compliant</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert into dashboard tab or create new container
        this.insertPricingSection(pricingHTML);
    }

    renderPricingTier(tier) {
        const colorClasses = {
            blue: 'border-blue-200 hover:border-blue-400',
            amber: 'border-amber-300 hover:border-amber-500 ring-4 ring-amber-100',
            purple: 'border-purple-200 hover:border-purple-400'
        };

        const buttonClasses = {
            blue: 'bg-blue-600 hover:bg-blue-700',
            amber: 'bg-amber-600 hover:bg-amber-700',
            purple: 'bg-purple-600 hover:bg-purple-700'
        };

        return `
            <div class="pricing-tier-card relative bg-white rounded-2xl shadow-xl ${colorClasses[tier.color]} border-2 transition-all duration-300 hover:shadow-2xl hover:scale-105">
                ${tier.popular ? `
                    <div class="absolute -top-4 left-1/2 transform -translate-x-1/2">
                        <span class="bg-gradient-to-r from-amber-500 to-amber-600 text-white px-6 py-2 rounded-full text-sm font-bold shadow-lg">
                            MOST POPULAR
                        </span>
                    </div>
                ` : ''}
                
                <div class="p-8">
                    <!-- Header -->
                    <div class="text-center mb-6">
                        <h3 class="text-2xl font-bold text-gray-900 mb-1">${tier.name}</h3>
                        <p class="text-sm font-semibold text-gray-500 uppercase tracking-wide">${tier.tier}</p>
                    </div>
                    
                    <!-- Pricing -->
                    <div class="text-center mb-6">
                        <div class="text-4xl font-bold text-gray-900 mb-2">
                            $${tier.minDeposit.toLocaleString()}
                        </div>
                        <div class="text-sm text-gray-600">Minimum deposit</div>
                    </div>
                    
                    <!-- Key Features -->
                    <div class="space-y-4 mb-8">
                        <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span class="text-sm font-medium text-gray-700">Daily Limit</span>
                            <span class="text-sm font-bold text-gray-900">$${tier.dailyLimit}</span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span class="text-sm font-medium text-gray-700">Leverage</span>
                            <span class="text-sm font-bold text-gray-900">${tier.leverage}</span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span class="text-sm font-medium text-gray-700">AI Features</span>
                            <span class="text-sm font-bold text-gray-900">${tier.features}</span>
                        </div>
                    </div>
                    
                    <!-- Benefits -->
                    <div class="mb-8">
                        <h4 class="text-sm font-semibold text-gray-900 mb-3">✨ Included Features</h4>
                        <ul class="space-y-2">
                            ${tier.benefits.slice(0, 5).map(benefit => `
                                <li class="flex items-center text-sm text-gray-600">
                                    <span class="text-green-600 mr-2">✓</span>
                                    ${benefit}
                                </li>
                            `).join('')}
                        </ul>
                        ${tier.benefits.length > 5 ? `
                            <div class="mt-2">
                                <button class="text-xs text-blue-600 hover:text-blue-700 font-medium" onclick="showAllFeatures('${tier.id}')">
                                    +${tier.benefits.length - 5} more features
                                </button>
                            </div>
                        ` : ''}
                    </div>
                    
                    <!-- CTA Button -->
                    <button 
                        class="w-full ${buttonClasses[tier.color]} text-white py-4 px-6 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg hover:shadow-xl"
                        onclick="selectTier('${tier.id}')"
                    >
                        ${tier.popular ? 'Start Trading Now' : 'Choose ' + tier.name}
                    </button>
                    
                    <!-- Additional Info -->
                    <div class="mt-4 text-center">
                        <p class="text-xs text-gray-500">
                            No setup fees • Cancel anytime • 30-day money back guarantee
                        </p>
                    </div>
                </div>
            </div>
        `;
    }

    createPaymentMethodsSection() {
        const paymentHTML = `
            <div id="payment-methods-dashboard" class="bg-white p-8 rounded-xl shadow-2xl">
                <div class="text-center mb-8">
                    <h2 class="text-3xl font-bold text-gray-900 mb-4">💳 Accepted Payment Methods</h2>
                    <p class="text-lg text-gray-600">Multiple secure funding options for your convenience</p>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    ${this.paymentMethods.map(category => this.renderPaymentCategory(category)).join('')}
                </div>
                
                <div class="mt-8 p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-xl border border-green-200">
                    <h3 class="text-lg font-bold text-gray-900 mb-4">🛡️ Security & Compliance</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div class="text-center">
                            <div class="text-2xl mb-2">🔒</div>
                            <div class="text-sm font-semibold text-gray-700">256-bit SSL</div>
                            <div class="text-xs text-gray-500">Bank-grade encryption</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl mb-2">🏦</div>
                            <div class="text-sm font-semibold text-gray-700">FDIC Insured</div>
                            <div class="text-xs text-gray-500">Up to $250,000</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl mb-2">🔐</div>
                            <div class="text-sm font-semibold text-gray-700">2FA Required</div>
                            <div class="text-xs text-gray-500">Multi-factor auth</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl mb-2">📋</div>
                            <div class="text-sm font-semibold text-gray-700">KYC/AML</div>
                            <div class="text-xs text-gray-500">Fully compliant</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert after pricing section
        this.insertPaymentSection(paymentHTML);
    }

    renderPaymentCategory(category) {
        return `
            <div class="payment-category">
                <h3 class="text-xl font-bold text-gray-900 mb-6 text-center">${category.category}</h3>
                <div class="space-y-4">
                    ${category.methods.map(method => this.renderPaymentMethod(method)).join('')}
                </div>
            </div>
        `;
    }

    renderPaymentMethod(method) {
        return `
            <div class="payment-method-card bg-gray-50 hover:bg-gray-100 p-4 rounded-xl border border-gray-200 hover:border-gray-300 transition-all duration-300 cursor-pointer"
                 onclick="selectPaymentMethod('${method.name}')">
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center gap-3">
                        <span class="text-2xl">${method.icon}</span>
                        <span class="font-semibold text-gray-900">${method.name}</span>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-2 text-xs text-gray-600">
                    <div>
                        <span class="font-medium">Fee:</span> ${method.fee}
                    </div>
                    <div>
                        <span class="font-medium">Time:</span> ${method.time}
                    </div>
                    <div class="col-span-2">
                        <span class="font-medium">Limits:</span> ${method.limits}
                    </div>
                </div>
            </div>
        `;
    }

    insertPricingSection(html) {
        // Insert into dashboard tab
        const dashboardTab = document.getElementById('dashboard-tab');
        if (dashboardTab) {
            // Insert before the key metrics grid
            const metricsGrid = dashboardTab.querySelector('.grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4');
            if (metricsGrid) {
                metricsGrid.insertAdjacentHTML('beforebegin', html);
            } else {
                dashboardTab.insertAdjacentHTML('afterbegin', html);
            }
        } else {
            // Create a new container if dashboard tab doesn't exist
            document.body.insertAdjacentHTML('beforeend', `
                <div class="max-w-7xl mx-auto px-6 py-6">
                    ${html}
                </div>
            `);
        }
    }

    insertPaymentSection(html) {
        const pricingDashboard = document.getElementById('pricing-dashboard');
        if (pricingDashboard) {
            pricingDashboard.insertAdjacentHTML('afterend', html);
        } else {
            document.body.insertAdjacentHTML('beforeend', `
                <div class="max-w-7xl mx-auto px-6 py-6">
                    ${html}
                </div>
            `);
        }
    }

    setupEventListeners() {
        // Global functions for tier and payment selection
        window.selectTier = (tierId) => {
            const tier = this.pricingTiers.find(t => t.id === tierId);
            if (tier) {
                this.showTierSelection(tier);
            }
        };

        window.selectPaymentMethod = (methodName) => {
            this.showPaymentMethodDetails(methodName);
        };

        window.showAllFeatures = (tierId) => {
            const tier = this.pricingTiers.find(t => t.id === tierId);
            if (tier) {
                this.showAllTierFeatures(tier);
            }
        };
    }

    showTierSelection(tier) {
        // Create tier selection modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4';
        modal.innerHTML = `
            <div class="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-90vh overflow-y-auto">
                <div class="p-8">
                    <div class="text-center mb-6">
                        <h2 class="text-3xl font-bold text-gray-900 mb-2">🚀 ${tier.name} Tier Selected</h2>
                        <p class="text-gray-600">Ready to start professional algorithmic trading?</p>
                    </div>
                    
                    <div class="bg-gradient-to-r from-cream-50 to-amber-50 p-6 rounded-xl mb-6">
                        <h3 class="text-xl font-bold text-gray-900 mb-4">Your Investment Package</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <span class="text-sm text-gray-600">Minimum Deposit</span>
                                <div class="text-2xl font-bold text-gray-900">$${tier.minDeposit.toLocaleString()}</div>
                            </div>
                            <div>
                                <span class="text-sm text-gray-600">Daily Trading Limit</span>
                                <div class="text-2xl font-bold text-gray-900">$${tier.dailyLimit}</div>
                            </div>
                            <div>
                                <span class="text-sm text-gray-600">Leverage Available</span>
                                <div class="text-2xl font-bold text-gray-900">${tier.leverage}</div>
                            </div>
                            <div>
                                <span class="text-sm text-gray-600">AI Features</span>
                                <div class="text-lg font-bold text-gray-900">${tier.features}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-6">
                        <h4 class="text-lg font-bold text-gray-900 mb-3">✨ All Included Features</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                            ${tier.benefits.map(benefit => `
                                <div class="flex items-center text-sm text-gray-600">
                                    <span class="text-green-600 mr-2">✓</span>
                                    ${benefit}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="flex gap-4">
                        <button class="flex-1 bg-gradient-to-r from-amber-600 to-amber-700 text-white py-4 px-6 rounded-xl font-bold text-lg hover:from-amber-700 hover:to-amber-800 transition-all"
                                onclick="proceedToRegistration('${tier.id}')">
                            Continue to Registration
                        </button>
                        <button class="px-6 py-4 border border-gray-300 text-gray-700 rounded-xl font-medium hover:bg-gray-50"
                                onclick="closeModal()">
                            Compare Tiers
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Global functions for modal
        window.closeModal = () => {
            modal.remove();
        };

        window.proceedToRegistration = (tierId) => {
            modal.remove();
            this.proceedToRegistrationWithTier(tierId);
        };

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    proceedToRegistrationWithTier(tierId) {
        const tier = this.pricingTiers.find(t => t.id === tierId);
        
        // If investor system exists, pre-fill the registration
        if (window.completeFunctionalSystem && window.completeFunctionalSystem.investorSystem) {
            const investorSystem = window.completeFunctionalSystem.investorSystem;
            
            // Show login modal with pre-selected investment amount
            investorSystem.showLogin();
            
            // Pre-select the investment amount based on tier
            setTimeout(() => {
                const investmentSelect = document.getElementById('investment-amount');
                if (investmentSelect) {
                    investmentSelect.value = tier.minDeposit.toString();
                }
            }, 100);
        } else {
            // Redirect to main platform with tier parameter
            const url = new URL(window.location.origin);
            url.searchParams.set('tier', tierId);
            window.location.href = url.toString();
        }
    }

    showPaymentMethodDetails(methodName) {
        const method = this.paymentMethods
            .flatMap(category => category.methods)
            .find(m => m.name === methodName);

        if (method) {
            alert(`${method.name}
            
Fee: ${method.fee}
Processing Time: ${method.time}
Daily Limits: ${method.limits}

This payment method will be available during account funding.`);
        }
    }

    showAllTierFeatures(tier) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4';
        modal.innerHTML = `
            <div class="bg-white rounded-xl shadow-2xl max-w-lg w-full max-h-90vh overflow-y-auto">
                <div class="p-6">
                    <h3 class="text-xl font-bold text-gray-900 mb-4">${tier.name} - All Features</h3>
                    <div class="space-y-2 mb-6">
                        ${tier.benefits.map(benefit => `
                            <div class="flex items-center text-sm text-gray-600">
                                <span class="text-green-600 mr-2">✓</span>
                                ${benefit}
                            </div>
                        `).join('')}
                    </div>
                    <button class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700"
                            onclick="this.closest('.fixed').remove()">
                        Close
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
}

// Initialize pricing dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for other systems to load
    setTimeout(() => {
        window.tieredPricingDashboard = new TieredPricingDashboard();
        console.log('💰 Tiered Pricing Dashboard initialized successfully');
    }, 2000);
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TieredPricingDashboard;
}