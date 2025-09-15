/**
 * GOMNA AI ACCOUNT REGISTRATION & PAYMENT SYSTEM
 * Complete user registration with integrated payment processing
 * Supports multiple payment methods and subscription tiers
 */

class AccountRegistrationSystem {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 4;
        this.registrationData = {
            personal: {},
            account: {},
            payment: {},
            subscription: {}
        };
        this.paymentProviders = {
            stripe: null,
            paypal: null,
            crypto: null
        };
        this.subscriptionPlans = [
            {
                id: 'starter',
                name: 'Starter',
                price: 99,
                period: 'month',
                features: [
                    'Access to Basic AI Trading Signals',
                    'Portfolio Analytics Dashboard',
                    '5 Trading Strategies',
                    'Email Support',
                    'Basic Risk Management Tools'
                ],
                tradingLimit: 10000,
                apiCalls: 1000
            },
            {
                id: 'professional',
                name: 'Professional',
                price: 499,
                period: 'month',
                features: [
                    'Advanced AI Trading Signals',
                    'Real-time Portfolio Optimization',
                    '20+ Trading Strategies',
                    'Priority Support',
                    'Advanced Risk Management',
                    'API Access',
                    'Custom Alerts',
                    'Backtesting Tools'
                ],
                tradingLimit: 100000,
                apiCalls: 10000,
                recommended: true
            },
            {
                id: 'institutional',
                name: 'Institutional',
                price: 2499,
                period: 'month',
                features: [
                    'Enterprise AI Trading Suite',
                    'Unlimited Trading Strategies',
                    'White-glove Support',
                    'Custom Strategy Development',
                    'Dedicated Account Manager',
                    'Full API Access',
                    'Multi-account Management',
                    'Compliance Reporting',
                    'Priority Execution'
                ],
                tradingLimit: 'unlimited',
                apiCalls: 'unlimited',
                enterprise: true
            }
        ];
        
        this.init();
    }

    init() {
        this.setupPaymentProviders();
        this.renderRegistrationModal();
        this.attachEventListeners();
        console.log('🚀 Account Registration System initialized');
    }

    setupPaymentProviders() {
        // Initialize Stripe
        if (typeof Stripe !== 'undefined') {
            this.paymentProviders.stripe = Stripe('pk_test_51234567890abcdef');
        }
        
        // Initialize PayPal
        if (typeof paypal !== 'undefined') {
            this.paymentProviders.paypal = paypal;
        }
    }

    renderRegistrationModal() {
        const modalHTML = `
            <div id="registrationModal" class="fixed inset-0 z-50 hidden">
                <!-- Backdrop -->
                <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick="accountRegistration.closeModal()"></div>
                
                <!-- Modal Content -->
                <div class="relative z-10 flex items-center justify-center min-h-screen p-4">
                    <div class="bg-gray-900 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl border border-gray-800">
                        <!-- Header -->
                        <div class="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 p-6">
                            <div class="flex justify-between items-center">
                                <div>
                                    <h2 class="text-2xl font-bold text-white">Create Your Gomna AI Account</h2>
                                    <p class="text-blue-100 mt-1">Join the future of AI-powered trading</p>
                                </div>
                                <button onclick="accountRegistration.closeModal()" class="text-white/80 hover:text-white">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                    </svg>
                                </button>
                            </div>
                            
                            <!-- Progress Bar -->
                            <div class="mt-6">
                                <div class="flex justify-between mb-2">
                                    ${this.renderProgressSteps()}
                                </div>
                                <div class="w-full bg-white/20 rounded-full h-2">
                                    <div id="progressBar" class="bg-white rounded-full h-2 transition-all duration-300" style="width: 25%"></div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Body -->
                        <div class="p-6 overflow-y-auto max-h-[60vh]">
                            <div id="registrationSteps">
                                ${this.renderCurrentStep()}
                            </div>
                        </div>
                        
                        <!-- Footer -->
                        <div class="border-t border-gray-800 p-6 bg-gray-900/50">
                            <div class="flex justify-between items-center">
                                <button id="prevBtn" onclick="accountRegistration.previousStep()" 
                                    class="px-6 py-2 text-gray-400 hover:text-white transition-colors ${this.currentStep === 1 ? 'invisible' : ''}">
                                    ← Previous
                                </button>
                                
                                <div class="flex gap-2">
                                    ${this.renderStepIndicators()}
                                </div>
                                
                                <button id="nextBtn" onclick="accountRegistration.nextStep()" 
                                    class="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all">
                                    ${this.currentStep === this.totalSteps ? 'Complete Registration' : 'Next →'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Append to body if not exists
        if (!document.getElementById('registrationModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
    }

    renderProgressSteps() {
        const steps = [
            { num: 1, label: 'Personal Info' },
            { num: 2, label: 'Account Setup' },
            { num: 3, label: 'Choose Plan' },
            { num: 4, label: 'Payment' }
        ];
        
        return steps.map(step => `
            <div class="flex items-center text-sm ${step.num <= this.currentStep ? 'text-white' : 'text-white/50'}">
                <span class="mr-1">${step.num}.</span> ${step.label}
            </div>
        `).join('');
    }

    renderStepIndicators() {
        let indicators = '';
        for (let i = 1; i <= this.totalSteps; i++) {
            const active = i === this.currentStep ? 'bg-white' : 'bg-white/30';
            indicators += `<div class="w-2 h-2 rounded-full ${active}"></div>`;
        }
        return indicators;
    }

    renderCurrentStep() {
        switch(this.currentStep) {
            case 1:
                return this.renderPersonalInfoStep();
            case 2:
                return this.renderAccountSetupStep();
            case 3:
                return this.renderSubscriptionStep();
            case 4:
                return this.renderPaymentStep();
            default:
                return '';
        }
    }

    renderPersonalInfoStep() {
        return `
            <div class="space-y-6">
                <div>
                    <h3 class="text-xl font-semibold text-white mb-4">Personal Information</h3>
                    <p class="text-gray-400 mb-6">Let's start with your basic information</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">First Name</label>
                        <input type="text" id="firstName" placeholder="John" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Last Name</label>
                        <input type="text" id="lastName" placeholder="Doe" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
                        <input type="email" id="email" placeholder="john@example.com" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Phone Number</label>
                        <input type="tel" id="phone" placeholder="+1 (555) 123-4567" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    
                    <div class="md:col-span-2">
                        <label class="block text-sm font-medium text-gray-300 mb-2">Country</label>
                        <select id="country" class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                            <option value="">Select your country</option>
                            <option value="US">United States</option>
                            <option value="GB">United Kingdom</option>
                            <option value="CA">Canada</option>
                            <option value="AU">Australia</option>
                            <option value="DE">Germany</option>
                            <option value="FR">France</option>
                            <option value="JP">Japan</option>
                            <option value="SG">Singapore</option>
                            <option value="HK">Hong Kong</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                </div>
                
                <div class="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
                    <div class="flex items-start">
                        <svg class="w-5 h-5 text-blue-500 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                        </svg>
                        <div>
                            <p class="text-sm text-gray-300">Your information is protected with bank-level encryption and will never be shared without your consent.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderAccountSetupStep() {
        return `
            <div class="space-y-6">
                <div>
                    <h3 class="text-xl font-semibold text-white mb-4">Account Setup</h3>
                    <p class="text-gray-400 mb-6">Create your secure trading account</p>
                </div>
                
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Username</label>
                        <input type="text" id="username" placeholder="Choose a unique username" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                        <p class="text-xs text-gray-500 mt-1">This will be your unique identifier on the platform</p>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
                        <input type="password" id="password" placeholder="Create a strong password" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                        <div class="mt-2">
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-gray-500">Password strength</span>
                                <span id="passwordStrength" class="text-gray-400">-</span>
                            </div>
                            <div class="w-full bg-gray-700 rounded-full h-1">
                                <div id="passwordStrengthBar" class="bg-gradient-to-r from-red-500 to-red-600 rounded-full h-1 transition-all" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Confirm Password</label>
                        <input type="password" id="confirmPassword" placeholder="Re-enter your password" 
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Trading Experience</label>
                        <select id="experience" class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                            <option value="">Select your experience level</option>
                            <option value="beginner">Beginner (< 1 year)</option>
                            <option value="intermediate">Intermediate (1-3 years)</option>
                            <option value="advanced">Advanced (3-5 years)</option>
                            <option value="expert">Expert (5+ years)</option>
                            <option value="institutional">Institutional Trader</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Investment Goals</label>
                        <div class="space-y-2">
                            <label class="flex items-center">
                                <input type="checkbox" class="mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500" value="growth">
                                <span class="text-gray-300">Long-term Growth</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" class="mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500" value="income">
                                <span class="text-gray-300">Passive Income</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" class="mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500" value="daytrading">
                                <span class="text-gray-300">Active Day Trading</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" class="mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500" value="hedging">
                                <span class="text-gray-300">Risk Hedging</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="bg-purple-900/20 border border-purple-800 rounded-lg p-4">
                        <div class="flex items-center mb-3">
                            <svg class="w-5 h-5 text-purple-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                            </svg>
                            <h4 class="text-sm font-semibold text-white">Security Features</h4>
                        </div>
                        <div class="space-y-2">
                            <label class="flex items-center">
                                <input type="checkbox" checked class="mr-3 rounded bg-gray-800 border-gray-700 text-purple-600 focus:ring-purple-500">
                                <span class="text-sm text-gray-300">Enable Two-Factor Authentication (2FA)</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" checked class="mr-3 rounded bg-gray-800 border-gray-700 text-purple-600 focus:ring-purple-500">
                                <span class="text-sm text-gray-300">Email notifications for account activity</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" checked class="mr-3 rounded bg-gray-800 border-gray-700 text-purple-600 focus:ring-purple-500">
                                <span class="text-sm text-gray-300">Require approval for withdrawals</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderSubscriptionStep() {
        return `
            <div class="space-y-6">
                <div>
                    <h3 class="text-xl font-semibold text-white mb-4">Choose Your Plan</h3>
                    <p class="text-gray-400 mb-6">Select the subscription that best fits your trading needs</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    ${this.subscriptionPlans.map(plan => this.renderPlanCard(plan)).join('')}
                </div>
                
                <div class="bg-gray-800 rounded-lg p-6">
                    <h4 class="text-lg font-semibold text-white mb-4">All Plans Include:</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div class="flex items-center text-gray-300">
                            <svg class="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                            </svg>
                            Real-time Market Data
                        </div>
                        <div class="flex items-center text-gray-300">
                            <svg class="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                            </svg>
                            AI-Powered Predictions
                        </div>
                        <div class="flex items-center text-gray-300">
                            <svg class="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                            </svg>
                            Secure Cloud Storage
                        </div>
                        <div class="flex items-center text-gray-300">
                            <svg class="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                            </svg>
                            Mobile App Access
                        </div>
                    </div>
                </div>
                
                <div class="space-y-4">
                    <div class="flex items-center justify-between">
                        <label class="flex items-center text-gray-300">
                            <input type="checkbox" id="annualBilling" class="mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500" onchange="accountRegistration.toggleBillingPeriod()">
                            <span>Annual billing (Save 20%)</span>
                        </label>
                        <span class="text-green-500 font-semibold hidden" id="annualSavings">Save $0</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderPlanCard(plan) {
        const isRecommended = plan.recommended;
        const borderClass = isRecommended ? 'border-blue-500' : 'border-gray-700';
        const bgClass = isRecommended ? 'bg-gradient-to-b from-blue-900/20 to-purple-900/20' : 'bg-gray-800/50';
        
        return `
            <div class="relative ${bgClass} border ${borderClass} rounded-lg p-6 cursor-pointer hover:border-blue-400 transition-all" 
                 onclick="accountRegistration.selectPlan('${plan.id}')">
                ${isRecommended ? '<div class="absolute -top-3 left-1/2 transform -translate-x-1/2"><span class="bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xs font-semibold px-3 py-1 rounded-full">RECOMMENDED</span></div>' : ''}
                
                <div class="mb-4">
                    <h4 class="text-lg font-semibold text-white">${plan.name}</h4>
                    <div class="mt-2 flex items-baseline">
                        <span class="text-3xl font-bold text-white">$${plan.price}</span>
                        <span class="text-gray-400 ml-2">/${plan.period}</span>
                    </div>
                    ${plan.tradingLimit ? `<p class="text-sm text-gray-400 mt-1">Trading limit: ${typeof plan.tradingLimit === 'number' ? '$' + plan.tradingLimit.toLocaleString() : plan.tradingLimit}</p>` : ''}
                </div>
                
                <ul class="space-y-2 mb-6">
                    ${plan.features.slice(0, 5).map(feature => `
                        <li class="flex items-start text-sm text-gray-300">
                            <svg class="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                            </svg>
                            ${feature}
                        </li>
                    `).join('')}
                </ul>
                
                <button class="w-full py-2 ${isRecommended ? 'bg-gradient-to-r from-blue-600 to-purple-600' : 'bg-gray-700'} text-white rounded-lg hover:opacity-90 transition-opacity">
                    Select ${plan.name}
                </button>
            </div>
        `;
    }

    renderPaymentStep() {
        const selectedPlan = this.subscriptionPlans.find(p => p.id === this.registrationData.subscription.planId) || this.subscriptionPlans[1];
        
        return `
            <div class="space-y-6">
                <div>
                    <h3 class="text-xl font-semibold text-white mb-4">Payment Information</h3>
                    <p class="text-gray-400 mb-6">Complete your registration with secure payment</p>
                </div>
                
                <!-- Order Summary -->
                <div class="bg-gray-800 rounded-lg p-6">
                    <h4 class="text-lg font-semibold text-white mb-4">Order Summary</h4>
                    <div class="space-y-3">
                        <div class="flex justify-between text-gray-300">
                            <span>${selectedPlan.name} Plan</span>
                            <span>$${selectedPlan.price}/month</span>
                        </div>
                        <div class="flex justify-between text-gray-300">
                            <span>Setup Fee</span>
                            <span class="text-green-500">FREE</span>
                        </div>
                        <div class="border-t border-gray-700 pt-3">
                            <div class="flex justify-between text-white font-semibold">
                                <span>Total Due Today</span>
                                <span>$${selectedPlan.price}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Payment Method Selection -->
                <div>
                    <h4 class="text-lg font-semibold text-white mb-4">Payment Method</h4>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <button onclick="accountRegistration.selectPaymentMethod('card')" 
                            class="payment-method-btn border border-gray-700 rounded-lg p-4 hover:border-blue-500 transition-all bg-gray-800/50">
                            <svg class="w-8 h-8 mx-auto mb-2 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z"/>
                            </svg>
                            <span class="text-sm text-gray-300">Credit/Debit Card</span>
                        </button>
                        
                        <button onclick="accountRegistration.selectPaymentMethod('bank')" 
                            class="payment-method-btn border border-gray-700 rounded-lg p-4 hover:border-blue-500 transition-all bg-gray-800/50">
                            <svg class="w-8 h-8 mx-auto mb-2 text-green-500" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M4 10v7h3v-7H4zm6 0v7h3v-7h-3zM2 22h19v-3H2v3zm14-12v7h3v-7h-3zm-4.5-9L2 6v2h19V6l-9.5-5z"/>
                            </svg>
                            <span class="text-sm text-gray-300">Bank Transfer</span>
                        </button>
                        
                        <button onclick="accountRegistration.selectPaymentMethod('crypto')" 
                            class="payment-method-btn border border-gray-700 rounded-lg p-4 hover:border-blue-500 transition-all bg-gray-800/50">
                            <svg class="w-8 h-8 mx-auto mb-2 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.31-8.86c-1.77-.45-2.34-.94-2.34-1.67 0-.84.79-1.43 2.1-1.43 1.38 0 1.9.66 1.94 1.64h1.71c-.05-1.34-.87-2.57-2.49-2.97V5H10.9v1.69c-1.51.32-2.72 1.3-2.72 2.81 0 1.79 1.49 2.69 3.66 3.21 1.95.46 2.34 1.15 2.34 1.87 0 .53-.39 1.39-2.1 1.39-1.6 0-2.23-.72-2.32-1.64H8.04c.1 1.7 1.36 2.66 2.86 2.97V19h2.34v-1.67c1.52-.29 2.72-1.16 2.73-2.77-.01-2.2-1.9-2.96-3.66-3.42z"/>
                            </svg>
                            <span class="text-sm text-gray-300">Cryptocurrency</span>
                        </button>
                    </div>
                    
                    <!-- Payment Form -->
                    <div id="paymentForm" class="space-y-4">
                        ${this.renderCardPaymentForm()}
                    </div>
                </div>
                
                <!-- Security Badge -->
                <div class="bg-green-900/20 border border-green-800 rounded-lg p-4">
                    <div class="flex items-center">
                        <svg class="w-6 h-6 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                        <div>
                            <p class="text-sm font-semibold text-white">Secure Payment Processing</p>
                            <p class="text-xs text-gray-400">Your payment information is encrypted and processed securely through our PCI-compliant payment partners.</p>
                        </div>
                    </div>
                </div>
                
                <!-- Terms and Conditions -->
                <div class="space-y-3">
                    <label class="flex items-start">
                        <input type="checkbox" id="termsAccepted" class="mt-1 mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500">
                        <span class="text-sm text-gray-300">
                            I agree to the <a href="#" class="text-blue-500 hover:text-blue-400">Terms of Service</a>, 
                            <a href="#" class="text-blue-500 hover:text-blue-400">Privacy Policy</a>, and 
                            <a href="#" class="text-blue-500 hover:text-blue-400">Risk Disclosure</a>
                        </span>
                    </label>
                    <label class="flex items-start">
                        <input type="checkbox" id="marketingAccepted" class="mt-1 mr-3 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500">
                        <span class="text-sm text-gray-300">
                            I would like to receive updates about new features and trading opportunities
                        </span>
                    </label>
                </div>
            </div>
        `;
    }

    renderCardPaymentForm() {
        return `
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Card Number</label>
                    <div class="relative">
                        <input type="text" id="cardNumber" placeholder="1234 5678 9012 3456" maxlength="19"
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none pr-12">
                        <div class="absolute right-3 top-1/2 transform -translate-y-1/2 flex gap-1">
                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAzMiAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjI0IiByeD0iNCIgZmlsbD0iIzAwNTFBNSIvPgo8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI3IiBmaWxsPSIjRUIwMDFCIi8+CjxjaXJjbGUgY3g9IjIwIiBjeT0iMTIiIHI9IjciIGZpbGw9IiNGRkE1MDAiIG9wYWNpdHk9IjAuOCIvPgo8L3N2Zz4=" alt="Card" class="h-6">
                        </div>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Expiry Date</label>
                        <input type="text" id="expiryDate" placeholder="MM/YY" maxlength="5"
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">CVV</label>
                        <input type="text" id="cvv" placeholder="123" maxlength="4"
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                    </div>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Cardholder Name</label>
                    <input type="text" id="cardholderName" placeholder="John Doe"
                        class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Billing Address</label>
                    <input type="text" id="billingAddress" placeholder="123 Main St, City, State 12345"
                        class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                </div>
            </div>
        `;
    }

    // Navigation Methods
    nextStep() {
        if (this.validateCurrentStep()) {
            this.saveStepData();
            
            if (this.currentStep === this.totalSteps) {
                this.submitRegistration();
            } else {
                this.currentStep++;
                this.updateUI();
            }
        }
    }

    previousStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateUI();
        }
    }

    updateUI() {
        // Update step content
        document.getElementById('registrationSteps').innerHTML = this.renderCurrentStep();
        
        // Update progress bar
        const progress = (this.currentStep / this.totalSteps) * 100;
        document.getElementById('progressBar').style.width = `${progress}%`;
        
        // Update navigation buttons
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (prevBtn) {
            prevBtn.classList.toggle('invisible', this.currentStep === 1);
        }
        
        if (nextBtn) {
            nextBtn.textContent = this.currentStep === this.totalSteps ? 'Complete Registration' : 'Next →';
        }
        
        // Re-attach event listeners for new elements
        this.attachStepEventListeners();
    }

    validateCurrentStep() {
        switch(this.currentStep) {
            case 1:
                return this.validatePersonalInfo();
            case 2:
                return this.validateAccountSetup();
            case 3:
                return this.validateSubscription();
            case 4:
                return this.validatePayment();
            default:
                return true;
        }
    }

    validatePersonalInfo() {
        const requiredFields = ['firstName', 'lastName', 'email', 'phone', 'country'];
        let isValid = true;
        
        requiredFields.forEach(field => {
            const element = document.getElementById(field);
            if (!element || !element.value.trim()) {
                isValid = false;
                if (element) {
                    element.classList.add('border-red-500');
                }
            } else {
                element.classList.remove('border-red-500');
            }
        });
        
        // Validate email format
        const email = document.getElementById('email');
        if (email && email.value && !this.isValidEmail(email.value)) {
            isValid = false;
            email.classList.add('border-red-500');
        }
        
        if (!isValid) {
            this.showError('Please fill in all required fields correctly');
        }
        
        return isValid;
    }

    validateAccountSetup() {
        const username = document.getElementById('username');
        const password = document.getElementById('password');
        const confirmPassword = document.getElementById('confirmPassword');
        
        if (!username || !username.value.trim()) {
            this.showError('Please enter a username');
            return false;
        }
        
        if (!password || password.value.length < 8) {
            this.showError('Password must be at least 8 characters');
            return false;
        }
        
        if (password.value !== confirmPassword.value) {
            this.showError('Passwords do not match');
            return false;
        }
        
        return true;
    }

    validateSubscription() {
        if (!this.registrationData.subscription.planId) {
            this.showError('Please select a subscription plan');
            return false;
        }
        return true;
    }

    validatePayment() {
        const termsAccepted = document.getElementById('termsAccepted');
        if (!termsAccepted || !termsAccepted.checked) {
            this.showError('Please accept the terms and conditions');
            return false;
        }
        
        // Validate payment fields based on selected method
        // This is simplified - in production, use proper payment validation
        const cardNumber = document.getElementById('cardNumber');
        const expiryDate = document.getElementById('expiryDate');
        const cvv = document.getElementById('cvv');
        const cardholderName = document.getElementById('cardholderName');
        
        if (!cardNumber || !cardNumber.value.replace(/\s/g, '').match(/^\d{16}$/)) {
            this.showError('Please enter a valid card number');
            return false;
        }
        
        if (!expiryDate || !expiryDate.value.match(/^\d{2}\/\d{2}$/)) {
            this.showError('Please enter a valid expiry date (MM/YY)');
            return false;
        }
        
        if (!cvv || !cvv.value.match(/^\d{3,4}$/)) {
            this.showError('Please enter a valid CVV');
            return false;
        }
        
        if (!cardholderName || !cardholderName.value.trim()) {
            this.showError('Please enter the cardholder name');
            return false;
        }
        
        return true;
    }

    saveStepData() {
        switch(this.currentStep) {
            case 1:
                this.registrationData.personal = {
                    firstName: document.getElementById('firstName').value,
                    lastName: document.getElementById('lastName').value,
                    email: document.getElementById('email').value,
                    phone: document.getElementById('phone').value,
                    country: document.getElementById('country').value
                };
                break;
            case 2:
                this.registrationData.account = {
                    username: document.getElementById('username').value,
                    password: document.getElementById('password').value,
                    experience: document.getElementById('experience').value,
                    goals: Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value)
                };
                break;
            case 3:
                // Subscription data is saved when plan is selected
                break;
            case 4:
                this.registrationData.payment = {
                    method: this.registrationData.payment.method || 'card',
                    cardNumber: document.getElementById('cardNumber').value.replace(/\s/g, ''),
                    expiryDate: document.getElementById('expiryDate').value,
                    cvv: document.getElementById('cvv').value,
                    cardholderName: document.getElementById('cardholderName').value,
                    billingAddress: document.getElementById('billingAddress').value
                };
                break;
        }
    }

    async submitRegistration() {
        try {
            // Show loading state
            this.showLoading();
            
            // Simulate API call
            await this.simulateApiCall();
            
            // Process payment
            const paymentResult = await this.processPayment();
            
            if (paymentResult.success) {
                // Create user account
                const accountResult = await this.createAccount();
                
                if (accountResult.success) {
                    // Show success message
                    this.showSuccess();
                    
                    // Close modal after delay
                    setTimeout(() => {
                        this.closeModal();
                        // Redirect to dashboard or login
                        window.location.href = '#dashboard';
                    }, 3000);
                }
            }
        } catch (error) {
            console.error('Registration failed:', error);
            this.showError('Registration failed. Please try again.');
        }
    }

    async simulateApiCall() {
        return new Promise(resolve => setTimeout(resolve, 2000));
    }

    async processPayment() {
        // Simulate payment processing
        await this.simulateApiCall();
        
        // In production, this would integrate with real payment gateway
        return {
            success: true,
            transactionId: 'TXN' + Date.now(),
            amount: this.subscriptionPlans.find(p => p.id === this.registrationData.subscription.planId).price
        };
    }

    async createAccount() {
        // Simulate account creation
        await this.simulateApiCall();
        
        return {
            success: true,
            userId: 'USER' + Date.now(),
            accountId: 'ACC' + Date.now()
        };
    }

    // UI Helper Methods
    selectPlan(planId) {
        this.registrationData.subscription.planId = planId;
        
        // Update UI to show selected plan
        document.querySelectorAll('.payment-method-btn').forEach(btn => {
            btn.classList.remove('border-blue-500', 'bg-blue-900/20');
        });
        
        event.currentTarget.classList.add('border-blue-500', 'bg-blue-900/20');
    }

    selectPaymentMethod(method) {
        this.registrationData.payment.method = method;
        
        // Update payment form based on method
        const paymentForm = document.getElementById('paymentForm');
        if (paymentForm) {
            switch(method) {
                case 'card':
                    paymentForm.innerHTML = this.renderCardPaymentForm();
                    break;
                case 'bank':
                    paymentForm.innerHTML = this.renderBankPaymentForm();
                    break;
                case 'crypto':
                    paymentForm.innerHTML = this.renderCryptoPaymentForm();
                    break;
            }
        }
        
        // Update button styles
        document.querySelectorAll('.payment-method-btn').forEach(btn => {
            btn.classList.remove('border-blue-500', 'bg-blue-900/20');
        });
        event.currentTarget.classList.add('border-blue-500', 'bg-blue-900/20');
    }

    renderBankPaymentForm() {
        return `
            <div class="space-y-4">
                <div class="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
                    <p class="text-sm text-gray-300">You will be redirected to our secure banking partner to complete the transfer.</p>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Account Holder Name</label>
                    <input type="text" id="accountHolder" placeholder="John Doe"
                        class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Bank Name</label>
                    <input type="text" id="bankName" placeholder="Chase Bank"
                        class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                </div>
            </div>
        `;
    }

    renderCryptoPaymentForm() {
        return `
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Select Cryptocurrency</label>
                    <select id="cryptoType" class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                        <option value="btc">Bitcoin (BTC)</option>
                        <option value="eth">Ethereum (ETH)</option>
                        <option value="usdt">Tether (USDT)</option>
                        <option value="usdc">USD Coin (USDC)</option>
                    </select>
                </div>
                <div class="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4">
                    <p class="text-sm text-gray-300">A unique wallet address will be generated for your payment. Please send the exact amount to complete registration.</p>
                </div>
            </div>
        `;
    }

    toggleBillingPeriod() {
        const checkbox = document.getElementById('annualBilling');
        const savingsElement = document.getElementById('annualSavings');
        
        if (checkbox && checkbox.checked) {
            savingsElement.classList.remove('hidden');
            // Calculate and show savings
            const plan = this.subscriptionPlans.find(p => p.id === this.registrationData.subscription.planId);
            if (plan) {
                const annualSavings = plan.price * 12 * 0.2;
                savingsElement.textContent = `Save $${annualSavings.toFixed(0)}`;
            }
        } else {
            savingsElement.classList.add('hidden');
        }
    }

    attachEventListeners() {
        // Password strength indicator
        const passwordInput = document.getElementById('password');
        if (passwordInput) {
            passwordInput.addEventListener('input', (e) => {
                this.updatePasswordStrength(e.target.value);
            });
        }
        
        // Card number formatting
        const cardNumberInput = document.getElementById('cardNumber');
        if (cardNumberInput) {
            cardNumberInput.addEventListener('input', (e) => {
                this.formatCardNumber(e.target);
            });
        }
        
        // Expiry date formatting
        const expiryInput = document.getElementById('expiryDate');
        if (expiryInput) {
            expiryInput.addEventListener('input', (e) => {
                this.formatExpiryDate(e.target);
            });
        }
    }

    attachStepEventListeners() {
        this.attachEventListeners();
    }

    updatePasswordStrength(password) {
        const strengthBar = document.getElementById('passwordStrengthBar');
        const strengthText = document.getElementById('passwordStrength');
        
        if (!strengthBar || !strengthText) return;
        
        let strength = 0;
        if (password.length >= 8) strength++;
        if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength++;
        if (password.match(/[0-9]/)) strength++;
        if (password.match(/[^a-zA-Z0-9]/)) strength++;
        
        const strengthLevels = ['Weak', 'Fair', 'Good', 'Strong'];
        const strengthColors = [
            'from-red-500 to-red-600',
            'from-yellow-500 to-yellow-600',
            'from-blue-500 to-blue-600',
            'from-green-500 to-green-600'
        ];
        
        strengthBar.style.width = `${(strength / 4) * 100}%`;
        strengthBar.className = `bg-gradient-to-r ${strengthColors[strength - 1] || strengthColors[0]} rounded-full h-1 transition-all`;
        strengthText.textContent = strengthLevels[strength - 1] || '-';
    }

    formatCardNumber(input) {
        let value = input.value.replace(/\s/g, '');
        let formattedValue = value.match(/.{1,4}/g)?.join(' ') || value;
        input.value = formattedValue;
    }

    formatExpiryDate(input) {
        let value = input.value.replace(/\D/g, '');
        if (value.length >= 2) {
            value = value.slice(0, 2) + '/' + value.slice(2, 4);
        }
        input.value = value;
    }

    isValidEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    showModal() {
        const modal = document.getElementById('registrationModal');
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }

    closeModal() {
        const modal = document.getElementById('registrationModal');
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
    }

    showLoading() {
        const nextBtn = document.getElementById('nextBtn');
        if (nextBtn) {
            nextBtn.disabled = true;
            nextBtn.innerHTML = `
                <svg class="animate-spin h-5 w-5 mx-auto" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
            `;
        }
    }

    showSuccess() {
        const stepsContainer = document.getElementById('registrationSteps');
        if (stepsContainer) {
            stepsContainer.innerHTML = `
                <div class="text-center py-12">
                    <div class="mb-6">
                        <svg class="w-20 h-20 mx-auto text-green-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <h3 class="text-2xl font-bold text-white mb-2">Registration Successful!</h3>
                    <p class="text-gray-400 mb-4">Welcome to Gomna AI Trading Platform</p>
                    <p class="text-sm text-gray-500">Redirecting to your dashboard...</p>
                </div>
            `;
        }
    }

    showError(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-red-600 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-pulse';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize the system when DOM is ready
if (typeof window !== 'undefined') {
    window.accountRegistration = null;
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.accountRegistration = new AccountRegistrationSystem();
            console.log('✅ Account Registration System ready');
        });
    } else {
        window.accountRegistration = new AccountRegistrationSystem();
        console.log('✅ Account Registration System ready');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccountRegistrationSystem;
}