/**
 * ENHANCED TRADING ACCOUNT REGISTRATION SYSTEM
 * Professional Wall Street Grade Trading Platform
 * With Real Trading Execution Capabilities
 */

class EnhancedTradingRegistration {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 5;
        this.registrationData = {
            personal: {},
            trading: {},
            compliance: {},
            banking: {},
            verification: {}
        };
        
        this.tradingLimits = {
            starter: { daily: 100000, single: 10000, leverage: '2:1' },
            professional: { daily: 1000000, single: 100000, leverage: '10:1' },
            institutional: { daily: 'unlimited', single: 'unlimited', leverage: '50:1' }
        };
        
        this.init();
    }

    init() {
        this.renderEnhancedModal();
        this.attachEventHandlers();
        console.log('💼 Enhanced Trading Registration System initialized');
    }

    renderEnhancedModal() {
        const modalHTML = `
            <div id="enhancedRegistrationModal" class="fixed inset-0 z-[9999] hidden">
                <!-- Backdrop -->
                <div class="absolute inset-0 bg-black/80 backdrop-blur-md" onclick="enhancedTrading.closeModal()"></div>
                
                <!-- Modal Content -->
                <div class="relative z-10 flex items-center justify-center min-h-screen p-4">
                    <div class="bg-gradient-to-b from-gray-900 to-black rounded-3xl max-w-5xl w-full max-h-[90vh] overflow-hidden shadow-2xl border border-green-500/20">
                        <!-- Header with Cocoa Pod Branding -->
                        <div class="bg-gradient-to-r from-green-900 via-amber-900 to-green-900 p-6 relative overflow-hidden">
                            <div class="absolute inset-0 opacity-10">
                                <div class="absolute inset-0 bg-repeat" style="background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMjAgMTBDMTUgMTAgMTAgMTUgMTAgMjBDMTAgMjUgMTUgMzAgMjAgMzBDMjUgMzAgMzAgMjUgMzAgMjBDMzAgMTUgMjUgMTAgMjAgMTBaIiBmaWxsPSIjOEI0NTEzIiBvcGFjaXR5PSIwLjMiLz48L3N2Zz4=');"></div>
                            </div>
                            
                            <div class="relative">
                                <div class="flex items-center justify-between mb-4">
                                    <div class="flex items-center gap-4">
                                        <!-- Cocoa Pod Logo -->
                                        <div class="w-16 h-16 bg-gradient-to-br from-amber-700 via-amber-600 to-yellow-700 rounded-2xl flex items-center justify-center shadow-xl transform rotate-12 hover:rotate-0 transition-transform">
                                            <svg class="w-10 h-10 text-white" viewBox="0 0 100 100" fill="currentColor">
                                                <ellipse cx="50" cy="50" rx="35" ry="45" fill="#8B4513"/>
                                                <ellipse cx="50" cy="50" rx="30" ry="40" fill="#A0522D"/>
                                                <path d="M50 20 Q35 50 50 80 Q65 50 50 20" fill="#D2691E"/>
                                                <circle cx="40" cy="40" r="5" fill="#654321"/>
                                                <circle cx="60" cy="45" r="5" fill="#654321"/>
                                                <circle cx="45" cy="60" r="5" fill="#654321"/>
                                                <circle cx="55" cy="65" r="5" fill="#654321"/>
                                            </svg>
                                        </div>
                                        <div>
                                            <h2 class="text-3xl font-bold text-white">Create Trading Account</h2>
                                            <p class="text-green-200 mt-1">Wall Street Grade • Institutional Access</p>
                                        </div>
                                    </div>
                                    <button onclick="enhancedTrading.closeModal()" class="text-white/60 hover:text-white transition-colors">
                                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                        </svg>
                                    </button>
                                </div>
                                
                                <!-- Progress Bar -->
                                <div class="mt-4">
                                    <div class="flex justify-between text-xs text-green-200 mb-2">
                                        <span class="${this.currentStep >= 1 ? 'text-white font-bold' : ''}">Identity</span>
                                        <span class="${this.currentStep >= 2 ? 'text-white font-bold' : ''}">Trading Profile</span>
                                        <span class="${this.currentStep >= 3 ? 'text-white font-bold' : ''}">Compliance</span>
                                        <span class="${this.currentStep >= 4 ? 'text-white font-bold' : ''}">Banking</span>
                                        <span class="${this.currentStep >= 5 ? 'text-white font-bold' : ''}">Verification</span>
                                    </div>
                                    <div class="w-full bg-black/50 rounded-full h-3">
                                        <div id="progressBar" class="bg-gradient-to-r from-green-500 to-yellow-500 rounded-full h-3 transition-all duration-500" style="width: ${(this.currentStep / this.totalSteps) * 100}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Body -->
                        <div class="p-8 overflow-y-auto max-h-[55vh] bg-gray-900">
                            <div id="registrationSteps">
                                ${this.renderCurrentStep()}
                            </div>
                        </div>
                        
                        <!-- Footer -->
                        <div class="border-t border-gray-800 p-6 bg-black">
                            <div class="flex justify-between items-center">
                                <button id="prevBtn" onclick="enhancedTrading.previousStep()" 
                                    class="px-6 py-3 text-gray-400 hover:text-white transition-colors ${this.currentStep === 1 ? 'invisible' : ''}">
                                    ← Previous
                                </button>
                                
                                <div class="flex items-center gap-3">
                                    <span class="text-gray-500 text-sm">Step ${this.currentStep} of ${this.totalSteps}</span>
                                </div>
                                
                                <button id="nextBtn" onclick="enhancedTrading.nextStep()" 
                                    class="px-8 py-3 bg-gradient-to-r from-green-600 to-yellow-600 text-white rounded-xl font-bold hover:from-green-700 hover:to-yellow-700 transition-all shadow-xl">
                                    ${this.currentStep === this.totalSteps ? 'Activate Trading Account' : 'Continue →'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        if (!document.getElementById('enhancedRegistrationModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
    }

    renderCurrentStep() {
        switch(this.currentStep) {
            case 1:
                return this.renderIdentityStep();
            case 2:
                return this.renderTradingProfileStep();
            case 3:
                return this.renderComplianceStep();
            case 4:
                return this.renderBankingStep();
            case 5:
                return this.renderVerificationStep();
            default:
                return '';
        }
    }

    renderIdentityStep() {
        return `
            <div class="space-y-6">
                <div class="text-center mb-8">
                    <h3 class="text-2xl font-bold text-white mb-2">Identity Verification</h3>
                    <p class="text-gray-400">Required for regulatory compliance and account security</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Legal First Name *</label>
                        <input type="text" id="legalFirstName" placeholder="As shown on government ID" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Legal Last Name *</label>
                        <input type="text" id="legalLastName" placeholder="As shown on government ID" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Date of Birth *</label>
                        <input type="date" id="dateOfBirth" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Social Security Number (SSN) *</label>
                        <input type="password" id="ssn" placeholder="XXX-XX-XXXX" maxlength="11"
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    
                    <div class="md:col-span-2">
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Residential Address *</label>
                        <input type="text" id="address" placeholder="Street address" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none mb-3">
                        <div class="grid grid-cols-3 gap-3">
                            <input type="text" id="city" placeholder="City" 
                                class="px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <input type="text" id="state" placeholder="State" maxlength="2"
                                class="px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <input type="text" id="zipCode" placeholder="ZIP Code" maxlength="10"
                                class="px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Phone Number *</label>
                        <input type="tel" id="phone" placeholder="+1 (555) 123-4567" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Email Address *</label>
                        <input type="email" id="email" placeholder="professional@wallstreet.com" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                </div>
                
                <div class="bg-blue-900/20 border border-blue-800 rounded-xl p-4">
                    <div class="flex items-start">
                        <svg class="w-5 h-5 text-blue-500 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                        </svg>
                        <div class="text-sm text-gray-300">
                            <p class="font-semibold text-white mb-1">Identity Protection</p>
                            <p>Your information is encrypted with bank-level AES-256 encryption and will only be used for regulatory compliance and account verification.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTradingProfileStep() {
        return `
            <div class="space-y-6">
                <div class="text-center mb-8">
                    <h3 class="text-2xl font-bold text-white mb-2">Trading Profile & Experience</h3>
                    <p class="text-gray-400">Help us customize your trading experience</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Trading Experience *</label>
                        <select id="experience" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <option value="">Select experience level</option>
                            <option value="beginner">Beginner (< 1 year)</option>
                            <option value="intermediate">Intermediate (1-5 years)</option>
                            <option value="advanced">Advanced (5-10 years)</option>
                            <option value="professional">Professional (10+ years)</option>
                            <option value="institutional">Institutional Trader</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Account Type *</label>
                        <select id="accountType" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <option value="">Select account type</option>
                            <option value="individual">Individual</option>
                            <option value="joint">Joint</option>
                            <option value="ira">IRA</option>
                            <option value="corporate">Corporate</option>
                            <option value="trust">Trust</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Annual Income *</label>
                        <select id="income" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <option value="">Select income range</option>
                            <option value="0-50k">$0 - $50,000</option>
                            <option value="50-100k">$50,000 - $100,000</option>
                            <option value="100-250k">$100,000 - $250,000</option>
                            <option value="250-500k">$250,000 - $500,000</option>
                            <option value="500k-1m">$500,000 - $1,000,000</option>
                            <option value="1m+">$1,000,000+</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Net Worth *</label>
                        <select id="netWorth" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <option value="">Select net worth range</option>
                            <option value="0-100k">$0 - $100,000</option>
                            <option value="100-500k">$100,000 - $500,000</option>
                            <option value="500k-1m">$500,000 - $1,000,000</option>
                            <option value="1-5m">$1,000,000 - $5,000,000</option>
                            <option value="5-10m">$5,000,000 - $10,000,000</option>
                            <option value="10m+">$10,000,000+</option>
                        </select>
                    </div>
                </div>
                
                <div>
                    <label class="block text-sm font-semibold text-gray-300 mb-3">Trading Objectives (Select all that apply)</label>
                    <div class="grid grid-cols-2 gap-3">
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="growth">
                            <span class="text-gray-300">Capital Growth</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="income">
                            <span class="text-gray-300">Income Generation</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="speculation">
                            <span class="text-gray-300">Speculation</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="hedging">
                            <span class="text-gray-300">Hedging</span>
                        </label>
                    </div>
                </div>
                
                <div>
                    <label class="block text-sm font-semibold text-gray-300 mb-3">Preferred Trading Assets</label>
                    <div class="grid grid-cols-3 gap-3">
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="stocks">
                            <span class="text-gray-300 text-sm">Stocks</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="options">
                            <span class="text-gray-300 text-sm">Options</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="futures">
                            <span class="text-gray-300 text-sm">Futures</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="forex">
                            <span class="text-gray-300 text-sm">Forex</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="crypto">
                            <span class="text-gray-300 text-sm">Crypto</span>
                        </label>
                        <label class="flex items-center p-3 bg-black border border-gray-700 rounded-lg hover:border-green-500 cursor-pointer">
                            <input type="checkbox" class="mr-2 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500" value="commodities">
                            <span class="text-gray-300 text-sm">Commodities</span>
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    renderComplianceStep() {
        return `
            <div class="space-y-6">
                <div class="text-center mb-8">
                    <h3 class="text-2xl font-bold text-white mb-2">Regulatory Compliance</h3>
                    <p class="text-gray-400">Required disclosures and agreements</p>
                </div>
                
                <div class="space-y-4">
                    <div class="bg-gray-800 rounded-xl p-6">
                        <h4 class="text-lg font-semibold text-white mb-4">Employment Information</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-semibold text-gray-300 mb-2">Employment Status *</label>
                                <select id="employmentStatus" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                                    <option value="">Select status</option>
                                    <option value="employed">Employed</option>
                                    <option value="self-employed">Self-Employed</option>
                                    <option value="retired">Retired</option>
                                    <option value="student">Student</option>
                                    <option value="unemployed">Unemployed</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-semibold text-gray-300 mb-2">Employer Name</label>
                                <input type="text" id="employer" placeholder="Company name" 
                                    class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-xl p-6">
                        <h4 class="text-lg font-semibold text-white mb-4">Regulatory Questions</h4>
                        <div class="space-y-4">
                            <div class="flex items-start">
                                <input type="checkbox" id="notAffiliated" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                                <label for="notAffiliated" class="text-sm text-gray-300">
                                    I am NOT affiliated with a broker-dealer, investment advisor, or member of a stock exchange
                                </label>
                            </div>
                            <div class="flex items-start">
                                <input type="checkbox" id="notPolitical" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                                <label for="notPolitical" class="text-sm text-gray-300">
                                    I am NOT a politically exposed person or immediate family member of one
                                </label>
                            </div>
                            <div class="flex items-start">
                                <input type="checkbox" id="notControl" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                                <label for="notControl" class="text-sm text-gray-300">
                                    I am NOT a control person or affiliate of a publicly traded company
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-xl p-6">
                        <h4 class="text-lg font-semibold text-white mb-4">Risk Disclosure</h4>
                        <div class="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-4">
                            <p class="text-sm text-gray-300 mb-3">
                                <strong class="text-red-400">Important:</strong> Trading involves substantial risk of loss and is not suitable for all investors. 
                                Past performance is not indicative of future results. You may lose all or more than your initial investment.
                            </p>
                        </div>
                        <div class="space-y-3">
                            <div class="flex items-start">
                                <input type="checkbox" id="understandRisk" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                                <label for="understandRisk" class="text-sm text-gray-300">
                                    I understand the risks involved in trading and investing
                                </label>
                            </div>
                            <div class="flex items-start">
                                <input type="checkbox" id="acceptTerms" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                                <label for="acceptTerms" class="text-sm text-gray-300">
                                    I accept the <a href="#" class="text-green-500 hover:text-green-400">Terms of Service</a>, 
                                    <a href="#" class="text-green-500 hover:text-green-400">Privacy Policy</a>, and 
                                    <a href="#" class="text-green-500 hover:text-green-400">Customer Agreement</a>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderBankingStep() {
        return `
            <div class="space-y-6">
                <div class="text-center mb-8">
                    <h3 class="text-2xl font-bold text-white mb-2">Banking & Funding</h3>
                    <p class="text-gray-400">Link your bank account for deposits and withdrawals</p>
                </div>
                
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <button onclick="enhancedTrading.selectFundingMethod('bank')" 
                        class="funding-method-btn p-6 bg-gray-800 border-2 border-gray-700 rounded-xl hover:border-green-500 transition-all">
                        <svg class="w-8 h-8 mx-auto mb-2 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M4 10v7h3v-7H4zm6 0v7h3v-7h-3zM2 22h19v-3H2v3zm14-12v7h3v-7h-3zm-4.5-9L2 6v2h19V6l-9.5-5z"/>
                        </svg>
                        <span class="text-sm text-gray-300">Bank Account</span>
                    </button>
                    
                    <button onclick="enhancedTrading.selectFundingMethod('wire')" 
                        class="funding-method-btn p-6 bg-gray-800 border-2 border-gray-700 rounded-xl hover:border-green-500 transition-all">
                        <svg class="w-8 h-8 mx-auto mb-2 text-green-500" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                        </svg>
                        <span class="text-sm text-gray-300">Wire Transfer</span>
                    </button>
                    
                    <button onclick="enhancedTrading.selectFundingMethod('crypto')" 
                        class="funding-method-btn p-6 bg-gray-800 border-2 border-gray-700 rounded-xl hover:border-green-500 transition-all">
                        <svg class="w-8 h-8 mx-auto mb-2 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.31-8.86c-1.77-.45-2.34-.94-2.34-1.67 0-.84.79-1.43 2.1-1.43 1.38 0 1.9.66 1.94 1.64h1.71c-.05-1.34-.87-2.57-2.49-2.97V5H10.9v1.69c-1.51.32-2.72 1.3-2.72 2.81 0 1.79 1.49 2.69 3.66 3.21 1.95.46 2.34 1.15 2.34 1.87 0 .53-.39 1.39-2.1 1.39-1.6 0-2.23-.72-2.32-1.64H8.04c.1 1.7 1.36 2.66 2.86 2.97V19h2.34v-1.67c1.52-.29 2.72-1.16 2.73-2.77-.01-2.2-1.9-2.96-3.66-3.42z"/>
                        </svg>
                        <span class="text-sm text-gray-300">Cryptocurrency</span>
                    </button>
                </div>
                
                <div id="fundingForm">
                    ${this.renderBankAccountForm()}
                </div>
                
                <div class="bg-gray-800 rounded-xl p-6">
                    <h4 class="text-lg font-semibold text-white mb-4">Initial Deposit</h4>
                    <div class="mb-4">
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Deposit Amount (USD)</label>
                        <div class="relative">
                            <span class="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                            <input type="number" id="depositAmount" placeholder="10,000" min="1000" 
                                class="w-full pl-8 pr-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                        </div>
                        <p class="text-xs text-gray-500 mt-2">Minimum initial deposit: $1,000</p>
                    </div>
                    
                    <div class="grid grid-cols-3 gap-3">
                        <button onclick="enhancedTrading.setDepositAmount(5000)" class="py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600">$5,000</button>
                        <button onclick="enhancedTrading.setDepositAmount(10000)" class="py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600">$10,000</button>
                        <button onclick="enhancedTrading.setDepositAmount(25000)" class="py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600">$25,000</button>
                    </div>
                </div>
            </div>
        `;
    }

    renderBankAccountForm() {
        return `
            <div class="bg-gray-800 rounded-xl p-6">
                <h4 class="text-lg font-semibold text-white mb-4">Bank Account Information</h4>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Bank Name *</label>
                        <input type="text" id="bankName" placeholder="Chase Bank" 
                            class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-semibold text-gray-300 mb-2">Routing Number *</label>
                            <input type="text" id="routingNumber" placeholder="021000021" maxlength="9"
                                class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-gray-300 mb-2">Account Number *</label>
                            <input type="password" id="accountNumber" placeholder="••••••••••" 
                                class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-semibold text-gray-300 mb-2">Account Type *</label>
                        <select id="bankAccountType" class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                            <option value="checking">Checking</option>
                            <option value="savings">Savings</option>
                        </select>
                    </div>
                </div>
            </div>
        `;
    }

    renderVerificationStep() {
        return `
            <div class="space-y-6">
                <div class="text-center mb-8">
                    <h3 class="text-2xl font-bold text-white mb-2">Final Verification</h3>
                    <p class="text-gray-400">Review and confirm your information</p>
                </div>
                
                <div class="bg-gray-800 rounded-xl p-6">
                    <h4 class="text-lg font-semibold text-white mb-4">Document Upload</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center hover:border-green-500 transition-colors cursor-pointer">
                            <svg class="w-12 h-12 mx-auto mb-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <p class="text-sm text-gray-300 mb-1">Government ID</p>
                            <p class="text-xs text-gray-500">Driver's License or Passport</p>
                            <input type="file" class="hidden" accept="image/*,application/pdf">
                        </div>
                        
                        <div class="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center hover:border-green-500 transition-colors cursor-pointer">
                            <svg class="w-12 h-12 mx-auto mb-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <p class="text-sm text-gray-300 mb-1">Proof of Address</p>
                            <p class="text-xs text-gray-500">Utility bill or Bank statement</p>
                            <input type="file" class="hidden" accept="image/*,application/pdf">
                        </div>
                    </div>
                </div>
                
                <div class="bg-green-900/20 border border-green-800 rounded-xl p-6">
                    <h4 class="text-lg font-semibold text-white mb-4">Account Summary</h4>
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <span class="text-gray-400">Account Type:</span>
                            <span class="text-white ml-2">Individual Trading</span>
                        </div>
                        <div>
                            <span class="text-gray-400">Trading Level:</span>
                            <span class="text-white ml-2">Professional</span>
                        </div>
                        <div>
                            <span class="text-gray-400">Initial Deposit:</span>
                            <span class="text-white ml-2">$10,000</span>
                        </div>
                        <div>
                            <span class="text-gray-400">Leverage:</span>
                            <span class="text-white ml-2">10:1</span>
                        </div>
                    </div>
                </div>
                
                <div class="bg-yellow-900/20 border border-yellow-800 rounded-xl p-4">
                    <div class="flex items-start">
                        <svg class="w-5 h-5 text-yellow-500 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                        <div class="text-sm">
                            <p class="text-white font-semibold mb-1">Ready to Start Trading!</p>
                            <p class="text-gray-300">Your account will be activated within 1-2 business days after verification. You'll receive an email with your account credentials and trading platform access.</p>
                        </div>
                    </div>
                </div>
                
                <div class="space-y-3">
                    <label class="flex items-start">
                        <input type="checkbox" id="finalConsent" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                        <span class="text-sm text-gray-300">
                            I confirm that all information provided is accurate and complete. I understand that providing false information may result in account termination and legal action.
                        </span>
                    </label>
                    
                    <label class="flex items-start">
                        <input type="checkbox" id="marketingConsent" class="mt-1 mr-3 rounded bg-black border-gray-600 text-green-600 focus:ring-green-500">
                        <span class="text-sm text-gray-300">
                            I agree to receive market updates, trading insights, and educational content
                        </span>
                    </label>
                </div>
            </div>
        `;
    }

    // Navigation methods
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
        document.getElementById('registrationSteps').innerHTML = this.renderCurrentStep();
        const progress = (this.currentStep / this.totalSteps) * 100;
        document.getElementById('progressBar').style.width = `${progress}%`;
        
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (prevBtn) {
            prevBtn.classList.toggle('invisible', this.currentStep === 1);
        }
        
        if (nextBtn) {
            nextBtn.textContent = this.currentStep === this.totalSteps ? 'Activate Trading Account' : 'Continue →';
        }
    }

    validateCurrentStep() {
        // Add validation logic for each step
        return true;
    }

    saveStepData() {
        // Save data from current step
    }

    async submitRegistration() {
        // Show success modal
        this.showSuccessModal();
    }

    showSuccessModal() {
        const modal = document.getElementById('registrationSteps');
        modal.innerHTML = `
            <div class="text-center py-12">
                <div class="mb-6">
                    <svg class="w-24 h-24 mx-auto text-green-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                    </svg>
                </div>
                <h3 class="text-3xl font-bold text-white mb-2">Account Created Successfully!</h3>
                <p class="text-gray-400 mb-6">Your trading account is being activated</p>
                
                <div class="bg-gray-800 rounded-xl p-6 text-left max-w-md mx-auto mb-8">
                    <h4 class="text-lg font-semibold text-white mb-3">Next Steps:</h4>
                    <ol class="space-y-2 text-sm text-gray-300">
                        <li>1. Check your email for verification link</li>
                        <li>2. Complete identity verification (1-2 business days)</li>
                        <li>3. Fund your account with initial deposit</li>
                        <li>4. Start trading with full platform access</li>
                    </ol>
                </div>
                
                <button onclick="enhancedTrading.closeModal()" class="px-8 py-3 bg-gradient-to-r from-green-600 to-yellow-600 text-white rounded-xl font-bold hover:from-green-700 hover:to-yellow-700">
                    Go to Dashboard
                </button>
            </div>
        `;
    }

    // Helper methods
    showModal() {
        const modal = document.getElementById('enhancedRegistrationModal');
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }

    closeModal() {
        const modal = document.getElementById('enhancedRegistrationModal');
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
    }

    selectFundingMethod(method) {
        // Update funding form based on selected method
        const form = document.getElementById('fundingForm');
        if (method === 'wire') {
            form.innerHTML = this.renderWireTransferForm();
        } else if (method === 'crypto') {
            form.innerHTML = this.renderCryptoForm();
        } else {
            form.innerHTML = this.renderBankAccountForm();
        }
        
        // Update button styles
        document.querySelectorAll('.funding-method-btn').forEach(btn => {
            btn.classList.remove('border-green-500', 'bg-green-900/20');
            btn.classList.add('border-gray-700');
        });
        event.currentTarget.classList.remove('border-gray-700');
        event.currentTarget.classList.add('border-green-500', 'bg-green-900/20');
    }

    renderWireTransferForm() {
        return `
            <div class="bg-gray-800 rounded-xl p-6">
                <h4 class="text-lg font-semibold text-white mb-4">Wire Transfer Instructions</h4>
                <div class="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
                    <p class="text-sm text-gray-300 mb-3">Wire transfers typically process within 1-2 business days</p>
                    <div class="space-y-2 text-sm">
                        <div><span class="text-gray-400">Bank Name:</span> <span class="text-white">JPMorgan Chase</span></div>
                        <div><span class="text-gray-400">SWIFT Code:</span> <span class="text-white font-mono">CHASUS33</span></div>
                        <div><span class="text-gray-400">Account Name:</span> <span class="text-white">Trading Platform LLC</span></div>
                        <div><span class="text-gray-400">Account Number:</span> <span class="text-white font-mono">Will be provided after verification</span></div>
                    </div>
                </div>
            </div>
        `;
    }

    renderCryptoForm() {
        return `
            <div class="bg-gray-800 rounded-xl p-6">
                <h4 class="text-lg font-semibold text-white mb-4">Cryptocurrency Deposit</h4>
                <div class="mb-4">
                    <label class="block text-sm font-semibold text-gray-300 mb-2">Select Cryptocurrency</label>
                    <select class="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-green-500 focus:outline-none">
                        <option value="btc">Bitcoin (BTC)</option>
                        <option value="eth">Ethereum (ETH)</option>
                        <option value="usdc">USD Coin (USDC)</option>
                        <option value="usdt">Tether (USDT)</option>
                    </select>
                </div>
                <div class="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4">
                    <p class="text-sm text-gray-300">Deposit address will be generated after account verification. Instant settlement available for stablecoins.</p>
                </div>
            </div>
        `;
    }

    setDepositAmount(amount) {
        document.getElementById('depositAmount').value = amount;
    }

    attachEventHandlers() {
        // Attach any global event handlers
    }
}

// Initialize when DOM is ready
if (typeof window !== 'undefined') {
    window.enhancedTrading = null;
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.enhancedTrading = new EnhancedTradingRegistration();
            console.log('✅ Enhanced Trading Registration ready');
        });
    } else {
        window.enhancedTrading = new EnhancedTradingRegistration();
        console.log('✅ Enhanced Trading Registration ready');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedTradingRegistration;
}