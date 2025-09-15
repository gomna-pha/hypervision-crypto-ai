/**
 * GOMNA USER ONBOARDING & REGISTRATION SYSTEM
 * Complete user registration with KYC, payment setup, and account verification
 */

class GomnaUserOnboardingSystem {
    constructor() {
        this.currentStep = 0;
        this.userProfile = {};
        this.onboardingSteps = [
            'welcome',
            'personal-info',
            'identity-verification',
            'payment-setup',
            'risk-assessment',
            'account-confirmation'
        ];
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Gomna User Onboarding System...');
        
        // Check if user is already registered
        const existingUser = this.checkExistingRegistration();
        if (existingUser) {
            console.log('👤 Existing user found, redirecting to dashboard...');
            this.showMainDashboard();
            return;
        }
        
        // Create onboarding overlay
        this.createOnboardingFlow();
    }
    
    checkExistingRegistration() {
        // Check localStorage for existing registration
        const userData = localStorage.getItem('gomna-user-profile');
        if (userData) {
            try {
                const profile = JSON.parse(userData);
                if (profile.registrationComplete && profile.verified) {
                    this.userProfile = profile;
                    return profile;
                }
            } catch (error) {
                console.warn('Invalid stored user data, clearing...');
                localStorage.removeItem('gomna-user-profile');
            }
        }
        return null;
    }
    
    createOnboardingFlow() {
        // Hide main content
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.style.display = 'none';
        }
        
        // Create onboarding overlay
        const overlay = document.createElement('div');
        overlay.id = 'gomna-onboarding-overlay';
        overlay.className = 'fixed inset-0 bg-gradient-to-br from-cream-50 to-cream-100 z-50 overflow-y-auto';
        overlay.innerHTML = this.getOnboardingHTML();
        
        document.body.appendChild(overlay);
        
        // Initialize first step
        this.showStep('welcome');
        
        // Setup event listeners
        this.setupOnboardingListeners();
    }
    
    getOnboardingHTML() {
        return `
            <div class="min-h-screen flex items-center justify-center p-4">
                <div class="max-w-4xl w-full">
                    <!-- Progress Bar -->
                    <div class="mb-8">
                        <div class="flex items-center justify-between mb-2">
                            <h1 class="text-2xl font-bold text-gray-900">Welcome to Gomna™ AI Trading</h1>
                            <div class="text-sm text-gray-600">
                                Step <span id="current-step-number">1</span> of ${this.onboardingSteps.length}
                            </div>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div id="progress-bar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 16.67%"></div>
                        </div>
                    </div>
                    
                    <!-- Step Container -->
                    <div id="onboarding-step-container" class="bg-white rounded-xl shadow-lg">
                        <!-- Steps will be inserted here -->
                    </div>
                </div>
            </div>
        `;
    }
    
    showStep(stepName) {
        const stepIndex = this.onboardingSteps.indexOf(stepName);
        if (stepIndex === -1) return;
        
        this.currentStep = stepIndex;
        
        // Update progress
        const progress = ((stepIndex + 1) / this.onboardingSteps.length) * 100;
        const progressBar = document.getElementById('progress-bar');
        const stepNumber = document.getElementById('current-step-number');
        
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (stepNumber) stepNumber.textContent = stepIndex + 1;
        
        // Get step content
        const stepContainer = document.getElementById('onboarding-step-container');
        if (stepContainer) {
            stepContainer.innerHTML = this.getStepContent(stepName);
        }
        
        // Setup step-specific listeners
        this.setupStepListeners(stepName);
    }
    
    getStepContent(stepName) {
        switch (stepName) {
            case 'welcome':
                return this.getWelcomeStep();
            case 'personal-info':
                return this.getPersonalInfoStep();
            case 'identity-verification':
                return this.getIdentityVerificationStep();
            case 'payment-setup':
                return this.getPaymentSetupStep();
            case 'risk-assessment':
                return this.getRiskAssessmentStep();
            case 'account-confirmation':
                return this.getAccountConfirmationStep();
            default:
                return '<div class="p-8"><h2>Unknown Step</h2></div>';
        }
    }
    
    getWelcomeStep() {
        return `
            <div class="p-8 text-center">
                <div class="mb-6">
                    <div class="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <i data-lucide="trending-up" class="w-12 h-12 text-blue-600"></i>
                    </div>
                    <h2 class="text-3xl font-bold text-gray-900 mb-4">Welcome to Professional AI Trading</h2>
                    <p class="text-lg text-gray-600 mb-8">
                        Join thousands of traders using advanced AI algorithms and hyperbolic portfolio optimization 
                        to maximize returns while minimizing risk.
                    </p>
                </div>
                
                <!-- Key Features -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="text-center">
                        <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                            <i data-lucide="shield-check" class="w-8 h-8 text-green-600"></i>
                        </div>
                        <h3 class="font-semibold text-gray-900 mb-2">Secure & Regulated</h3>
                        <p class="text-sm text-gray-600">Bank-level security with full regulatory compliance</p>
                    </div>
                    
                    <div class="text-center">
                        <div class="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                            <i data-lucide="brain" class="w-8 h-8 text-purple-600"></i>
                        </div>
                        <h3 class="font-semibold text-gray-900 mb-2">AI-Powered Trading</h3>
                        <p class="text-sm text-gray-600">Advanced algorithms with overfitting prevention</p>
                    </div>
                    
                    <div class="text-center">
                        <div class="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-3">
                            <i data-lucide="dollar-sign" class="w-8 h-8 text-orange-600"></i>
                        </div>
                        <h3 class="font-semibold text-gray-900 mb-2">Transparent Fees</h3>
                        <p class="text-sm text-gray-600">No hidden costs, competitive pricing</p>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="flex gap-4 justify-center">
                    <button id="start-registration" class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors">
                        Get Started
                    </button>
                    <button id="demo-mode" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-8 py-3 rounded-lg font-semibold transition-colors">
                        Try Demo First
                    </button>
                </div>
                
                <p class="text-xs text-gray-500 mt-6">
                    By continuing, you agree to our Terms of Service and Privacy Policy
                </p>
            </div>
        `;
    }
    
    getPersonalInfoStep() {
        return `
            <div class="p-8">
                <div class="mb-6">
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Personal Information</h2>
                    <p class="text-gray-600">Please provide your basic information to create your account.</p>
                </div>
                
                <form id="personal-info-form" class="space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">First Name *</label>
                            <input type="text" id="firstName" required 
                                   class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Last Name *</label>
                            <input type="text" id="lastName" required
                                   class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Email Address *</label>
                        <input type="email" id="email" required
                               class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Phone Number *</label>
                        <input type="tel" id="phone" required
                               class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Date of Birth *</label>
                        <input type="date" id="dateOfBirth" required
                               class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Country of Residence *</label>
                        <select id="country" required
                                class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            <option value="">Select Country</option>
                            <option value="US">United States</option>
                            <option value="CA">Canada</option>
                            <option value="GB">United Kingdom</option>
                            <option value="AU">Australia</option>
                            <option value="DE">Germany</option>
                            <option value="FR">France</option>
                            <option value="JP">Japan</option>
                            <option value="SG">Singapore</option>
                            <!-- Add more countries as needed -->
                        </select>
                    </div>
                </form>
                
                <!-- Navigation -->
                <div class="flex justify-between items-center mt-8 pt-6 border-t">
                    <button id="back-btn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                        Back
                    </button>
                    <button id="next-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                        Continue
                    </button>
                </div>
            </div>
        `;
    }
    
    getIdentityVerificationStep() {
        return `
            <div class="p-8">
                <div class="mb-6">
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Identity Verification (KYC)</h2>
                    <p class="text-gray-600">For security and regulatory compliance, we need to verify your identity.</p>
                </div>
                
                <div class="space-y-6">
                    <!-- Document Type Selection -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Document Type *</label>
                        <select id="documentType" required
                                class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            <option value="">Select Document</option>
                            <option value="passport">Passport</option>
                            <option value="drivers_license">Driver's License</option>
                            <option value="national_id">National ID Card</option>
                        </select>
                    </div>
                    
                    <!-- Document Upload -->
                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                        <div class="mb-4">
                            <i data-lucide="upload" class="w-12 h-12 text-gray-400 mx-auto mb-2"></i>
                            <h3 class="text-lg font-semibold text-gray-900 mb-2">Upload Document</h3>
                            <p class="text-sm text-gray-600 mb-4">
                                Upload a clear photo of your ID document. Accepted formats: JPG, PNG, PDF (max 10MB)
                            </p>
                        </div>
                        
                        <input type="file" id="documentUpload" accept=".jpg,.jpeg,.png,.pdf" class="hidden">
                        <button id="upload-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                            Choose File
                        </button>
                        
                        <div id="upload-status" class="mt-4 hidden">
                            <div class="inline-flex items-center text-green-600">
                                <i data-lucide="check-circle" class="w-5 h-5 mr-2"></i>
                                Document uploaded successfully
                            </div>
                        </div>
                    </div>
                    
                    <!-- Address Verification -->
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Address Verification</h3>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Street Address *</label>
                                <input type="text" id="streetAddress" required
                                       class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">City *</label>
                                <input type="text" id="city" required
                                       class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">State/Province *</label>
                                <input type="text" id="state" required
                                       class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">ZIP/Postal Code *</label>
                                <input type="text" id="zipCode" required
                                       class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation -->
                <div class="flex justify-between items-center mt-8 pt-6 border-t">
                    <button id="back-btn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                        Back
                    </button>
                    <button id="next-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                        Verify Identity
                    </button>
                </div>
            </div>
        `;
    }
    
    getPaymentSetupStep() {
        return `
            <div class="p-8">
                <div class="mb-6">
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Payment Setup</h2>
                    <p class="text-gray-600">Connect your bank account or payment method to fund your trading account.</p>
                </div>
                
                <!-- Payment Method Selection -->
                <div class="space-y-6">
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Choose Payment Method</h3>
                        
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                            <!-- Bank Account -->
                            <div class="payment-method-card border-2 border-gray-200 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition-colors" data-method="bank">
                                <div class="text-center">
                                    <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                                        <i data-lucide="building-2" class="w-6 h-6 text-blue-600"></i>
                                    </div>
                                    <h4 class="font-semibold text-gray-900 mb-1">Bank Account</h4>
                                    <p class="text-xs text-gray-600">Connect via Plaid (Secure)</p>
                                </div>
                            </div>
                            
                            <!-- Debit Card -->
                            <div class="payment-method-card border-2 border-gray-200 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition-colors" data-method="card">
                                <div class="text-center">
                                    <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                                        <i data-lucide="credit-card" class="w-6 h-6 text-green-600"></i>
                                    </div>
                                    <h4 class="font-semibold text-gray-900 mb-1">Debit Card</h4>
                                    <p class="text-xs text-gray-600">Instant funding</p>
                                </div>
                            </div>
                            
                            <!-- Wire Transfer -->
                            <div class="payment-method-card border-2 border-gray-200 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition-colors" data-method="wire">
                                <div class="text-center">
                                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                                        <i data-lucide="zap" class="w-6 h-6 text-purple-600"></i>
                                    </div>
                                    <h4 class="font-semibold text-gray-900 mb-1">Wire Transfer</h4>
                                    <p class="text-xs text-gray-600">Large deposits</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Payment Form (Initially Hidden) -->
                    <div id="payment-form-container" class="hidden">
                        <!-- Bank Account Form -->
                        <div id="bank-form" class="payment-form hidden">
                            <h4 class="font-semibold text-gray-900 mb-4">Connect Bank Account</h4>
                            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                                <div class="flex items-start">
                                    <i data-lucide="shield-check" class="w-5 h-5 text-blue-600 mr-3 mt-0.5"></i>
                                    <div>
                                        <h5 class="font-semibold text-blue-900 mb-1">Secure Connection via Plaid</h5>
                                        <p class="text-sm text-blue-700">
                                            Your banking credentials are encrypted and never stored on our servers. 
                                            Plaid is trusted by thousands of financial institutions.
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <button id="connect-bank-btn" class="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-semibold transition-colors">
                                Connect with Plaid
                            </button>
                        </div>
                        
                        <!-- Card Form -->
                        <div id="card-form" class="payment-form hidden">
                            <h4 class="font-semibold text-gray-900 mb-4">Add Debit Card</h4>
                            <div class="space-y-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Card Number</label>
                                    <input type="text" id="cardNumber" placeholder="1234 5678 9012 3456"
                                           class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                </div>
                                
                                <div class="grid grid-cols-2 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">Expiry Date</label>
                                        <input type="text" id="cardExpiry" placeholder="MM/YY"
                                               class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                    </div>
                                    
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">CVV</label>
                                        <input type="text" id="cardCvv" placeholder="123"
                                               class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                    </div>
                                </div>
                                
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Cardholder Name</label>
                                    <input type="text" id="cardName" placeholder="John Smith"
                                           class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Wire Transfer Info -->
                        <div id="wire-form" class="payment-form hidden">
                            <h4 class="font-semibold text-gray-900 mb-4">Wire Transfer Instructions</h4>
                            <div class="bg-gray-50 border border-gray-200 rounded-lg p-6">
                                <div class="space-y-3">
                                    <div class="flex justify-between">
                                        <span class="font-medium text-gray-700">Bank Name:</span>
                                        <span class="text-gray-900">Gomna Trading Bank</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="font-medium text-gray-700">Account Name:</span>
                                        <span class="text-gray-900">Gomna Trading LLC</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="font-medium text-gray-700">Routing Number:</span>
                                        <span class="text-gray-900">021000021</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="font-medium text-gray-700">Account Number:</span>
                                        <span class="text-gray-900">1234567890</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="font-medium text-gray-700">Reference:</span>
                                        <span class="text-gray-900" id="wire-reference">Loading...</span>
                                    </div>
                                </div>
                            </div>
                            <p class="text-sm text-gray-600 mt-4">
                                Please include the reference number in your wire transfer. 
                                Funds typically arrive within 1-2 business days.
                            </p>
                        </div>
                    </div>
                    
                    <!-- Initial Deposit Amount -->
                    <div id="deposit-amount-section" class="hidden">
                        <h4 class="font-semibold text-gray-900 mb-4">Initial Deposit Amount</h4>
                        <div class="grid grid-cols-3 gap-4 mb-4">
                            <button class="deposit-amount-btn border-2 border-gray-200 rounded-lg p-4 text-center hover:border-blue-500 transition-colors" data-amount="1000">
                                <div class="text-2xl font-bold text-gray-900">$1,000</div>
                                <div class="text-sm text-gray-600">Starter</div>
                            </button>
                            
                            <button class="deposit-amount-btn border-2 border-blue-500 bg-blue-50 rounded-lg p-4 text-center" data-amount="5000">
                                <div class="text-2xl font-bold text-blue-900">$5,000</div>
                                <div class="text-sm text-blue-600">Popular</div>
                            </button>
                            
                            <button class="deposit-amount-btn border-2 border-gray-200 rounded-lg p-4 text-center hover:border-blue-500 transition-colors" data-amount="10000">
                                <div class="text-2xl font-bold text-gray-900">$10,000</div>
                                <div class="text-sm text-gray-600">Professional</div>
                            </button>
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Or enter custom amount</label>
                            <input type="number" id="customAmount" placeholder="Enter amount" min="100" step="100"
                                   class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        </div>
                    </div>
                </div>
                
                <!-- Navigation -->
                <div class="flex justify-between items-center mt-8 pt-6 border-t">
                    <button id="back-btn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                        Back
                    </button>
                    <button id="next-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors" disabled>
                        Continue
                    </button>
                </div>
            </div>
        `;
    }
    
    getRiskAssessmentStep() {
        return `
            <div class="p-8">
                <div class="mb-6">
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Risk Assessment</h2>
                    <p class="text-gray-600">Help us understand your investment experience and risk tolerance.</p>
                </div>
                
                <form id="risk-assessment-form" class="space-y-8">
                    <!-- Investment Experience -->
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Investment Experience</h3>
                        <div class="space-y-3">
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="experience" value="beginner" class="w-4 h-4 text-blue-600">
                                <span>Beginner (Less than 1 year)</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="experience" value="intermediate" class="w-4 h-4 text-blue-600">
                                <span>Intermediate (1-5 years)</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="experience" value="experienced" class="w-4 h-4 text-blue-600">
                                <span>Experienced (5+ years)</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="experience" value="professional" class="w-4 h-4 text-blue-600">
                                <span>Professional Trader</span>
                            </label>
                        </div>
                    </div>
                    
                    <!-- Risk Tolerance -->
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Risk Tolerance</h3>
                        <div class="space-y-3">
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="riskTolerance" value="conservative" class="w-4 h-4 text-blue-600">
                                <span>Conservative - Prefer steady, low-risk returns</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="riskTolerance" value="moderate" class="w-4 h-4 text-blue-600">
                                <span>Moderate - Balanced risk and return</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="riskTolerance" value="aggressive" class="w-4 h-4 text-blue-600">
                                <span>Aggressive - Higher risk for higher returns</span>
                            </label>
                        </div>
                    </div>
                    
                    <!-- Investment Goals -->
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Investment Goals</h3>
                        <div class="space-y-3">
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="goals" value="retirement" class="w-4 h-4 text-blue-600">
                                <span>Retirement Planning</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="goals" value="income" class="w-4 h-4 text-blue-600">
                                <span>Regular Income Generation</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="goals" value="growth" class="w-4 h-4 text-blue-600">
                                <span>Capital Growth</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="goals" value="diversification" class="w-4 h-4 text-blue-600">
                                <span>Portfolio Diversification</span>
                            </label>
                        </div>
                    </div>
                    
                    <!-- Investment Timeline -->
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Investment Timeline</h3>
                        <div class="space-y-3">
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="timeline" value="short" class="w-4 h-4 text-blue-600">
                                <span>Short-term (Less than 2 years)</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="timeline" value="medium" class="w-4 h-4 text-blue-600">
                                <span>Medium-term (2-5 years)</span>
                            </label>
                            <label class="flex items-center space-x-3 cursor-pointer">
                                <input type="radio" name="timeline" value="long" class="w-4 h-4 text-blue-600">
                                <span>Long-term (5+ years)</span>
                            </label>
                        </div>
                    </div>
                </form>
                
                <!-- Navigation -->
                <div class="flex justify-between items-center mt-8 pt-6 border-t">
                    <button id="back-btn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                        Back
                    </button>
                    <button id="next-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                        Complete Assessment
                    </button>
                </div>
            </div>
        `;
    }
    
    getAccountConfirmationStep() {
        return `
            <div class="p-8 text-center">
                <div class="mb-8">
                    <div class="w-24 h-24 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i data-lucide="check-circle" class="w-12 h-12 text-green-600"></i>
                    </div>
                    
                    <h2 class="text-3xl font-bold text-gray-900 mb-4">Account Setup Complete!</h2>
                    <p class="text-lg text-gray-600 mb-8">
                        Congratulations! Your Gomna™ AI Trading account has been successfully created and verified.
                    </p>
                </div>
                
                <!-- Account Summary -->
                <div class="bg-gray-50 rounded-xl p-6 mb-8">
                    <h3 class="text-xl font-semibold text-gray-900 mb-6">Account Summary</h3>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-left">
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-3">Personal Information</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Name:</span>
                                    <span id="confirm-name">-</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Email:</span>
                                    <span id="confirm-email">-</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Country:</span>
                                    <span id="confirm-country">-</span>
                                </div>
                            </div>
                        </div>
                        
                        <div>
                            <h4 class="font-semibold text-gray-700 mb-3">Account Details</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Account Type:</span>
                                    <span class="text-blue-600 font-semibold">Professional Trading</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Risk Profile:</span>
                                    <span id="confirm-risk">-</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Initial Deposit:</span>
                                    <span id="confirm-deposit">-</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Next Steps -->
                <div class="text-left mb-8">
                    <h3 class="text-xl font-semibold text-gray-900 mb-4">What's Next?</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bg-blue-50 rounded-lg p-4">
                            <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center mb-3">
                                <span class="text-white font-bold text-sm">1</span>
                            </div>
                            <h4 class="font-semibold text-gray-900 mb-2">Fund Your Account</h4>
                            <p class="text-sm text-gray-600">Your payment method is set up. Make your first deposit to start trading.</p>
                        </div>
                        
                        <div class="bg-green-50 rounded-lg p-4">
                            <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center mb-3">
                                <span class="text-white font-bold text-sm">2</span>
                            </div>
                            <h4 class="font-semibold text-gray-900 mb-2">Explore AI Features</h4>
                            <p class="text-sm text-gray-600">Set up your AI trading agents and explore advanced portfolio optimization.</p>
                        </div>
                        
                        <div class="bg-purple-50 rounded-lg p-4">
                            <div class="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center mb-3">
                                <span class="text-white font-bold text-sm">3</span>
                            </div>
                            <h4 class="font-semibold text-gray-900 mb-2">Start Trading</h4>
                            <p class="text-sm text-gray-600">Begin with manual trades or let our AI agents optimize your portfolio.</p>
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="flex gap-4 justify-center">
                    <button id="enter-platform-btn" class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors">
                        Enter Trading Platform
                    </button>
                    <button id="start-demo-btn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-8 py-4 rounded-lg font-semibold text-lg transition-colors">
                        Try Demo Mode
                    </button>
                </div>
            </div>
        `;
    }
    
    setupOnboardingListeners() {
        // This will be called after overlay is created
        console.log('🔧 Setting up onboarding listeners...');
    }
    
    setupStepListeners(stepName) {
        // Setup listeners specific to each step
        switch (stepName) {
            case 'welcome':
                this.setupWelcomeListeners();
                break;
            case 'personal-info':
                this.setupPersonalInfoListeners();
                break;
            case 'identity-verification':
                this.setupIdentityListeners();
                break;
            case 'payment-setup':
                this.setupPaymentListeners();
                break;
            case 'risk-assessment':
                this.setupRiskListeners();
                break;
            case 'account-confirmation':
                this.setupConfirmationListeners();
                break;
        }
        
        // Always setup Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    setupWelcomeListeners() {
        const startBtn = document.getElementById('start-registration');
        const demoBtn = document.getElementById('demo-mode');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.showStep('personal-info');
            });
        }
        
        if (demoBtn) {
            demoBtn.addEventListener('click', () => {
                this.skipToDemo();
            });
        }
    }
    
    setupPersonalInfoListeners() {
        const nextBtn = document.getElementById('next-btn');
        const backBtn = document.getElementById('back-btn');
        const form = document.getElementById('personal-info-form');
        
        if (nextBtn && form) {
            nextBtn.addEventListener('click', () => {
                if (this.validatePersonalInfo()) {
                    this.savePersonalInfo();
                    this.showStep('identity-verification');
                }
            });
        }
        
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showStep('welcome');
            });
        }
    }
    
    setupIdentityListeners() {
        const nextBtn = document.getElementById('next-btn');
        const backBtn = document.getElementById('back-btn');
        const uploadBtn = document.getElementById('upload-btn');
        const fileInput = document.getElementById('documentUpload');
        
        if (uploadBtn && fileInput) {
            uploadBtn.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    const fileName = e.target.files[0].name;
                    uploadBtn.textContent = fileName;
                    const status = document.getElementById('upload-status');
                    if (status) {
                        status.classList.remove('hidden');
                    }
                }
            });
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                if (this.validateIdentityVerification()) {
                    this.saveIdentityInfo();
                    this.showStep('payment-setup');
                }
            });
        }
        
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showStep('personal-info');
            });
        }
    }
    
    setupPaymentListeners() {
        const nextBtn = document.getElementById('next-btn');
        const backBtn = document.getElementById('back-btn');
        const methodCards = document.querySelectorAll('.payment-method-card');
        const connectBankBtn = document.getElementById('connect-bank-btn');
        const amountBtns = document.querySelectorAll('.deposit-amount-btn');
        
        // Payment method selection
        methodCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove active class from all cards
                methodCards.forEach(c => c.classList.remove('border-blue-500', 'bg-blue-50'));
                
                // Add active class to selected card
                card.classList.add('border-blue-500', 'bg-blue-50');
                
                // Show appropriate form
                const method = card.dataset.method;
                this.showPaymentForm(method);
            });
        });
        
        // Amount selection
        amountBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                amountBtns.forEach(b => b.classList.remove('border-blue-500', 'bg-blue-50'));
                btn.classList.add('border-blue-500', 'bg-blue-50');
                
                this.userProfile.depositAmount = parseInt(btn.dataset.amount);
                if (nextBtn) nextBtn.disabled = false;
            });
        });
        
        if (connectBankBtn) {
            connectBankBtn.addEventListener('click', () => {
                this.simulateBankConnection();
            });
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                if (this.validatePaymentSetup()) {
                    this.savePaymentInfo();
                    this.showStep('risk-assessment');
                }
            });
        }
        
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showStep('identity-verification');
            });
        }
    }
    
    setupRiskListeners() {
        const nextBtn = document.getElementById('next-btn');
        const backBtn = document.getElementById('back-btn');
        const form = document.getElementById('risk-assessment-form');
        
        if (nextBtn && form) {
            nextBtn.addEventListener('click', () => {
                if (this.validateRiskAssessment()) {
                    this.saveRiskAssessment();
                    this.showStep('account-confirmation');
                }
            });
        }
        
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showStep('payment-setup');
            });
        }
    }
    
    setupConfirmationListeners() {
        const enterBtn = document.getElementById('enter-platform-btn');
        const demoBtn = document.getElementById('start-demo-btn');
        
        // Populate confirmation details
        this.populateConfirmationDetails();
        
        if (enterBtn) {
            enterBtn.addEventListener('click', () => {
                this.completeRegistration();
            });
        }
        
        if (demoBtn) {
            demoBtn.addEventListener('click', () => {
                this.skipToDemo();
            });
        }
    }
    
    // Validation Methods
    validatePersonalInfo() {
        const requiredFields = ['firstName', 'lastName', 'email', 'phone', 'dateOfBirth', 'country'];
        const missingFields = [];
        
        for (const field of requiredFields) {
            const element = document.getElementById(field);
            if (!element || !element.value.trim()) {
                missingFields.push(field);
            }
        }
        
        if (missingFields.length > 0) {
            alert(`Please fill in all required fields: ${missingFields.join(', ')}`);
            return false;
        }
        
        // Email validation
        const email = document.getElementById('email').value;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            alert('Please enter a valid email address');
            return false;
        }
        
        return true;
    }
    
    validateIdentityVerification() {
        const documentType = document.getElementById('documentType').value;
        const fileInput = document.getElementById('documentUpload');
        const requiredAddressFields = ['streetAddress', 'city', 'state', 'zipCode'];
        
        if (!documentType) {
            alert('Please select a document type');
            return false;
        }
        
        if (!fileInput.files.length) {
            alert('Please upload your identity document');
            return false;
        }
        
        for (const field of requiredAddressFields) {
            const element = document.getElementById(field);
            if (!element || !element.value.trim()) {
                alert('Please fill in all address fields');
                return false;
            }
        }
        
        return true;
    }
    
    validatePaymentSetup() {
        if (!this.userProfile.paymentMethod) {
            alert('Please select a payment method');
            return false;
        }
        
        if (!this.userProfile.depositAmount || this.userProfile.depositAmount < 100) {
            alert('Please select a deposit amount (minimum $100)');
            return false;
        }
        
        return true;
    }
    
    validateRiskAssessment() {
        const experience = document.querySelector('input[name="experience"]:checked');
        const riskTolerance = document.querySelector('input[name="riskTolerance"]:checked');
        const timeline = document.querySelector('input[name="timeline"]:checked');
        
        if (!experience || !riskTolerance || !timeline) {
            alert('Please answer all risk assessment questions');
            return false;
        }
        
        return true;
    }
    
    // Save Methods
    savePersonalInfo() {
        const fields = ['firstName', 'lastName', 'email', 'phone', 'dateOfBirth', 'country'];
        fields.forEach(field => {
            const element = document.getElementById(field);
            if (element) {
                this.userProfile[field] = element.value;
            }
        });
    }
    
    saveIdentityInfo() {
        this.userProfile.documentType = document.getElementById('documentType').value;
        this.userProfile.documentUploaded = true;
        
        const addressFields = ['streetAddress', 'city', 'state', 'zipCode'];
        addressFields.forEach(field => {
            const element = document.getElementById(field);
            if (element) {
                this.userProfile[field] = element.value;
            }
        });
        
        this.userProfile.identityVerified = true;
    }
    
    savePaymentInfo() {
        // Payment information is already saved in showPaymentForm and event listeners
    }
    
    saveRiskAssessment() {
        const experience = document.querySelector('input[name="experience"]:checked');
        const riskTolerance = document.querySelector('input[name="riskTolerance"]:checked');
        const timeline = document.querySelector('input[name="timeline"]:checked');
        
        if (experience) this.userProfile.experience = experience.value;
        if (riskTolerance) this.userProfile.riskTolerance = riskTolerance.value;
        if (timeline) this.userProfile.timeline = timeline.value;
        
        // Save selected goals
        const goals = [];
        document.querySelectorAll('input[name="goals"]:checked').forEach(checkbox => {
            goals.push(checkbox.value);
        });
        this.userProfile.goals = goals;
    }
    
    // Helper Methods
    showPaymentForm(method) {
        // Hide all forms
        document.querySelectorAll('.payment-form').forEach(form => {
            form.classList.add('hidden');
        });
        
        // Show selected form
        const formContainer = document.getElementById('payment-form-container');
        const depositSection = document.getElementById('deposit-amount-section');
        
        if (formContainer) {
            formContainer.classList.remove('hidden');
            
            const selectedForm = document.getElementById(`${method}-form`);
            if (selectedForm) {
                selectedForm.classList.remove('hidden');
            }
        }
        
        if (depositSection) {
            depositSection.classList.remove('hidden');
        }
        
        // Generate wire reference if needed
        if (method === 'wire') {
            const wireRef = document.getElementById('wire-reference');
            if (wireRef) {
                wireRef.textContent = `GOMNA-${Date.now().toString().slice(-6)}`;
            }
        }
        
        this.userProfile.paymentMethod = method;
    }
    
    simulateBankConnection() {
        const btn = document.getElementById('connect-bank-btn');
        if (btn) {
            btn.innerHTML = '<i data-lucide="loader" class="inline w-4 h-4 mr-2 animate-spin"></i>Connecting...';
            btn.disabled = true;
            
            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="check" class="inline w-4 h-4 mr-2"></i>Bank Connected';
                btn.classList.replace('bg-blue-600', 'bg-green-600');
                btn.classList.replace('hover:bg-blue-700', 'hover:bg-green-700');
                
                this.userProfile.bankConnected = true;
                
                const nextBtn = document.getElementById('next-btn');
                if (nextBtn) nextBtn.disabled = false;
            }, 2000);
        }
    }
    
    populateConfirmationDetails() {
        const confirmName = document.getElementById('confirm-name');
        const confirmEmail = document.getElementById('confirm-email');
        const confirmCountry = document.getElementById('confirm-country');
        const confirmRisk = document.getElementById('confirm-risk');
        const confirmDeposit = document.getElementById('confirm-deposit');
        
        if (confirmName) confirmName.textContent = `${this.userProfile.firstName} ${this.userProfile.lastName}`;
        if (confirmEmail) confirmEmail.textContent = this.userProfile.email;
        if (confirmCountry) confirmCountry.textContent = this.userProfile.country;
        if (confirmRisk) confirmRisk.textContent = this.userProfile.riskTolerance || 'Not specified';
        if (confirmDeposit) confirmDeposit.textContent = this.userProfile.depositAmount ? `$${this.userProfile.depositAmount.toLocaleString()}` : 'Not specified';
    }
    
    skipToDemo() {
        // Create minimal demo profile
        this.userProfile = {
            firstName: 'Demo',
            lastName: 'User',
            email: 'demo@example.com',
            accountType: 'demo',
            registrationComplete: true,
            verified: false,
            demoMode: true
        };
        
        this.completeRegistration(true);
    }
    
    completeRegistration(isDemoMode = false) {
        // Mark registration as complete
        this.userProfile.registrationComplete = true;
        this.userProfile.verified = !isDemoMode;
        this.userProfile.registrationDate = new Date().toISOString();
        
        // Save to localStorage
        localStorage.setItem('gomna-user-profile', JSON.stringify(this.userProfile));
        
        // Show success message
        if (!isDemoMode) {
            alert('🎉 Registration complete! Welcome to Gomna™ AI Trading Platform.');
        }
        
        // Remove onboarding overlay and show main dashboard
        this.showMainDashboard();
    }
    
    showMainDashboard() {
        // Remove onboarding overlay
        const overlay = document.getElementById('gomna-onboarding-overlay');
        if (overlay) {
            overlay.remove();
        }
        
        // Show main content
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.style.display = 'block';
        }
        
        // Update UI to reflect user registration
        this.updateUIForRegisteredUser();
    }
    
    updateUIForRegisteredUser() {
        // Update header to show user info
        this.updateUserInterface();
        
        // Enable trading features
        this.enableTradingFeatures();
        
        console.log('✅ User registration complete, platform ready');
    }
    
    updateUserInterface() {
        // Add user profile section to header or create user menu
        const header = document.querySelector('header, .header, .main-header');
        if (header && this.userProfile.firstName) {
            const userInfo = document.createElement('div');
            userInfo.className = 'user-info flex items-center gap-2 text-sm';
            userInfo.innerHTML = `
                <div class="flex items-center gap-2">
                    <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <span class="text-blue-600 font-semibold">${this.userProfile.firstName.charAt(0)}</span>
                    </div>
                    <span class="text-gray-700">Welcome, ${this.userProfile.firstName}</span>
                    ${this.userProfile.demoMode ? '<span class="bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded">Demo</span>' : ''}
                </div>
            `;
            
            // Find a suitable place to insert user info
            const existingUserInfo = header.querySelector('.user-info');
            if (existingUserInfo) {
                existingUserInfo.replaceWith(userInfo);
            } else {
                header.appendChild(userInfo);
            }
        }
    }
    
    enableTradingFeatures() {
        // Remove any restrictions on trading features
        const restrictedElements = document.querySelectorAll('[data-requires-registration]');
        restrictedElements.forEach(element => {
            element.removeAttribute('disabled');
            element.classList.remove('opacity-50', 'cursor-not-allowed');
        });
        
        // Show welcome notification
        setTimeout(() => {
            this.showWelcomeNotification();
        }, 1000);
    }
    
    showWelcomeNotification() {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white p-4 rounded-lg shadow-lg z-50 animate-pulse-slow';
        notification.innerHTML = `
            <div class="flex items-center gap-3">
                <i data-lucide="check-circle" class="w-5 h-5"></i>
                <div>
                    <div class="font-semibold">Welcome to Gomna™ AI Trading!</div>
                    <div class="text-sm opacity-90">${this.userProfile.demoMode ? 'Demo mode active' : 'Account ready for trading'}</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Small delay to ensure other systems are loaded
    setTimeout(() => {
        window.gomnaOnboarding = new GomnaUserOnboardingSystem();
    }, 100);
});

// Export for global access
window.GomnaUserOnboardingSystem = GomnaUserOnboardingSystem;

console.log('🚀 Gomna User Onboarding System loaded successfully');