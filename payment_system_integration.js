/**
 * GOMNA COMPREHENSIVE PAYMENT SYSTEM INTEGRATION
 * Complete payment processing with KYC, user onboarding, and trading wallet management
 * Supports Stripe, Plaid, and modern fintech onboarding flows
 */

// Browser compatibility fix
if (typeof process === 'undefined') {
    window.process = { env: {} };
}

console.log('🔄 Loading Gomna Payment System Integration...');

class GomnaPaymentSystem {
    constructor(config = {}) {
        this.config = {
            stripePublicKey: config.stripePublicKey || (typeof process !== 'undefined' ? process.env.STRIPE_PUBLIC_KEY : null),
            plaidPublicKey: config.plaidPublicKey || (typeof process !== 'undefined' ? process.env.PLAID_PUBLIC_KEY : null),
            environment: config.environment || 'sandbox',
            apiBaseUrl: config.apiBaseUrl || '/api',
            kycProvider: config.kycProvider || 'plaid-identity',
            ...config
        };
        
        this.stripe = null;
        this.plaid = null;
        this.currentUser = null;
        this.paymentMethods = new Map();
        this.transactions = new Map();
        
        this.init();
    }

    async init() {
        try {
            await this.loadPaymentProviders();
            this.setupEventListeners();
            console.log('🏦 Gomna Payment System initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize payment system:', error);
        }
    }

    /**
     * PAYMENT PROVIDER SETUP
     */
    async loadPaymentProviders() {
        // Load Stripe
        if (this.config.stripePublicKey) {
            if (!window.Stripe) {
                await this.loadScript('https://js.stripe.com/v3/');
            }
            this.stripe = Stripe(this.config.stripePublicKey);
        }

        // Load Plaid
        if (this.config.plaidPublicKey) {
            if (!window.Plaid) {
                await this.loadScript('https://cdn.plaid.com/link/v2/stable/link-initialize.js');
            }
        }
    }

    async loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    /**
     * USER ONBOARDING & KYC SYSTEM
     */
    async startUserOnboarding(userData) {
        const onboardingFlow = new UserOnboardingFlow(this, userData);
        return await onboardingFlow.start();
    }

    /**
     * KYC VERIFICATION PROCESS
     */
    async initiateKYC(userId, identityData) {
        try {
            const kycResult = await this.apiCall('/kyc/initiate', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    identityData,
                    provider: this.config.kycProvider
                })
            });

            return {
                success: true,
                kycId: kycResult.kycId,
                status: kycResult.status,
                nextSteps: kycResult.nextSteps
            };
        } catch (error) {
            console.error('KYC initiation failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * BANK ACCOUNT CONNECTION (PLAID INTEGRATION)
     */
    async connectBankAccount(userId) {
        if (!window.Plaid) {
            throw new Error('Plaid not loaded');
        }

        return new Promise((resolve, reject) => {
            const handler = Plaid.create({
                token: this.config.plaidPublicKey,
                env: this.config.environment === 'production' ? 'production' : 'sandbox',
                product: ['auth', 'identity', 'transactions'],
                onSuccess: async (public_token, metadata) => {
                    try {
                        // Exchange public token for access token
                        const result = await this.apiCall('/plaid/exchange-token', {
                            method: 'POST',
                            body: JSON.stringify({
                                public_token,
                                userId,
                                metadata
                            })
                        });

                        resolve({
                            success: true,
                            accountId: result.accountId,
                            accounts: result.accounts,
                            institution: metadata.institution
                        });
                    } catch (error) {
                        reject(error);
                    }
                },
                onExit: (err, metadata) => {
                    if (err) {
                        reject(new Error(`Plaid Link exited with error: ${err.error_message}`));
                    } else {
                        resolve({ success: false, cancelled: true });
                    }
                },
                onEvent: (eventName, metadata) => {
                    console.log('Plaid event:', eventName, metadata);
                }
            });

            handler.open();
        });
    }

    /**
     * PAYMENT METHOD MANAGEMENT
     */
    async addPaymentMethod(userId, paymentMethodData) {
        try {
            const { type, data } = paymentMethodData;
            let result;

            switch (type) {
                case 'card':
                    result = await this.addCardPaymentMethod(userId, data);
                    break;
                case 'bank':
                    result = await this.addBankPaymentMethod(userId, data);
                    break;
                case 'ach':
                    result = await this.addACHPaymentMethod(userId, data);
                    break;
                default:
                    throw new Error(`Unsupported payment method type: ${type}`);
            }

            if (result.success) {
                this.paymentMethods.set(result.paymentMethodId, {
                    id: result.paymentMethodId,
                    userId,
                    type,
                    data: result.data,
                    createdAt: new Date().toISOString()
                });
            }

            return result;
        } catch (error) {
            console.error('Failed to add payment method:', error);
            return { success: false, error: error.message };
        }
    }

    async addCardPaymentMethod(userId, cardData) {
        const { card } = await this.stripe.createPaymentMethod({
            type: 'card',
            card: cardData.cardElement,
            billing_details: cardData.billingDetails
        });

        // Attach to customer
        const result = await this.apiCall('/stripe/attach-payment-method', {
            method: 'POST',
            body: JSON.stringify({
                paymentMethodId: card.id,
                userId
            })
        });

        return {
            success: true,
            paymentMethodId: card.id,
            data: {
                brand: card.card.brand,
                last4: card.card.last4,
                expMonth: card.card.exp_month,
                expYear: card.card.exp_year
            }
        };
    }

    async addBankPaymentMethod(userId, bankData) {
        // Use Plaid for bank verification
        const plaidResult = await this.connectBankAccount(userId);
        
        if (!plaidResult.success) {
            return plaidResult;
        }

        return {
            success: true,
            paymentMethodId: plaidResult.accountId,
            data: {
                institutionName: plaidResult.institution.name,
                accountType: bankData.accountType,
                mask: bankData.mask
            }
        };
    }

    /**
     * TRADING WALLET MANAGEMENT
     */
    async createTradingWallet(userId, walletConfig = {}) {
        try {
            const walletData = {
                userId,
                currency: walletConfig.currency || 'USD',
                type: walletConfig.type || 'trading',
                features: {
                    instantDeposits: walletConfig.instantDeposits || false,
                    fractionalShares: walletConfig.fractionalShares || true,
                    cryptoTrading: walletConfig.cryptoTrading || true,
                    marginTrading: walletConfig.marginTrading || false
                }
            };

            const result = await this.apiCall('/wallet/create', {
                method: 'POST',
                body: JSON.stringify(walletData)
            });

            return {
                success: true,
                walletId: result.walletId,
                balance: result.balance || 0,
                features: result.features
            };
        } catch (error) {
            console.error('Failed to create trading wallet:', error);
            return { success: false, error: error.message };
        }
    }

    async getWalletBalance(walletId) {
        try {
            const result = await this.apiCall(`/wallet/${walletId}/balance`);
            return {
                success: true,
                balance: result.balance,
                availableBalance: result.availableBalance,
                pendingDeposits: result.pendingDeposits,
                pendingWithdrawals: result.pendingWithdrawals
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * DEPOSIT & WITHDRAWAL PROCESSING
     */
    async processDeposit(depositData) {
        const { userId, walletId, amount, paymentMethodId, currency = 'USD' } = depositData;

        try {
            // Validate minimum deposit amount
            if (amount < 1) {
                throw new Error('Minimum deposit amount is $1.00');
            }

            // Create deposit transaction
            const depositResult = await this.apiCall('/transactions/deposit', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    walletId,
                    amount: Math.round(amount * 100), // Convert to cents
                    currency,
                    paymentMethodId,
                    description: 'Trading account deposit'
                })
            });

            // Track transaction
            this.transactions.set(depositResult.transactionId, {
                id: depositResult.transactionId,
                type: 'deposit',
                amount,
                currency,
                status: depositResult.status,
                createdAt: new Date().toISOString()
            });

            return {
                success: true,
                transactionId: depositResult.transactionId,
                status: depositResult.status,
                estimatedSettlement: depositResult.estimatedSettlement
            };
        } catch (error) {
            console.error('Deposit processing failed:', error);
            return { success: false, error: error.message };
        }
    }

    async processWithdrawal(withdrawalData) {
        const { userId, walletId, amount, paymentMethodId, currency = 'USD' } = withdrawalData;

        try {
            // Check available balance
            const balanceResult = await this.getWalletBalance(walletId);
            if (!balanceResult.success || balanceResult.availableBalance < amount) {
                throw new Error('Insufficient available balance');
            }

            // Create withdrawal transaction
            const withdrawalResult = await this.apiCall('/transactions/withdrawal', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    walletId,
                    amount: Math.round(amount * 100), // Convert to cents
                    currency,
                    paymentMethodId,
                    description: 'Trading account withdrawal'
                })
            });

            // Track transaction
            this.transactions.set(withdrawalResult.transactionId, {
                id: withdrawalResult.transactionId,
                type: 'withdrawal',
                amount,
                currency,
                status: withdrawalResult.status,
                createdAt: new Date().toISOString()
            });

            return {
                success: true,
                transactionId: withdrawalResult.transactionId,
                status: withdrawalResult.status,
                estimatedSettlement: withdrawalResult.estimatedSettlement
            };
        } catch (error) {
            console.error('Withdrawal processing failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * TRADING FEES & COMMISSION PROCESSING
     */
    async processTradingFee(feeData) {
        const { userId, walletId, tradeId, feeType, amount, currency = 'USD' } = feeData;

        try {
            const feeResult = await this.apiCall('/transactions/trading-fee', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    walletId,
                    tradeId,
                    feeType, // 'commission', 'regulatory', 'spread', etc.
                    amount: Math.round(amount * 100),
                    currency,
                    description: `Trading fee - ${feeType}`
                })
            });

            return {
                success: true,
                transactionId: feeResult.transactionId,
                feeAmount: amount,
                newBalance: feeResult.newBalance
            };
        } catch (error) {
            console.error('Trading fee processing failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * TRANSACTION HISTORY & REPORTING
     */
    async getTransactionHistory(userId, filters = {}) {
        try {
            const queryParams = new URLSearchParams({
                userId,
                ...filters
            });

            const result = await this.apiCall(`/transactions/history?${queryParams}`);
            
            return {
                success: true,
                transactions: result.transactions,
                pagination: result.pagination,
                summary: result.summary
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * COMPLIANCE & RISK MANAGEMENT
     */
    async performAMLCheck(transactionData) {
        try {
            const amlResult = await this.apiCall('/compliance/aml-check', {
                method: 'POST',
                body: JSON.stringify(transactionData)
            });

            return {
                success: true,
                riskScore: amlResult.riskScore,
                flags: amlResult.flags,
                approved: amlResult.approved
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async checkDailyLimits(userId, amount, transactionType) {
        try {
            const limitResult = await this.apiCall('/compliance/check-limits', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    amount,
                    transactionType
                })
            });

            return {
                success: true,
                withinLimits: limitResult.withinLimits,
                dailyRemaining: limitResult.dailyRemaining,
                monthlyRemaining: limitResult.monthlyRemaining
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * UTILITY METHODS
     */
    async apiCall(endpoint, options = {}) {
        const url = `${this.config.apiBaseUrl}${endpoint}`;
        const defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        const response = await fetch(url, {
            headers: { ...defaultHeaders, ...(options.headers || {}) },
            ...options
        });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    setupEventListeners() {
        // Global payment system events
        document.addEventListener('payment:deposit-initiated', (e) => {
            console.log('Deposit initiated:', e.detail);
        });

        document.addEventListener('payment:withdrawal-initiated', (e) => {
            console.log('Withdrawal initiated:', e.detail);
        });

        document.addEventListener('payment:transaction-completed', (e) => {
            console.log('Transaction completed:', e.detail);
        });
    }

    /**
     * STATIC UTILITY METHODS
     */
    static formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    }

    static validatePaymentAmount(amount, min = 1, max = 50000) {
        const numAmount = parseFloat(amount);
        return !isNaN(numAmount) && numAmount >= min && numAmount <= max;
    }
}

/**
 * USER ONBOARDING FLOW CLASS
 */
class UserOnboardingFlow {
    constructor(paymentSystem, userData) {
        this.paymentSystem = paymentSystem;
        this.userData = userData;
        this.currentStep = 1;
        this.maxSteps = 5;
        this.onboardingData = {};
    }

    async start() {
        try {
            console.log('🚀 Starting user onboarding flow...');
            
            // Step 1: Basic Registration
            await this.step1_BasicRegistration();
            
            // Step 2: Identity Verification (KYC)
            await this.step2_IdentityVerification();
            
            // Step 3: Bank Account Connection
            await this.step3_BankConnection();
            
            // Step 4: Trading Preferences
            await this.step4_TradingPreferences();
            
            // Step 5: Account Activation
            await this.step5_AccountActivation();
            
            return {
                success: true,
                userId: this.onboardingData.userId,
                walletId: this.onboardingData.walletId,
                onboardingComplete: true
            };
        } catch (error) {
            console.error('Onboarding failed:', error);
            return { success: false, error: error.message, step: this.currentStep };
        }
    }

    async step1_BasicRegistration() {
        console.log('📝 Step 1: Basic Registration');
        
        // Create user account
        const userResult = await this.paymentSystem.apiCall('/users/create', {
            method: 'POST',
            body: JSON.stringify({
                email: this.userData.email,
                phone: this.userData.phone,
                firstName: this.userData.firstName,
                lastName: this.userData.lastName,
                dateOfBirth: this.userData.dateOfBirth
            })
        });

        this.onboardingData.userId = userResult.userId;
        this.currentStep = 2;
    }

    async step2_IdentityVerification() {
        console.log('🔍 Step 2: Identity Verification (KYC)');
        
        const kycResult = await this.paymentSystem.initiateKYC(
            this.onboardingData.userId,
            {
                firstName: this.userData.firstName,
                lastName: this.userData.lastName,
                dateOfBirth: this.userData.dateOfBirth,
                ssn: this.userData.ssn,
                address: this.userData.address
            }
        );

        if (!kycResult.success) {
            throw new Error('KYC verification failed');
        }

        this.onboardingData.kycId = kycResult.kycId;
        this.currentStep = 3;
    }

    async step3_BankConnection() {
        console.log('🏦 Step 3: Bank Account Connection');
        
        const bankResult = await this.paymentSystem.connectBankAccount(
            this.onboardingData.userId
        );

        if (!bankResult.success) {
            throw new Error('Bank account connection failed');
        }

        this.onboardingData.bankAccountId = bankResult.accountId;
        this.currentStep = 4;
    }

    async step4_TradingPreferences() {
        console.log('⚙️ Step 4: Trading Preferences');
        
        const walletResult = await this.paymentSystem.createTradingWallet(
            this.onboardingData.userId,
            {
                currency: 'USD',
                instantDeposits: true,
                fractionalShares: true,
                cryptoTrading: true,
                marginTrading: false
            }
        );

        if (!walletResult.success) {
            throw new Error('Trading wallet creation failed');
        }

        this.onboardingData.walletId = walletResult.walletId;
        this.currentStep = 5;
    }

    async step5_AccountActivation() {
        console.log('✅ Step 5: Account Activation');
        
        // Activate the account
        await this.paymentSystem.apiCall('/users/activate', {
            method: 'POST',
            body: JSON.stringify({
                userId: this.onboardingData.userId,
                kycId: this.onboardingData.kycId,
                walletId: this.onboardingData.walletId
            })
        });

        this.currentStep = 6; // Complete
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GomnaPaymentSystem, UserOnboardingFlow };
}

// Global access
window.GomnaPaymentSystem = GomnaPaymentSystem;
window.UserOnboardingFlow = UserOnboardingFlow;

console.log('💳 Gomna Payment System loaded successfully');