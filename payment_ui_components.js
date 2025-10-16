/**
 * GOMNA PAYMENT UI COMPONENTS
 * Interactive UI components for payment system integration
 * User-friendly onboarding, deposit/withdrawal, and account management
 */

class PaymentUIManager {
    constructor(paymentSystem) {
        this.paymentSystem = paymentSystem;
        this.currentModal = null;
        this.components = new Map();
        this.init();
    }

    init() {
        this.createBaseStyles();
        this.setupGlobalEventListeners();
    }

    /**
     * ONBOARDING UI COMPONENTS
     */
    createOnboardingModal(userData = {}) {
        const modalId = 'gomna-onboarding-modal';
        
        const modalHTML = `
            <div id="${modalId}" class="gomna-modal-overlay">
                <div class="gomna-modal-container onboarding-modal">
                    <div class="modal-header">
                        <div class="cocoa-pod-branding">
                            <div class="gomna-logo-container">
                                <div class="gomna-logo-3d gomna-logo-small">
                                    <div class="cocoa-pod-shell"></div>
                                    <div class="pod-interior">
                                        <div class="pod-placenta"></div>
                                        ${Array.from({length: 10}, () => '<div class="cocoa-seed"></div>').join('')}
                                        <div class="pod-pulp"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <h2>Welcome to Gomna™ Trading</h2>
                        <p>Start your organic growth journey with secure account setup</p>
                        <button class="modal-close" onclick="this.closest('.gomna-modal-overlay').remove()">&times;</button>
                    </div>
                    
                    <div class="onboarding-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 20%"></div>
                        </div>
                        <div class="progress-steps">
                            <div class="step active">1. Personal Info</div>
                            <div class="step">2. Identity</div>
                            <div class="step">3. Bank Account</div>
                            <div class="step">4. Preferences</div>
                            <div class="step">5. Activate</div>
                        </div>
                    </div>

                    <div class="onboarding-content">
                        <div id="step-1" class="onboarding-step active">
                            <h3>Personal Information</h3>
                            <form id="personal-info-form" class="gomna-form">
                                <div class="form-row">
                                    <div class="form-group">
                                        <label for="firstName">First Name</label>
                                        <input type="text" id="firstName" name="firstName" required 
                                               value="${userData.firstName || ''}" />
                                    </div>
                                    <div class="form-group">
                                        <label for="lastName">Last Name</label>
                                        <input type="text" id="lastName" name="lastName" required 
                                               value="${userData.lastName || ''}" />
                                    </div>
                                </div>
                                <div class="form-row">
                                    <div class="form-group">
                                        <label for="email">Email Address</label>
                                        <input type="email" id="email" name="email" required 
                                               value="${userData.email || ''}" />
                                    </div>
                                    <div class="form-group">
                                        <label for="phone">Phone Number</label>
                                        <input type="tel" id="phone" name="phone" required 
                                               value="${userData.phone || ''}" />
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="dateOfBirth">Date of Birth</label>
                                    <input type="date" id="dateOfBirth" name="dateOfBirth" required 
                                           value="${userData.dateOfBirth || ''}" />
                                </div>
                            </form>
                        </div>

                        <div id="step-2" class="onboarding-step">
                            <h3>Identity Verification</h3>
                            <div class="kyc-section">
                                <div class="info-box">
                                    <p>📋 We need to verify your identity for security and regulatory compliance.</p>
                                </div>
                                <form id="identity-form" class="gomna-form">
                                    <div class="form-group">
                                        <label for="ssn">Social Security Number</label>
                                        <input type="password" id="ssn" name="ssn" 
                                               placeholder="XXX-XX-XXXX" required />
                                    </div>
                                    <div class="form-group">
                                        <label for="address">Street Address</label>
                                        <input type="text" id="address" name="address" required />
                                    </div>
                                    <div class="form-row">
                                        <div class="form-group">
                                            <label for="city">City</label>
                                            <input type="text" id="city" name="city" required />
                                        </div>
                                        <div class="form-group">
                                            <label for="state">State</label>
                                            <select id="state" name="state" required>
                                                <option value="">Select State</option>
                                                <!-- Add US states -->
                                                <option value="CA">California</option>
                                                <option value="NY">New York</option>
                                                <option value="TX">Texas</option>
                                                <!-- Add more states -->
                                            </select>
                                        </div>
                                        <div class="form-group">
                                            <label for="zipCode">ZIP Code</label>
                                            <input type="text" id="zipCode" name="zipCode" required />
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </div>

                        <div id="step-3" class="onboarding-step">
                            <h3>Connect Bank Account</h3>
                            <div class="bank-connection-section">
                                <div class="info-box">
                                    <p>🏦 Connect your bank account for secure deposits and withdrawals.</p>
                                </div>
                                <div class="plaid-connect-container">
                                    <button id="connect-bank-btn" class="btn btn-primary btn-large">
                                        <span class="btn-icon">🔗</span>
                                        Connect Bank Account
                                    </button>
                                    <p class="security-note">🔒 Secured by 256-bit encryption and bank-level security</p>
                                </div>
                                <div id="bank-connection-status" class="connection-status" style="display: none;">
                                    <div class="status-success">
                                        ✅ Bank account connected successfully!
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div id="step-4" class="onboarding-step">
                            <h3>Trading Preferences</h3>
                            <form id="preferences-form" class="gomna-form">
                                <div class="preference-group">
                                    <h4>Account Features</h4>
                                    <div class="checkbox-group">
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="instantDeposits" checked />
                                            <span class="checkbox-custom"></span>
                                            Enable instant deposits (up to $1,000)
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="fractionalShares" checked />
                                            <span class="checkbox-custom"></span>
                                            Allow fractional share trading
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="cryptoTrading" checked />
                                            <span class="checkbox-custom"></span>
                                            Enable cryptocurrency trading
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="marginTrading" />
                                            <span class="checkbox-custom"></span>
                                            Enable margin trading (requires additional approval)
                                        </label>
                                    </div>
                                </div>
                                <div class="preference-group">
                                    <h4>Investment Profile</h4>
                                    <div class="radio-group">
                                        <label class="radio-label">
                                            <input type="radio" name="riskProfile" value="conservative" />
                                            <span class="radio-custom"></span>
                                            Conservative - Focus on capital preservation
                                        </label>
                                        <label class="radio-label">
                                            <input type="radio" name="riskProfile" value="moderate" checked />
                                            <span class="radio-custom"></span>
                                            Moderate - Balanced growth and risk
                                        </label>
                                        <label class="radio-label">
                                            <input type="radio" name="riskProfile" value="aggressive" />
                                            <span class="radio-custom"></span>
                                            Aggressive - Maximum growth potential
                                        </label>
                                    </div>
                                </div>
                            </form>
                        </div>

                        <div id="step-5" class="onboarding-step">
                            <h3>Account Activation</h3>
                            <div class="activation-section">
                                <div class="success-animation">
                                    <div class="cocoa-pod-celebration">
                                        🌱 → 🌿 → 🌳
                                    </div>
                                </div>
                                <h4>Welcome to Gomna™ Trading!</h4>
                                <p>Your account has been successfully created and verified.</p>
                                <div class="account-summary">
                                    <div class="summary-item">
                                        <span class="label">Account ID:</span>
                                        <span id="account-id" class="value">Loading...</span>
                                    </div>
                                    <div class="summary-item">
                                        <span class="label">Wallet ID:</span>
                                        <span id="wallet-id" class="value">Loading...</span>
                                    </div>
                                    <div class="summary-item">
                                        <span class="label">Starting Balance:</span>
                                        <span class="value">$0.00</span>
                                    </div>
                                </div>
                                <button id="start-trading-btn" class="btn btn-success btn-large">
                                    🚀 Start Trading
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="onboarding-actions">
                        <button id="prev-step-btn" class="btn btn-secondary" disabled>Previous</button>
                        <button id="next-step-btn" class="btn btn-primary">Next</button>
                    </div>
                </div>
            </div>
        `;

        // Insert modal into DOM
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Setup onboarding logic
        this.setupOnboardingFlow(modalId);
        
        return document.getElementById(modalId);
    }

    /**
     * DEPOSIT UI COMPONENT
     */
    createDepositModal(walletId) {
        const modalId = 'gomna-deposit-modal';
        
        const modalHTML = `
            <div id="${modalId}" class="gomna-modal-overlay">
                <div class="gomna-modal-container deposit-modal">
                    <div class="modal-header">
                        <h2>💰 Deposit Funds</h2>
                        <p>Add money to your trading account</p>
                        <button class="modal-close" onclick="this.closest('.gomna-modal-overlay').remove()">&times;</button>
                    </div>

                    <div class="deposit-content">
                        <form id="deposit-form" class="gomna-form">
                            <div class="form-group">
                                <label for="deposit-amount">Deposit Amount</label>
                                <div class="amount-input-container">
                                    <span class="currency-symbol">$</span>
                                    <input type="number" id="deposit-amount" name="amount" 
                                           placeholder="0.00" min="1" max="50000" step="0.01" required />
                                </div>
                                <div class="amount-suggestions">
                                    <button type="button" class="amount-btn" data-amount="25">$25</button>
                                    <button type="button" class="amount-btn" data-amount="100">$100</button>
                                    <button type="button" class="amount-btn" data-amount="500">$500</button>
                                    <button type="button" class="amount-btn" data-amount="1000">$1,000</button>
                                </div>
                            </div>

                            <div class="form-group">
                                <label for="payment-method">Payment Method</label>
                                <div id="payment-methods-list" class="payment-methods">
                                    <div class="payment-method-item" data-method-id="bank-001">
                                        <div class="method-icon">🏦</div>
                                        <div class="method-details">
                                            <div class="method-name">Chase Bank ****1234</div>
                                            <div class="method-type">Checking Account</div>
                                        </div>
                                        <div class="method-fee">Free</div>
                                    </div>
                                    <div class="add-payment-method">
                                        <button type="button" id="add-new-payment" class="btn btn-outline">
                                            + Add New Payment Method
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <div class="deposit-summary">
                                <div class="summary-row">
                                    <span>Deposit Amount:</span>
                                    <span id="summary-amount">$0.00</span>
                                </div>
                                <div class="summary-row">
                                    <span>Processing Fee:</span>
                                    <span id="summary-fee">$0.00</span>
                                </div>
                                <div class="summary-row total">
                                    <span>Total:</span>
                                    <span id="summary-total">$0.00</span>
                                </div>
                            </div>

                            <div class="deposit-timeline">
                                <h4>Processing Timeline</h4>
                                <div class="timeline-item">
                                    <div class="timeline-icon">⚡</div>
                                    <div class="timeline-text">
                                        <strong>Instant:</strong> Up to $1,000 available immediately
                                    </div>
                                </div>
                                <div class="timeline-item">
                                    <div class="timeline-icon">🏦</div>
                                    <div class="timeline-text">
                                        <strong>1-3 Business Days:</strong> Full amount cleared
                                    </div>
                                </div>
                            </div>

                            <button type="submit" class="btn btn-primary btn-large">
                                Deposit Funds
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.setupDepositFlow(modalId, walletId);
        
        return document.getElementById(modalId);
    }

    /**
     * WITHDRAWAL UI COMPONENT
     */
    createWithdrawalModal(walletId) {
        const modalId = 'gomna-withdrawal-modal';
        
        const modalHTML = `
            <div id="${modalId}" class="gomna-modal-overlay">
                <div class="gomna-modal-container withdrawal-modal">
                    <div class="modal-header">
                        <h2>💸 Withdraw Funds</h2>
                        <p>Transfer money to your bank account</p>
                        <button class="modal-close" onclick="this.closest('.gomna-modal-overlay').remove()">&times;</button>
                    </div>

                    <div class="withdrawal-content">
                        <div class="balance-info">
                            <div class="balance-item">
                                <span class="balance-label">Available Balance:</span>
                                <span id="available-balance" class="balance-value">Loading...</span>
                            </div>
                            <div class="balance-item">
                                <span class="balance-label">Pending Withdrawals:</span>
                                <span id="pending-withdrawals" class="balance-value">$0.00</span>
                            </div>
                        </div>

                        <form id="withdrawal-form" class="gomna-form">
                            <div class="form-group">
                                <label for="withdrawal-amount">Withdrawal Amount</label>
                                <div class="amount-input-container">
                                    <span class="currency-symbol">$</span>
                                    <input type="number" id="withdrawal-amount" name="amount" 
                                           placeholder="0.00" min="1" step="0.01" required />
                                </div>
                                <div class="amount-suggestions">
                                    <button type="button" class="amount-btn" data-action="withdraw-all">
                                        Withdraw All
                                    </button>
                                </div>
                            </div>

                            <div class="form-group">
                                <label for="withdrawal-method">Withdrawal Method</label>
                                <div id="withdrawal-methods-list" class="payment-methods">
                                    <div class="payment-method-item" data-method-id="bank-001">
                                        <div class="method-icon">🏦</div>
                                        <div class="method-details">
                                            <div class="method-name">Chase Bank ****1234</div>
                                            <div class="method-type">Checking Account</div>
                                        </div>
                                        <div class="method-timing">1-3 Days</div>
                                    </div>
                                </div>
                            </div>

                            <div class="withdrawal-summary">
                                <div class="summary-row">
                                    <span>Withdrawal Amount:</span>
                                    <span id="withdrawal-summary-amount">$0.00</span>
                                </div>
                                <div class="summary-row">
                                    <span>Processing Fee:</span>
                                    <span id="withdrawal-summary-fee">$0.00</span>
                                </div>
                                <div class="summary-row total">
                                    <span>You'll Receive:</span>
                                    <span id="withdrawal-summary-total">$0.00</span>
                                </div>
                            </div>

                            <div class="info-box warning">
                                <p>⚠️ Withdrawals typically take 1-3 business days to process. You cannot cancel a withdrawal once initiated.</p>
                            </div>

                            <button type="submit" class="btn btn-primary btn-large">
                                Withdraw Funds
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.setupWithdrawalFlow(modalId, walletId);
        
        return document.getElementById(modalId);
    }

    /**
     * SETUP METHODS
     */
    setupOnboardingFlow(modalId) {
        const modal = document.getElementById(modalId);
        let currentStep = 1;
        const maxSteps = 5;
        let onboardingData = {};

        // Navigation buttons
        const prevBtn = modal.querySelector('#prev-step-btn');
        const nextBtn = modal.querySelector('#next-step-btn');
        const steps = modal.querySelectorAll('.onboarding-step');
        const progressSteps = modal.querySelectorAll('.progress-steps .step');
        const progressBar = modal.querySelector('.progress-fill');

        const updateStep = (step) => {
            // Hide all steps
            steps.forEach(s => s.classList.remove('active'));
            progressSteps.forEach(s => s.classList.remove('active'));
            
            // Show current step
            modal.querySelector(`#step-${step}`).classList.add('active');
            progressSteps[step - 1].classList.add('active');
            
            // Update progress bar
            progressBar.style.width = `${(step / maxSteps) * 100}%`;
            
            // Update navigation buttons
            prevBtn.disabled = step === 1;
            nextBtn.textContent = step === maxSteps ? 'Complete Setup' : 'Next';
            
            currentStep = step;
        };

        nextBtn.addEventListener('click', async () => {
            if (currentStep < maxSteps) {
                // Validate current step
                const isValid = await this.validateOnboardingStep(currentStep, modal);
                if (isValid) {
                    updateStep(currentStep + 1);
                }
            } else {
                // Complete onboarding
                await this.completeOnboarding(modal, onboardingData);
            }
        });

        prevBtn.addEventListener('click', () => {
            if (currentStep > 1) {
                updateStep(currentStep - 1);
            }
        });

        // Bank connection
        modal.querySelector('#connect-bank-btn').addEventListener('click', async () => {
            try {
                const result = await this.paymentSystem.connectBankAccount('temp-user-id');
                if (result.success) {
                    modal.querySelector('#bank-connection-status').style.display = 'block';
                    onboardingData.bankAccountId = result.accountId;
                }
            } catch (error) {
                alert('Failed to connect bank account: ' + error.message);
            }
        });
    }

    setupDepositFlow(modalId, walletId) {
        const modal = document.getElementById(modalId);
        const form = modal.querySelector('#deposit-form');
        const amountInput = modal.querySelector('#deposit-amount');
        
        // Amount suggestions
        modal.querySelectorAll('.amount-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const amount = btn.dataset.amount;
                amountInput.value = amount;
                this.updateDepositSummary(modal, parseFloat(amount));
            });
        });

        // Amount input changes
        amountInput.addEventListener('input', (e) => {
            const amount = parseFloat(e.target.value) || 0;
            this.updateDepositSummary(modal, amount);
        });

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.processDeposit(modal, walletId);
        });
    }

    setupWithdrawalFlow(modalId, walletId) {
        const modal = document.getElementById(modalId);
        
        // Load wallet balance
        this.loadWalletBalance(modal, walletId);
        
        // Setup form handlers similar to deposit
        // ... (similar to deposit setup)
    }

    /**
     * HELPER METHODS
     */
    async validateOnboardingStep(step, modal) {
        switch (step) {
            case 1:
                return this.validatePersonalInfo(modal);
            case 2:
                return this.validateIdentityInfo(modal);
            case 3:
                return this.validateBankConnection(modal);
            case 4:
                return this.validatePreferences(modal);
            default:
                return true;
        }
    }

    updateDepositSummary(modal, amount) {
        const fee = 0; // No fee for bank transfers
        const total = amount + fee;
        
        modal.querySelector('#summary-amount').textContent = GomnaPaymentSystem.formatCurrency(amount);
        modal.querySelector('#summary-fee').textContent = GomnaPaymentSystem.formatCurrency(fee);
        modal.querySelector('#summary-total').textContent = GomnaPaymentSystem.formatCurrency(total);
    }

    async loadWalletBalance(modal, walletId) {
        try {
            const balanceResult = await this.paymentSystem.getWalletBalance(walletId);
            if (balanceResult.success) {
                modal.querySelector('#available-balance').textContent = 
                    GomnaPaymentSystem.formatCurrency(balanceResult.availableBalance);
                modal.querySelector('#pending-withdrawals').textContent = 
                    GomnaPaymentSystem.formatCurrency(balanceResult.pendingWithdrawals);
            }
        } catch (error) {
            console.error('Failed to load wallet balance:', error);
        }
    }

    /**
     * STYLING
     */
    createBaseStyles() {
        const styles = `
            <style id="gomna-payment-ui-styles">
                .gomna-modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                    animation: modal-fade-in 0.3s ease;
                }

                .gomna-modal-container {
                    background: white;
                    border-radius: 12px;
                    max-width: 600px;
                    max-height: 90vh;
                    overflow-y: auto;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                    animation: modal-slide-up 0.3s ease;
                }

                .modal-header {
                    padding: 24px;
                    border-bottom: 1px solid #e0e0e0;
                    position: relative;
                }

                .modal-close {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: none;
                    border: none;
                    font-size: 24px;
                    cursor: pointer;
                    color: #666;
                }

                .gomna-form {
                    padding: 24px;
                }

                .form-group {
                    margin-bottom: 20px;
                }

                .form-row {
                    display: flex;
                    gap: 16px;
                }

                .form-row .form-group {
                    flex: 1;
                }

                .form-group label {
                    display: block;
                    font-weight: 600;
                    margin-bottom: 8px;
                    color: #333;
                }

                .form-group input,
                .form-group select {
                    width: 100%;
                    padding: 12px;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    font-size: 14px;
                    transition: border-color 0.3s ease;
                }

                .form-group input:focus,
                .form-group select:focus {
                    outline: none;
                    border-color: #8B4513;
                    box-shadow: 0 0 0 3px rgba(139, 69, 19, 0.1);
                }

                .btn {
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-decoration: none;
                    display: inline-block;
                    text-align: center;
                }

                .btn-primary {
                    background: #8B4513;
                    color: white;
                }

                .btn-primary:hover {
                    background: #A0522D;
                }

                .btn-large {
                    padding: 16px 32px;
                    font-size: 16px;
                    width: 100%;
                }

                .onboarding-progress {
                    padding: 24px;
                    border-bottom: 1px solid #e0e0e0;
                }

                .progress-bar {
                    height: 4px;
                    background: #e0e0e0;
                    border-radius: 2px;
                    margin-bottom: 16px;
                }

                .progress-fill {
                    height: 100%;
                    background: #8B4513;
                    border-radius: 2px;
                    transition: width 0.3s ease;
                }

                .progress-steps {
                    display: flex;
                    justify-content: space-between;
                }

                .step {
                    font-size: 12px;
                    color: #666;
                    font-weight: 500;
                }

                .step.active {
                    color: #8B4513;
                    font-weight: 600;
                }

                .onboarding-step {
                    display: none;
                    min-height: 300px;
                }

                .onboarding-step.active {
                    display: block;
                }

                .amount-input-container {
                    position: relative;
                }

                .currency-symbol {
                    position: absolute;
                    left: 12px;
                    top: 12px;
                    color: #666;
                    font-weight: 600;
                }

                .amount-input-container input {
                    padding-left: 32px;
                }

                .amount-suggestions {
                    display: flex;
                    gap: 8px;
                    margin-top: 8px;
                }

                .amount-btn {
                    padding: 8px 16px;
                    background: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                }

                .amount-btn:hover {
                    background: #e0e0e0;
                }

                @keyframes modal-fade-in {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }

                @keyframes modal-slide-up {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
            </style>
        `;

        if (!document.querySelector('#gomna-payment-ui-styles')) {
            document.head.insertAdjacentHTML('beforeend', styles);
        }
    }

    setupGlobalEventListeners() {
        // Close modals on overlay click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('gomna-modal-overlay')) {
                e.target.remove();
            }
        });

        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.querySelector('.gomna-modal-overlay');
                if (modal) {
                    modal.remove();
                }
            }
        });
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PaymentUIManager;
}

// Global access
window.PaymentUIManager = PaymentUIManager;

console.log('🎨 Payment UI Components loaded successfully');