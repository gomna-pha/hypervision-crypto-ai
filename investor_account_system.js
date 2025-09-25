/**
 * GOMNA INVESTOR ACCOUNT REGISTRATION & AUTHENTICATION SYSTEM
 * Secure platform for investors to register, verify identity, and purchase algorithms
 * Real-time strategy performance drives investment decisions
 */

class InvestorAccountSystem {
    constructor() {
        this.currentUser = null;
        this.sessionToken = null;
        this.encryptionKey = null;
        this.apiCredentials = new Map();
        this.kycDocuments = new Map();
        this.investmentLimits = {
            unverified: 1000,    // $1,000 limit for unverified users
            verified: 50000,     // $50,000 limit for KYC verified users
            institutional: 1000000 // $1M limit for institutional accounts
        };
        
        this.init();
    }

    async init() {
        console.log('👤 Initializing Investor Account System...');
        
        // Initialize encryption for secure credential storage
        await this.initializeEncryption();
        
        // Check for existing session
        this.restoreSession();
        
        // Setup authentication UI
        this.setupAuthenticationUI();
        
        console.log('✅ Investor Account System ready');
    }

    async initializeEncryption() {
        // Generate encryption key for local credential storage
        if (!localStorage.getItem('encryption_salt')) {
            const salt = crypto.getRandomValues(new Uint8Array(16));
            localStorage.setItem('encryption_salt', Array.from(salt).join(','));
        }
    }

    setupAuthenticationUI() {
        // Add authentication panel to marketplace
        this.createAuthenticationPanel();
        this.setupAuthEventListeners();
    }

    createAuthenticationPanel() {
        const authPanel = document.createElement('div');
        authPanel.id = 'investor-auth-panel';
        authPanel.className = 'fixed top-0 left-0 w-full h-full bg-black bg-opacity-50 flex items-center justify-center z-50 hidden';
        authPanel.innerHTML = `
            <div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-8">
                <!-- Login/Register Tabs -->
                <div class="flex mb-6">
                    <button class="auth-tab flex-1 py-2 px-4 text-center border-b-2 border-blue-500 text-blue-500 font-medium" data-tab="login">
                        Sign In
                    </button>
                    <button class="auth-tab flex-1 py-2 px-4 text-center border-b-2 border-gray-200 text-gray-500" data-tab="register">
                        Create Account
                    </button>
                </div>

                <!-- Login Form -->
                <form id="login-form" class="auth-form">
                    <h2 class="text-2xl font-bold text-center mb-6">Welcome Back, Investor</h2>
                    <p class="text-center text-gray-600 mb-6">Access your algorithm portfolio</p>
                    
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                            <input type="email" name="email" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="investor@example.com">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Password</label>
                            <input type="password" name="password" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="••••••••">
                        </div>
                        <div class="flex items-center justify-between">
                            <label class="flex items-center">
                                <input type="checkbox" name="remember" class="text-blue-500">
                                <span class="ml-2 text-sm text-gray-600">Remember me</span>
                            </label>
                            <a href="#" class="text-sm text-blue-500 hover:underline">Forgot password?</a>
                        </div>
                    </div>
                    
                    <button type="submit" class="w-full mt-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium">
                        Sign In to Trade
                    </button>
                </form>

                <!-- Registration Form -->
                <form id="register-form" class="auth-form hidden">
                    <h2 class="text-2xl font-bold text-center mb-6">Join GOMNA Trading</h2>
                    <p class="text-center text-gray-600 mb-6">Start investing in profitable algorithms</p>
                    
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">First Name</label>
                                <input type="text" name="firstName" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="John">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Last Name</label>
                                <input type="text" name="lastName" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="Doe">
                            </div>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                            <input type="email" name="email" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="john@example.com">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Phone Number</label>
                            <input type="tel" name="phone" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="+1 (555) 123-4567">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Password</label>
                            <input type="password" name="password" required minlength="8" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="••••••••">
                            <div class="text-xs text-gray-500 mt-1">Minimum 8 characters</div>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Confirm Password</label>
                            <input type="password" name="confirmPassword" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="••••••••">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Investment Experience</label>
                            <select name="experience" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500">
                                <option value="">Select experience level</option>
                                <option value="beginner">Beginner (0-1 years)</option>
                                <option value="intermediate">Intermediate (1-5 years)</option>
                                <option value="advanced">Advanced (5+ years)</option>
                                <option value="professional">Professional/Institutional</option>
                            </select>
                        </div>
                        <div class="flex items-start space-x-2">
                            <input type="checkbox" name="terms" required class="mt-1">
                            <span class="text-xs text-gray-600">
                                I agree to the <a href="#" class="text-blue-500 hover:underline">Terms of Service</a> and 
                                <a href="#" class="text-blue-500 hover:underline">Privacy Policy</a>. 
                                I understand the risks involved in algorithmic trading.
                            </span>
                        </div>
                    </div>
                    
                    <button type="submit" class="w-full mt-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors font-medium">
                        Create Investor Account
                    </button>
                </form>

                <!-- Close Button -->
                <button class="close-auth absolute top-4 right-4 text-gray-400 hover:text-gray-600">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;
        
        document.body.appendChild(authPanel);
    }

    setupAuthEventListeners() {
        // Tab switching
        document.addEventListener('click', (e) => {
            if (e.target.matches('.auth-tab')) {
                const tab = e.target.getAttribute('data-tab');
                this.switchAuthTab(tab);
            }
        });

        // Form submissions
        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin(new FormData(e.target));
        });

        document.getElementById('register-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegistration(new FormData(e.target));
        });

        // Close modal
        document.querySelector('.close-auth').addEventListener('click', () => {
            this.closeAuthPanel();
        });

        // Show auth when trying to purchase without login
        document.addEventListener('click', (e) => {
            if (e.target.matches('.buy-algorithm-btn') && !this.currentUser) {
                e.preventDefault();
                this.showAuthPanel('register');
                this.showNotification('Please create an account to purchase algorithms', 'info');
            }
        });
    }

    switchAuthTab(tab) {
        // Update tab styles
        document.querySelectorAll('.auth-tab').forEach(tabBtn => {
            if (tabBtn.getAttribute('data-tab') === tab) {
                tabBtn.classList.add('border-blue-500', 'text-blue-500');
                tabBtn.classList.remove('border-gray-200', 'text-gray-500');
            } else {
                tabBtn.classList.remove('border-blue-500', 'text-blue-500');
                tabBtn.classList.add('border-gray-200', 'text-gray-500');
            }
        });

        // Show/hide forms
        document.querySelectorAll('.auth-form').forEach(form => {
            if (form.id === `${tab}-form`) {
                form.classList.remove('hidden');
            } else {
                form.classList.add('hidden');
            }
        });
    }

    async handleLogin(formData) {
        const email = formData.get('email');
        const password = formData.get('password');
        const remember = formData.get('remember');

        this.showNotification('Authenticating investor account...', 'info');

        try {
            // Simulate API login call
            const loginResult = await this.authenticateUser(email, password);
            
            if (loginResult.success) {
                this.currentUser = loginResult.user;
                this.sessionToken = loginResult.token;
                
                if (remember) {
                    localStorage.setItem('investor_session', JSON.stringify({
                        token: this.sessionToken,
                        user: this.currentUser,
                        expires: Date.now() + (30 * 24 * 60 * 60 * 1000) // 30 days
                    }));
                }

                this.closeAuthPanel();
                this.updateUIForLoggedInUser();
                this.showNotification(`Welcome back, ${this.currentUser.firstName}! Ready to trade algorithms.`, 'success');
                
                // Load user's algorithm portfolio
                await this.loadUserPortfolio();
                
            } else {
                this.showNotification('Invalid credentials. Please check your email and password.', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showNotification('Login failed. Please try again.', 'error');
        }
    }

    async handleRegistration(formData) {
        const userData = {
            firstName: formData.get('firstName'),
            lastName: formData.get('lastName'),
            email: formData.get('email'),
            phone: formData.get('phone'),
            password: formData.get('password'),
            confirmPassword: formData.get('confirmPassword'),
            experience: formData.get('experience'),
            terms: formData.get('terms')
        };

        // Validate passwords match
        if (userData.password !== userData.confirmPassword) {
            this.showNotification('Passwords do not match', 'error');
            return;
        }

        this.showNotification('Creating your investor account...', 'info');

        try {
            // Simulate API registration call
            const registrationResult = await this.createUser(userData);
            
            if (registrationResult.success) {
                this.currentUser = registrationResult.user;
                this.sessionToken = registrationResult.token;
                
                // Auto-login after registration
                localStorage.setItem('investor_session', JSON.stringify({
                    token: this.sessionToken,
                    user: this.currentUser,
                    expires: Date.now() + (30 * 24 * 60 * 60 * 1000) // 30 days
                }));

                this.closeAuthPanel();
                this.updateUIForLoggedInUser();
                this.showNotification(`Account created successfully! Welcome to GOMNA, ${this.currentUser.firstName}!`, 'success');
                
                // Show onboarding for new users
                setTimeout(() => {
                    this.showInvestorOnboarding();
                }, 2000);
                
            } else {
                this.showNotification(registrationResult.error || 'Registration failed', 'error');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showNotification('Registration failed. Please try again.', 'error');
        }
    }

    async authenticateUser(email, password) {
        // Simulate API call with realistic delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock authentication - in production, this would be a real API call
        if (email.includes('@') && password.length >= 8) {
            return {
                success: true,
                user: {
                    id: 'inv_' + Math.random().toString(36).substr(2, 9),
                    firstName: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1),
                    lastName: 'Investor',
                    email: email,
                    phone: '+1 (555) 123-4567',
                    experience: 'intermediate',
                    kycStatus: 'pending',
                    verificationLevel: 'email',
                    investmentLimit: this.investmentLimits.unverified,
                    portfolioValue: 10000,
                    totalInvested: 0,
                    algorithmsOwned: [],
                    joinedDate: new Date().toISOString(),
                    lastLogin: new Date().toISOString()
                },
                token: 'jwt_' + Math.random().toString(36).substr(2, 24)
            };
        } else {
            return {
                success: false,
                error: 'Invalid credentials'
            };
        }
    }

    async createUser(userData) {
        // Simulate API call with realistic delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Mock user creation - in production, this would be a real API call
        return {
            success: true,
            user: {
                id: 'inv_' + Math.random().toString(36).substr(2, 9),
                firstName: userData.firstName,
                lastName: userData.lastName,
                email: userData.email,
                phone: userData.phone,
                experience: userData.experience,
                kycStatus: 'pending',
                verificationLevel: 'email',
                investmentLimit: this.investmentLimits.unverified,
                portfolioValue: 10000, // Starting bonus
                totalInvested: 0,
                algorithmsOwned: [],
                joinedDate: new Date().toISOString(),
                lastLogin: new Date().toISOString()
            },
            token: 'jwt_' + Math.random().toString(36).substr(2, 24)
        };
    }

    restoreSession() {
        const sessionData = localStorage.getItem('investor_session');
        if (sessionData) {
            try {
                const session = JSON.parse(sessionData);
                if (session.expires > Date.now()) {
                    this.currentUser = session.user;
                    this.sessionToken = session.token;
                    this.updateUIForLoggedInUser();
                    console.log('✅ Session restored for', this.currentUser.email);
                } else {
                    localStorage.removeItem('investor_session');
                }
            } catch (error) {
                console.error('Error restoring session:', error);
                localStorage.removeItem('investor_session');
            }
        }
    }

    updateUIForLoggedInUser() {
        // Update header to show user info
        this.addUserHeaderInfo();
        
        // Update marketplace to show user's algorithms
        if (window.marketplaceUI) {
            window.marketplaceUI.marketplace.userPortfolio = {
                balance: this.currentUser.portfolioValue,
                ownedAlgorithms: new Set(this.currentUser.algorithmsOwned),
                activePositions: new Map(),
                totalPnL: 0,
                trades: []
            };
        }
    }

    addUserHeaderInfo() {
        // Add user info to header
        const headerRight = document.querySelector('.flex.items-center.gap-4');
        if (headerRight && !document.getElementById('user-info')) {
            const userInfo = document.createElement('div');
            userInfo.id = 'user-info';
            userInfo.className = 'flex items-center gap-3 bg-white bg-opacity-20 rounded-full px-4 py-2';
            userInfo.innerHTML = `
                <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                    ${this.currentUser.firstName.charAt(0)}${this.currentUser.lastName.charAt(0)}
                </div>
                <div class="text-sm">
                    <div class="font-medium text-gray-900">${this.currentUser.firstName} ${this.currentUser.lastName}</div>
                    <div class="text-gray-600">${this.currentUser.verificationLevel.toUpperCase()}</div>
                </div>
                <div class="dropdown-menu hidden absolute top-full right-0 mt-2 bg-white rounded-lg shadow-lg border p-2 min-w-48 z-50">
                    <a href="#" class="block px-4 py-2 text-sm hover:bg-gray-100 rounded user-profile">Profile & KYC</a>
                    <a href="#" class="block px-4 py-2 text-sm hover:bg-gray-100 rounded user-portfolio">My Algorithms</a>
                    <a href="#" class="block px-4 py-2 text-sm hover:bg-gray-100 rounded user-settings">Settings</a>
                    <hr class="my-2">
                    <a href="#" class="block px-4 py-2 text-sm hover:bg-gray-100 rounded text-red-600 user-logout">Sign Out</a>
                </div>
            `;
            
            // Add click handler for dropdown
            userInfo.addEventListener('click', () => {
                const dropdown = userInfo.querySelector('.dropdown-menu');
                dropdown.classList.toggle('hidden');
            });
            
            // Add logout handler
            userInfo.querySelector('.user-logout').addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
            
            headerRight.appendChild(userInfo);
        }
    }

    showAuthPanel(defaultTab = 'login') {
        const authPanel = document.getElementById('investor-auth-panel');
        if (authPanel) {
            authPanel.classList.remove('hidden');
            this.switchAuthTab(defaultTab);
        }
    }

    closeAuthPanel() {
        const authPanel = document.getElementById('investor-auth-panel');
        if (authPanel) {
            authPanel.classList.add('hidden');
        }
    }

    logout() {
        this.currentUser = null;
        this.sessionToken = null;
        localStorage.removeItem('investor_session');
        
        // Remove user info from header
        const userInfo = document.getElementById('user-info');
        if (userInfo) {
            userInfo.remove();
        }
        
        this.showNotification('Signed out successfully', 'info');
        
        // Reset marketplace to default state
        if (window.marketplaceUI) {
            window.marketplaceUI.marketplace.userPortfolio = {
                balance: 0,
                ownedAlgorithms: new Set(),
                activePositions: new Map(),
                totalPnL: 0,
                trades: []
            };
        }
    }

    async loadUserPortfolio() {
        if (!this.currentUser) return;
        
        // Load user's owned algorithms and portfolio data
        // In production, this would fetch from API
        console.log('📊 Loading portfolio for', this.currentUser.email);
    }

    showInvestorOnboarding() {
        // Show welcome modal for new investors
        const onboardingModal = document.createElement('div');
        onboardingModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        onboardingModal.innerHTML = `
            <div class="bg-white rounded-xl shadow-2xl max-w-lg w-full mx-4 p-8">
                <h2 class="text-2xl font-bold text-center mb-4">Welcome to GOMNA Trading! 🎉</h2>
                <p class="text-gray-600 text-center mb-6">
                    You've successfully joined the world's most advanced algorithmic trading marketplace.
                </p>
                
                <div class="space-y-4 mb-6">
                    <div class="flex items-start space-x-3">
                        <div class="w-8 h-8 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-sm">✓</div>
                        <div>
                            <div class="font-medium">$10,000 Starting Balance</div>
                            <div class="text-sm text-gray-600">Use this to purchase your first algorithms</div>
                        </div>
                    </div>
                    <div class="flex items-start space-x-3">
                        <div class="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm">📊</div>
                        <div>
                            <div class="font-medium">Real-Time Performance Data</div>
                            <div class="text-sm text-gray-600">All algorithms show live performance metrics</div>
                        </div>
                    </div>
                    <div class="flex items-start space-x-3">
                        <div class="w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-sm">🤖</div>
                        <div>
                            <div class="font-medium">6 Professional Strategies</div>
                            <div class="text-sm text-gray-600">From basic arbitrage to advanced HFT algorithms</div>
                        </div>
                    </div>
                </div>
                
                <div class="text-center">
                    <button class="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors" onclick="this.parentElement.parentElement.parentElement.remove()">
                        Start Trading Algorithms
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(onboardingModal);
    }

    showNotification(message, type = 'info') {
        // Use existing notification system
        if (window.marketplaceUI) {
            window.marketplaceUI.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    // API Methods for live data integration
    async connectExchangeAPI(exchange, credentials) {
        // Store encrypted credentials for exchange connections
        const encryptedCreds = await this.encryptCredentials(credentials);
        this.apiCredentials.set(exchange, encryptedCreds);
        
        // Test connection
        return await this.testExchangeConnection(exchange, credentials);
    }

    async encryptCredentials(credentials) {
        // Simple encryption for demo - use proper encryption in production
        const encoded = btoa(JSON.stringify(credentials));
        return encoded;
    }

    async testExchangeConnection(exchange, credentials) {
        // Simulate API test
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        return {
            success: true,
            exchange: exchange,
            status: 'connected',
            permissions: ['read', 'trade'],
            testnet: credentials.testnet || false
        };
    }

    // Public API for integration
    isLoggedIn() {
        return !!this.currentUser;
    }

    getCurrentUser() {
        return this.currentUser;
    }

    getUserInvestmentLimit() {
        return this.currentUser?.investmentLimit || 0;
    }

    async purchaseAlgorithm(algorithmId, amount) {
        if (!this.currentUser) {
            this.showAuthPanel('register');
            return { success: false, error: 'Please login first' };
        }

        if (amount > this.currentUser.investmentLimit) {
            return { 
                success: false, 
                error: `Purchase exceeds your limit of $${this.currentUser.investmentLimit.toLocaleString()}. Complete KYC to increase limit.` 
            };
        }

        // Process purchase
        this.currentUser.algorithmsOwned.push(algorithmId);
        this.currentUser.totalInvested += amount;
        this.currentUser.portfolioValue -= amount;

        return { success: true, message: 'Algorithm purchased successfully' };
    }
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.InvestorAccountSystem = InvestorAccountSystem;
}

console.log('👤 Investor Account System loaded successfully');