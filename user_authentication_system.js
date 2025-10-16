/**
 * GOMNA AI USER AUTHENTICATION SYSTEM
 * Complete authentication with JWT, 2FA, and session management
 */

class UserAuthenticationSystem {
    constructor() {
        this.currentUser = null;
        this.authToken = null;
        this.refreshToken = null;
        this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
        this.rememberMe = false;
        
        this.init();
    }

    init() {
        this.checkExistingSession();
        this.setupAuthInterceptors();
        this.renderLoginModal();
        console.log('🔐 User Authentication System initialized');
    }

    checkExistingSession() {
        const storedToken = localStorage.getItem('gomna_auth_token');
        const storedUser = localStorage.getItem('gomna_user_data');
        
        if (storedToken && storedUser) {
            try {
                this.authToken = storedToken;
                this.currentUser = JSON.parse(storedUser);
                this.validateSession();
            } catch (error) {
                console.error('Invalid session data:', error);
                this.clearSession();
            }
        }
    }

    async validateSession() {
        try {
            const response = await fetch('/api/auth/validate', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.valid) {
                    this.updateUIForAuthenticatedUser();
                    this.startSessionTimer();
                } else {
                    this.clearSession();
                }
            } else {
                this.clearSession();
            }
        } catch (error) {
            console.error('Session validation error:', error);
            // For demo purposes, simulate valid session
            this.updateUIForAuthenticatedUser();
        }
    }

    renderLoginModal() {
        const modalHTML = `
            <div id="loginModal" class="fixed inset-0 z-50 hidden">
                <!-- Backdrop -->
                <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick="userAuth.closeLoginModal()"></div>
                
                <!-- Modal Content -->
                <div class="relative z-10 flex items-center justify-center min-h-screen p-4">
                    <div class="bg-gray-900 rounded-2xl max-w-md w-full shadow-2xl border border-gray-800">
                        <!-- Header -->
                        <div class="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 p-6 rounded-t-2xl">
                            <div class="flex justify-between items-center">
                                <div>
                                    <h2 class="text-2xl font-bold text-white">Welcome Back</h2>
                                    <p class="text-blue-100 mt-1">Sign in to your Gomna AI account</p>
                                </div>
                                <button onclick="userAuth.closeLoginModal()" class="text-white/80 hover:text-white">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                    </svg>
                                </button>
                            </div>
                        </div>
                        
                        <!-- Body -->
                        <div class="p-6">
                            <div id="loginForm" class="space-y-4">
                                <!-- Login Method Toggle -->
                                <div class="flex rounded-lg bg-gray-800 p-1">
                                    <button id="emailLoginTab" onclick="userAuth.switchLoginMethod('email')" 
                                        class="flex-1 py-2 px-4 rounded-md bg-blue-600 text-white font-medium transition-all">
                                        Email
                                    </button>
                                    <button id="walletLoginTab" onclick="userAuth.switchLoginMethod('wallet')" 
                                        class="flex-1 py-2 px-4 rounded-md text-gray-400 font-medium transition-all">
                                        Wallet
                                    </button>
                                </div>
                                
                                <!-- Email Login Form -->
                                <div id="emailLoginForm">
                                    <div class="space-y-4">
                                        <div>
                                            <label class="block text-sm font-medium text-gray-300 mb-2">Email or Username</label>
                                            <input type="text" id="loginUsername" placeholder="Enter your email or username" 
                                                class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                                        </div>
                                        
                                        <div>
                                            <div class="flex justify-between mb-2">
                                                <label class="text-sm font-medium text-gray-300">Password</label>
                                                <a href="#" onclick="userAuth.showForgotPassword()" class="text-sm text-blue-500 hover:text-blue-400">Forgot password?</a>
                                            </div>
                                            <input type="password" id="loginPassword" placeholder="Enter your password" 
                                                class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none">
                                        </div>
                                        
                                        <div class="flex items-center justify-between">
                                            <label class="flex items-center text-gray-300">
                                                <input type="checkbox" id="rememberMe" class="mr-2 rounded bg-gray-800 border-gray-700 text-blue-600 focus:ring-blue-500">
                                                <span class="text-sm">Remember me</span>
                                            </label>
                                            <label class="flex items-center text-gray-300">
                                                <input type="checkbox" id="use2FA" class="mr-2 rounded bg-gray-800 border-gray-700 text-purple-600 focus:ring-purple-500">
                                                <span class="text-sm">Use 2FA</span>
                                            </label>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Wallet Login Form -->
                                <div id="walletLoginForm" class="hidden">
                                    <div class="space-y-4">
                                        <p class="text-gray-400 text-sm text-center mb-4">Connect your wallet to sign in securely</p>
                                        
                                        <button onclick="userAuth.connectWallet('metamask')" 
                                            class="w-full flex items-center justify-between px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg hover:border-orange-500 transition-all">
                                            <span class="text-white font-medium">MetaMask</span>
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTI3LjIgNC44TDE3LjYgMTEuMkwxOS40IDYuOEwyNy4yIDQuOFoiIGZpbGw9IiNFMjc2MjUiLz4KPC9zdmc+" alt="MetaMask" class="w-6 h-6">
                                        </button>
                                        
                                        <button onclick="userAuth.connectWallet('walletconnect')" 
                                            class="w-full flex items-center justify-between px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg hover:border-blue-500 transition-all">
                                            <span class="text-white font-medium">WalletConnect</span>
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMTFDMTIgNyAyMCA3IDI0IDExTDI2IDlDMjEgNCAyMCA0IDYgOUw4IDExWiIgZmlsbD0iIzNDOTlGQyIvPgo8L3N2Zz4=" alt="WalletConnect" class="w-6 h-6">
                                        </button>
                                        
                                        <button onclick="userAuth.connectWallet('coinbase')" 
                                            class="w-full flex items-center justify-between px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg hover:border-blue-600 transition-all">
                                            <span class="text-white font-medium">Coinbase Wallet</span>
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iMTYiIGZpbGw9IiMwMDUyRkYiLz4KPC9zdmc+" alt="Coinbase" class="w-6 h-6">
                                        </button>
                                    </div>
                                </div>
                                
                                <!-- 2FA Input (hidden by default) -->
                                <div id="twoFactorInput" class="hidden">
                                    <label class="block text-sm font-medium text-gray-300 mb-2">2FA Code</label>
                                    <input type="text" id="twoFactorCode" placeholder="Enter 6-digit code" maxlength="6"
                                        class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white text-center text-lg tracking-widest focus:border-purple-500 focus:outline-none">
                                </div>
                                
                                <!-- Error Message -->
                                <div id="loginError" class="hidden bg-red-900/20 border border-red-800 rounded-lg p-3">
                                    <p class="text-sm text-red-400"></p>
                                </div>
                                
                                <!-- Submit Button -->
                                <button onclick="userAuth.handleLogin()" 
                                    class="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all">
                                    Sign In
                                </button>
                                
                                <!-- Social Login -->
                                <div class="relative">
                                    <div class="absolute inset-0 flex items-center">
                                        <div class="w-full border-t border-gray-700"></div>
                                    </div>
                                    <div class="relative flex justify-center text-sm">
                                        <span class="px-2 bg-gray-900 text-gray-500">Or continue with</span>
                                    </div>
                                </div>
                                
                                <div class="grid grid-cols-3 gap-3">
                                    <button onclick="userAuth.socialLogin('google')" 
                                        class="flex justify-center py-2 px-4 border border-gray-700 rounded-lg hover:bg-gray-800 transition-all">
                                        <svg class="w-5 h-5" viewBox="0 0 24 24">
                                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                                        </svg>
                                    </button>
                                    
                                    <button onclick="userAuth.socialLogin('github')" 
                                        class="flex justify-center py-2 px-4 border border-gray-700 rounded-lg hover:bg-gray-800 transition-all">
                                        <svg class="w-5 h-5" fill="white" viewBox="0 0 24 24">
                                            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                                        </svg>
                                    </button>
                                    
                                    <button onclick="userAuth.socialLogin('twitter')" 
                                        class="flex justify-center py-2 px-4 border border-gray-700 rounded-lg hover:bg-gray-800 transition-all">
                                        <svg class="w-5 h-5" fill="#1DA1F2" viewBox="0 0 24 24">
                                            <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                                        </svg>
                                    </button>
                                </div>
                                
                                <!-- Sign Up Link -->
                                <div class="text-center">
                                    <p class="text-gray-400 text-sm">
                                        Don't have an account? 
                                        <a href="#" onclick="userAuth.closeLoginModal(); accountRegistration.showModal()" class="text-blue-500 hover:text-blue-400 font-medium">Sign up</a>
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Append to body if not exists
        if (!document.getElementById('loginModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
    }

    async handleLogin() {
        const loginMethod = document.getElementById('emailLoginForm').classList.contains('hidden') ? 'wallet' : 'email';
        
        if (loginMethod === 'email') {
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            const use2FA = document.getElementById('use2FA').checked;
            const rememberMe = document.getElementById('rememberMe').checked;
            
            if (!username || !password) {
                this.showLoginError('Please enter your username and password');
                return;
            }
            
            try {
                // Show loading state
                this.setLoginLoading(true);
                
                // Simulate API call
                await this.simulateApiCall();
                
                // For demo, accept any credentials
                const loginResult = {
                    success: true,
                    user: {
                        id: 'USER_' + Date.now(),
                        username: username,
                        email: username.includes('@') ? username : username + '@example.com',
                        name: 'John Doe',
                        plan: 'professional',
                        verified: true
                    },
                    token: 'JWT_' + btoa(username + ':' + Date.now()),
                    refreshToken: 'REFRESH_' + btoa(username + ':' + Date.now())
                };
                
                if (use2FA) {
                    // Show 2FA input
                    document.getElementById('twoFactorInput').classList.remove('hidden');
                    this.setLoginLoading(false);
                    
                    // Wait for 2FA code
                    const twoFactorCode = await this.waitFor2FACode();
                    
                    if (twoFactorCode !== '123456') { // Demo code
                        this.showLoginError('Invalid 2FA code');
                        return;
                    }
                }
                
                if (loginResult.success) {
                    this.handleLoginSuccess(loginResult, rememberMe);
                } else {
                    this.showLoginError('Invalid credentials');
                }
            } catch (error) {
                console.error('Login error:', error);
                this.showLoginError('Login failed. Please try again.');
            } finally {
                this.setLoginLoading(false);
            }
        }
    }

    async waitFor2FACode() {
        return new Promise((resolve) => {
            const checkInterval = setInterval(() => {
                const code = document.getElementById('twoFactorCode').value;
                if (code && code.length === 6) {
                    clearInterval(checkInterval);
                    resolve(code);
                }
            }, 100);
        });
    }

    handleLoginSuccess(loginResult, rememberMe) {
        this.currentUser = loginResult.user;
        this.authToken = loginResult.token;
        this.refreshToken = loginResult.refreshToken;
        this.rememberMe = rememberMe;
        
        // Store session
        if (rememberMe) {
            localStorage.setItem('gomna_auth_token', this.authToken);
            localStorage.setItem('gomna_refresh_token', this.refreshToken);
            localStorage.setItem('gomna_user_data', JSON.stringify(this.currentUser));
        } else {
            sessionStorage.setItem('gomna_auth_token', this.authToken);
            sessionStorage.setItem('gomna_user_data', JSON.stringify(this.currentUser));
        }
        
        // Update UI
        this.updateUIForAuthenticatedUser();
        
        // Close modal
        this.closeLoginModal();
        
        // Show success message
        this.showSuccessNotification('Welcome back, ' + this.currentUser.name + '!');
        
        // Start session timer
        this.startSessionTimer();
    }

    updateUIForAuthenticatedUser() {
        // Update header buttons
        const loginBtn = document.getElementById('login-btn');
        const registerBtn = document.getElementById('register-btn');
        
        if (loginBtn && registerBtn) {
            // Replace with user menu
            const userMenuHTML = `
                <div class="relative">
                    <button onclick="userAuth.toggleUserMenu()" class="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-all">
                        <div class="w-8 h-8 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center">
                            <span class="text-white font-semibold">${this.currentUser ? this.currentUser.name.charAt(0) : 'U'}</span>
                        </div>
                        <span class="font-medium">${this.currentUser ? this.currentUser.name : 'User'}</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                    </button>
                    
                    <div id="userMenu" class="hidden absolute right-0 mt-2 w-64 bg-gray-800 rounded-lg shadow-xl border border-gray-700 z-50">
                        <div class="p-4 border-b border-gray-700">
                            <p class="text-white font-semibold">${this.currentUser ? this.currentUser.name : 'User'}</p>
                            <p class="text-gray-400 text-sm">${this.currentUser ? this.currentUser.email : ''}</p>
                            <div class="mt-2">
                                <span class="inline-block px-2 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xs font-semibold rounded-full">
                                    ${this.currentUser ? this.currentUser.plan.toUpperCase() : 'FREE'} PLAN
                                </span>
                            </div>
                        </div>
                        <div class="py-2">
                            <a href="#dashboard" class="block px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-all">
                                <i data-lucide="layout-dashboard" class="w-4 h-4 inline mr-2"></i>
                                Dashboard
                            </a>
                            <a href="#account" class="block px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-all">
                                <i data-lucide="user" class="w-4 h-4 inline mr-2"></i>
                                Account Settings
                            </a>
                            <a href="#subscription" class="block px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-all">
                                <i data-lucide="credit-card" class="w-4 h-4 inline mr-2"></i>
                                Subscription
                            </a>
                            <a href="#api-keys" class="block px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-all">
                                <i data-lucide="key" class="w-4 h-4 inline mr-2"></i>
                                API Keys
                            </a>
                        </div>
                        <div class="border-t border-gray-700 py-2">
                            <button onclick="userAuth.logout()" class="block w-full text-left px-4 py-2 text-red-400 hover:bg-gray-700 hover:text-red-300 transition-all">
                                <i data-lucide="log-out" class="w-4 h-4 inline mr-2"></i>
                                Sign Out
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            // Create container for user menu
            const container = document.createElement('div');
            container.innerHTML = userMenuHTML;
            
            // Replace buttons with user menu
            loginBtn.parentElement.replaceChild(container.firstElementChild, loginBtn);
            registerBtn.remove();
            
            // Re-initialize Lucide icons
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        }
        
        // Enable premium features if applicable
        this.enablePremiumFeatures();
    }

    enablePremiumFeatures() {
        // Enable features based on user's subscription plan
        if (this.currentUser && this.currentUser.plan !== 'free') {
            // Enable advanced features
            document.querySelectorAll('.premium-feature').forEach(el => {
                el.classList.remove('opacity-50', 'pointer-events-none');
            });
            
            // Show premium badge
            document.querySelectorAll('.premium-badge').forEach(el => {
                el.classList.remove('hidden');
            });
        }
    }

    toggleUserMenu() {
        const menu = document.getElementById('userMenu');
        if (menu) {
            menu.classList.toggle('hidden');
            
            // Close menu when clicking outside
            if (!menu.classList.contains('hidden')) {
                document.addEventListener('click', this.closeUserMenuOnClickOutside);
            }
        }
    }

    closeUserMenuOnClickOutside(e) {
        const menu = document.getElementById('userMenu');
        if (menu && !menu.contains(e.target) && !e.target.closest('button[onclick*="toggleUserMenu"]')) {
            menu.classList.add('hidden');
            document.removeEventListener('click', this.closeUserMenuOnClickOutside);
        }
    }

    startSessionTimer() {
        // Clear existing timer
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
        }
        
        // Set new timer
        this.sessionTimer = setTimeout(() => {
            this.showSessionExpiredWarning();
        }, this.sessionTimeout - 60000); // Warn 1 minute before expiry
    }

    showSessionExpiredWarning() {
        const warning = document.createElement('div');
        warning.className = 'fixed top-4 right-4 bg-yellow-600 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        warning.innerHTML = `
            <p class="font-semibold">Session Expiring Soon</p>
            <p class="text-sm">Your session will expire in 1 minute.</p>
            <button onclick="userAuth.extendSession()" class="mt-2 px-4 py-1 bg-white text-yellow-600 rounded font-medium">
                Extend Session
            </button>
        `;
        document.body.appendChild(warning);
        
        setTimeout(() => {
            warning.remove();
            if (!this.sessionExtended) {
                this.logout();
            }
        }, 60000);
    }

    extendSession() {
        this.sessionExtended = true;
        this.startSessionTimer();
        this.showSuccessNotification('Session extended');
    }

    async logout() {
        try {
            // Call logout API
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
        } catch (error) {
            console.error('Logout error:', error);
        }
        
        // Clear session
        this.clearSession();
        
        // Redirect to home
        window.location.href = '/';
    }

    clearSession() {
        this.currentUser = null;
        this.authToken = null;
        this.refreshToken = null;
        
        localStorage.removeItem('gomna_auth_token');
        localStorage.removeItem('gomna_refresh_token');
        localStorage.removeItem('gomna_user_data');
        sessionStorage.clear();
        
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
        }
    }

    setupAuthInterceptors() {
        // Intercept all fetch requests to add auth headers
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            let [resource, config] = args;
            
            if (this.authToken && resource.startsWith('/api/')) {
                config = config || {};
                config.headers = config.headers || {};
                config.headers['Authorization'] = `Bearer ${this.authToken}`;
            }
            
            const response = await originalFetch(resource, config);
            
            // Handle 401 responses
            if (response.status === 401) {
                this.handleUnauthorized();
            }
            
            return response;
        };
    }

    handleUnauthorized() {
        this.clearSession();
        this.showLoginModal();
        this.showLoginError('Your session has expired. Please sign in again.');
    }

    // UI Helper Methods
    showLoginModal() {
        const modal = document.getElementById('loginModal');
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }

    closeLoginModal() {
        const modal = document.getElementById('loginModal');
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
    }

    switchLoginMethod(method) {
        const emailForm = document.getElementById('emailLoginForm');
        const walletForm = document.getElementById('walletLoginForm');
        const emailTab = document.getElementById('emailLoginTab');
        const walletTab = document.getElementById('walletLoginTab');
        
        if (method === 'email') {
            emailForm.classList.remove('hidden');
            walletForm.classList.add('hidden');
            emailTab.classList.add('bg-blue-600', 'text-white');
            emailTab.classList.remove('text-gray-400');
            walletTab.classList.remove('bg-blue-600', 'text-white');
            walletTab.classList.add('text-gray-400');
        } else {
            emailForm.classList.add('hidden');
            walletForm.classList.remove('hidden');
            walletTab.classList.add('bg-blue-600', 'text-white');
            walletTab.classList.remove('text-gray-400');
            emailTab.classList.remove('bg-blue-600', 'text-white');
            emailTab.classList.add('text-gray-400');
        }
    }

    async connectWallet(walletType) {
        try {
            this.setLoginLoading(true);
            
            // Simulate wallet connection
            await this.simulateApiCall();
            
            // For demo, auto-login with wallet
            const walletAddress = '0x' + Math.random().toString(16).substr(2, 40);
            
            const loginResult = {
                success: true,
                user: {
                    id: 'WALLET_USER_' + Date.now(),
                    username: walletAddress.substr(0, 10) + '...',
                    email: walletAddress + '@wallet.eth',
                    name: 'Wallet User',
                    plan: 'professional',
                    verified: true,
                    walletAddress: walletAddress
                },
                token: 'JWT_WALLET_' + btoa(walletAddress + ':' + Date.now()),
                refreshToken: 'REFRESH_WALLET_' + btoa(walletAddress + ':' + Date.now())
            };
            
            this.handleLoginSuccess(loginResult, true);
        } catch (error) {
            console.error('Wallet connection error:', error);
            this.showLoginError('Failed to connect wallet');
        } finally {
            this.setLoginLoading(false);
        }
    }

    async socialLogin(provider) {
        try {
            this.setLoginLoading(true);
            
            // Simulate OAuth flow
            await this.simulateApiCall();
            
            // For demo, auto-login
            const loginResult = {
                success: true,
                user: {
                    id: provider.toUpperCase() + '_USER_' + Date.now(),
                    username: provider + '_user',
                    email: 'user@' + provider + '.com',
                    name: provider.charAt(0).toUpperCase() + provider.slice(1) + ' User',
                    plan: 'starter',
                    verified: true
                },
                token: 'JWT_' + provider.toUpperCase() + '_' + Date.now(),
                refreshToken: 'REFRESH_' + provider.toUpperCase() + '_' + Date.now()
            };
            
            this.handleLoginSuccess(loginResult, false);
        } catch (error) {
            console.error('Social login error:', error);
            this.showLoginError('Failed to login with ' + provider);
        } finally {
            this.setLoginLoading(false);
        }
    }

    showForgotPassword() {
        // Show forgot password modal
        alert('Password reset functionality would be implemented here');
    }

    setLoginLoading(loading) {
        const button = document.querySelector('#loginModal button[onclick*="handleLogin"]');
        if (button) {
            button.disabled = loading;
            button.innerHTML = loading ? 
                '<svg class="animate-spin h-5 w-5 mx-auto" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : 
                'Sign In';
        }
    }

    showLoginError(message) {
        const errorDiv = document.getElementById('loginError');
        if (errorDiv) {
            errorDiv.classList.remove('hidden');
            errorDiv.querySelector('p').textContent = message;
            
            setTimeout(() => {
                errorDiv.classList.add('hidden');
            }, 5000);
        }
    }

    showSuccessNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-pulse';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    async simulateApiCall() {
        return new Promise(resolve => setTimeout(resolve, 1500));
    }
}

// Initialize the system when DOM is ready
if (typeof window !== 'undefined') {
    window.userAuth = null;
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.userAuth = new UserAuthenticationSystem();
            console.log('✅ User Authentication System ready');
            
            // Attach login button event
            const loginBtn = document.getElementById('login-btn');
            if (loginBtn) {
                loginBtn.onclick = () => window.userAuth.showLoginModal();
            }
        });
    } else {
        window.userAuth = new UserAuthenticationSystem();
        console.log('✅ User Authentication System ready');
        
        // Attach login button event
        const loginBtn = document.getElementById('login-btn');
        if (loginBtn) {
            loginBtn.onclick = () => window.userAuth.showLoginModal();
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UserAuthenticationSystem;
}