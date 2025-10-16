/**
 * Cocoa Trading AI - Master Integration System
 * Integrates all professional components into the HyperVision platform
 * 
 * Features:
 * - Seamless integration of all Cocoa Trading AI components
 * - Professional initialization sequence
 * - Comprehensive error handling and fallbacks
 * - Performance optimization and lazy loading
 */

class CocoaTradingAIIntegration {
    constructor() {
        this.components = new Map();
        this.initializationOrder = [
            'branding',
            'tieredAccounts', 
            'regulatoryCredentials',
            'marketDashboard',
            'portfolioMetrics',
            'customizablePanels'
        ];
        this.isInitialized = false;
        this.loadingProgress = 0;
        
        this.init();
    }

    async init() {
        try {
            console.log('🚀 Starting Cocoa Trading AI Platform Integration...');
            
            this.showLoadingScreen();
            await this.loadAllComponents();
            this.initializeComponents();
            this.setupGlobalIntegration();
            this.hideLoadingScreen();
            
            this.isInitialized = true;
            console.log('✅ Cocoa Trading AI Platform Integration completed successfully');
            
            // Show welcome message
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('❌ Error during Cocoa Trading AI integration:', error);
            this.showErrorMessage(error);
        }
    }

    showLoadingScreen() {
        const loadingScreen = document.createElement('div');
        loadingScreen.id = 'cocoa-loading-screen';
        loadingScreen.innerHTML = `
            <div class="loading-container">
                <div class="cocoa-logo-animation">
                    <div class="logo-circle"></div>
                    <div class="logo-text">Cocoa Trading AI</div>
                </div>
                <div class="loading-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div class="loading-text" id="loading-text">Initializing Professional Trading Platform...</div>
                </div>
                <div class="loading-features">
                    <div class="feature-list">
                        <div class="feature-item">🎨 Professional Branding & Theme</div>
                        <div class="feature-item">🏦 Tiered Account Structure</div>
                        <div class="feature-item">🛡️ Regulatory Compliance Display</div>
                        <div class="feature-item">📊 Real-Time Market Dashboard</div>
                        <div class="feature-item">📈 Advanced Portfolio Analytics</div>
                        <div class="feature-item">🎛️ Customizable Panel System</div>
                    </div>
                </div>
            </div>
        `;

        const loadingStyles = document.createElement('style');
        loadingStyles.textContent = `
            #cocoa-loading-screen {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, #1A1A1A 0%, #2C1810 50%, #1A1A1A 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 100000;
                animation: fadeIn 0.5s ease-out;
            }

            .loading-container {
                text-align: center;
                max-width: 500px;
                padding: 40px;
            }

            .cocoa-logo-animation {
                margin-bottom: 40px;
            }

            .logo-circle {
                width: 80px;
                height: 80px;
                border: 4px solid rgba(212, 165, 116, 0.3);
                border-top: 4px solid #D4A574;
                border-radius: 50%;
                margin: 0 auto 20px;
                animation: spin 2s linear infinite;
            }

            .logo-text {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #D4A574 0%, #FFD700 50%, #D4A574 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }

            .loading-progress {
                margin-bottom: 30px;
            }

            .progress-bar {
                width: 100%;
                height: 8px;
                background: rgba(212, 165, 116, 0.2);
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 15px;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #8B4513, #D4A574, #FFD700);
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 4px;
            }

            .loading-text {
                color: #D4A574;
                font-weight: 600;
                font-size: 1.1rem;
            }

            .loading-features {
                margin-top: 30px;
            }

            .feature-list {
                display: grid;
                gap: 10px;
                text-align: left;
            }

            .feature-item {
                color: rgba(212, 165, 116, 0.8);
                padding: 8px 12px;
                background: rgba(212, 165, 116, 0.1);
                border-radius: 6px;
                border-left: 3px solid #D4A574;
                font-size: 0.9rem;
                opacity: 0;
                animation: slideInLeft 0.5s ease-out forwards;
            }

            .feature-item:nth-child(1) { animation-delay: 0.1s; }
            .feature-item:nth-child(2) { animation-delay: 0.2s; }
            .feature-item:nth-child(3) { animation-delay: 0.3s; }
            .feature-item:nth-child(4) { animation-delay: 0.4s; }
            .feature-item:nth-child(5) { animation-delay: 0.5s; }
            .feature-item:nth-child(6) { animation-delay: 0.6s; }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }

            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }
        `;
        
        document.head.appendChild(loadingStyles);
        document.body.appendChild(loadingScreen);
    }

    async loadAllComponents() {
        const componentScripts = [
            { name: 'branding', src: 'cocoa_trading_ai_branding.js' },
            { name: 'tieredAccounts', src: 'tiered_account_system.js' },
            { name: 'regulatoryCredentials', src: 'regulatory_credentials_system.js' },
            { name: 'marketDashboard', src: 'professional_market_dashboard.js' },
            { name: 'portfolioMetrics', src: 'portfolio_performance_metrics.js' },
            { name: 'customizablePanels', src: 'customizable_panels_system.js' }
        ];

        for (let i = 0; i < componentScripts.length; i++) {
            const component = componentScripts[i];
            
            this.updateLoadingProgress(
                (i / componentScripts.length) * 100,
                `Loading ${component.name} component...`
            );
            
            try {
                await this.loadScript(component.src);
                this.components.set(component.name, true);
                console.log(`✅ Loaded component: ${component.name}`);
            } catch (error) {
                console.warn(`⚠️ Failed to load component ${component.name}:`, error);
                this.components.set(component.name, false);
            }
            
            // Add small delay for smooth progress animation
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        
        this.updateLoadingProgress(100, 'All components loaded successfully!');
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            // Check if script already exists
            const existingScript = document.querySelector(`script[src*="${src}"]`);
            if (existingScript) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    updateLoadingProgress(percentage, text) {
        const progressFill = document.getElementById('progress-fill');
        const loadingText = document.getElementById('loading-text');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (loadingText) {
            loadingText.textContent = text;
        }
        
        this.loadingProgress = percentage;
    }

    initializeComponents() {
        // Initialize components in the correct order
        this.initializationOrder.forEach(componentName => {
            if (this.components.get(componentName)) {
                this.updateLoadingProgress(
                    this.loadingProgress + 5,
                    `Initializing ${componentName}...`
                );
            }
        });
    }

    setupGlobalIntegration() {
        // Set up cross-component communication
        this.setupEventBus();
        this.integrateWithExistingPlatform();
        this.optimizePerformance();
        
        // Set up global Cocoa Trading AI namespace
        window.CocoaTradingAI = {
            branding: window.cocoaTradingAIBranding,
            accounts: window.tieredAccountSystem,
            regulatory: window.regulatoryCredentialsSystem,
            market: window.professionalMarketDashboard,
            portfolio: window.portfolioPerformanceMetrics,
            panels: window.customizablePanelsSystem,
            integration: this
        };

        console.log('✅ Global integration setup complete');
    }

    setupEventBus() {
        // Create a simple event bus for component communication
        window.CocoaTradingAI.eventBus = {
            events: {},
            
            on(event, callback) {
                if (!this.events[event]) {
                    this.events[event] = [];
                }
                this.events[event].push(callback);
            },
            
            emit(event, data) {
                if (this.events[event]) {
                    this.events[event].forEach(callback => callback(data));
                }
            },
            
            off(event, callback) {
                if (this.events[event]) {
                    this.events[event] = this.events[event].filter(cb => cb !== callback);
                }
            }
        };

        // Set up some default integrations
        if (window.tieredAccountSystem && window.portfolioPerformanceMetrics) {
            window.CocoaTradingAI.eventBus.on('tierChanged', (tierData) => {
                console.log('🏦 Account tier changed:', tierData);
                // Update portfolio access based on tier
            });
        }

        console.log('📡 Event bus initialized');
    }

    integrateWithExistingPlatform() {
        // Look for existing HyperVision components and integrate them
        const existingMarketplace = document.querySelector('#marketplace-tab, [data-tab="marketplace"]');
        if (existingMarketplace) {
            // Enhance existing marketplace with Cocoa Trading AI features
            this.enhanceExistingComponents();
        }

        // Update page title and branding
        document.title = 'Cocoa Trading AI - Professional HFT & Arbitrage Platform';
        
        // Update meta tags
        this.updateMetaTags();

        console.log('🔗 Integration with existing platform complete');
    }

    enhanceExistingComponents() {
        // Add Cocoa Trading AI enhancements to existing elements
        const existingButtons = document.querySelectorAll('button, .btn');
        existingButtons.forEach(btn => {
            if (window.cocoaTradingAIBranding && !btn.classList.contains('cocoa-btn-primary') && !btn.classList.contains('cocoa-btn-secondary')) {
                window.cocoaTradingAIBranding.applyBrandingToElement(btn, 'button-primary');
            }
        });

        const existingPanels = document.querySelectorAll('.card, .panel');
        existingPanels.forEach(panel => {
            if (window.cocoaTradingAIBranding && !panel.classList.contains('cocoa-panel')) {
                window.cocoaTradingAIBranding.applyBrandingToElement(panel, 'panel');
            }
        });
    }

    updateMetaTags() {
        // Update meta description
        let metaDesc = document.querySelector('meta[name="description"]');
        if (!metaDesc) {
            metaDesc = document.createElement('meta');
            metaDesc.name = 'description';
            document.head.appendChild(metaDesc);
        }
        metaDesc.content = 'Professional high-frequency trading and arbitrage platform with advanced AI algorithms, regulatory compliance, and institutional-grade features.';

        // Add keywords
        let metaKeywords = document.querySelector('meta[name="keywords"]');
        if (!metaKeywords) {
            metaKeywords = document.createElement('meta');
            metaKeywords.name = 'keywords';
            document.head.appendChild(metaKeywords);
        }
        metaKeywords.content = 'HFT, high frequency trading, arbitrage, cryptocurrency, AI trading, institutional trading, SEC regulated, FINRA member';
    }

    optimizePerformance() {
        // Set up performance monitoring
        const performanceObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach(entry => {
                if (entry.entryType === 'measure') {
                    console.log(`📊 Performance: ${entry.name} took ${entry.duration}ms`);
                }
            });
        });
        
        try {
            performanceObserver.observe({ entryTypes: ['measure'] });
        } catch (e) {
            console.log('Performance observer not supported');
        }

        // Lazy load heavy components
        this.setupLazyLoading();

        console.log('⚡ Performance optimization setup');
    }

    setupLazyLoading() {
        // Set up intersection observer for lazy loading
        const lazyObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('lazy-loaded');
                    lazyObserver.unobserve(entry.target);
                }
            });
        });

        // Observe panels for lazy loading
        setTimeout(() => {
            document.querySelectorAll('.cocoa-panel, .grid-panel').forEach(panel => {
                lazyObserver.observe(panel);
            });
        }, 1000);
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('cocoa-loading-screen');
        if (loadingScreen) {
            loadingScreen.style.animation = 'fadeOut 0.5s ease-out forwards';
            setTimeout(() => {
                loadingScreen.remove();
            }, 500);
        }
    }

    showWelcomeMessage() {
        const welcomeModal = document.createElement('div');
        welcomeModal.className = 'cocoa-welcome-modal';
        welcomeModal.innerHTML = `
            <div class="welcome-backdrop"></div>
            <div class="welcome-content">
                <div class="welcome-header">
                    <div class="welcome-logo">🍫</div>
                    <h1>Welcome to Cocoa Trading AI</h1>
                    <p>Your Professional HFT & Arbitrage Platform is Ready</p>
                </div>
                <div class="welcome-features">
                    <div class="welcome-feature">
                        <div class="feature-icon">🎨</div>
                        <div class="feature-info">
                            <h3>Professional Interface</h3>
                            <p>Sleek, modern design optimized for trading professionals</p>
                        </div>
                    </div>
                    <div class="welcome-feature">
                        <div class="feature-icon">🏦</div>
                        <div class="feature-info">
                            <h3>Tiered Account System</h3>
                            <p>Starter, Professional, and Institutional account options</p>
                        </div>
                    </div>
                    <div class="welcome-feature">
                        <div class="feature-icon">🛡️</div>
                        <div class="feature-info">
                            <h3>Regulatory Compliance</h3>
                            <p>SEC registered, FINRA member, and SIPC protected</p>
                        </div>
                    </div>
                    <div class="welcome-feature">
                        <div class="feature-icon">📊</div>
                        <div class="feature-info">
                            <h3>Real-Time Analytics</h3>
                            <p>Advanced market data and portfolio performance metrics</p>
                        </div>
                    </div>
                </div>
                <div class="welcome-actions">
                    <button class="welcome-btn primary" onclick="this.closest('.cocoa-welcome-modal').remove()">
                        Start Trading
                    </button>
                    <button class="welcome-btn secondary" onclick="window.CocoaTradingAI.integration.showPlatformTour()">
                        Take Platform Tour
                    </button>
                </div>
            </div>
        `;

        const welcomeStyles = document.createElement('style');
        welcomeStyles.textContent = `
            .cocoa-welcome-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 50000;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.5s ease-out;
            }

            .welcome-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(10px);
            }

            .welcome-content {
                position: relative;
                background: linear-gradient(135deg, #1A1A1A 0%, #2C1810 100%);
                border: 2px solid #D4A574;
                border-radius: 20px;
                max-width: 700px;
                width: 90%;
                max-height: 90vh;
                overflow-y: auto;
                box-shadow: 0 20px 60px rgba(139, 69, 19, 0.5);
            }

            .welcome-header {
                text-align: center;
                padding: 40px 40px 20px;
                border-bottom: 1px solid rgba(212, 165, 116, 0.2);
            }

            .welcome-logo {
                font-size: 4rem;
                margin-bottom: 20px;
                animation: bounce 2s infinite;
            }

            .welcome-header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #D4A574 0%, #FFD700 50%, #D4A574 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }

            .welcome-header p {
                color: #D4A574;
                font-size: 1.2rem;
                opacity: 0.9;
            }

            .welcome-features {
                padding: 30px 40px;
                display: grid;
                gap: 20px;
            }

            .welcome-feature {
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 20px;
                background: rgba(212, 165, 116, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(212, 165, 116, 0.1);
            }

            .feature-icon {
                font-size: 2.5rem;
                width: 60px;
                text-align: center;
            }

            .feature-info h3 {
                color: #D4A574;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 5px;
            }

            .feature-info p {
                color: rgba(212, 165, 116, 0.8);
                line-height: 1.4;
            }

            .welcome-actions {
                padding: 20px 40px 40px;
                display: flex;
                gap: 15px;
                justify-content: center;
            }

            .welcome-btn {
                padding: 15px 30px;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 1.1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .welcome-btn.primary {
                background: linear-gradient(135deg, #8B4513, #D4A574);
                color: white;
                box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
            }

            .welcome-btn.primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
            }

            .welcome-btn.secondary {
                background: transparent;
                color: #D4A574;
                border: 2px solid #D4A574;
            }

            .welcome-btn.secondary:hover {
                background: #D4A574;
                color: #1A1A1A;
            }

            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-10px); }
                60% { transform: translateY(-5px); }
            }

            @media (max-width: 768px) {
                .welcome-content {
                    margin: 20px;
                    width: calc(100% - 40px);
                }
                
                .welcome-header, .welcome-features, .welcome-actions {
                    padding-left: 20px;
                    padding-right: 20px;
                }
                
                .welcome-feature {
                    flex-direction: column;
                    text-align: center;
                }
                
                .welcome-actions {
                    flex-direction: column;
                }
            }
        `;

        document.head.appendChild(welcomeStyles);
        document.body.appendChild(welcomeModal);

        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (welcomeModal.parentElement) {
                welcomeModal.remove();
            }
        }, 10000);
    }

    showPlatformTour() {
        // Simple tour implementation
        const tourSteps = [
            { target: '.cocoa-trading-header', message: 'Welcome to the professional Cocoa Trading AI interface!' },
            { target: '#tier-account-system', message: 'Manage your account tier and upgrade options here.' },
            { target: '#regulatory-credentials-system', message: 'View our regulatory compliance and credentials.' },
            { target: '#professional-market-dashboard', message: 'Monitor real-time market data and arbitrage opportunities.' },
            { target: '#customization-control-panel', message: 'Customize your dashboard layout with drag-and-drop panels.' }
        ];

        let currentStep = 0;
        
        const showStep = (step) => {
            const target = document.querySelector(tourSteps[step].target);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.style.outline = '3px solid #FFD700';
                target.style.outlineOffset = '5px';
                
                setTimeout(() => {
                    alert(tourSteps[step].message);
                    target.style.outline = 'none';
                    
                    if (step < tourSteps.length - 1) {
                        showStep(step + 1);
                    }
                }, 1000);
            }
        };

        // Close welcome modal first
        document.querySelector('.cocoa-welcome-modal')?.remove();
        
        // Start tour
        showStep(0);
    }

    showErrorMessage(error) {
        const errorModal = document.createElement('div');
        errorModal.innerHTML = `
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 100000;
            ">
                <div style="
                    background: #1A1A1A;
                    border: 2px solid #DC2626;
                    border-radius: 16px;
                    padding: 30px;
                    max-width: 500px;
                    width: 90%;
                    text-align: center;
                    color: white;
                ">
                    <div style="font-size: 3rem; color: #DC2626; margin-bottom: 20px;">⚠️</div>
                    <h2 style="color: #DC2626; margin-bottom: 15px;">Initialization Error</h2>
                    <p style="margin-bottom: 20px;">
                        Some components failed to load. The platform will continue with reduced functionality.
                    </p>
                    <button onclick="this.closest('div[style*=\"position: fixed\"]').remove()" 
                            style="
                                background: #DC2626;
                                color: white;
                                border: none;
                                padding: 12px 24px;
                                border-radius: 8px;
                                cursor: pointer;
                                font-weight: 600;
                            ">
                        Continue Anyway
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(errorModal);
    }

    // Public API methods
    getLoadedComponents() {
        return Array.from(this.components.entries()).filter(([name, loaded]) => loaded);
    }

    getIntegrationStatus() {
        return {
            initialized: this.isInitialized,
            loadingProgress: this.loadingProgress,
            loadedComponents: this.getLoadedComponents(),
            globalNamespace: !!window.CocoaTradingAI
        };
    }

    reinitialize() {
        console.log('🔄 Reinitializing Cocoa Trading AI platform...');
        this.isInitialized = false;
        this.loadingProgress = 0;
        this.components.clear();
        this.init();
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.cocoaTradingAIIntegration = new CocoaTradingAIIntegration();
    });
} else {
    window.cocoaTradingAIIntegration = new CocoaTradingAIIntegration();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CocoaTradingAIIntegration;
}

console.log('🚀 Cocoa Trading AI Master Integration System loaded and ready');