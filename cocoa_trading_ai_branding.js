/**
 * Cocoa Trading AI - Professional Branding Integration System
 * Enhances HyperVision platform with professional Cocoa Trading AI interface
 * 
 * Features:
 * - Professional branding overlay system
 * - Seamless integration with existing HyperVision platform
 * - Cocoa Trading AI theme and styling
 * - Professional typography and color scheme
 */

class CocoaTradingAIBranding {
    constructor() {
        this.isInitialized = false;
        this.brandingConfig = {
            primaryColor: '#8B4513',      // Rich brown
            secondaryColor: '#D4A574',    // Light cocoa
            accentColor: '#FFD700',       // Gold accent
            backgroundColor: '#1A1A1A',   // Dark background
            textColor: '#FFFFFF',         // White text
            successColor: '#10B981',      // Green
            warningColor: '#F59E0B',      // Amber
            errorColor: '#EF4444'         // Red
        };
        
        this.init();
    }

    async init() {
        try {
            console.log('🎨 Initializing Cocoa Trading AI Branding System...');
            
            await this.injectProfessionalStyles();
            this.updateBrandingElements();
            this.initializeProfessionalHeader();
            this.setupBrandingAnimations();
            
            this.isInitialized = true;
            console.log('✅ Cocoa Trading AI Branding System initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing branding system:', error);
        }
    }

    async injectProfessionalStyles() {
        const style = document.createElement('style');
        style.id = 'cocoa-trading-ai-styles';
        style.textContent = `
            /* Cocoa Trading AI Professional Styles */
            :root {
                --cocoa-primary: ${this.brandingConfig.primaryColor};
                --cocoa-secondary: ${this.brandingConfig.secondaryColor};
                --cocoa-accent: ${this.brandingConfig.accentColor};
                --cocoa-bg: ${this.brandingConfig.backgroundColor};
                --cocoa-text: ${this.brandingConfig.textColor};
                --cocoa-success: ${this.brandingConfig.successColor};
                --cocoa-warning: ${this.brandingConfig.warningColor};
                --cocoa-error: ${this.brandingConfig.errorColor};
            }

            /* Professional Header Enhancement */
            .cocoa-trading-header {
                background: linear-gradient(135deg, var(--cocoa-primary) 0%, var(--cocoa-secondary) 100%);
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(139, 69, 19, 0.3);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(212, 165, 116, 0.2);
            }

            .cocoa-trading-logo {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 15px;
            }

            .cocoa-trading-title {
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--cocoa-accent) 0%, #FFF 50%, var(--cocoa-accent) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .cocoa-trading-subtitle {
                color: var(--cocoa-text);
                text-align: center;
                font-size: 1.1rem;
                font-weight: 300;
                opacity: 0.9;
                margin: 5px 0 0 0;
            }

            /* Professional Panel Enhancements */
            .cocoa-panel {
                background: rgba(26, 26, 26, 0.95);
                border: 1px solid rgba(212, 165, 116, 0.2);
                border-radius: 12px;
                backdrop-filter: blur(15px);
                box-shadow: 0 8px 32px rgba(139, 69, 19, 0.2);
                transition: all 0.3s ease;
            }

            .cocoa-panel:hover {
                border-color: rgba(212, 165, 116, 0.4);
                box-shadow: 0 12px 40px rgba(139, 69, 19, 0.3);
                transform: translateY(-2px);
            }

            .cocoa-panel-header {
                background: linear-gradient(135deg, var(--cocoa-primary) 0%, var(--cocoa-secondary) 100%);
                color: var(--cocoa-text);
                padding: 15px 20px;
                border-radius: 12px 12px 0 0;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .cocoa-panel-content {
                padding: 20px;
                color: var(--cocoa-text);
            }

            /* Professional Buttons */
            .cocoa-btn-primary {
                background: linear-gradient(135deg, var(--cocoa-primary) 0%, var(--cocoa-secondary) 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
            }

            .cocoa-btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
            }

            .cocoa-btn-secondary {
                background: transparent;
                color: var(--cocoa-secondary);
                border: 2px solid var(--cocoa-secondary);
                padding: 10px 22px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .cocoa-btn-secondary:hover {
                background: var(--cocoa-secondary);
                color: var(--cocoa-bg);
            }

            /* Professional Status Indicators */
            .cocoa-status-indicator {
                display: inline-flex;
                align-items: center;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
            }

            .cocoa-status-live {
                background: rgba(16, 185, 129, 0.2);
                color: var(--cocoa-success);
                border: 1px solid var(--cocoa-success);
            }

            .cocoa-status-demo {
                background: rgba(245, 158, 11, 0.2);
                color: var(--cocoa-warning);
                border: 1px solid var(--cocoa-warning);
            }

            /* Professional Trading Cards */
            .cocoa-trading-card {
                background: rgba(26, 26, 26, 0.9);
                border: 1px solid rgba(212, 165, 116, 0.2);
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
                transition: all 0.3s ease;
            }

            .cocoa-trading-card:hover {
                border-color: var(--cocoa-accent);
                box-shadow: 0 8px 25px rgba(255, 215, 0, 0.2);
            }

            /* Professional Metrics */
            .cocoa-metric {
                text-align: center;
                padding: 15px;
                border-radius: 8px;
                background: rgba(139, 69, 19, 0.1);
                border: 1px solid rgba(139, 69, 19, 0.2);
            }

            .cocoa-metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--cocoa-accent);
                display: block;
            }

            .cocoa-metric-label {
                font-size: 0.9rem;
                color: var(--cocoa-text);
                opacity: 0.8;
                margin-top: 5px;
            }

            /* Professional Animation Classes */
            .cocoa-fade-in {
                animation: cocoaFadeIn 0.6s ease-out;
            }

            .cocoa-slide-in {
                animation: cocoaSlideIn 0.8s ease-out;
            }

            @keyframes cocoaFadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes cocoaSlideIn {
                from { opacity: 0; transform: translateX(-30px); }
                to { opacity: 1; transform: translateX(0); }
            }

            /* Professional Typography */
            .cocoa-heading-1 {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--cocoa-accent);
                margin-bottom: 20px;
            }

            .cocoa-heading-2 {
                font-size: 2rem;
                font-weight: 600;
                color: var(--cocoa-text);
                margin-bottom: 15px;
            }

            .cocoa-heading-3 {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--cocoa-secondary);
                margin-bottom: 10px;
            }

            /* Professional Form Elements */
            .cocoa-input {
                background: rgba(26, 26, 26, 0.8);
                border: 1px solid rgba(212, 165, 116, 0.3);
                border-radius: 8px;
                padding: 12px 16px;
                color: var(--cocoa-text);
                font-size: 1rem;
                width: 100%;
                transition: all 0.3s ease;
            }

            .cocoa-input:focus {
                outline: none;
                border-color: var(--cocoa-accent);
                box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
            }

            .cocoa-select {
                background: rgba(26, 26, 26, 0.8);
                border: 1px solid rgba(212, 165, 116, 0.3);
                border-radius: 8px;
                padding: 12px 16px;
                color: var(--cocoa-text);
                font-size: 1rem;
                width: 100%;
                cursor: pointer;
            }

            /* Professional Navigation */
            .cocoa-nav {
                background: rgba(139, 69, 19, 0.1);
                border-radius: 12px;
                padding: 10px;
                margin: 20px 0;
                border: 1px solid rgba(139, 69, 19, 0.2);
            }

            .cocoa-nav-item {
                display: inline-block;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 8px;
                color: var(--cocoa-text);
                text-decoration: none;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .cocoa-nav-item:hover,
            .cocoa-nav-item.active {
                background: var(--cocoa-primary);
                color: white;
            }

            /* Professional Responsive Design */
            @media (max-width: 768px) {
                .cocoa-trading-title {
                    font-size: 1.8rem;
                }
                
                .cocoa-panel {
                    margin: 10px 5px;
                }
                
                .cocoa-btn-primary,
                .cocoa-btn-secondary {
                    width: 100%;
                    margin: 5px 0;
                }
            }
        `;
        
        document.head.appendChild(style);
        console.log('✅ Professional Cocoa Trading AI styles injected');
    }

    updateBrandingElements() {
        // Update existing elements with Cocoa Trading AI branding
        const panels = document.querySelectorAll('.card, .panel, [class*="panel"]');
        panels.forEach(panel => {
            if (!panel.classList.contains('cocoa-panel')) {
                panel.classList.add('cocoa-panel');
                panel.classList.add('cocoa-fade-in');
            }
        });

        // Update buttons
        const buttons = document.querySelectorAll('button, .btn');
        buttons.forEach(btn => {
            if (!btn.classList.contains('cocoa-btn-primary') && !btn.classList.contains('cocoa-btn-secondary')) {
                if (btn.classList.contains('btn-primary') || btn.textContent.toLowerCase().includes('trade') || btn.textContent.toLowerCase().includes('start')) {
                    btn.classList.add('cocoa-btn-primary');
                } else {
                    btn.classList.add('cocoa-btn-secondary');
                }
            }
        });

        // Update form inputs
        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (!input.classList.contains('cocoa-input') && !input.classList.contains('cocoa-select')) {
                if (input.tagName.toLowerCase() === 'select') {
                    input.classList.add('cocoa-select');
                } else {
                    input.classList.add('cocoa-input');
                }
            }
        });

        console.log('✅ Branding elements updated with Cocoa Trading AI styling');
    }

    initializeProfessionalHeader() {
        // Create or update the main header with Cocoa Trading AI branding
        let headerContainer = document.querySelector('.cocoa-trading-header');
        
        if (!headerContainer) {
            headerContainer = document.createElement('div');
            headerContainer.className = 'cocoa-trading-header cocoa-fade-in';
            
            // Insert at the beginning of the main content
            const mainContent = document.querySelector('.container, .main-content, body > div') || document.body;
            mainContent.insertBefore(headerContainer, mainContent.firstChild);
        }

        headerContainer.innerHTML = `
            <div class="cocoa-trading-logo">
                <h1 class="cocoa-trading-title">Cocoa Trading AI</h1>
            </div>
            <p class="cocoa-trading-subtitle">
                Professional High-Frequency Trading & Arbitrage Platform
            </p>
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px; gap: 20px;">
                <div class="cocoa-status-indicator cocoa-status-live">
                    <span style="display: inline-block; width: 8px; height: 8px; background: var(--cocoa-success); border-radius: 50%; margin-right: 8px;"></span>
                    Live Market Data
                </div>
                <div class="cocoa-status-indicator cocoa-status-live">
                    <span style="display: inline-block; width: 8px; height: 8px; background: var(--cocoa-success); border-radius: 50%; margin-right: 8px;"></span>
                    AI Systems Active
                </div>
                <div class="cocoa-status-indicator cocoa-status-live">
                    <span style="display: inline-block; width: 8px; height: 8px; background: var(--cocoa-success); border-radius: 50%; margin-right: 8px;"></span>
                    Arbitrage Engines Online
                </div>
            </div>
        `;

        console.log('✅ Professional Cocoa Trading AI header initialized');
    }

    setupBrandingAnimations() {
        // Add entrance animations to elements
        const animateElements = () => {
            const elements = document.querySelectorAll('.cocoa-panel, .cocoa-trading-card');
            elements.forEach((element, index) => {
                setTimeout(() => {
                    element.classList.add('cocoa-fade-in');
                }, index * 100);
            });
        };

        // Run animations after a short delay
        setTimeout(animateElements, 500);

        // Set up intersection observer for scroll animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('cocoa-slide-in');
                }
            });
        }, { threshold: 0.1 });

        // Observe new elements as they're added
        const observeNewElements = () => {
            const elements = document.querySelectorAll('.cocoa-panel:not(.observed)');
            elements.forEach(element => {
                observer.observe(element);
                element.classList.add('observed');
            });
        };

        observeNewElements();
        setInterval(observeNewElements, 2000);

        console.log('✅ Branding animations initialized');
    }

    // Method to apply branding to specific elements
    applyBrandingToElement(element, type = 'panel') {
        if (!element) return;

        switch (type) {
            case 'panel':
                element.classList.add('cocoa-panel', 'cocoa-fade-in');
                break;
            case 'button-primary':
                element.classList.add('cocoa-btn-primary');
                break;
            case 'button-secondary':
                element.classList.add('cocoa-btn-secondary');
                break;
            case 'card':
                element.classList.add('cocoa-trading-card', 'cocoa-fade-in');
                break;
            case 'metric':
                element.classList.add('cocoa-metric');
                break;
        }
    }

    // Method to create professional metric display
    createMetricDisplay(value, label, container) {
        const metricDiv = document.createElement('div');
        metricDiv.className = 'cocoa-metric cocoa-fade-in';
        metricDiv.innerHTML = `
            <span class="cocoa-metric-value">${value}</span>
            <div class="cocoa-metric-label">${label}</div>
        `;
        
        if (container) {
            container.appendChild(metricDiv);
        }
        
        return metricDiv;
    }

    // Method to create professional status indicator
    createStatusIndicator(text, isActive = true, container) {
        const statusDiv = document.createElement('div');
        statusDiv.className = `cocoa-status-indicator ${isActive ? 'cocoa-status-live' : 'cocoa-status-demo'}`;
        statusDiv.innerHTML = `
            <span style="display: inline-block; width: 8px; height: 8px; background: var(--cocoa-${isActive ? 'success' : 'warning'}); border-radius: 50%; margin-right: 8px;"></span>
            ${text}
        `;
        
        if (container) {
            container.appendChild(statusDiv);
        }
        
        return statusDiv;
    }

    // Method to update theme dynamically
    updateTheme(newConfig) {
        Object.assign(this.brandingConfig, newConfig);
        
        // Update CSS variables
        const root = document.documentElement;
        Object.keys(this.brandingConfig).forEach(key => {
            const cssVar = `--cocoa-${key.replace(/([A-Z])/g, '-$1').toLowerCase().replace('color', '').replace('background-', 'bg')}`;
            root.style.setProperty(cssVar, this.brandingConfig[key]);
        });
        
        console.log('✅ Cocoa Trading AI theme updated');
    }

    // Method to get current branding status
    getStatus() {
        return {
            initialized: this.isInitialized,
            config: this.brandingConfig,
            elementsStyled: document.querySelectorAll('.cocoa-panel, .cocoa-btn-primary').length
        };
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.cocoaTradingAIBranding = new CocoaTradingAIBranding();
    });
} else {
    window.cocoaTradingAIBranding = new CocoaTradingAIBranding();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CocoaTradingAIBranding;
}

console.log('🎨 Cocoa Trading AI Branding System loaded and ready for professional enhancement');