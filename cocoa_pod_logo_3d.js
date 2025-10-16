/**
 * GOMNA 3D COCOA POD LOGO COMPONENT
 * Organic growth inspired logo with seeds inside cocoa pod
 * Advanced 3D CSS rendering and interaction system
 */

class CocoaPodLogo3D {
    constructor() {
        this.instances = new Map();
        this.animationFrame = null;
        this.init();
    }

    init() {
        // Auto-initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeLogos());
        } else {
            this.initializeLogos();
        }
    }

    /**
     * Create a 3D cocoa pod logo instance
     * @param {Object} options - Configuration options
     * @param {string} options.container - Selector for container element
     * @param {string} options.size - Size variant: 'small', 'medium', 'large'
     * @param {boolean} options.premium - Enable premium effects
     * @param {string} options.text - Brand text to display
     * @param {boolean} options.trademark - Show trademark symbol
     * @param {Object} options.animation - Animation settings
     */
    static create(options = {}) {
        const defaults = {
            container: '.logo-container',
            size: 'medium',
            premium: false,
            text: 'GOMNA',
            trademark: true,
            animation: {
                enabled: true,
                speed: 1,
                hover: true
            }
        };

        const config = { ...defaults, ...options };
        const container = document.querySelector(config.container);
        
        if (!container) {
            console.warn(`CocoaPodLogo3D: Container "${config.container}" not found`);
            return null;
        }

        return new CocoaPodLogo3D().createLogo(container, config);
    }

    createLogo(container, config) {
        // Clear existing content
        container.innerHTML = '';
        
        // Create logo HTML structure
        const logoHTML = this.generateLogoHTML(config);
        container.innerHTML = logoHTML;
        
        // Apply size classes
        const logoElement = container.querySelector('.gomna-logo-3d');
        this.applySizeClass(logoElement, config.size);
        
        // Apply premium effects if enabled
        if (config.premium) {
            logoElement.closest('.gomna-logo-container').classList.add('gomna-logo-premium');
        }
        
        // Setup interactions
        this.setupInteractions(logoElement, config);
        
        // Store instance
        const instanceId = this.generateInstanceId();
        this.instances.set(instanceId, {
            element: logoElement,
            config: config,
            container: container
        });
        
        return {
            id: instanceId,
            element: logoElement,
            update: (newConfig) => this.updateLogo(instanceId, newConfig),
            destroy: () => this.destroyLogo(instanceId)
        };
    }

    generateLogoHTML(config) {
        // Generate seeds HTML
        const seedsHTML = Array.from({ length: 17 }, (_, i) => 
            `<div class="cocoa-seed"></div>`
        ).join('');

        return `
            <div class="gomna-logo-container">
                <div class="gomna-logo-3d">
                    <!-- Cocoa Pod Outer Shell -->
                    <div class="cocoa-pod-shell"></div>
                    
                    <!-- Pod Interior with Seeds -->
                    <div class="pod-interior">
                        <!-- Central Placenta/Spine -->
                        <div class="pod-placenta"></div>
                        
                        <!-- Cocoa Seeds arranged naturally -->
                        ${seedsHTML}
                        
                        <!-- Pulp/Mucilage around seeds -->
                        <div class="pod-pulp"></div>
                    </div>
                    
                    <!-- Brand Text -->
                    <div class="logo-brand-text">
                        ${config.text}
                        ${config.trademark ? '<sup class="logo-trademark">™</sup>' : ''}
                    </div>
                </div>
            </div>
        `;
    }

    applySizeClass(element, size) {
        // Remove existing size classes
        element.classList.remove('gomna-logo-small', 'gomna-logo-large');
        
        // Apply new size class
        switch (size) {
            case 'small':
                element.classList.add('gomna-logo-small');
                break;
            case 'large':
                element.classList.add('gomna-logo-large');
                break;
            case 'medium':
            default:
                // Default size, no additional class needed
                break;
        }
    }

    setupInteractions(logoElement, config) {
        if (!config.animation.enabled) return;

        // Enhanced hover effects
        if (config.animation.hover) {
            logoElement.addEventListener('mouseenter', (e) => {
                this.onHoverStart(e.target, config);
            });

            logoElement.addEventListener('mouseleave', (e) => {
                this.onHoverEnd(e.target, config);
            });
        }

        // Click effects
        logoElement.addEventListener('click', (e) => {
            this.onClickEffect(e.target, config);
        });

        // Touch support for mobile
        logoElement.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.onHoverStart(e.target, config);
        });

        logoElement.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.onClickEffect(e.target, config);
            setTimeout(() => this.onHoverEnd(e.target, config), 100);
        });
    }

    onHoverStart(element, config) {
        // Add hover state classes
        element.classList.add('logo-hover-active');
        
        // Trigger seed shimmer sequence
        const seeds = element.querySelectorAll('.cocoa-seed');
        seeds.forEach((seed, index) => {
            setTimeout(() => {
                seed.style.animationPlayState = 'running';
                seed.style.filter = 'brightness(1.3) saturate(1.2)';
            }, index * 50);
        });

        // Emit custom event
        element.dispatchEvent(new CustomEvent('cocoapod:hover', { 
            detail: { config, action: 'start' } 
        }));
    }

    onHoverEnd(element, config) {
        // Remove hover state
        element.classList.remove('logo-hover-active');
        
        // Reset seed effects
        const seeds = element.querySelectorAll('.cocoa-seed');
        seeds.forEach(seed => {
            seed.style.filter = '';
        });

        // Emit custom event
        element.dispatchEvent(new CustomEvent('cocoapod:hover', { 
            detail: { config, action: 'end' } 
        }));
    }

    onClickEffect(element, config) {
        // Add click animation class
        element.classList.add('logo-click-active');
        
        // Create ripple effect
        this.createRippleEffect(element);
        
        // Remove click class after animation
        setTimeout(() => {
            element.classList.remove('logo-click-active');
        }, 300);

        // Emit custom event
        element.dispatchEvent(new CustomEvent('cocoapod:click', { 
            detail: { config } 
        }));
    }

    createRippleEffect(element) {
        const ripple = document.createElement('div');
        ripple.className = 'logo-ripple-effect';
        ripple.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            background: rgba(205, 133, 63, 0.3);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 20;
            animation: ripple-expand 0.6s ease-out forwards;
        `;

        // Add ripple keyframes if not exists
        if (!document.querySelector('#ripple-styles')) {
            const style = document.createElement('style');
            style.id = 'ripple-styles';
            style.textContent = `
                @keyframes ripple-expand {
                    0% {
                        width: 10px;
                        height: 10px;
                        opacity: 0.8;
                    }
                    100% {
                        width: 100px;
                        height: 100px;
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        element.appendChild(ripple);
        
        // Remove ripple after animation
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }

    initializeLogos() {
        // Auto-initialize logos with data attributes
        const autoLogos = document.querySelectorAll('[data-cocoa-logo]');
        
        autoLogos.forEach(container => {
            const config = this.parseDataAttributes(container);
            this.createLogo(container, config);
        });
    }

    parseDataAttributes(element) {
        const config = {
            size: element.dataset.cocoaSize || 'medium',
            premium: element.dataset.cocoaPremium === 'true',
            text: element.dataset.cocoaText || 'GOMNA',
            trademark: element.dataset.cocoaTrademark !== 'false',
            animation: {
                enabled: element.dataset.cocoaAnimation !== 'false',
                speed: parseFloat(element.dataset.cocoaSpeed) || 1,
                hover: element.dataset.cocoaHover !== 'false'
            }
        };

        return config;
    }

    updateLogo(instanceId, newConfig) {
        const instance = this.instances.get(instanceId);
        if (!instance) return false;

        const mergedConfig = { ...instance.config, ...newConfig };
        this.createLogo(instance.container, mergedConfig);
        
        // Update stored config
        instance.config = mergedConfig;
        return true;
    }

    destroyLogo(instanceId) {
        const instance = this.instances.get(instanceId);
        if (!instance) return false;

        // Clean up event listeners
        const element = instance.element;
        const clonedElement = element.cloneNode(true);
        element.parentNode.replaceChild(clonedElement, element);
        
        // Remove from instances
        this.instances.delete(instanceId);
        
        return true;
    }

    generateInstanceId() {
        return 'cocoa-logo-' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Utility methods for branding integration
     */
    static updateFavicon(size = 32) {
        // Create canvas for favicon generation
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        // Simple cocoa pod favicon representation
        ctx.fillStyle = '#8B4513';
        ctx.fillRect(0, 0, size, size);
        
        // Convert to favicon
        const link = document.querySelector("link[rel*='icon']") || document.createElement('link');
        link.type = 'image/x-icon';
        link.rel = 'shortcut icon';
        link.href = canvas.toDataURL();
        document.getElementsByTagName('head')[0].appendChild(link);
    }

    static updatePageTitle(title) {
        document.title = title + ' | Gomna™ Trading AI';
        
        // Update meta tags
        let metaDesc = document.querySelector('meta[name="description"]');
        if (!metaDesc) {
            metaDesc = document.createElement('meta');
            metaDesc.name = 'description';
            document.head.appendChild(metaDesc);
        }
        metaDesc.content = `${title} - Powered by Gomna™ organic growth trading platform`;
    }

    /**
     * Advanced animation controls
     */
    pauseAllAnimations() {
        document.querySelectorAll('.gomna-logo-3d').forEach(logo => {
            logo.style.animationPlayState = 'paused';
        });
    }

    resumeAllAnimations() {
        document.querySelectorAll('.gomna-logo-3d').forEach(logo => {
            logo.style.animationPlayState = 'running';
        });
    }

    setAnimationSpeed(speed) {
        document.querySelectorAll('.gomna-logo-3d').forEach(logo => {
            logo.style.animationDuration = `${6/speed}s`;
        });
    }
}

// Auto-initialize global instance
const cocoaPodLogo3D = new CocoaPodLogo3D();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CocoaPodLogo3D;
}

// Global access
window.CocoaPodLogo3D = CocoaPodLogo3D;

// Initialize common branding elements
document.addEventListener('DOMContentLoaded', function() {
    // Update favicon
    CocoaPodLogo3D.updateFavicon(32);
    
    // Update page branding if not already set
    if (document.title === '' || document.title === 'Document') {
        CocoaPodLogo3D.updatePageTitle('Organic Growth Trading');
    }
    
    console.log('🌱 CocoaPodLogo3D initialized - Organic growth branding active');
});