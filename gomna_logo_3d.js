/**
 * GOMNA 3D LOGO COMPONENT
 * Advanced 3D logo rendering based on GM trademark
 * Interactive animations and premium effects
 */

class GomnaLogo3D {
    constructor(options = {}) {
        this.options = {
            size: options.size || 'medium', // 'small', 'medium', 'large'
            premium: options.premium || false,
            animated: options.animated !== false, // Default true
            trademark: options.trademark !== false, // Default true
            onClick: options.onClick || null,
            className: options.className || ''
        };
    }

    render() {
        const sizeClass = this.getSizeClass();
        const premiumClass = this.options.premium ? 'gomna-logo-premium' : '';
        const customClass = this.options.className;

        return `
            <div class="gomna-logo-container ${customClass}">
                <div class="gomna-logo-3d ${sizeClass} ${premiumClass}" 
                     onclick="${this.options.onClick ? this.options.onClick.name + '()' : ''}"
                     title="Gomna Trading AI - Advanced Portfolio Optimization">
                    <div class="logo-oval"></div>
                    <div class="logo-letters">GM</div>
                    ${this.options.trademark ? '<div class="logo-trademark">™</div>' : ''}
                </div>
            </div>
        `;
    }

    getSizeClass() {
        switch (this.options.size) {
            case 'small': return 'gomna-logo-small';
            case 'large': return 'gomna-logo-large';
            default: return '';
        }
    }

    // Static method to create logo HTML
    static create(options = {}) {
        const logo = new GomnaLogo3D(options);
        return logo.render();
    }

    // Insert logo into DOM element
    static insertInto(elementId, options = {}) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = GomnaLogo3D.create(options);
        }
    }

    // Replace existing logos in the page
    static replaceLogos() {
        // Replace common logo selectors with 3D version
        const logoSelectors = [
            '.logo',
            '.brand-logo',
            '[class*="logo"]',
            '.navbar-brand',
            '.header-logo'
        ];

        logoSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                // Determine size based on element dimensions
                const rect = element.getBoundingClientRect();
                let size = 'medium';
                if (rect.width < 50) size = 'small';
                else if (rect.width > 100) size = 'large';

                element.innerHTML = GomnaLogo3D.create({
                    size: size,
                    premium: true,
                    className: element.className
                });
            });
        });
    }

    // Add to navigation/header
    static addToHeader() {
        const headerSelectors = [
            'header',
            '.header',
            '.navbar',
            '.navigation',
            '.top-bar'
        ];

        headerSelectors.forEach(selector => {
            const header = document.querySelector(selector);
            if (header) {
                const logoContainer = document.createElement('div');
                logoContainer.className = 'gomna-logo-header';
                logoContainer.style.cssText = `
                    position: absolute;
                    top: 10px;
                    left: 20px;
                    z-index: 1000;
                `;
                
                logoContainer.innerHTML = GomnaLogo3D.create({
                    size: 'large',
                    premium: true,
                    trademark: true
                });

                header.appendChild(logoContainer);
                return; // Only add to first found header
            }
        });
    }

    // Add floating logo
    static addFloatingLogo() {
        const floatingContainer = document.createElement('div');
        floatingContainer.id = 'gomna-floating-logo';
        floatingContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            pointer-events: none;
        `;

        floatingContainer.innerHTML = GomnaLogo3D.create({
            size: 'medium',
            premium: true,
            trademark: true,
            className: 'floating-logo'
        });

        document.body.appendChild(floatingContainer);
    }

    // Initialize all logo placements
    static initializeAll() {
        // Load CSS if not already loaded
        if (!document.querySelector('link[href*="logo_3d.css"]')) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = './logo_3d.css';
            document.head.appendChild(cssLink);
        }

        // Add to header
        setTimeout(() => {
            this.addToHeader();
            this.addFloatingLogo();
            this.replaceLogos();
        }, 100);
    }

    // Update page title and favicon
    static updateBranding() {
        // Update page title
        if (document.title.indexOf('Gomna') === -1) {
            document.title = 'Gomna Trading AI - ' + document.title;
        }

        // Create SVG favicon
        const faviconSvg = `
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="48" viewBox="0 0 32 48">
                <defs>
                    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#2563eb"/>
                        <stop offset="50%" style="stop-color:#3b82f6"/>
                        <stop offset="100%" style="stop-color:#60a5fa"/>
                    </linearGradient>
                </defs>
                <ellipse cx="16" cy="24" rx="15" ry="22" fill="url(#logoGrad)" stroke="#1e40af" stroke-width="1"/>
                <text x="16" y="28" font-family="Arial Black" font-size="12" font-weight="900" 
                      fill="white" text-anchor="middle" stroke="#1e40af" stroke-width="0.5">GM</text>
            </svg>
        `;

        // Convert SVG to data URL and set as favicon
        const faviconDataUrl = 'data:image/svg+xml;base64,' + btoa(faviconSvg);
        
        let favicon = document.querySelector('link[rel="icon"]') || 
                     document.querySelector('link[rel="shortcut icon"]');
        
        if (!favicon) {
            favicon = document.createElement('link');
            favicon.rel = 'icon';
            document.head.appendChild(favicon);
        }
        
        favicon.href = faviconDataUrl;
    }
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            GomnaLogo3D.initializeAll();
            GomnaLogo3D.updateBranding();
        });
    } else {
        GomnaLogo3D.initializeAll();
        GomnaLogo3D.updateBranding();
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.GomnaLogo3D = GomnaLogo3D;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GomnaLogo3D;
}