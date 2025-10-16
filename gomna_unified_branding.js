// GOMNA Unified Branding - Single consistent branding across platform
// This script ensures only one GOMNA branding appears with proper styling

class GomnaUnifiedBranding {
    constructor() {
        this.removeDuplicateBranding();
        this.setupUnifiedBranding();
    }

    removeDuplicateBranding() {
        // Remove any duplicate logos or branding elements
        const existingLogos = document.querySelectorAll('.logo-container, .gomna-branding');
        existingLogos.forEach(element => {
            element.remove();
        });

        // Also hide the old logo in the header if it exists
        const oldLogos = document.querySelectorAll('.bg-gradient-to-br.from-amber-600, .bg-gradient-to-br.from-amber-500');
        oldLogos.forEach(logo => {
            const parent = logo.parentElement;
            if (parent) {
                parent.style.display = 'none';
            }
        });
    }

    setupUnifiedBranding() {
        // Create the single unified branding element
        const brandingWrapper = document.createElement('div');
        brandingWrapper.id = 'gomna-unified-branding';
        brandingWrapper.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 20px;
        `;

        // Create logo container (cocoa pod)
        const logoContainer = document.createElement('div');
        logoContainer.className = 'gomna-logo';
        logoContainer.style.cssText = `
            background: #FAF7F0;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(107, 68, 35, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            cursor: pointer;
            transition: all 0.3s ease;
        `;

        // Cocoa pod SVG logo
        const logoSVG = `
            <svg width="44" height="44" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="podGradientUnified" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#8B6F47" />
                        <stop offset="100%" style="stop-color:#5D4037" />
                    </linearGradient>
                    <filter id="shadowUnified">
                        <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
                    </filter>
                </defs>
                <!-- Cocoa pod shape -->
                <ellipse cx="100" cy="100" rx="50" ry="75" fill="url(#podGradientUnified)" filter="url(#shadowUnified)" transform="rotate(12 100 100)" />
                <!-- Center line -->
                <path d="M 100 40 L 100 160" stroke="#6B4423" stroke-width="3" opacity="0.5" />
                <!-- Cocoa seeds with better visibility -->
                <ellipse cx="85" cy="80" rx="14" ry="18" fill="#FAF7F0" opacity="0.95" />
                <ellipse cx="115" cy="80" rx="14" ry="18" fill="#F5E6D3" opacity="0.95" />
                <ellipse cx="100" cy="110" rx="15" ry="19" fill="#FAF7F0" opacity="0.95" />
                <ellipse cx="85" cy="140" rx="14" ry="18" fill="#F5E6D3" opacity="0.95" />
                <ellipse cx="115" cy="140" rx="14" ry="18" fill="#FAF7F0" opacity="0.95" />
                <!-- Seed details -->
                <circle cx="85" cy="80" r="4" fill="#8B6F47" opacity="0.8" />
                <circle cx="115" cy="80" r="4" fill="#8B6F47" opacity="0.8" />
                <circle cx="100" cy="110" r="4" fill="#8B6F47" opacity="0.8" />
                <circle cx="85" cy="140" r="4" fill="#8B6F47" opacity="0.8" />
                <circle cx="115" cy="140" r="4" fill="#8B6F47" opacity="0.8" />
            </svg>
        `;

        logoContainer.innerHTML = logoSVG;

        // Add hover effects to logo
        logoContainer.onmouseenter = () => {
            logoContainer.style.transform = 'scale(1.1) rotate(5deg)';
            logoContainer.style.boxShadow = '0 6px 16px rgba(107, 68, 35, 0.3)';
        };
        logoContainer.onmouseleave = () => {
            logoContainer.style.transform = 'scale(1) rotate(0deg)';
            logoContainer.style.boxShadow = '0 4px 12px rgba(107, 68, 35, 0.2)';
        };

        // Add click handler for logo selection
        logoContainer.onclick = () => {
            window.open('cocoa_logos.html', '_blank');
        };

        // Create text branding
        const textBranding = document.createElement('div');
        textBranding.style.cssText = `
            display: flex;
            flex-direction: column;
            justify-content: center;
        `;
        
        textBranding.innerHTML = `
            <div style="
                font-size: 36px; 
                font-weight: bold; 
                color: #3E2723; 
                letter-spacing: 5px; 
                font-family: 'Georgia', 'Times New Roman', serif; 
                font-style: italic;
                text-shadow: 1px 1px 2px rgba(107, 68, 35, 0.2);
                line-height: 1;
            ">GOMNA</div>
            <div style="
                font-size: 13px; 
                color: #6B4423; 
                letter-spacing: 0.8px; 
                margin-top: 4px; 
                font-family: 'Georgia', 'Times New Roman', serif;
                font-weight: 500;
            ">Wall Street Grade Quantitative Trading Platform</div>
        `;

        // Assemble the branding
        brandingWrapper.appendChild(logoContainer);
        brandingWrapper.appendChild(textBranding);

        // Add to page
        document.body.appendChild(brandingWrapper);

        // Add animation styles
        const animationStyle = document.createElement('style');
        animationStyle.textContent = `
            @keyframes gentleFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-2px); }
            }
            
            #gomna-unified-branding .gomna-logo svg {
                animation: gentleFloat 4s ease-in-out infinite;
            }
            
            /* Ensure proper spacing for main content */
            body > .container:first-of-type,
            body > div:first-of-type > .container,
            main,
            .glass-effect:first-of-type {
                margin-top: 100px !important;
            }
            
            /* Hide any duplicate h1 with GOMNA */
            h1:contains("GOMNA") {
                visibility: hidden;
                height: 0;
                margin: 0;
            }
        `;
        document.head.appendChild(animationStyle);

        // Update any remaining GOMNA text in headers to avoid duplication
        this.updateHeaders();
    }

    updateHeaders() {
        // Find and update h1 elements that contain GOMNA
        const headers = document.querySelectorAll('h1');
        headers.forEach(header => {
            if (header.textContent && header.textContent.includes('GOMNA')) {
                // Instead of hiding, replace with empty or update with different content
                const parent = header.parentElement;
                if (parent) {
                    // Check if this is in the main header area
                    const isMainHeader = parent.closest('.glass-effect') || parent.closest('header');
                    if (isMainHeader) {
                        // Remove the h1 but keep the subtitle if it exists
                        const subtitle = parent.querySelector('p');
                        if (subtitle && subtitle.textContent.includes('Wall Street')) {
                            subtitle.style.display = 'none'; // Hide subtitle since we have it in the unified branding
                        }
                        header.style.display = 'none'; // Hide the h1
                    }
                }
            }
        });

        // Also update the document title
        document.title = 'GOMNA - Wall Street Grade Quantitative Trading Platform';
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a bit for other scripts to load
        setTimeout(() => {
            window.gomnaUnifiedBranding = new GomnaUnifiedBranding();
        }, 100);
    });
} else {
    // Small delay to ensure other elements are loaded
    setTimeout(() => {
        window.gomnaUnifiedBranding = new GomnaUnifiedBranding();
    }, 100);
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GomnaUnifiedBranding;
}