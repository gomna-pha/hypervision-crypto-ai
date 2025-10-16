// GOMNA Platform Theme Application Script
// This script applies the cream and brown theme and integrates the 3D cocoa pod logo

class GomnaThemeManager {
    constructor() {
        this.selectedLogo = null;
        this.initializeTheme();
    }

    initializeTheme() {
        // Apply cream and brown color scheme
        this.applyColorTheme();
        
        // Add 3D cocoa pod logo
        this.addCocoaLogo();
        
        // Update branding to GOMNA
        this.updateBranding();
        
        // Remove conflicting colors
        this.removeOldColors();
    }

    applyColorTheme() {
        // Create and inject theme stylesheet
        const themeLink = document.createElement('link');
        themeLink.rel = 'stylesheet';
        themeLink.href = 'cream_brown_theme.css';
        document.head.appendChild(themeLink);

        // Override existing color variables
        const style = document.createElement('style');
        style.textContent = `
            /* Override all existing colors with cream and brown theme */
            :root {
                /* Primary Cream Colors */
                --cream-50: #FEFDFB;
                --cream-100: #FAF7F0;
                --cream-200: #F5E6D3;
                --cream-300: #E8DCC7;
                --cream-400: #DDD0B8;
                --cream-500: #D4C4A8;
                
                /* Primary Brown Colors */
                --brown-100: #D4A574;
                --brown-200: #BC9A6A;
                --brown-300: #A0826D;
                --brown-400: #8B6F47;
                --brown-500: #6B4423;
                --brown-600: #5D4037;
                --brown-700: #4A2C2A;
                --brown-800: #3E2723;
                --brown-900: #2E1A1A;
                
                /* Accent Colors */
                --gold: #D4AF37;
                --bronze: #CD7F32;
                --cocoa: #7B3F00;
                
                /* Background Gradients */
                --bg-gradient-main: linear-gradient(135deg, #FAF7F0 0%, #F5E6D3 100%);
                --bg-gradient-dark: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
                --bg-gradient-premium: linear-gradient(135deg, #D4AF37 0%, #CD7F32 100%);
            }

            /* Override body background */
            body {
                background: var(--bg-gradient-main) !important;
                color: var(--brown-700) !important;
                font-family: 'Georgia', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
            }

            /* Override all blue/purple/green colors */
            .bg-blue-600, .bg-blue-700, .bg-blue-800,
            .bg-purple-600, .bg-purple-700, .bg-purple-800,
            .bg-green-600, .bg-green-700, .bg-green-800,
            .bg-indigo-600, .bg-indigo-700, .bg-indigo-800 {
                background: var(--bg-gradient-dark) !important;
            }

            .text-blue-600, .text-blue-700, .text-blue-800,
            .text-purple-600, .text-purple-700, .text-purple-800,
            .text-green-600, .text-green-700, .text-green-800,
            .text-indigo-600, .text-indigo-700, .text-indigo-800 {
                color: var(--brown-600) !important;
            }

            /* Update button styles */
            .btn-primary, 
            button[class*="bg-blue"], 
            button[class*="bg-purple"],
            button[class*="bg-green"],
            button[class*="bg-indigo"] {
                background: var(--bg-gradient-dark) !important;
                color: var(--cream-100) !important;
                border: none !important;
            }

            .btn-primary:hover,
            button[class*="bg-blue"]:hover,
            button[class*="bg-purple"]:hover,
            button[class*="bg-green"]:hover,
            button[class*="bg-indigo"]:hover {
                background: linear-gradient(135deg, #4A2C2A 0%, #2E1A1A 100%) !important;
            }

            /* Update card backgrounds */
            .bg-white {
                background-color: var(--cream-50) !important;
            }

            .bg-gray-50 {
                background-color: var(--cream-100) !important;
            }

            .bg-gray-100 {
                background-color: var(--cream-200) !important;
            }

            /* Update borders */
            .border-gray-200, .border-gray-300 {
                border-color: var(--brown-200) !important;
            }

            /* Update text colors */
            .text-gray-600, .text-gray-700 {
                color: var(--brown-600) !important;
            }

            .text-gray-800, .text-gray-900 {
                color: var(--brown-800) !important;
            }

            /* Update shadows */
            [class*="shadow"] {
                box-shadow: 0 4px 6px rgba(107, 68, 35, 0.1) !important;
            }

            /* Navigation bar */
            nav, header {
                background: var(--cream-50) !important;
                border-bottom: 2px solid var(--brown-300) !important;
            }

            /* Metric cards */
            .metric-card {
                background: linear-gradient(135deg, var(--cream-50) 0%, var(--cream-100) 100%) !important;
                border: 1px solid var(--brown-200) !important;
            }

            /* Charts color scheme */
            .chart-container {
                background: var(--cream-50) !important;
                border-radius: 12px;
                padding: 15px;
            }
        `;
        document.head.appendChild(style);
    }

    addCocoaLogo() {
        // Create logo container
        const logoContainer = document.createElement('div');
        logoContainer.className = 'logo-container';
        logoContainer.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 10000;
            background: #FAF7F0;
            border-radius: 15px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(107, 68, 35, 0.2);
            display: flex;
            align-items: center;
            gap: 15px;
            cursor: pointer;
            transition: transform 0.3s ease;
        `;

        // Add hover effect
        logoContainer.onmouseenter = () => {
            logoContainer.style.transform = 'scale(1.05)';
        };
        logoContainer.onmouseleave = () => {
            logoContainer.style.transform = 'scale(1)';
        };

        // Create 3D cocoa pod SVG logo (Premium Embossed style as default)
        const logoSVG = `
            <svg class="logo-3d" width="60" height="60" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="podGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#8B6F47" />
                        <stop offset="100%" style="stop-color:#5D4037" />
                    </linearGradient>
                    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#D4AF37" />
                        <stop offset="100%" style="stop-color:#CD7F32" />
                    </linearGradient>
                    <filter id="emboss">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="blur" />
                        <feSpecularLighting result="specOut" in="blur" specularConstant="1.5" specularExponent="20" lighting-color="white">
                            <fePointLight x="-50" y="30" z="200" />
                        </feSpecularLighting>
                        <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut2" />
                        <feComposite in="SourceGraphic" in2="specOut2" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" />
                    </filter>
                    <filter id="goldShine">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="5" result="blur" />
                        <feSpecularLighting result="specOut" in="blur" specularConstant="2" specularExponent="30" lighting-color="#FFD700">
                            <fePointLight x="-50" y="30" z="300" />
                        </feSpecularLighting>
                        <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut2" />
                        <feComposite in="SourceGraphic" in2="specOut2" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" />
                    </filter>
                </defs>
                
                <!-- Outer gold ring -->
                <circle cx="100" cy="100" r="90" fill="none" stroke="url(#goldGradient)" stroke-width="4" filter="url(#goldShine)" opacity="0.9" />
                
                <!-- Premium cocoa pod shape -->
                <ellipse cx="100" cy="100" rx="45" ry="70" fill="url(#podGradient)" filter="url(#emboss)" transform="rotate(15 100 100)" />
                
                <!-- Center ridge with embossing -->
                <path d="M 85 50 Q 100 55 115 50 L 110 150 Q 100 155 90 150 Z" fill="#6B4423" opacity="0.7" filter="url(#emboss)" />
                
                <!-- Premium cocoa seeds -->
                <ellipse cx="85" cy="75" rx="12" ry="16" fill="#FAF7F0" opacity="0.95" filter="url(#emboss)" />
                <ellipse cx="115" cy="75" rx="12" ry="16" fill="#F5E6D3" opacity="0.95" filter="url(#emboss)" />
                <ellipse cx="100" cy="100" rx="13" ry="17" fill="#FAF7F0" opacity="0.95" filter="url(#emboss)" />
                <ellipse cx="85" cy="125" rx="12" ry="16" fill="#F5E6D3" opacity="0.95" filter="url(#emboss)" />
                <ellipse cx="115" cy="125" rx="12" ry="16" fill="#FAF7F0" opacity="0.95" filter="url(#emboss)" />
                
                <!-- Gold accent details -->
                <circle cx="85" cy="75" r="3" fill="url(#goldGradient)" opacity="0.7" />
                <circle cx="115" cy="75" r="3" fill="url(#goldGradient)" opacity="0.7" />
                <circle cx="100" cy="100" r="3" fill="url(#goldGradient)" opacity="0.7" />
                <circle cx="85" cy="125" r="3" fill="url(#goldGradient)" opacity="0.7" />
                <circle cx="115" cy="125" r="3" fill="url(#goldGradient)" opacity="0.7" />
                
                <!-- Shine effect -->
                <ellipse cx="90" cy="65" rx="15" ry="25" fill="white" opacity="0.2" filter="blur(3px)" />
            </svg>
        `;

        logoContainer.innerHTML = logoSVG;

        // Add GOMNA text
        const logoText = document.createElement('div');
        logoText.innerHTML = `
            <div style="display: flex; flex-direction: column;">
                <span style="font-size: 24px; font-weight: bold; color: #3E2723; letter-spacing: 2px;">GOMNA</span>
                <span style="font-size: 10px; color: #6B4423; text-transform: uppercase; letter-spacing: 1px;">Trading Platform</span>
            </div>
        `;
        logoContainer.appendChild(logoText);

        // Add click handler to open logo selector
        logoContainer.onclick = () => {
            window.open('cocoa_logos.html', '_blank');
        };

        // Add to page
        document.body.appendChild(logoContainer);

        // Add floating animation
        const animationStyle = document.createElement('style');
        animationStyle.textContent = `
            @keyframes gentle-float {
                0%, 100% { transform: translateY(0px) rotateZ(0deg); }
                25% { transform: translateY(-3px) rotateZ(1deg); }
                50% { transform: translateY(-2px) rotateZ(0deg); }
                75% { transform: translateY(-3px) rotateZ(-1deg); }
            }
            .logo-3d {
                animation: gentle-float 4s ease-in-out infinite;
                filter: drop-shadow(0 3px 6px rgba(107, 68, 35, 0.3));
            }
        `;
        document.head.appendChild(animationStyle);
    }

    updateBranding() {
        // Update all text references from "Gomna AI" or "Cocoa Trading AI" to "GOMNA"
        const textNodes = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const replacements = [];
        while (textNodes.nextNode()) {
            const node = textNodes.currentNode;
            if (node.nodeValue && (
                node.nodeValue.includes('Gomna AI') || 
                node.nodeValue.includes('Cocoa Trading AI') ||
                node.nodeValue.includes('Gomna') ||
                node.nodeValue.includes('Cocoa Trading')
            )) {
                replacements.push(node);
            }
        }

        replacements.forEach(node => {
            node.nodeValue = node.nodeValue
                .replace(/Gomna AI/gi, 'GOMNA')
                .replace(/Cocoa Trading AI/gi, 'GOMNA')
                .replace(/Cocoa Trading/gi, 'GOMNA')
                .replace(/Gomna(?!\s*[A-Z])/gi, 'GOMNA');
        });

        // Update page title
        if (document.title.includes('Gomna') || document.title.includes('Cocoa')) {
            document.title = 'GOMNA - Wall Street Grade Quantitative Trading Platform';
        }

        // Update any logos or brand elements
        const brandElements = document.querySelectorAll('[class*="brand"], [class*="logo"], h1, .navbar-brand');
        brandElements.forEach(element => {
            if (element.textContent && (
                element.textContent.includes('Gomna') || 
                element.textContent.includes('Cocoa')
            )) {
                element.textContent = element.textContent
                    .replace(/Gomna AI/gi, 'GOMNA')
                    .replace(/Cocoa Trading AI/gi, 'GOMNA')
                    .replace(/Cocoa Trading/gi, 'GOMNA')
                    .replace(/Gomna/gi, 'GOMNA');
            }
        });
    }

    removeOldColors() {
        // Remove any remaining blue, purple, or green color classes
        const colorClasses = [
            'blue', 'purple', 'green', 'indigo', 'teal', 'cyan', 'pink', 'rose'
        ];

        colorClasses.forEach(color => {
            // Find all elements with these color classes
            const elements = document.querySelectorAll(`[class*="${color}-"]`);
            elements.forEach(element => {
                const classes = element.className.split(' ');
                const newClasses = classes.map(cls => {
                    // Replace color classes with brown equivalents
                    if (cls.includes(`${color}-600`)) return cls.replace(`${color}-600`, 'brown-600');
                    if (cls.includes(`${color}-700`)) return cls.replace(`${color}-700`, 'brown-700');
                    if (cls.includes(`${color}-800`)) return cls.replace(`${color}-800`, 'brown-800');
                    if (cls.includes(`bg-${color}`)) return 'bg-brown-600';
                    if (cls.includes(`text-${color}`)) return 'text-brown-600';
                    if (cls.includes(`border-${color}`)) return 'border-brown-400';
                    return cls;
                });
                element.className = newClasses.join(' ');
            });
        });

        // Update inline styles
        const elementsWithStyle = document.querySelectorAll('[style]');
        elementsWithStyle.forEach(element => {
            let style = element.getAttribute('style');
            if (style) {
                // Replace hex colors
                style = style.replace(/#4F46E5/gi, '#8B6F47'); // Indigo to Brown
                style = style.replace(/#10B981/gi, '#8B6F47'); // Green to Brown
                style = style.replace(/#8B5CF6/gi, '#6B4423'); // Purple to Dark Brown
                style = style.replace(/#3B82F6/gi, '#5D4037'); // Blue to Dark Brown
                
                // Replace rgb colors
                style = style.replace(/rgb\(79,\s*70,\s*229\)/gi, 'rgb(139, 111, 71)'); // Indigo
                style = style.replace(/rgb\(16,\s*185,\s*129\)/gi, 'rgb(139, 111, 71)'); // Green
                style = style.replace(/rgb\(139,\s*92,\s*246\)/gi, 'rgb(107, 68, 35)'); // Purple
                style = style.replace(/rgb\(59,\s*130,\s*246\)/gi, 'rgb(93, 64, 55)'); // Blue
                
                element.setAttribute('style', style);
            }
        });
    }

    // Method to update charts with cream and brown colors
    updateChartColors() {
        // This will be called by chart initialization code
        return {
            primary: '#8B6F47',
            secondary: '#D4A574',
            accent: '#D4AF37',
            background: '#FAF7F0',
            grid: '#E8DCC7',
            text: '#3E2723',
            datasets: [
                '#8B6F47', // Brown
                '#D4A574', // Light Brown
                '#D4AF37', // Gold
                '#CD7F32', // Bronze
                '#6B4423', // Dark Brown
                '#7B3F00'  // Cocoa
            ]
        };
    }
}

// Initialize theme when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.gomnaTheme = new GomnaThemeManager();
    });
} else {
    window.gomnaTheme = new GomnaThemeManager();
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GomnaThemeManager;
}