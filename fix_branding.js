// Fix GOMNA Branding - Ensures single, italic GOMNA with proper subtitle
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        // Remove all duplicate branding elements
        const duplicates = document.querySelectorAll('.gomna-branding, .logo-container');
        duplicates.forEach(el => el.remove());
        
        // Create single unified branding
        const unifiedBranding = document.createElement('div');
        unifiedBranding.id = 'gomna-main-branding';
        unifiedBranding.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 18px;
        `;
        
        // Logo
        unifiedBranding.innerHTML = `
            <div id="gomna-logo" style="
                background: #FAF7F0;
                border-radius: 12px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(107, 68, 35, 0.2);
                width: 60px;
                height: 60px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <svg width="44" height="44" viewBox="0 0 200 200">
                    <defs>
                        <linearGradient id="cocoaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#8B6F47" />
                            <stop offset="100%" style="stop-color:#5D4037" />
                        </linearGradient>
                    </defs>
                    <ellipse cx="100" cy="100" rx="50" ry="75" fill="url(#cocoaGrad)" transform="rotate(12 100 100)" />
                    <path d="M 100 40 L 100 160" stroke="#6B4423" stroke-width="3" opacity="0.5" />
                    <ellipse cx="85" cy="80" rx="14" ry="18" fill="#FAF7F0" opacity="0.95" />
                    <ellipse cx="115" cy="80" rx="14" ry="18" fill="#F5E6D3" opacity="0.95" />
                    <ellipse cx="100" cy="110" rx="15" ry="19" fill="#FAF7F0" opacity="0.95" />
                    <ellipse cx="85" cy="140" rx="14" ry="18" fill="#F5E6D3" opacity="0.95" />
                    <ellipse cx="115" cy="140" rx="14" ry="18" fill="#FAF7F0" opacity="0.95" />
                    <circle cx="85" cy="80" r="4" fill="#8B6F47" opacity="0.8" />
                    <circle cx="115" cy="80" r="4" fill="#8B6F47" opacity="0.8" />
                    <circle cx="100" cy="110" r="4" fill="#8B6F47" opacity="0.8" />
                    <circle cx="85" cy="140" r="4" fill="#8B6F47" opacity="0.8" />
                    <circle cx="115" cy="140" r="4" fill="#8B6F47" opacity="0.8" />
                </svg>
            </div>
            <div>
                <div style="
                    font-size: 36px;
                    font-weight: bold;
                    color: #3E2723;
                    letter-spacing: 5px;
                    font-family: Georgia, 'Times New Roman', serif;
                    font-style: italic;
                    text-shadow: 1px 1px 2px rgba(107, 68, 35, 0.2);
                    line-height: 1;
                ">GOMNA</div>
                <div style="
                    font-size: 13px;
                    color: #6B4423;
                    letter-spacing: 0.8px;
                    margin-top: 4px;
                    font-family: Georgia, 'Times New Roman', serif;
                    font-weight: 500;
                ">Wall Street Grade Quantitative Trading Platform</div>
            </div>
        `;
        
        document.body.appendChild(unifiedBranding);
        
        // Add hover effect
        const logo = document.getElementById('gomna-logo');
        logo.onmouseenter = () => {
            logo.style.transform = 'scale(1.1) rotate(5deg)';
        };
        logo.onmouseleave = () => {
            logo.style.transform = 'scale(1) rotate(0deg)';
        };
        logo.onclick = () => {
            window.open('cocoa_logos.html', '_blank');
        };
        
        // Hide duplicate headers
        document.querySelectorAll('h1').forEach(h1 => {
            if (h1.textContent.includes('GOMNA')) {
                h1.style.visibility = 'hidden';
                h1.style.height = '0';
            }
        });
        
        // Hide duplicate subtitles
        document.querySelectorAll('p').forEach(p => {
            if (p.textContent.includes('Wall Street Grade Quantitative Trading Platform')) {
                const parent = p.parentElement;
                if (parent && parent.querySelector('h1')) {
                    p.style.visibility = 'hidden';
                    p.style.height = '0';
                }
            }
        });
    }, 200);
});