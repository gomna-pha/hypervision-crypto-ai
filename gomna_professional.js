// GOMNA Professional Platform - Original, Sophisticated Design
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        // Clear all existing branding
        document.querySelectorAll('.logo-container, .gomna-branding, #gomna-main-branding, #gomna-unified-branding').forEach(el => el.remove());
        
        // Hide duplicate headers
        document.querySelectorAll('h1').forEach(h1 => {
            if (h1.textContent.includes('GOMNA')) {
                h1.parentElement.style.display = 'none';
            }
        });
        
        // Create professional header bar
        const headerBar = document.createElement('div');
        headerBar.id = 'gomna-professional-header';
        headerBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 80px;
            background: linear-gradient(135deg, #3E2723 0%, #5D4037 50%, #3E2723 100%);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            display: flex;
            align-items: center;
            padding: 0 30px;
            border-bottom: 3px solid #D4AF37;
        `;
        
        // Professional logo and branding layout
        headerBar.innerHTML = `
            <div style="display: flex; align-items: center; gap: 25px; flex: 1;">
                <!-- Premium Cocoa Logo -->
                <div id="gomna-premium-logo" style="
                    background: linear-gradient(135deg, #D4AF37, #CD7F32);
                    border-radius: 50%;
                    padding: 12px;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.4);
                    cursor: pointer;
                    transition: all 0.3s ease;
                    position: relative;
                ">
                    <div style="
                        background: #FAF7F0;
                        border-radius: 50%;
                        padding: 8px;
                        width: 48px;
                        height: 48px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <svg width="32" height="32" viewBox="0 0 200 200">
                            <defs>
                                <linearGradient id="premiumGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#8B6F47" />
                                    <stop offset="50%" style="stop-color:#D4AF37" />
                                    <stop offset="100%" style="stop-color:#8B6F47" />
                                </linearGradient>
                                <filter id="goldGlow">
                                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                                    <feMerge>
                                        <feMergeNode in="coloredBlur"/>
                                        <feMergeNode in="SourceGraphic"/>
                                    </feMerge>
                                </filter>
                            </defs>
                            <ellipse cx="100" cy="100" rx="60" ry="80" fill="url(#premiumGrad)" filter="url(#goldGlow)" />
                            <path d="M 100 30 L 100 170" stroke="#5D4037" stroke-width="4" opacity="0.6" />
                            <!-- Premium cocoa seeds -->
                            <ellipse cx="80" cy="70" rx="16" ry="20" fill="#FAF7F0" opacity="0.95" />
                            <ellipse cx="120" cy="70" rx="16" ry="20" fill="#F5E6D3" opacity="0.95" />
                            <ellipse cx="100" cy="100" rx="18" ry="22" fill="#FAF7F0" opacity="0.95" />
                            <ellipse cx="80" cy="130" rx="16" ry="20" fill="#F5E6D3" opacity="0.95" />
                            <ellipse cx="120" cy="130" rx="16" ry="20" fill="#FAF7F0" opacity="0.95" />
                            <!-- Seed details -->
                            <circle cx="80" cy="70" r="5" fill="#8B6F47" />
                            <circle cx="120" cy="70" r="5" fill="#8B6F47" />
                            <circle cx="100" cy="100" r="5" fill="#8B6F47" />
                            <circle cx="80" cy="130" r="5" fill="#8B6F47" />
                            <circle cx="120" cy="130" r="5" fill="#8B6F47" />
                        </svg>
                    </div>
                </div>
                
                <!-- Professional Branding -->
                <div style="border-left: 3px solid #D4AF37; padding-left: 25px; height: 50px; display: flex; align-items: center;">
                    <div>
                        <div style="
                            font-size: 42px;
                            font-weight: 300;
                            color: #FAF7F0;
                            letter-spacing: 8px;
                            font-family: 'Didot', 'Bodoni MT', 'Georgia', serif;
                            font-style: italic;
                            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3), 0 0 20px rgba(212, 175, 55, 0.3);
                            margin-bottom: -5px;
                        ">GOMNA</div>
                        <div style="
                            font-size: 11px;
                            color: #D4AF37;
                            letter-spacing: 3px;
                            font-family: 'Futura', 'Helvetica Neue', sans-serif;
                            font-weight: 300;
                            text-transform: uppercase;
                        ">Quantitative Trading Platform</div>
                    </div>
                </div>
                
                <!-- Status Indicators -->
                <div style="margin-left: auto; display: flex; gap: 20px; align-items: center;">
                    <div style="
                        background: rgba(250, 247, 240, 0.1);
                        border: 1px solid #D4AF37;
                        border-radius: 20px;
                        padding: 8px 16px;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    ">
                        <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                        <span style="color: #FAF7F0; font-size: 12px; font-weight: 500;">LIVE TRADING</span>
                    </div>
                    
                    <div style="
                        background: rgba(250, 247, 240, 0.1);
                        border: 1px solid #D4AF37;
                        border-radius: 20px;
                        padding: 8px 16px;
                    ">
                        <span style="color: #D4AF37; font-size: 12px; font-weight: 500;">v2.0 PREMIUM</span>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(headerBar);
        
        // Add premium hover effect to logo
        const logo = document.getElementById('gomna-premium-logo');
        logo.onmouseenter = () => {
            logo.style.transform = 'scale(1.1) rotate(360deg)';
            logo.style.boxShadow = '0 0 30px rgba(212, 175, 55, 0.6)';
        };
        logo.onmouseleave = () => {
            logo.style.transform = 'scale(1) rotate(0deg)';
            logo.style.boxShadow = '0 0 20px rgba(212, 175, 55, 0.4)';
        };
        logo.onclick = () => {
            window.open('cocoa_logos.html', '_blank');
        };
        
        // Adjust main content to account for header
        document.body.style.paddingTop = '100px';
        
        // Style all containers to be more professional
        const containers = document.querySelectorAll('.glass-effect, .container, main');
        containers.forEach(container => {
            if (container) {
                container.style.marginTop = '100px';
            }
        });
        
        // Create professional info bar
        const infoBar = document.createElement('div');
        infoBar.style.cssText = `
            position: fixed;
            top: 80px;
            left: 0;
            right: 0;
            height: 30px;
            background: linear-gradient(90deg, #FAF7F0 0%, #F5E6D3 50%, #FAF7F0 100%);
            border-bottom: 1px solid #D4C4A8;
            display: flex;
            align-items: center;
            padding: 0 30px;
            z-index: 9999;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        `;
        
        infoBar.innerHTML = `
            <div style="display: flex; gap: 30px; align-items: center; width: 100%; font-size: 11px; color: #5D4037;">
                <span><strong>Market Status:</strong> <span style="color: #10B981;">● Open</span></span>
                <span><strong>Server Time:</strong> <span id="server-time">${new Date().toLocaleTimeString()}</span></span>
                <span><strong>Active Users:</strong> <span style="color: #8B6F47; font-weight: bold;">2,847</span></span>
                <span style="margin-left: auto;"><strong>Next Update:</strong> <span id="next-update">in 2s</span></span>
            </div>
        `;
        
        document.body.appendChild(infoBar);
        
        // Update time
        setInterval(() => {
            const timeEl = document.getElementById('server-time');
            if (timeEl) timeEl.textContent = new Date().toLocaleTimeString();
        }, 1000);
        
        // Add professional animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            #gomna-professional-header {
                background-size: 200% 100%;
                animation: shimmer 10s infinite linear;
            }
            
            body {
                padding-top: 120px !important;
            }
            
            .glass-effect:first-of-type,
            .container:first-of-type {
                margin-top: 130px !important;
            }
            
            /* Professional scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #FAF7F0;
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #8B6F47, #D4AF37);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #6B4423, #CD7F32);
            }
        `;
        document.head.appendChild(style);
        
    }, 100);
});

// Professional notification system
function showProfessionalNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 140px;
        right: 30px;
        background: ${type === 'success' ? 'linear-gradient(135deg, #10B981, #059669)' : 'linear-gradient(135deg, #EF4444, #DC2626)'};
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        z-index: 11000;
        animation: slideIn 0.5s ease;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 14px;
        font-weight: 500;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.5s ease';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
}

// Add slide animations
const animStyle = document.createElement('style');
animStyle.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(animStyle);

// Show welcome message
setTimeout(() => {
    showProfessionalNotification('Welcome to GOMNA Premium Trading Platform', 'success');
}, 1000);