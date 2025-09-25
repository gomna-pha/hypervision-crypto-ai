/**
 * GOMNA AI Trading - Preserved Platform with Clean Branding
 * Applies clean GOMNA branding while preserving all existing functionality
 */

// Initialize GOMNA branding without destroying existing content
function initializeGOMNABranding() {
    console.log('Applying GOMNA branding to preserved platform...');
    
    // Wait for DOM to be ready
    if (document.readyState !== 'complete') {
        setTimeout(initializeGOMNABranding, 100);
        return;
    }

    try {
        // Apply GOMNA branding updates without destroying content
        updateHeaderBranding();
        updateColorScheme();
        removeEmojis();
        addRegistrationModal();
        initializeEnhancedFeatures();
        
        console.log('GOMNA branding applied successfully!');
        
    } catch (error) {
        console.log('Error applying branding:', error);
    }
}

function updateHeaderBranding() {
    // Update main title to "AGENTIC AI TRADING"
    const mainTitle = document.querySelector('h1');
    if (mainTitle && mainTitle.textContent.includes('Gomna AI Trading')) {
        mainTitle.innerHTML = 'AGENTIC AI TRADING';
    }
    
    // Update subtitle to match user request
    const subtitle = document.querySelector('p.text-sm.text-gray-600');
    if (subtitle && subtitle.textContent.includes('Institutional Quantitative Trading')) {
        subtitle.innerHTML = '<em>Professional High-Frequency Trading & Arbitrage Platform</em>';
    }
    
    // Update document title
    document.title = 'GOMNA AI Trading - Professional High-Frequency Trading & Arbitrage Platform';
    
    // Update any "HyperVision AI" references to "GOMNA AI"
    const allElements = document.querySelectorAll('*');
    allElements.forEach(element => {
        if (element.textContent && element.textContent.includes('HyperVision AI')) {
            element.innerHTML = element.innerHTML.replace(/HyperVision AI/g, 'GOMNA AI');
        }
    });
}

function updateColorScheme() {
    // Apply 90% cream color scheme
    const style = document.createElement('style');
    style.id = 'gomna-clean-colors';
    style.textContent = `
        /* 90% Cream Color Scheme - Only 1.5% brown accent */
        .gradient-bg {
            background: linear-gradient(135deg, #fefbf3 0%, #fdf6e3 50%, #f5e6d3 100%) !important;
        }
        
        .glass-effect {
            background: rgba(254, 251, 243, 0.98) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(245, 230, 211, 0.4) !important;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #fefbf3 0%, #fdf6e3 100%) !important;
            border: 1px solid rgba(245, 230, 211, 0.3) !important;
        }
        
        .bg-white {
            background-color: #fefbf3 !important;
        }
        
        .bg-gradient-to-r.from-cream-50.to-cream-100,
        .bg-gradient-to-r.from-cream-100.to-cream-200 {
            background: linear-gradient(to right, #fefbf3, #fdf6e3) !important;
        }
        
        /* Brown accents - only 1.5% usage */
        .text-amber-800, .text-amber-700, .text-cream-800 {
            color: #8b7355 !important;
        }
        
        .bg-amber-600, .bg-amber-700 {
            background-color: #8b7355 !important;
        }
        
        .border-amber-200, .border-cream-200 {
            border-color: rgba(139, 115, 85, 0.2) !important;
        }
        
        /* Ensure all content areas use cream colors */
        .tab-content, #dashboard-tab, #performance-tab, #analytics-tab, 
        #portfolio-tab, #payments-tab, #agentic-tab, #transparency-tab {
            background: #fefbf3 !important;
        }
        
        /* Professional button styling */
        .btn-primary, button.bg-gradient-to-r {
            background: linear-gradient(135deg, #8b7355 0%, #6d5d48 100%) !important;
            color: #fefbf3 !important;
        }
        
        .btn-primary:hover, button.bg-gradient-to-r:hover {
            background: linear-gradient(135deg, #6d5d48 0%, #5a4d3b 100%) !important;
        }
    `;
    document.head.appendChild(style);
}

function removeEmojis() {
    // Remove all emojis from the platform
    const allElements = document.querySelectorAll('*');
    allElements.forEach(element => {
        if (element.textContent) {
            // Remove common emojis while preserving the rest of the text
            const emojiPattern = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|📡|📈|🐦|⚡|✅|🚀|🧠|⚡|📊|💰|💼|⭐|🔒|🎯|📱|💻|🌐/gu;
            
            if (element.children.length === 0) { // Only update text nodes
                element.textContent = element.textContent.replace(emojiPattern, '').trim();
            }
        }
    });
}

function addRegistrationModal() {
    // Check if modal already exists
    if (document.getElementById('registration-modal')) return;
    
    // Create registration modal HTML
    const modalHTML = `
    <div id="registration-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center p-4">
        <div class="bg-gradient-to-br from-cream-50 to-cream-100 rounded-2xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div class="p-8">
                <div class="text-center mb-6">
                    <h2 class="text-2xl font-bold text-gray-900 mb-2">Join GOMNA AI Trading</h2>
                    <p class="text-sm text-gray-600">Professional Trading Platform Registration</p>
                </div>
                
                <form id="registration-form" class="space-y-4">
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Account Type</label>
                        <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500" name="accountType" required>
                            <option value="">Select Account Type</option>
                            <option value="starter">Starter - $10K minimum</option>
                            <option value="professional">Professional - $100K minimum</option>
                            <option value="institutional">Institutional - $1M minimum</option>
                        </select>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">First Name</label>
                            <input type="text" name="firstName" required class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500">
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Last Name</label>
                            <input type="text" name="lastName" required class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500">
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Email Address</label>
                        <input type="email" name="email" required class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Investment Capital</label>
                        <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500" name="capital" required>
                            <option value="">Select Investment Range</option>
                            <option value="10k-50k">$10K - $50K</option>
                            <option value="50k-100k">$50K - $100K</option>
                            <option value="100k-500k">$100K - $500K</option>
                            <option value="500k-1m">$500K - $1M</option>
                            <option value="1m+">$1M+</option>
                        </select>
                    </div>
                    
                    <div class="flex items-center">
                        <input type="checkbox" id="terms" name="terms" required class="h-4 w-4 text-amber-600 focus:ring-amber-500 border-gray-300 rounded">
                        <label for="terms" class="ml-2 block text-sm text-gray-700">
                            I agree to the <a href="#" class="text-amber-600 hover:text-amber-700">Terms of Service</a> and <a href="#" class="text-amber-600 hover:text-amber-700">Privacy Policy</a>
                        </label>
                    </div>
                    
                    <button type="submit" class="w-full bg-gradient-to-r from-amber-600 to-amber-700 text-cream-50 py-3 px-6 rounded-lg font-semibold hover:from-amber-700 hover:to-amber-800 transition-all duration-200 shadow-lg">
                        Create Account
                    </button>
                </form>
                
                <button id="close-modal" class="absolute top-4 right-4 text-gray-500 hover:text-gray-700">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        </div>
    </div>`;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Add registration button to header if it doesn't exist
    const headerActions = document.querySelector('.flex.items-center.gap-4:last-child');
    if (headerActions && !document.getElementById('register-btn')) {
        const registerBtn = document.createElement('button');
        registerBtn.id = 'register-btn';
        registerBtn.className = 'px-6 py-2 bg-gradient-to-r from-green-600 to-green-700 text-cream-50 rounded-lg font-semibold hover:from-green-700 hover:to-green-800 transition-all shadow-lg';
        registerBtn.innerHTML = 'Register Account';
        registerBtn.onclick = openRegistration;
        headerActions.appendChild(registerBtn);
    }
}

function openRegistration() {
    document.getElementById('registration-modal').classList.remove('hidden');
}

function closeRegistration() {
    document.getElementById('registration-modal').classList.add('hidden');
}

function initializeEnhancedFeatures() {
    // Add event listeners for modal
    const modal = document.getElementById('registration-modal');
    const closeBtn = document.getElementById('close-modal');
    const form = document.getElementById('registration-form');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeRegistration);
    }
    
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) closeRegistration();
        });
    }
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            handleRegistration(new FormData(form));
        });
    }
    
    // Add trading execution functionality
    addTradeExecution();
}

function handleRegistration(formData) {
    const data = Object.fromEntries(formData);
    console.log('Registration data:', data);
    
    // Show success notification
    showNotification('Registration successful! Account pending approval.', 'success');
    closeRegistration();
}

function addTradeExecution() {
    // Add trade execution buttons throughout the platform
    const tradingButtons = document.querySelectorAll('[data-action="trade"]');
    
    // If no existing trading buttons, add some to key sections
    const sections = document.querySelectorAll('.bg-white.p-6.rounded-xl.shadow-lg');
    sections.forEach((section, index) => {
        if (index % 3 === 0) { // Add to every 3rd section
            const executeBtn = document.createElement('button');
            executeBtn.className = 'mt-4 px-4 py-2 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg text-sm hover:from-green-700 hover:to-green-800 transition-all';
            executeBtn.textContent = 'Execute Trade';
            executeBtn.onclick = () => executeTrade();
            section.appendChild(executeBtn);
        }
    });
}

function executeTrade() {
    // Simulate trade execution
    const assets = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AAPL', 'TSLA', 'NVDA'];
    const actions = ['BUY', 'SELL'];
    const asset = assets[Math.floor(Math.random() * assets.length)];
    const action = actions[Math.floor(Math.random() * actions.length)];
    const amount = (Math.random() * 10000 + 1000).toFixed(0);
    
    showNotification(`${action} ${asset} - $${amount} executed successfully!`, 'success');
    
    // Add to live trading feed if it exists
    const tradesTable = document.getElementById('trades-table');
    if (tradesTable) {
        const now = new Date().toLocaleTimeString();
        const pnl = (Math.random() * 2000 - 1000).toFixed(0);
        const pnlClass = pnl > 0 ? 'text-green-600' : 'text-red-600';
        const pnlSign = pnl > 0 ? '+' : '';
        
        const row = `
            <tr class="border-b hover:bg-cream-50">
                <td class="py-2 text-sm">${now}</td>
                <td class="py-2 text-sm font-medium">${asset}</td>
                <td class="py-2 text-sm"><span class="px-2 py-1 rounded-full text-xs ${action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">${action}</span></td>
                <td class="py-2 text-sm text-right font-mono">$${(Math.random() * 100000 + 10000).toFixed(2)}</td>
                <td class="py-2 text-sm text-right">$${amount}</td>
                <td class="py-2 text-sm text-right ${pnlClass}">${pnlSign}$${Math.abs(pnl)}</td>
                <td class="py-2 text-sm">GOMNA AI</td>
            </tr>
        `;
        tradesTable.insertAdjacentHTML('afterbegin', row);
        
        // Keep only last 10 trades
        const rows = tradesTable.querySelectorAll('tr');
        if (rows.length > 10) {
            rows[rows.length - 1].remove();
        }
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-2xl border-l-4 transition-all duration-300 transform translate-x-full ${
        type === 'success' ? 'bg-green-50 border-green-500 text-green-800' :
        type === 'error' ? 'bg-red-50 border-red-500 text-red-800' :
        'bg-blue-50 border-blue-500 text-blue-800'
    }`;
    
    notification.innerHTML = `
        <div class="flex items-center">
            <div class="flex-1">
                <p class="font-medium">${message}</p>
            </div>
            <button class="ml-4 text-gray-500 hover:text-gray-700" onclick="this.parentElement.parentElement.remove()">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeGOMNABranding);
} else {
    initializeGOMNABranding();
}

// Export for manual initialization if needed
window.initializeGOMNABranding = initializeGOMNABranding;
window.openRegistration = openRegistration;
window.executeTrade = executeTrade;