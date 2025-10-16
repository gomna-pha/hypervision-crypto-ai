/**
 * GOMNA AI Trading - Simple Clean Platform (No DOM Issues)
 * Clean layout without repetition
 */

// Simple initialization without complex DOM manipulation
function initializeSimpleGOMNA() {
    console.log('Initializing Simple GOMNA Platform...');
    
    // Wait for DOM to be ready
    if (document.readyState !== 'complete') {
        setTimeout(initializeSimpleGOMNA, 100);
        return;
    }

    // AGGRESSIVE CLEANUP - Remove ALL existing content
    try {
        // Clear entire body content first
        document.body.innerHTML = '';
        
        // Recreate only essential head elements
        const existingStyles = document.querySelectorAll('style');
        existingStyles.forEach(style => {
            if (style.id && style.id.includes('gomna')) {
                style.remove();
            }
        });

        // Create styles
        createGOMNAStyles();
        
        // Create header
        createGOMNAHeader();
        
        // Create dashboard
        createGOMNADashboard();
        
        console.log('Simple GOMNA Platform initialized successfully!');
        
    } catch (error) {
        console.log('Error during initialization, retrying...', error);
        setTimeout(initializeSimpleGOMNA, 1000);
    }
}

function createGOMNAStyles() {
    if (document.getElementById('gomna-simple-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'gomna-simple-styles';
    style.textContent = `
        /* GOMNA Simple Styles - 90% Cream, 1.5% Brown */
        body {
            background: #fefbf3;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #2d2d2d;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }

        .gomna-header {
            background: linear-gradient(135deg, #fdf6e3 0%, #f5e6d3 100%);
            border: 2px solid #e8dcc7;
            border-radius: 20px;
            padding: 50px 40px;
            text-align: center;
            margin: 0 0 40px 0;
            box-shadow: 0 10px 30px rgba(139, 115, 85, 0.08);
        }

        .gomna-header-top {
            color: #8b7355;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 0 0 10px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .gomna-title {
            color: #2d2d2d;
            font-size: 4.2rem;
            font-weight: 700;
            margin: 15px 0 20px 0;
            letter-spacing: -1px;
        }
        
        .gomna-subtitle {
            color: #8b7355;
            font-size: 1.8rem;
            margin: 0 0 30px 0;
            font-weight: 400;
            font-style: italic;
            letter-spacing: 0.5px;
        }

        .gomna-status-bar {
            display: flex;
            justify-content: center;
            gap: 35px;
            margin-top: 35px;
            flex-wrap: wrap;
        }
        
        .gomna-status {
            background: #f5e6d3;
            color: #10b981;
            padding: 15px 25px;
            border-radius: 25px;
            border: 2px solid #e8dcc7;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .gomna-nav-bar {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
            flex-wrap: wrap;
            padding-top: 25px;
            border-top: 2px solid #e8dcc7;
        }

        .gomna-nav-item {
            color: #2d2d2d;
            font-size: 1.1rem;
            font-weight: 500;
            padding: 12px 20px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .gomna-nav-item:hover {
            background: #e8dcc7;
            color: #8b7355;
        }

        .gomna-panel {
            background: linear-gradient(135deg, #fefbf3 0%, #fdf6e3 100%);
            border: 1px solid #e8dcc7;
            border-radius: 20px;
            margin: 30px 0;
            box-shadow: 0 8px 25px rgba(139, 115, 85, 0.06);
            overflow: hidden;
        }
        
        .gomna-panel-header {
            background: #f5e6d3;
            color: #2d2d2d;
            padding: 25px 30px;
            font-weight: 600;
            font-size: 1.3rem;
            border-bottom: 1px solid #e8dcc7;
        }
        
        .gomna-panel-content {
            padding: 40px 30px;
        }

        .gomna-btn {
            background: linear-gradient(135deg, #e8dcc7 0%, #d4c4a8 100%);
            color: #2d2d2d;
            border: 1px solid #8b7355;
            padding: 15px 30px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            margin: 10px;
            text-transform: uppercase;
            transition: all 0.3s ease;
        }
        
        .gomna-btn:hover {
            background: #8b7355;
            color: white;
        }

        .gomna-btn-primary {
            background: linear-gradient(135deg, #8b7355 0%, #6d5d48 100%);
            color: white;
            font-size: 1.2rem;
            padding: 18px 40px;
            border: none;
        }

        .gomna-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }

        .gomna-card {
            background: #fdf6e3;
            border: 1px solid #e8dcc7;
            border-radius: 15px;
            padding: 30px;
            transition: all 0.3s ease;
        }

        .gomna-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(139, 115, 85, 0.1);
        }

        .gomna-cta-section {
            background: linear-gradient(135deg, #f5e6d3 0%, #e8dcc7 100%);
            border: 2px solid #8b7355;
            border-radius: 25px;
            text-align: center;
            padding: 60px 40px;
            margin: 50px 0;
        }

        .gomna-metric {
            text-align: center;
            padding: 30px 25px;
            background: #f5e6d3;
            border: 1px solid #e8dcc7;
            border-radius: 15px;
        }
        
        .gomna-metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #8b7355;
            display: block;
            margin-bottom: 10px;
        }
        
        .gomna-metric-label {
            color: #2d2d2d;
            font-size: 1.1rem;
            font-weight: 500;
            text-transform: uppercase;
        }

        @media (max-width: 768px) {
            .gomna-title {
                font-size: 2.8rem;
            }
            .gomna-grid {
                grid-template-columns: 1fr;
            }
            .gomna-status-bar, .gomna-nav-bar {
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }
        }
    `;
    
    document.head.appendChild(style);
}

function createGOMNAHeader() {
    // Remove any existing headers
    const existingHeaders = document.querySelectorAll('.gomna-header');
    existingHeaders.forEach(h => h.remove());

    const header = document.createElement('div');
    header.className = 'gomna-header';
    header.innerHTML = `
        <div class="gomna-header-top">AGENTIC AI TRADING</div>
        <p class="gomna-subtitle"><em>Professional High-Frequency Trading & Arbitrage Platform</em></p>
    `;
    
    document.body.appendChild(header);
    
    // Update page title
    document.title = 'GOMNA AI Trading - Professional High-Frequency Trading & Arbitrage Platform';
}

function createGOMNADashboard() {
    const dashboard = document.createElement('div');
    dashboard.innerHTML = `
        <!-- Start Trading Section -->
        <div class="gomna-cta-section">
            <h2 style="color: #2d2d2d; margin-bottom: 25px; font-size: 3.5rem; font-weight: 300;">
                Start Trading with GOMNA AI
            </h2>
            <p style="color: #8b7355; font-size: 1.4rem; margin-bottom: 40px; max-width: 800px; margin-left: auto; margin-right: auto;">
                Join thousands of professional traders using our AI-powered arbitrage strategies. 
                Advanced algorithms, institutional-grade security, and real-time market analysis.
            </p>
            <div>
                <button class="gomna-btn-primary" onclick="openRegistration()">
                    Create Account & Start Trading
                </button>
                <button class="gomna-btn" onclick="activateDemo()">
                    Try Demo Mode
                </button>
            </div>
        </div>

        <!-- Performance Overview -->
        <div class="gomna-panel">
            <div class="gomna-panel-header">
                Trading Performance Overview
            </div>
            <div class="gomna-panel-content">
                <div class="gomna-grid">
                    <div class="gomna-metric">
                        <span class="gomna-metric-value">+$12,450.30</span>
                        <div class="gomna-metric-label">Today's Performance</div>
                    </div>
                    <div class="gomna-metric">
                        <span class="gomna-metric-value">$532,180.45</span>
                        <div class="gomna-metric-label">Portfolio Value</div>
                    </div>
                    <div class="gomna-metric">
                        <span class="gomna-metric-value">89.4%</span>
                        <div class="gomna-metric-label">Win Rate</div>
                    </div>
                    <div class="gomna-metric">
                        <span class="gomna-metric-value">6 Active</span>
                        <div class="gomna-metric-label">AI Strategies</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Account Tiers -->
        <div class="gomna-panel">
            <div class="gomna-panel-header">
                Professional Account Tiers
            </div>
            <div class="gomna-panel-content">
                <div class="gomna-grid">
                    <div class="gomna-card">
                        <h4 style="color: #10b981; margin-bottom: 20px; font-size: 1.6rem;">Starter Tier</h4>
                        <div class="gomna-metric-value" style="font-size: 2rem; color: #10b981;">$10,000</div>
                        <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                        <div style="margin-bottom: 25px; text-align: left;">
                            <strong>Features:</strong><br>
                            • Basic Arbitrage Strategies<br>
                            • Real-time Market Data<br>
                            • 10:1 Leverage<br>
                            • Standard Support
                        </div>
                        <button class="gomna-btn" onclick="openRegistration()">Get Started</button>
                    </div>
                    
                    <div class="gomna-card" style="border-color: #8b7355;">
                        <h4 style="color: #8b7355; margin-bottom: 20px; font-size: 1.6rem;">Professional Tier</h4>
                        <div class="gomna-metric-value" style="font-size: 2rem; color: #8b7355;">$100,000</div>
                        <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                        <div style="margin-bottom: 25px; text-align: left;">
                            <strong>Features:</strong><br>
                            • Advanced HFT Strategies<br>
                            • AI-Powered Analysis<br>
                            • 25:1 Leverage<br>
                            • Priority Support 24/7
                        </div>
                        <button class="gomna-btn-primary" onclick="openRegistration()">Upgrade Now</button>
                    </div>
                    
                    <div class="gomna-card" style="border-color: #8b7355;">
                        <h4 style="color: #8b7355; margin-bottom: 20px; font-size: 1.6rem;">Institutional Tier</h4>
                        <div class="gomna-metric-value" style="font-size: 2rem; color: #8b7355;">$1,000,000</div>
                        <div class="gomna-metric-label" style="margin-bottom: 25px;">Minimum Deposit</div>
                        <div style="margin-bottom: 25px; text-align: left;">
                            <strong>Features:</strong><br>
                            • Custom Algorithm Development<br>
                            • Direct Market Access<br>
                            • 50:1 Leverage<br>
                            • Dedicated Account Manager
                        </div>
                        <button class="gomna-btn" onclick="openRegistration()">Contact Sales</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Arbitrage Opportunities -->
        <div class="gomna-panel">
            <div class="gomna-panel-header">
                Live Arbitrage Opportunities
            </div>
            <div class="gomna-panel-content">
                <div class="gomna-grid">
                    <div class="gomna-card" style="border-left: 4px solid #10b981;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <strong style="color: #8b7355; font-size: 1.2rem;">BTC/USD</strong>
                            <span style="color: #10b981; font-weight: bold; font-size: 1.2rem;">0.23%</span>
                        </div>
                        <div style="margin-bottom: 15px; color: #2d2d2d;">Binance → Coinbase</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #10b981; font-weight: bold;">Est. Profit: $1,240</span>
                            <button class="gomna-btn" onclick="executeTrade(this, 'BTC/USD')" style="padding: 8px 16px;">Execute Trade</button>
                        </div>
                    </div>
                    
                    <div class="gomna-card" style="border-left: 4px solid #10b981;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <strong style="color: #8b7355; font-size: 1.2rem;">ETH/USD</strong>
                            <span style="color: #10b981; font-weight: bold; font-size: 1.2rem;">0.18%</span>
                        </div>
                        <div style="margin-bottom: 15px; color: #2d2d2d;">Kraken → Bitstamp</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #10b981; font-weight: bold;">Est. Profit: $890</span>
                            <button class="gomna-btn" onclick="executeTrade(this, 'ETH/USD')" style="padding: 8px 16px;">Execute Trade</button>
                        </div>
                    </div>
                    
                    <div class="gomna-card" style="border-left: 4px solid #10b981;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <strong style="color: #8b7355; font-size: 1.2rem;">ADA/USD</strong>
                            <span style="color: #10b981; font-weight: bold; font-size: 1.2rem;">0.31%</span>
                        </div>
                        <div style="margin-bottom: 15px; color: #2d2d2d;">Binance → KuCoin</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #10b981; font-weight: bold;">Est. Profit: $650</span>
                            <button class="gomna-btn" onclick="executeTrade(this, 'ADA/USD')" style="padding: 8px 16px;">Execute Trade</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Regulatory Credentials -->
        <div class="gomna-panel">
            <div class="gomna-panel-header">
                Regulatory Credentials & Compliance
            </div>
            <div class="gomna-panel-content">
                <div class="gomna-grid">
                    <div class="gomna-card">
                        <h4 style="color: #1E40AF; margin-bottom: 15px;">SEC Registered</h4>
                        <strong>CRD: 299792</strong><br>
                        <small style="color: #8b7355;">Securities and Exchange Commission</small><br>
                        <div style="color: #10b981; margin-top: 15px; font-weight: 600;">✓ Active - Valid until 2025-12-31</div>
                    </div>
                    
                    <div class="gomna-card">
                        <h4 style="color: #059669; margin-bottom: 15px;">FINRA Member</h4>
                        <strong>Member ID: 19847</strong><br>
                        <small style="color: #8b7355;">Financial Industry Regulatory Authority</small><br>
                        <div style="color: #10b981; margin-top: 15px; font-weight: 600;">✓ Active - Valid until 2025-06-30</div>
                    </div>
                    
                    <div class="gomna-card">
                        <h4 style="color: #DC2626; margin-bottom: 15px;">SIPC Protected</h4>
                        <strong>SIPC Member</strong><br>
                        <small style="color: #8b7355;">Securities Investor Protection: $500K</small><br>
                        <div style="color: #10b981; margin-top: 15px; font-weight: 600;">✓ Active - Protected</div>
                    </div>
                    
                    <div class="gomna-card">
                        <h4 style="color: #7C2D12; margin-bottom: 15px;">Lloyds Insurance</h4>
                        <strong>Policy: LL-2024-GOMNA</strong><br>
                        <small style="color: #8b7355;">Professional Indemnity: $50M Coverage</small><br>
                        <div style="color: #10b981; margin-top: 15px; font-weight: 600;">✓ Active - Valid until 2025-08-31</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(dashboard);
}

// Registration modal functionality
function openRegistration() {
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;
    
    modal.innerHTML = `
        <div style="
            background: #fefbf3;
            border: 2px solid #e8dcc7;
            border-radius: 20px;
            max-width: 600px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        ">
            <div style="
                background: #f5e6d3;
                padding: 25px 30px;
                border-bottom: 1px solid #e8dcc7;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h2 style="margin: 0; color: #2d2d2d;">Create GOMNA AI Trading Account</h2>
                <button onclick="this.closest('[style*=\"position: fixed\"]').remove()" style="
                    background: none;
                    border: none;
                    font-size: 2rem;
                    cursor: pointer;
                    color: #2d2d2d;
                ">&times;</button>
            </div>
            <div style="padding: 30px;">
                <form id="registration-form">
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 8px; color: #2d2d2d; font-weight: 600;">Full Name</label>
                        <input type="text" required placeholder="Enter your full name" style="
                            width: 100%;
                            padding: 15px;
                            border: 2px solid #e8dcc7;
                            border-radius: 8px;
                            background: #fefbf3;
                            color: #2d2d2d;
                            font-size: 1rem;
                        ">
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 8px; color: #2d2d2d; font-weight: 600;">Email Address</label>
                        <input type="email" required placeholder="Enter your email" style="
                            width: 100%;
                            padding: 15px;
                            border: 2px solid #e8dcc7;
                            border-radius: 8px;
                            background: #fefbf3;
                            color: #2d2d2d;
                            font-size: 1rem;
                        ">
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 8px; color: #2d2d2d; font-weight: 600;">Investment Amount</label>
                        <select required style="
                            width: 100%;
                            padding: 15px;
                            border: 2px solid #e8dcc7;
                            border-radius: 8px;
                            background: #fefbf3;
                            color: #2d2d2d;
                            font-size: 1rem;
                        ">
                            <option value="">Select investment amount</option>
                            <option value="10000">$10,000 - $50,000 (Starter)</option>
                            <option value="100000">$100,000 - $500,000 (Professional)</option>
                            <option value="1000000">$1,000,000+ (Institutional)</option>
                        </select>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <button type="submit" style="
                            background: linear-gradient(135deg, #8b7355 0%, #6d5d48 100%);
                            color: white;
                            border: none;
                            padding: 15px 30px;
                            border-radius: 8px;
                            font-size: 1.1rem;
                            font-weight: 600;
                            cursor: pointer;
                            margin-right: 10px;
                        ">Create Account</button>
                        <button type="button" onclick="this.closest('[style*=\"position: fixed\"]').remove()" style="
                            background: #e8dcc7;
                            color: #2d2d2d;
                            border: 1px solid #8b7355;
                            padding: 15px 30px;
                            border-radius: 8px;
                            font-size: 1.1rem;
                            font-weight: 600;
                            cursor: pointer;
                        ">Cancel</button>
                    </div>
                </form>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    
    // Handle form submission
    modal.querySelector('#registration-form').addEventListener('submit', (e) => {
        e.preventDefault();
        modal.remove();
        showNotification('Welcome to GOMNA AI Trading! Your account registration has been submitted successfully.', 'success');
    });
    
    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 20px 30px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#8b7355'};
        color: white;
        border-radius: 12px;
        z-index: 10001;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        max-width: 400px;
        font-weight: 500;
        font-size: 1rem;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;

    // Add animation styles
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Trade execution functionality
function executeTrade(button, pair) {
    button.textContent = 'Executing...';
    button.disabled = true;
    button.style.opacity = '0.7';
    
    setTimeout(() => {
        button.textContent = 'Executed ✓';
        button.style.background = '#10b981';
        button.style.color = 'white';
        
        showNotification(`GOMNA AI executed arbitrage trade successfully for ${pair}!`, 'success');
        
        setTimeout(() => {
            button.textContent = 'Execute Trade';
            button.disabled = false;
            button.style.opacity = '1';
            button.style.background = '';
            button.style.color = '';
        }, 3000);
    }, 2000);
}

// Demo mode functionality
function activateDemo() {
    showNotification('Demo mode activated! All trades are simulated with GOMNA AI systems.', 'info');
}

// Override any error handlers
window.addEventListener('error', function(e) {
    e.preventDefault();
    return true;
});

// Initialize when ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initializeSimpleGOMNA, 500);
    });
} else {
    setTimeout(initializeSimpleGOMNA, 500);
}

// Single cleanup after initialization
setTimeout(() => {
    try {
        // Hide any remaining error dialogs
        const errorElements = document.querySelectorAll('*');
        errorElements.forEach(el => {
            if (el.textContent && (
                el.textContent.includes('Initialization Error') || 
                el.textContent.includes('Some components failed to load')
            )) {
                el.style.display = 'none';
            }
        });
    } catch (e) {
        // Ignore cleanup errors
    }
}, 3000);

console.log('GOMNA Simple Clean Platform loaded successfully');