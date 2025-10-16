// GOMNA Professional Dashboard Components
class ProfessionalDashboard {
    constructor() {
        this.initProfessionalUI();
        this.initTradingCards();
        this.initPerformanceMetrics();
    }

    initProfessionalUI() {
        // Create floating action menu
        const actionMenu = document.createElement('div');
        actionMenu.id = 'professional-action-menu';
        actionMenu.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
        `;

        actionMenu.innerHTML = `
            <div style="
                background: linear-gradient(135deg, #3E2723, #5D4037);
                border: 2px solid #D4AF37;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            " id="menu-toggle">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D4AF37" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </div>
            
            <div id="menu-items" style="
                position: absolute;
                bottom: 70px;
                right: 0;
                display: none;
                flex-direction: column;
                gap: 10px;
            ">
                <button onclick="toggleDashboard()" style="
                    background: linear-gradient(135deg, #8B6F47, #D4AF37);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 25px;
                    cursor: pointer;
                    white-space: nowrap;
                    font-weight: 500;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                ">📊 Dashboard</button>
                
                <button onclick="togglePortfolio()" style="
                    background: linear-gradient(135deg, #8B6F47, #D4AF37);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 25px;
                    cursor: pointer;
                    white-space: nowrap;
                    font-weight: 500;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                ">💼 Portfolio</button>
                
                <button onclick="toggleAnalytics()" style="
                    background: linear-gradient(135deg, #8B6F47, #D4AF37);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 25px;
                    cursor: pointer;
                    white-space: nowrap;
                    font-weight: 500;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                ">📈 Analytics</button>
            </div>
        `;

        document.body.appendChild(actionMenu);

        // Toggle menu
        document.getElementById('menu-toggle').onclick = function() {
            const items = document.getElementById('menu-items');
            items.style.display = items.style.display === 'none' ? 'flex' : 'none';
            this.style.transform = items.style.display === 'none' ? 'rotate(0deg)' : 'rotate(45deg)';
        };
    }

    initTradingCards() {
        const tradingPanel = document.createElement('div');
        tradingPanel.id = 'professional-trading-panel';
        tradingPanel.style.cssText = `
            position: fixed;
            top: 140px;
            right: 30px;
            width: 320px;
            max-height: 70vh;
            overflow-y: auto;
            z-index: 900;
        `;

        tradingPanel.innerHTML = `
            <div style="
                background: linear-gradient(135deg, #FAF7F0, #F5E6D3);
                border: 2px solid #8B6F47;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                margin-bottom: 15px;
            ">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 15px;">
                    <h3 style="font-size: 16px; color: #3E2723; margin: 0; flex: 1;">Quick Trade</h3>
                    <span style="font-size: 10px; color: #8B6F47; background: #FAF7F0; padding: 3px 8px; border-radius: 10px;">PRO</span>
                </div>
                
                <div style="display: grid; gap: 10px;">
                    <select style="
                        padding: 10px;
                        border: 1px solid #D4C4A8;
                        border-radius: 8px;
                        background: white;
                        color: #3E2723;
                        font-size: 14px;
                    ">
                        <option>BTC/USD - Bitcoin</option>
                        <option>ETH/USD - Ethereum</option>
                        <option>SOL/USD - Solana</option>
                    </select>
                    
                    <input type="number" placeholder="Amount" style="
                        padding: 10px;
                        border: 1px solid #D4C4A8;
                        border-radius: 8px;
                        background: white;
                        color: #3E2723;
                        font-size: 14px;
                    ">
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <button style="
                            background: linear-gradient(135deg, #10B981, #059669);
                            color: white;
                            border: none;
                            padding: 12px;
                            border-radius: 8px;
                            cursor: pointer;
                            font-weight: bold;
                            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
                        ">BUY</button>
                        
                        <button style="
                            background: linear-gradient(135deg, #EF4444, #DC2626);
                            color: white;
                            border: none;
                            padding: 12px;
                            border-radius: 8px;
                            cursor: pointer;
                            font-weight: bold;
                            box-shadow: 0 5px 15px rgba(239, 68, 68, 0.3);
                        ">SELL</button>
                    </div>
                </div>
            </div>
            
            <!-- Market Overview -->
            <div style="
                background: linear-gradient(135deg, #FAF7F0, #F5E6D3);
                border: 2px solid #8B6F47;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            ">
                <h3 style="font-size: 16px; color: #3E2723; margin: 0 0 15px 0;">Market Overview</h3>
                <div id="market-overview" style="display: grid; gap: 10px;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>
        `;

        document.body.appendChild(tradingPanel);
        this.updateMarketOverview();
    }

    updateMarketOverview() {
        const markets = [
            { symbol: 'BTC', price: 67432.50, change: 2.34 },
            { symbol: 'ETH', price: 3521.80, change: -1.12 },
            { symbol: 'SOL', price: 142.65, change: 5.67 }
        ];

        const container = document.getElementById('market-overview');
        if (container) {
            container.innerHTML = markets.map(market => `
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    background: white;
                    border-radius: 8px;
                    border: 1px solid #E8DCC7;
                ">
                    <div>
                        <div style="font-weight: bold; color: #3E2723;">${market.symbol}/USD</div>
                        <div style="font-size: 14px; color: #8B6F47;">$${market.price.toLocaleString()}</div>
                    </div>
                    <div style="
                        color: ${market.change > 0 ? '#10B981' : '#EF4444'};
                        font-weight: bold;
                        font-size: 14px;
                    ">
                        ${market.change > 0 ? '↑' : '↓'} ${Math.abs(market.change)}%
                    </div>
                </div>
            `).join('');
        }

        // Update every 5 seconds
        setTimeout(() => this.updateMarketOverview(), 5000);
    }

    initPerformanceMetrics() {
        const metricsBar = document.createElement('div');
        metricsBar.id = 'performance-metrics-bar';
        metricsBar.style.cssText = `
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 50px;
            background: linear-gradient(90deg, #3E2723 0%, #5D4037 50%, #3E2723 100%);
            border-top: 2px solid #D4AF37;
            display: flex;
            align-items: center;
            padding: 0 30px;
            z-index: 999;
            box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.2);
        `;

        metricsBar.innerHTML = `
            <div style="display: flex; gap: 40px; align-items: center; width: 100%;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #D4AF37; font-size: 12px;">Portfolio:</span>
                    <span style="color: #FAF7F0; font-size: 16px; font-weight: bold;">$2.84M</span>
                    <span style="color: #10B981; font-size: 12px;">+12.4%</span>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #D4AF37; font-size: 12px;">Daily P&L:</span>
                    <span style="color: #10B981; font-size: 16px; font-weight: bold;">+$45,230</span>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #D4AF37; font-size: 12px;">Win Rate:</span>
                    <span style="color: #FAF7F0; font-size: 16px; font-weight: bold;">78.5%</span>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #D4AF37; font-size: 12px;">Sharpe:</span>
                    <span style="color: #FAF7F0; font-size: 16px; font-weight: bold;">2.41</span>
                </div>
                
                <div style="margin-left: auto; display: flex; align-items: center; gap: 15px;">
                    <button style="
                        background: linear-gradient(135deg, #D4AF37, #CD7F32);
                        color: #3E2723;
                        border: none;
                        padding: 8px 20px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-weight: bold;
                        font-size: 12px;
                        box-shadow: 0 3px 10px rgba(212, 175, 55, 0.3);
                    ">EXECUTE TRADE</button>
                </div>
            </div>
        `;

        document.body.appendChild(metricsBar);
        
        // Adjust body padding for bottom bar
        document.body.style.paddingBottom = '70px';
    }
}

// Global functions for menu
window.toggleDashboard = () => {
    const panel = document.getElementById('realtime-portfolio');
    if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
};

window.togglePortfolio = () => {
    const panel = document.getElementById('professional-trading-panel');
    if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
};

window.toggleAnalytics = () => {
    const panel = document.getElementById('model-transparency');
    if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
};

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            window.professionalDashboard = new ProfessionalDashboard();
        }, 500);
    });
} else {
    setTimeout(() => {
        window.professionalDashboard = new ProfessionalDashboard();
    }, 500);
}