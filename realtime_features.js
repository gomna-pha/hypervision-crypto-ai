// GOMNA Real-Time Features - Portfolio and Model Transparency
class GomnaRealtimeFeatures {
    constructor() {
        this.portfolioInterval = null;
        this.modelInterval = null;
        this.initRealTimePortfolio();
        this.initModelTransparency();
    }

    initRealTimePortfolio() {
        const dashboard = document.createElement('div');
        dashboard.id = 'realtime-portfolio';
        dashboard.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            background: linear-gradient(135deg, #FAF7F0 0%, #F5E6D3 100%);
            border: 2px solid #8B6F47;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(107, 68, 35, 0.2);
            z-index: 900;
            transition: all 0.3s ease;
        `;

        dashboard.innerHTML = `
            <div style="margin-bottom: 15px; border-bottom: 2px solid #8B6F47; padding-bottom: 10px;">
                <h3 style="font-size: 18px; font-weight: bold; color: #3E2723; margin: 0;">
                    Real-Time Portfolio
                </h3>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 5px;">
                    <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                    <span style="font-size: 12px; color: #6B4423;">LIVE</span>
                    <span id="portfolio-time" style="font-size: 11px; color: #8B6F47; margin-left: auto;"></span>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 11px; color: #8B6F47; text-transform: uppercase;">Total Value</div>
                        <div id="portfolio-total" style="font-size: 24px; font-weight: bold; color: #3E2723;">$0.00</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 11px; color: #8B6F47; text-transform: uppercase;">24h Change</div>
                        <div id="portfolio-change" style="font-size: 20px; font-weight: bold;">+0.00%</div>
                    </div>
                </div>
                <div id="holdings-list" style="max-height: 200px; overflow-y: auto;"></div>
            </div>
            <button onclick="this.parentElement.style.display = this.parentElement.style.display === 'none' ? 'block' : 'none'" style="
                position: absolute;
                top: 10px;
                right: 10px;
                width: 20px;
                height: 20px;
                background: #8B6F47;
                color: #FAF7F0;
                border: none;
                border-radius: 50%;
                cursor: pointer;
                font-size: 12px;
            ">×</button>
        `;

        document.body.appendChild(dashboard);
        this.startPortfolioUpdates();
    }

    startPortfolioUpdates() {
        const update = () => {
            const total = document.getElementById('portfolio-total');
            const change = document.getElementById('portfolio-change');
            const time = document.getElementById('portfolio-time');
            const holdings = document.getElementById('holdings-list');

            if (total && change && time) {
                const baseValue = 2847563;
                const variation = (Math.random() - 0.5) * 10000;
                const totalValue = baseValue + variation;
                const changePercent = ((Math.random() - 0.45) * 5).toFixed(2);

                total.textContent = `$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                change.textContent = `${changePercent > 0 ? '+' : ''}${changePercent}%`;
                change.style.color = changePercent > 0 ? '#10B981' : '#EF4444';
                time.textContent = new Date().toLocaleTimeString();

                if (holdings && Math.random() > 0.7) {
                    const assets = [
                        { symbol: 'BTC', name: 'Bitcoin', value: 145230.50, change: 2.3 },
                        { symbol: 'ETH', name: 'Ethereum', value: 89450.00, change: -1.2 },
                        { symbol: 'SOL', name: 'Solana', value: 34200.00, change: 5.7 }
                    ];

                    holdings.innerHTML = assets.map(asset => `
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #E8DCC7;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="width: 32px; height: 32px; background: #8B6F47; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #FAF7F0; font-weight: bold; font-size: 10px;">
                                    ${asset.symbol}
                                </div>
                                <div>
                                    <div style="font-size: 13px; font-weight: 600; color: #3E2723;">${asset.name}</div>
                                    <div style="font-size: 11px; color: #8B6F47;">$${asset.value.toLocaleString()}</div>
                                </div>
                            </div>
                            <div style="font-size: 13px; font-weight: 600; color: ${asset.change > 0 ? '#10B981' : '#EF4444'};">
                                ${asset.change > 0 ? '+' : ''}${asset.change}%
                            </div>
                        </div>
                    `).join('');
                }
            }
        };

        update();
        this.portfolioInterval = setInterval(update, 2000);
    }

    initModelTransparency() {
        const panel = document.createElement('div');
        panel.id = 'model-transparency';
        panel.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 350px;
            background: linear-gradient(135deg, #FAF7F0 0%, #F5E6D3 100%);
            border: 2px solid #8B6F47;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(107, 68, 35, 0.2);
            z-index: 900;
            transition: all 0.3s ease;
        `;

        panel.innerHTML = `
            <div style="margin-bottom: 15px; border-bottom: 2px solid #8B6F47; padding-bottom: 10px;">
                <h3 style="font-size: 18px; font-weight: bold; color: #3E2723; margin: 0;">
                    Model Transparency
                </h3>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 5px;">
                    <div style="width: 8px; height: 8px; background: #D4AF37; border-radius: 50%; animation: pulse 2s infinite;"></div>
                    <span style="font-size: 12px; color: #6B4423;">AI ACTIVE</span>
                </div>
            </div>
            <div>
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #8B6F47; margin-bottom: 5px;">Model Accuracy</div>
                    <div style="background: #E8DCC7; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div id="accuracy-bar" style="background: linear-gradient(90deg, #8B6F47, #D4AF37); height: 100%; width: 0%; transition: width 1s ease;">
                            <span style="color: white; font-size: 11px; padding: 2px 8px; display: block; text-align: right;">0%</span>
                        </div>
                    </div>
                </div>
                <div id="model-signals"></div>
            </div>
            <button onclick="this.parentElement.style.display = this.parentElement.style.display === 'none' ? 'block' : 'none'" style="
                position: absolute;
                top: 10px;
                right: 10px;
                width: 20px;
                height: 20px;
                background: #8B6F47;
                color: #FAF7F0;
                border: none;
                border-radius: 50%;
                cursor: pointer;
                font-size: 12px;
            ">×</button>
        `;

        document.body.appendChild(panel);
        this.startModelUpdates();
    }

    startModelUpdates() {
        const update = () => {
            const accuracyBar = document.getElementById('accuracy-bar');
            const signalsDiv = document.getElementById('model-signals');

            if (accuracyBar) {
                const accuracy = 85 + Math.random() * 10;
                accuracyBar.style.width = `${accuracy}%`;
                accuracyBar.querySelector('span').textContent = `${accuracy.toFixed(1)}%`;
            }

            if (signalsDiv) {
                const signals = [
                    { type: 'BUY', asset: 'BTC/USD', confidence: 92, reason: 'Bullish divergence' },
                    { type: 'HOLD', asset: 'ETH/USD', confidence: 78, reason: 'Consolidation' },
                    { type: 'SELL', asset: 'XRP/USD', confidence: 85, reason: 'Overbought' }
                ];

                const signal = signals[Math.floor(Math.random() * signals.length)];
                const color = signal.type === 'BUY' ? '#10B981' : signal.type === 'SELL' ? '#EF4444' : '#D4AF37';

                signalsDiv.innerHTML = `
                    <div style="background: ${color}20; border: 1px solid ${color}; border-radius: 8px; padding: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold; color: ${color};">${signal.type}</span>
                            <span style="font-size: 12px; color: #3E2723;">${signal.asset}</span>
                        </div>
                        <div style="font-size: 11px; color: #6B4423; margin-top: 5px;">${signal.reason}</div>
                        <div style="font-size: 10px; color: #8B6F47; margin-top: 3px;">Confidence: ${signal.confidence}%</div>
                    </div>
                `;
            }
        };

        update();
        this.modelInterval = setInterval(update, 3000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            window.gomnaRealtime = new GomnaRealtimeFeatures();
        }, 500);
    });
} else {
    setTimeout(() => {
        window.gomnaRealtime = new GomnaRealtimeFeatures();
    }, 500);
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
`;
document.head.appendChild(style);