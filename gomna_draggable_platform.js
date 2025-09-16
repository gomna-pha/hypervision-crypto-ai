// GOMNA Draggable Platform - All panels can be moved and repositioned
class GomnaDraggablePlatform {
    constructor() {
        this.clearExistingPanels();
        this.initHeader();
        this.initDraggablePanels();
        this.initDragHandler();
        this.startRealTimeUpdates();
    }

    clearExistingPanels() {
        // Remove any existing panels to prevent duplicates
        const panels = [
            '#gomna-main-header',
            '#hyperbolic-cnn-dashboard',
            '.draggable-panel',
            '#gomna-draggable-header'
        ];
        panels.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => el.remove());
        });
    }

    initHeader() {
        const header = document.createElement('div');
        header.id = 'gomna-draggable-header';
        header.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(135deg, #1A0F0A 0%, #2C1810 50%, #1A0F0A 100%);
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.5);
            z-index: 10000;
            display: flex;
            align-items: center;
            padding: 0 20px;
            border-bottom: 2px solid #D4AF37;
        `;

        header.innerHTML = `
            <div style="display: flex; align-items: center; gap: 20px; width: 100%;">
                <div style="
                    background: radial-gradient(circle, #D4AF37, #8B6F47);
                    border-radius: 50%;
                    padding: 8px;
                    cursor: pointer;
                    box-shadow: 0 0 10px rgba(212, 175, 55, 0.5);
                ">
                    <svg width="32" height="32" viewBox="0 0 200 200">
                        <defs>
                            <linearGradient id="cocoaDrag" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#8B6F47" />
                                <stop offset="100%" style="stop-color:#5D4037" />
                            </linearGradient>
                        </defs>
                        <ellipse cx="100" cy="100" rx="55" ry="75" fill="url(#cocoaDrag)" />
                        <path d="M 100 35 L 100 165" stroke="#3E2723" stroke-width="3" opacity="0.5" />
                        <ellipse cx="85" cy="75" rx="15" ry="18" fill="#FAF7F0" opacity="0.9" />
                        <ellipse cx="115" cy="75" rx="15" ry="18" fill="#F5E6D3" opacity="0.9" />
                        <ellipse cx="100" cy="105" rx="16" ry="19" fill="#FAF7F0" opacity="0.9" />
                        <ellipse cx="85" cy="135" rx="15" ry="18" fill="#F5E6D3" opacity="0.9" />
                        <ellipse cx="115" cy="135" rx="15" ry="18" fill="#FAF7F0" opacity="0.9" />
                    </svg>
                </div>

                <div>
                    <div style="
                        font-size: 28px;
                        font-weight: 300;
                        color: #FAF7F0;
                        letter-spacing: 5px;
                        font-family: Georgia, serif;
                        font-style: italic;
                    ">GOMNA</div>
                    <div style="
                        font-size: 9px;
                        color: #D4AF37;
                        letter-spacing: 1.5px;
                        text-transform: uppercase;
                        margin-top: -2px;
                    ">Hyperbolic CNN Trading • Drag Panels to Customize</div>
                </div>

                <div style="margin-left: auto; display: flex; gap: 15px; align-items: center;">
                    <button onclick="resetPanelPositions()" style="
                        background: linear-gradient(135deg, #D4AF37, #CD7F32);
                        color: #2C1810;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 15px;
                        font-size: 11px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">Reset Layout</button>
                    
                    <button onclick="togglePanelsVisibility()" style="
                        background: linear-gradient(135deg, #8B6F47, #6B4423);
                        color: #FAF7F0;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 15px;
                        font-size: 11px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">Toggle Panels</button>
                </div>
            </div>
        `;

        document.body.appendChild(header);
        document.body.style.paddingTop = '70px';
    }

    initDraggablePanels() {
        // Panel 1: Hyperbolic CNN Model
        this.createPanel({
            id: 'hyperbolic-cnn-panel',
            title: 'Hyperbolic CNN Model',
            position: { top: 80, left: 20 },
            content: `
                <div style="display: grid; gap: 10px; padding: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #FAF7F0; font-size: 12px;">Curvature:</span>
                        <span style="color: #10B981; font-size: 13px; font-weight: bold;">-1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #FAF7F0; font-size: 12px;">Dimension:</span>
                        <span style="color: #10B981; font-size: 13px; font-weight: bold;">128</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #FAF7F0; font-size: 12px;">Accuracy:</span>
                        <span style="color: #10B981; font-size: 13px; font-weight: bold;" id="accuracy-value">94.7%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #FAF7F0; font-size: 12px;">Latency:</span>
                        <span style="color: #D4AF37; font-size: 13px; font-weight: bold;">125ms</span>
                    </div>
                </div>
            `
        });

        // Panel 2: Multimodal Data Sources
        this.createPanel({
            id: 'multimodal-panel',
            title: 'Multimodal Data Sources',
            position: { top: 250, left: 20 },
            content: `
                <div id="multimodal-content" style="display: grid; gap: 8px; padding: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></div>
                            <span style="color: #FAF7F0; font-size: 12px;">Equity Indices</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 11px;">25%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></div>
                            <span style="color: #FAF7F0; font-size: 12px;">Commodities</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 11px;">20%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></div>
                            <span style="color: #FAF7F0; font-size: 12px;">Cryptocurrency</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 11px;">30%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></div>
                            <span style="color: #FAF7F0; font-size: 12px;">Economic Data</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 11px;">15%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="width: 6px; height: 6px; background: #D4AF37; border-radius: 50%; animation: pulse 2s infinite;"></div>
                            <span style="color: #FAF7F0; font-size: 12px;">Sentiment</span>
                        </div>
                        <span style="color: #8B6F47; font-size: 11px;">10%</span>
                    </div>
                </div>
            `
        });

        // Panel 3: Live Signals
        this.createPanel({
            id: 'live-signals-panel',
            title: 'Live Signals',
            position: { top: 450, left: 20 },
            content: `
                <div id="signals-content" style="display: grid; gap: 10px; padding: 10px;">
                    <div style="
                        background: #10B98120;
                        border: 1px solid #10B981;
                        border-radius: 6px;
                        padding: 10px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div>
                            <div style="color: #FAF7F0; font-size: 12px;">BTC/USD</div>
                            <div style="color: #10B981; font-size: 14px; font-weight: bold;">BUY</div>
                        </div>
                        <div style="color: #FAF7F0; font-size: 11px;">92%</div>
                    </div>
                    <div style="
                        background: #D4AF3720;
                        border: 1px solid #D4AF37;
                        border-radius: 6px;
                        padding: 10px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div>
                            <div style="color: #FAF7F0; font-size: 12px;">ETH/USD</div>
                            <div style="color: #D4AF37; font-size: 14px; font-weight: bold;">HOLD</div>
                        </div>
                        <div style="color: #FAF7F0; font-size: 11px;">78%</div>
                    </div>
                </div>
            `
        });

        // Panel 4: Portfolio Performance
        this.createPanel({
            id: 'portfolio-panel',
            title: 'Portfolio Performance',
            position: { top: 80, right: 20 },
            content: `
                <div style="display: grid; gap: 12px; padding: 10px;">
                    <div>
                        <div style="color: #8B6F47; font-size: 11px; text-transform: uppercase;">Total Value</div>
                        <div style="color: #10B981; font-size: 24px; font-weight: bold;" id="portfolio-total">$2,847,563</div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <div style="color: #8B6F47; font-size: 10px;">24h Change</div>
                            <div style="color: #10B981; font-size: 16px; font-weight: bold;">+2.34%</div>
                        </div>
                        <div>
                            <div style="color: #8B6F47; font-size: 10px;">Sharpe Ratio</div>
                            <div style="color: #D4AF37; font-size: 16px; font-weight: bold;">2.89</div>
                        </div>
                    </div>
                </div>
            `
        });

        // Panel 5: Active Positions
        this.createPanel({
            id: 'positions-panel',
            title: 'Active Positions',
            position: { top: 250, right: 20 },
            content: `
                <div id="positions-content" style="display: grid; gap: 10px; padding: 10px; max-height: 200px; overflow-y: auto;">
                    <div style="
                        background: rgba(250, 247, 240, 0.05);
                        border: 1px solid #3E2723;
                        border-radius: 6px;
                        padding: 10px;
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 5px;
                    ">
                        <div>
                            <div style="color: #D4AF37; font-size: 13px; font-weight: bold;">BTC</div>
                            <div style="color: #8B6F47; font-size: 11px;">2.5 units</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #FAF7F0; font-size: 13px;">$168,581</div>
                            <div style="color: #10B981; font-size: 11px;">+12.4%</div>
                        </div>
                    </div>
                    <div style="
                        background: rgba(250, 247, 240, 0.05);
                        border: 1px solid #3E2723;
                        border-radius: 6px;
                        padding: 10px;
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 5px;
                    ">
                        <div>
                            <div style="color: #D4AF37; font-size: 13px; font-weight: bold;">ETH</div>
                            <div style="color: #8B6F47; font-size: 11px;">45.2 units</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #FAF7F0; font-size: 13px;">$159,252</div>
                            <div style="color: #10B981; font-size: 11px;">+8.7%</div>
                        </div>
                    </div>
                    <div style="
                        background: rgba(250, 247, 240, 0.05);
                        border: 1px solid #3E2723;
                        border-radius: 6px;
                        padding: 10px;
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 5px;
                    ">
                        <div>
                            <div style="color: #D4AF37; font-size: 13px; font-weight: bold;">SOL</div>
                            <div style="color: #8B6F47; font-size: 11px;">1,250 units</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #FAF7F0; font-size: 13px;">$178,125</div>
                            <div style="color: #10B981; font-size: 11px;">+15.3%</div>
                        </div>
                    </div>
                </div>
            `
        });

        // Panel 6: Execute Trade Button
        this.createPanel({
            id: 'execute-panel',
            title: '',
            position: { bottom: 30, right: 20 },
            content: `
                <button onclick="executeAITrade()" style="
                    width: 100%;
                    background: linear-gradient(135deg, #D4AF37, #CD7F32);
                    color: #2C1810;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(212, 175, 55, 0.3);
                ">EXECUTE AI TRADE</button>
            `
        });
    }

    createPanel(config) {
        const panel = document.createElement('div');
        panel.id = config.id;
        panel.className = 'draggable-panel';
        
        // Calculate position
        let positionStyle = 'position: fixed; ';
        if (config.position.top !== undefined) positionStyle += `top: ${config.position.top}px; `;
        if (config.position.bottom !== undefined) positionStyle += `bottom: ${config.position.bottom}px; `;
        if (config.position.left !== undefined) positionStyle += `left: ${config.position.left}px; `;
        if (config.position.right !== undefined) positionStyle += `right: ${config.position.right}px; `;
        
        panel.style.cssText = positionStyle + `
            width: 280px;
            background: linear-gradient(135deg, #1A1A1A, #2C1810);
            border: 1px solid #D4AF37;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            z-index: 1000;
            cursor: move;
            transition: box-shadow 0.3s ease;
            overflow: hidden;
        `;

        panel.innerHTML = `
            ${config.title ? `
                <div class="panel-header" style="
                    background: linear-gradient(135deg, #2C1810, #3E2723);
                    padding: 10px 15px;
                    border-bottom: 1px solid #D4AF37;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: move;
                ">
                    <h3 style="color: #D4AF37; font-size: 13px; margin: 0; font-weight: 600;">
                        ${config.title}
                    </h3>
                    <button onclick="minimizePanel('${config.id}')" style="
                        background: transparent;
                        border: none;
                        color: #D4AF37;
                        cursor: pointer;
                        font-size: 16px;
                        padding: 0;
                        width: 20px;
                        height: 20px;
                    ">−</button>
                </div>
            ` : ''}
            <div class="panel-content">
                ${config.content}
            </div>
        `;

        // Add hover effect
        panel.onmouseenter = () => {
            panel.style.boxShadow = '0 12px 30px rgba(212, 175, 55, 0.2)';
        };
        panel.onmouseleave = () => {
            panel.style.boxShadow = '0 8px 20px rgba(0, 0, 0, 0.4)';
        };

        document.body.appendChild(panel);
        this.makeDraggable(panel);
    }

    makeDraggable(element) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        let isDragging = false;
        
        const dragHeader = element.querySelector('.panel-header') || element;
        
        dragHeader.onmousedown = dragMouseDown;

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            isDragging = true;
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
            
            // Bring to front
            element.style.zIndex = '2000';
            
            // Add dragging style
            element.style.transition = 'none';
            element.style.opacity = '0.9';
        }

        function elementDrag(e) {
            if (!isDragging) return;
            e = e || window.event;
            e.preventDefault();
            
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            
            const newTop = element.offsetTop - pos2;
            const newLeft = element.offsetLeft - pos1;
            
            // Boundary checking
            const maxTop = window.innerHeight - element.offsetHeight;
            const maxLeft = window.innerWidth - element.offsetWidth;
            
            element.style.top = Math.min(Math.max(0, newTop), maxTop) + "px";
            element.style.left = Math.min(Math.max(0, newLeft), maxLeft) + "px";
            
            // Clear right/bottom positioning when dragging
            element.style.right = 'auto';
            element.style.bottom = 'auto';
        }

        function closeDragElement() {
            isDragging = false;
            document.onmouseup = null;
            document.onmousemove = null;
            
            // Reset styles
            element.style.zIndex = '1000';
            element.style.transition = 'box-shadow 0.3s ease';
            element.style.opacity = '1';
            
            // Save position to localStorage
            const positions = JSON.parse(localStorage.getItem('panelPositions') || '{}');
            positions[element.id] = {
                top: element.style.top,
                left: element.style.left
            };
            localStorage.setItem('panelPositions', JSON.stringify(positions));
        }
    }

    initDragHandler() {
        // Load saved positions
        const savedPositions = JSON.parse(localStorage.getItem('panelPositions') || '{}');
        Object.keys(savedPositions).forEach(panelId => {
            const panel = document.getElementById(panelId);
            if (panel && savedPositions[panelId]) {
                panel.style.top = savedPositions[panelId].top;
                panel.style.left = savedPositions[panelId].left;
                panel.style.right = 'auto';
                panel.style.bottom = 'auto';
            }
        });
    }

    startRealTimeUpdates() {
        // Update accuracy
        setInterval(() => {
            const accuracy = (94 + Math.random() * 2).toFixed(1);
            const el = document.getElementById('accuracy-value');
            if (el) el.textContent = `${accuracy}%`;
        }, 2000);

        // Update portfolio value
        setInterval(() => {
            const base = 2847563;
            const variation = Math.floor((Math.random() - 0.5) * 10000);
            const value = base + variation;
            const el = document.getElementById('portfolio-total');
            if (el) el.textContent = `$${value.toLocaleString()}`;
        }, 3000);

        // Rotate signals
        setInterval(() => {
            const signals = [
                { pair: 'BTC/USD', action: 'BUY', confidence: 92, color: '#10B981' },
                { pair: 'ETH/USD', action: 'HOLD', confidence: 78, color: '#D4AF37' },
                { pair: 'SOL/USD', action: 'SELL', confidence: 85, color: '#EF4444' },
                { pair: 'ADA/USD', action: 'BUY', confidence: 88, color: '#10B981' }
            ];
            
            const container = document.getElementById('signals-content');
            if (container) {
                const randomSignals = signals.sort(() => Math.random() - 0.5).slice(0, 2);
                container.innerHTML = randomSignals.map(signal => `
                    <div style="
                        background: ${signal.color}20;
                        border: 1px solid ${signal.color};
                        border-radius: 6px;
                        padding: 10px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div>
                            <div style="color: #FAF7F0; font-size: 12px;">${signal.pair}</div>
                            <div style="color: ${signal.color}; font-size: 14px; font-weight: bold;">${signal.action}</div>
                        </div>
                        <div style="color: #FAF7F0; font-size: 11px;">${signal.confidence}%</div>
                    </div>
                `).join('');
            }
        }, 4000);
    }
}

// Global functions
window.minimizePanel = (panelId) => {
    const panel = document.getElementById(panelId);
    const content = panel.querySelector('.panel-content');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        panel.querySelector('button').textContent = '−';
    } else {
        content.style.display = 'none';
        panel.querySelector('button').textContent = '+';
    }
};

window.resetPanelPositions = () => {
    localStorage.removeItem('panelPositions');
    location.reload();
};

window.togglePanelsVisibility = () => {
    const panels = document.querySelectorAll('.draggable-panel');
    panels.forEach(panel => {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    });
};

window.executeAITrade = () => {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 70px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(16, 185, 129, 0.4);
        z-index: 11000;
        animation: slideDown 0.5s ease;
    `;
    notification.textContent = '✓ AI Trade Executed via Hyperbolic CNN!';
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideUp 0.5s ease';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
};

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.gomnaDraggable = new GomnaDraggablePlatform();
    });
} else {
    window.gomnaDraggable = new GomnaDraggablePlatform();
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    @keyframes slideDown {
        from { transform: translate(-50%, -100px); opacity: 0; }
        to { transform: translate(-50%, 0); opacity: 1; }
    }
    @keyframes slideUp {
        from { transform: translate(-50%, 0); opacity: 1; }
        to { transform: translate(-50%, -100px); opacity: 0; }
    }
    body {
        background: linear-gradient(135deg, #0A0A0A, #1A1A1A) !important;
        padding-top: 70px !important;
        margin: 0;
    }
    .draggable-panel {
        user-select: none;
    }
    .draggable-panel:active {
        cursor: grabbing !important;
    }
    /* Hide previous panels */
    #hyperbolic-cnn-dashboard,
    #gomna-main-header,
    #gomna-professional-header,
    #gomna-unified-branding {
        display: none !important;
    }
`;
document.head.appendChild(style);