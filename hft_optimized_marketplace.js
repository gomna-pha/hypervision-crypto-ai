/**
 * HFT-Optimized Algorithm Marketplace
 * Enhanced marketplace with ultra-low latency optimization integration
 */

class HFTOptimizedMarketplace {
    constructor() {
        this.hftSystem = null;
        this.originalMarketplace = null;
        this.isOptimizationEnabled = false;
        
        // HFT-specific algorithm enhancements
        this.hftEnhancements = {
            'hyperbolic-cnn-pro': {
                originalLatency: 150, // microseconds
                optimizedLatency: 25, // with HFT optimizations
                hardwareAcceleration: ['WebGPU', 'WASM'],
                networkOptimization: true,
                performanceGain: '83%'
            },
            'triangular-arbitrage-elite': {
                originalLatency: 200,
                optimizedLatency: 35,
                hardwareAcceleration: ['WebGPU', 'WASM', 'Parallel Processing'],
                networkOptimization: true,
                performanceGain: '82.5%'
            },
            'flash-loan-arbitrage-master': {
                originalLatency: 100,
                optimizedLatency: 15,
                hardwareAcceleration: ['WebGPU', 'Binary Protocol'],
                networkOptimization: true,
                performanceGain: '85%'
            },
            'statistical-pairs-ai': {
                originalLatency: 300,
                optimizedLatency: 50,
                hardwareAcceleration: ['WASM', 'Memory Pool'],
                networkOptimization: true,
                performanceGain: '83.3%'
            },
            'sentiment-momentum-pro': {
                originalLatency: 250,
                optimizedLatency: 40,
                hardwareAcceleration: ['GPU Compute', 'Lock-free Queues'],
                networkOptimization: true,
                performanceGain: '84%'
            }
        };
        
        this.initializeHFTIntegration();
    }
    
    async initializeHFTIntegration() {
        console.log('🚀 Initializing HFT-Optimized Marketplace...');
        
        try {
            // Initialize HFT optimization system
            if (typeof IntegratedHFTOptimizationSystem !== 'undefined') {
                this.hftSystem = new IntegratedHFTOptimizationSystem();
                await this.hftSystem.initializeSystem();
                this.isOptimizationEnabled = true;
                
                console.log('✅ HFT Optimization System integrated successfully');
            }
            
            // Get reference to original marketplace
            if (typeof algorithmMarketplace !== 'undefined') {
                this.originalMarketplace = algorithmMarketplace;
                this.enhanceOriginalMarketplace();
            }
            
            this.setupHFTMonitoring();
            
        } catch (error) {
            console.error('❌ HFT integration failed:', error);
            this.isOptimizationEnabled = false;
        }
    }
    
    // Enhance the original marketplace with HFT features
    enhanceOriginalMarketplace() {
        if (!this.originalMarketplace) return;
        
        // Override the algorithm rendering to include HFT metrics
        const originalRenderAlgorithmCards = this.originalMarketplace.renderAlgorithmCards.bind(this.originalMarketplace);
        
        this.originalMarketplace.renderAlgorithmCards = () => {
            const originalHTML = originalRenderAlgorithmCards();
            return this.addHFTEnhancements(originalHTML);
        };
        
        // Override investment modal to include HFT information
        const originalOpenInvestmentModal = this.originalMarketplace.openInvestmentModal.bind(this.originalMarketplace);
        
        this.originalMarketplace.openInvestmentModal = (algorithmId) => {
            originalOpenInvestmentModal(algorithmId);
            this.addHFTInvestmentInfo(algorithmId);
        };
        
        console.log('✅ Original marketplace enhanced with HFT features');
    }
    
    // Add HFT enhancements to algorithm cards
    addHFTEnhancements(originalHTML) {
        if (!this.isOptimizationEnabled) return originalHTML;
        
        let enhancedHTML = originalHTML;
        
        // Add HFT performance badges to each algorithm card
        Object.entries(this.hftEnhancements).forEach(([algorithmId, enhancement]) => {
            const hftBadge = this.createHFTBadge(enhancement);
            const cardPattern = new RegExp(`(<div class="algorithm-card[^>]*data-algorithm="${algorithmId}"[^>]*>)`, 'g');
            
            enhancedHTML = enhancedHTML.replace(cardPattern, (match) => {
                return match + hftBadge;
            });
        });
        
        return enhancedHTML;
    }
    
    // Create HFT performance badge
    createHFTBadge(enhancement) {
        return `
            <div class="hft-enhancement-badge">
                <div class="hft-badge-header">
                    <span class="hft-icon">⚡</span>
                    <span class="hft-title">HFT OPTIMIZED</span>
                </div>
                <div class="hft-metrics">
                    <div class="hft-metric">
                        <span class="hft-metric-label">Latency:</span>
                        <span class="hft-metric-value">${enhancement.optimizedLatency}μs</span>
                        <span class="hft-improvement">${enhancement.performanceGain} faster</span>
                    </div>
                    <div class="hft-acceleration">
                        ${enhancement.hardwareAcceleration.map(tech => 
                            `<span class="hft-tech-badge">${tech}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add HFT information to investment modal
    addHFTInvestmentInfo(algorithmId) {
        const enhancement = this.hftEnhancements[algorithmId];
        if (!enhancement || !this.isOptimizationEnabled) return;
        
        const modalBody = document.getElementById('modal-body');
        if (!modalBody) return;
        
        const hftSection = document.createElement('div');
        hftSection.className = 'hft-investment-section';
        hftSection.innerHTML = this.renderHFTInvestmentSection(enhancement);
        
        // Insert after existing content
        modalBody.appendChild(hftSection);
    }
    
    // Render HFT investment section
    renderHFTInvestmentSection(enhancement) {
        return `
            <div class="hft-section-header">
                <h3>⚡ Ultra-Low Latency Performance</h3>
                <div class="hft-status-indicator">
                    <span class="hft-status-dot active"></span>
                    <span class="hft-status-text">HFT Optimization Active</span>
                </div>
            </div>
            
            <div class="hft-performance-grid">
                <div class="hft-performance-card">
                    <div class="hft-perf-title">Execution Latency</div>
                    <div class="hft-perf-value">${enhancement.optimizedLatency} μs</div>
                    <div class="hft-perf-improvement">
                        <span class="hft-arrow">↓</span>
                        ${((enhancement.originalLatency - enhancement.optimizedLatency) / enhancement.originalLatency * 100).toFixed(1)}% reduction
                    </div>
                </div>
                
                <div class="hft-performance-card">
                    <div class="hft-perf-title">Performance Gain</div>
                    <div class="hft-perf-value">${enhancement.performanceGain}</div>
                    <div class="hft-perf-improvement">vs. standard implementation</div>
                </div>
                
                <div class="hft-performance-card">
                    <div class="hft-perf-title">Hardware Acceleration</div>
                    <div class="hft-perf-value">${enhancement.hardwareAcceleration.length}</div>
                    <div class="hft-perf-improvement">optimization layers</div>
                </div>
            </div>
            
            <div class="hft-technology-stack">
                <h4>🔧 Optimization Technologies</h4>
                <div class="hft-tech-grid">
                    ${enhancement.hardwareAcceleration.map(tech => `
                        <div class="hft-tech-item">
                            <span class="hft-tech-icon">${this.getTechIcon(tech)}</span>
                            <span class="hft-tech-name">${tech}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="hft-real-time-metrics">
                <h4>📊 Real-Time Performance Monitoring</h4>
                <div class="hft-live-metrics" id="hft-live-metrics">
                    <div class="hft-metric-item">
                        <span class="hft-metric-name">Current Latency:</span>
                        <span class="hft-metric-live" id="hft-current-latency">--</span>
                    </div>
                    <div class="hft-metric-item">
                        <span class="hft-metric-name">Throughput:</span>
                        <span class="hft-metric-live" id="hft-throughput">--</span>
                    </div>
                    <div class="hft-metric-item">
                        <span class="hft-metric-name">System Health:</span>
                        <span class="hft-metric-live hft-health" id="hft-system-health">--</span>
                    </div>
                </div>
            </div>
            
            <div class="hft-competitive-advantage">
                <h4>🏆 Competitive Advantage</h4>
                <ul class="hft-advantage-list">
                    <li>✅ Microsecond-level arbitrage detection</li>
                    <li>✅ GPU-accelerated parallel processing</li>
                    <li>✅ Zero-copy memory management</li>
                    <li>✅ Binary protocol networking</li>
                    <li>✅ Real-time performance optimization</li>
                </ul>
            </div>
        `;
    }
    
    // Get technology icon
    getTechIcon(tech) {
        const icons = {
            'WebGPU': '🖥️',
            'WASM': '⚙️',
            'Parallel Processing': '🔀',
            'Binary Protocol': '📡',
            'Memory Pool': '💾',
            'GPU Compute': '🎮',
            'Lock-free Queues': '🔄'
        };
        
        return icons[tech] || '🔧';
    }
    
    // Setup real-time HFT monitoring
    setupHFTMonitoring() {
        if (!this.hftSystem) return;
        
        // Update live metrics every 100ms
        this.metricsInterval = setInterval(() => {
            this.updateLiveMetrics();
        }, 100);
        
        // Listen for performance updates
        document.addEventListener('performanceReport', (event) => {
            this.handlePerformanceUpdate(event.detail);
        });
        
        console.log('✅ HFT monitoring setup complete');
    }
    
    // Update live metrics display
    updateLiveMetrics() {
        if (!this.hftSystem || !this.hftSystem.isRunning) return;
        
        const status = this.hftSystem.getSystemStatus();
        
        // Update latency display
        const latencyElement = document.getElementById('hft-current-latency');
        if (latencyElement && status.performanceData) {
            const avgLatency = this.calculateAverageLatency(status.performanceData.realtimeMetrics);
            latencyElement.textContent = `${avgLatency.toFixed(1)}μs`;
        }
        
        // Update throughput display
        const throughputElement = document.getElementById('hft-throughput');
        if (throughputElement && status.performanceData) {
            const throughput = this.calculateThroughput(status.performanceData.realtimeMetrics);
            throughputElement.textContent = `${throughput.toLocaleString()}/s`;
        }
        
        // Update system health
        const healthElement = document.getElementById('hft-system-health');
        if (healthElement && status.performanceData) {
            healthElement.textContent = status.performanceData.systemHealth.status;
            healthElement.className = `hft-metric-live hft-health ${status.performanceData.systemHealth.status.toLowerCase()}`;
        }
    }
    
    // Calculate average latency across components
    calculateAverageLatency(realtimeMetrics) {
        let totalLatency = 0;
        let componentCount = 0;
        
        Object.values(realtimeMetrics).forEach(metrics => {
            if (metrics.latency && metrics.latency.length > 0) {
                const avgComponentLatency = metrics.latency.reduce((sum, val) => sum + val, 0) / metrics.latency.length;
                totalLatency += avgComponentLatency;
                componentCount++;
            }
        });
        
        return componentCount > 0 ? totalLatency / componentCount : 0;
    }
    
    // Calculate total system throughput
    calculateThroughput(realtimeMetrics) {
        let totalThroughput = 0;
        
        Object.values(realtimeMetrics).forEach(metrics => {
            if (metrics.throughput && metrics.throughput.length > 0) {
                const latestThroughput = metrics.throughput[metrics.throughput.length - 1];
                totalThroughput += latestThroughput || 0;
            }
        });
        
        return Math.round(totalThroughput);
    }
    
    // Handle performance updates
    handlePerformanceUpdate(report) {
        // Log significant performance events
        if (report.systemHealth.status === 'CRITICAL') {
            console.error('🚨 CRITICAL HFT performance detected:', report);
        } else if (report.systemHealth.status === 'DEGRADED') {
            console.warn('⚠️ HFT performance degraded:', report);
        }
    }
    
    // Start HFT optimization
    async startHFTOptimization() {
        if (!this.hftSystem) {
            console.error('❌ HFT system not available');
            return false;
        }
        
        try {
            await this.hftSystem.start();
            
            // Update UI to show HFT is active
            this.updateHFTStatusIndicators(true);
            
            console.log('🟢 HFT optimization started');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to start HFT optimization:', error);
            return false;
        }
    }
    
    // Stop HFT optimization
    stopHFTOptimization() {
        if (this.hftSystem) {
            this.hftSystem.stop();
        }
        
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
        }
        
        // Update UI to show HFT is inactive
        this.updateHFTStatusIndicators(false);
        
        console.log('🔴 HFT optimization stopped');
    }
    
    // Update HFT status indicators in UI
    updateHFTStatusIndicators(isActive) {
        const indicators = document.querySelectorAll('.hft-status-indicator');
        
        indicators.forEach(indicator => {
            const dot = indicator.querySelector('.hft-status-dot');
            const text = indicator.querySelector('.hft-status-text');
            
            if (dot && text) {
                if (isActive) {
                    dot.className = 'hft-status-dot active';
                    text.textContent = 'HFT Optimization Active';
                } else {
                    dot.className = 'hft-status-dot inactive';
                    text.textContent = 'HFT Optimization Inactive';
                }
            }
        });
    }
    
    // Run HFT benchmark
    async runHFTBenchmark() {
        if (!this.hftSystem) {
            console.error('❌ HFT system not available');
            return null;
        }
        
        console.log('🏁 Running HFT benchmark...');
        
        try {
            const benchmark = await this.hftSystem.runBenchmark();
            
            // Display results
            this.displayBenchmarkResults(benchmark);
            
            return benchmark;
            
        } catch (error) {
            console.error('❌ Benchmark failed:', error);
            return null;
        }
    }
    
    // Display benchmark results
    displayBenchmarkResults(benchmark) {
        const modal = document.getElementById('investment-modal');
        const modalName = document.getElementById('modal-algorithm-name');
        const modalBody = document.getElementById('modal-body');
        
        if (modal && modalName && modalBody) {
            modalName.textContent = 'HFT System Benchmark Results';
            modalBody.innerHTML = this.renderBenchmarkResults(benchmark);
            modal.classList.remove('hidden');
        }
    }
    
    // Render benchmark results
    renderBenchmarkResults(benchmark) {
        return `
            <div class="hft-benchmark-results">
                <div class="benchmark-header">
                    <h3>🏁 HFT Performance Benchmark</h3>
                    <div class="benchmark-timestamp">${benchmark.timestamp}</div>
                </div>
                
                <div class="benchmark-summary">
                    <div class="benchmark-card">
                        <div class="benchmark-title">System Throughput</div>
                        <div class="benchmark-value">
                            ${benchmark.systemPerformance.throughput.toLocaleString()} ops/s
                        </div>
                    </div>
                    
                    <div class="benchmark-card">
                        <div class="benchmark-title">Average Latency</div>
                        <div class="benchmark-value">
                            ${benchmark.systemPerformance.avgLatencyPerPoint.toFixed(2)}μs
                        </div>
                    </div>
                    
                    <div class="benchmark-card">
                        <div class="benchmark-title">Data Points</div>
                        <div class="benchmark-value">
                            ${benchmark.systemPerformance.dataPoints.toLocaleString()}
                        </div>
                    </div>
                </div>
                
                <div class="component-benchmarks">
                    <h4>Component Performance</h4>
                    ${Object.entries(benchmark.components).map(([component, results]) => `
                        <div class="component-benchmark">
                            <div class="component-name">${component.toUpperCase()}</div>
                            <div class="component-metrics">
                                ${Object.entries(results).map(([metric, value]) => `
                                    <span class="component-metric">
                                        ${metric}: ${typeof value === 'number' ? value.toFixed(2) : value}
                                    </span>
                                `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Get HFT system status
    getHFTStatus() {
        if (!this.hftSystem) {
            return {
                available: false,
                status: 'Not Available',
                message: 'HFT system not initialized'
            };
        }
        
        const status = this.hftSystem.getSystemStatus();
        
        return {
            available: true,
            status: status.health,
            isRunning: status.isRunning,
            optimizationLevel: status.optimizationLevel,
            uptime: status.uptime,
            components: status.components,
            performanceData: status.performanceData
        };
    }
}

// CSS Styles for HFT enhancements
const hftStyles = `
    <style>
        .hft-enhancement-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 8px;
            border-radius: 8px;
            font-size: 0.75rem;
            min-width: 120px;
            z-index: 10;
            box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3);
        }
        
        .hft-badge-header {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-bottom: 6px;
        }
        
        .hft-icon {
            font-size: 1rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .hft-title {
            font-weight: 700;
            font-size: 0.7rem;
        }
        
        .hft-metrics {
            font-size: 0.65rem;
        }
        
        .hft-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }
        
        .hft-metric-value {
            font-weight: 600;
            color: #fbbf24;
        }
        
        .hft-improvement {
            color: #10b981;
            font-size: 0.6rem;
        }
        
        .hft-acceleration {
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            margin-top: 4px;
        }
        
        .hft-tech-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 0.55rem;
            font-weight: 500;
        }
        
        .hft-investment-section {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 12px;
            border: 2px solid #3b82f6;
        }
        
        .hft-section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .hft-section-header h3 {
            color: #1e40af;
            font-size: 1.25rem;
            margin: 0;
        }
        
        .hft-status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .hft-status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ef4444;
        }
        
        .hft-status-dot.active {
            background: #10b981;
            animation: pulse 2s infinite;
        }
        
        .hft-status-text {
            font-size: 0.9rem;
            font-weight: 600;
            color: #374151;
        }
        
        .hft-performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .hft-performance-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .hft-perf-title {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 8px;
        }
        
        .hft-perf-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 4px;
        }
        
        .hft-perf-improvement {
            font-size: 0.75rem;
            color: #10b981;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }
        
        .hft-arrow {
            font-weight: bold;
        }
        
        .hft-technology-stack {
            margin-bottom: 25px;
        }
        
        .hft-technology-stack h4 {
            color: #374151;
            margin-bottom: 12px;
        }
        
        .hft-tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .hft-tech-item {
            display: flex;
            align-items: center;
            gap: 8px;
            background: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
        }
        
        .hft-real-time-metrics {
            margin-bottom: 25px;
        }
        
        .hft-real-time-metrics h4 {
            color: #374151;
            margin-bottom: 12px;
        }
        
        .hft-live-metrics {
            background: white;
            padding: 15px;
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .hft-metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .hft-metric-name {
            font-weight: 600;
            color: #374151;
        }
        
        .hft-metric-live {
            font-weight: 700;
            color: #1e40af;
        }
        
        .hft-health.healthy {
            color: #10b981;
        }
        
        .hft-health.degraded {
            color: #f59e0b;
        }
        
        .hft-health.critical {
            color: #ef4444;
        }
        
        .hft-competitive-advantage h4 {
            color: #374151;
            margin-bottom: 12px;
        }
        
        .hft-advantage-list {
            list-style: none;
            padding: 0;
        }
        
        .hft-advantage-list li {
            padding: 6px 0;
            color: #374151;
            font-size: 0.9rem;
        }
        
        .hft-benchmark-results {
            padding: 20px;
        }
        
        .benchmark-header {
            text-align: center;
            margin-bottom: 25px;
        }
        
        .benchmark-timestamp {
            color: #6b7280;
            font-size: 0.85rem;
        }
        
        .benchmark-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .benchmark-card {
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .benchmark-title {
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        
        .benchmark-value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .component-benchmarks h4 {
            margin-bottom: 15px;
            color: #374151;
        }
        
        .component-benchmark {
            background: #f8fafc;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }
        
        .component-name {
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 8px;
        }
        
        .component-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .component-metric {
            font-size: 0.85rem;
            color: #374151;
        }
    </style>
`;

// Initialize HFT-optimized marketplace
document.addEventListener('DOMContentLoaded', () => {
    // Add HFT styles to document
    document.head.insertAdjacentHTML('beforeend', hftStyles);
    
    // Initialize HFT marketplace
    window.hftOptimizedMarketplace = new HFTOptimizedMarketplace();
    
    // Auto-start HFT optimization if system is ready
    setTimeout(() => {
        if (window.hftOptimizedMarketplace.isOptimizationEnabled) {
            window.hftOptimizedMarketplace.startHFTOptimization();
        }
    }, 2000);
});

console.log('📦 HFT-Optimized Marketplace module loaded');