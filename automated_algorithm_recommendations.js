/**
 * AUTOMATED ALGORITHM RECOMMENDATION SYSTEM
 * Intelligent system that analyzes investor profiles and automatically recommends
 * optimal algorithm packages based on risk tolerance, capital, and performance goals
 */

class AlgorithmRecommendationEngine {
    constructor() {
        this.riskProfiles = this.initializeRiskProfiles();
        this.algorithmPackages = this.initializeAlgorithmPackages();
        this.performanceTargets = this.initializePerformanceTargets();
        this.recommendations = new Map();
    }

    initializeRiskProfiles() {
        return {
            conservative: {
                name: 'Conservative',
                description: 'Low risk, steady returns',
                minSharpe: 2.0,
                maxDrawdown: 0.05,
                targetAPY: { min: 0.15, max: 0.20 },
                volatilityTolerance: 'low',
                riskScore: 1
            },
            balanced: {
                name: 'Balanced',
                description: 'Moderate risk/reward',
                minSharpe: 1.5,
                maxDrawdown: 0.10,
                targetAPY: { min: 0.25, max: 0.35 },
                volatilityTolerance: 'medium',
                riskScore: 2
            },
            aggressive: {
                name: 'Aggressive',
                description: 'High risk, high reward',
                minSharpe: 1.0,
                maxDrawdown: 0.20,
                targetAPY: { min: 0.40, max: 0.60 },
                volatilityTolerance: 'high',
                riskScore: 3
            }
        };
    }

    initializeAlgorithmPackages() {
        return {
            // TRADITIONAL STRATEGY PACKAGES
            conservativePackage: {
                id: 'conservative-package',
                name: 'Conservative Growth Package',
                description: 'Low-risk algorithms focused on steady, consistent returns',
                price: 1499,
                monthlyFee: 249,
                algorithms: [
                    'statistical-pairs',      // Sharpe: 2.89
                    'options-volatility'      // Sharpe: 3.21
                ],
                targetProfile: 'conservative',
                expectedAPY: 0.18,
                maxDrawdown: 0.041,
                minCapital: 10000,
                features: [
                    'Conservative risk management',
                    'Steady monthly returns',
                    'Low volatility strategies',
                    'Capital preservation focus',
                    'Automated rebalancing'
                ]
            },

            balancedPackage: {
                id: 'balanced-package', 
                name: 'Balanced Growth Package',
                description: 'Optimal risk/reward with diversified strategies',
                price: 2199,
                monthlyFee: 349,
                algorithms: [
                    'triangular-arbitrage',   // Sharpe: 3.42
                    'news-sentiment',         // Sharpe: 2.54
                    'crypto-defi'            // Sharpe: 2.97
                ],
                targetProfile: 'balanced',
                expectedAPY: 0.29,
                maxDrawdown: 0.057,
                minCapital: 50000,
                features: [
                    'Multi-strategy diversification',
                    'Real-time sentiment analysis',
                    'Cross-market arbitrage',
                    'Adaptive risk management',
                    'Performance optimization'
                ]
            },

            aggressivePackage: {
                id: 'aggressive-package',
                name: 'Aggressive Alpha Package', 
                description: 'High-performance algorithms for maximum returns',
                price: 2799,
                monthlyFee: 499,
                algorithms: [
                    'hft-latency',           // Sharpe: 4.17
                    'triangular-arbitrage',   // Sharpe: 3.42
                    'options-volatility',     // Sharpe: 3.21
                    'crypto-defi'            // Sharpe: 2.97
                ],
                targetProfile: 'aggressive',
                expectedAPY: 0.48,
                maxDrawdown: 0.12,
                minCapital: 100000,
                features: [
                    'Ultra-high-frequency trading',
                    'Maximum alpha generation',
                    'Advanced arbitrage strategies',
                    'Institutional-grade execution',
                    'Dynamic portfolio optimization'
                ]
            },

            // SPECIALIZED HFT PACKAGES
            arbitrageMasterPackage: {
                id: 'arbitrage-master-package',
                name: 'Arbitrage Master Suite',
                description: 'Complete arbitrage-focused algorithm collection',
                price: 3499,
                monthlyFee: 599,
                algorithms: [
                    'triangular-arbitrage',   // Cross-exchange
                    'hft-latency',           // Ultra-low latency
                    'statistical-pairs',      // Statistical arbitrage
                    'news-sentiment',         // Sentiment arbitrage
                    'options-volatility',     // Volatility arbitrage
                    'crypto-defi'            // DeFi arbitrage
                ],
                targetProfile: 'professional',
                expectedAPY: 0.52,
                maxDrawdown: 0.089,
                minCapital: 250000,
                features: [
                    'Complete arbitrage coverage',
                    'Multi-timeframe execution',
                    'Cross-asset opportunities',
                    'Advanced risk controls',
                    'Institutional performance'
                ]
            },

            institutionalPackage: {
                id: 'institutional-package',
                name: 'Institutional Elite Package',
                description: 'Premium institutional-grade complete solution',
                price: 4999,
                monthlyFee: 799,
                algorithms: [
                    'hft-latency',           // Premium HFT
                    'triangular-arbitrage',   // Cross-exchange arbitrage
                    'statistical-pairs',      // Pairs trading
                    'news-sentiment',         // Sentiment analysis
                    'options-volatility',     // Options strategies
                    'crypto-defi'            // DeFi optimization
                ],
                targetProfile: 'institutional',
                expectedAPY: 0.58,
                maxDrawdown: 0.078,
                minCapital: 1000000,
                features: [
                    'Complete algorithm suite',
                    'Dedicated account management',
                    'Custom parameter tuning',
                    'Priority execution access',
                    'Advanced reporting & analytics'
                ]
            }
        };
    }

    initializePerformanceTargets() {
        return {
            capital_preservation: {
                name: 'Capital Preservation',
                targetAPY: 0.08,
                maxDrawdown: 0.03,
                recommendedAlgorithms: ['statistical-pairs']
            },
            steady_growth: {
                name: 'Steady Growth',
                targetAPY: 0.18,
                maxDrawdown: 0.05,
                recommendedAlgorithms: ['statistical-pairs', 'options-volatility']
            },
            balanced_growth: {
                name: 'Balanced Growth', 
                targetAPY: 0.29,
                maxDrawdown: 0.08,
                recommendedAlgorithms: ['triangular-arbitrage', 'news-sentiment', 'crypto-defi']
            },
            aggressive_growth: {
                name: 'Aggressive Growth',
                targetAPY: 0.45,
                maxDrawdown: 0.15,
                recommendedAlgorithms: ['hft-latency', 'triangular-arbitrage', 'options-volatility']
            },
            maximum_alpha: {
                name: 'Maximum Alpha Generation',
                targetAPY: 0.60,
                maxDrawdown: 0.20,
                recommendedAlgorithms: ['hft-latency', 'triangular-arbitrage', 'options-volatility', 'crypto-defi']
            }
        };
    }

    // MAIN RECOMMENDATION ENGINE
    generateRecommendations(investorProfile) {
        const recommendations = {
            primary: null,
            alternatives: [],
            customBundle: null,
            reasoning: [],
            expectedOutcomes: {}
        };

        // Analyze investor profile
        const analysis = this.analyzeInvestorProfile(investorProfile);
        
        // Generate primary recommendation
        recommendations.primary = this.selectPrimaryPackage(analysis);
        
        // Generate alternatives
        recommendations.alternatives = this.selectAlternativePackages(analysis);
        
        // Create custom bundle if needed
        recommendations.customBundle = this.createCustomBundle(analysis);
        
        // Generate reasoning
        recommendations.reasoning = this.generateReasoning(analysis, recommendations);
        
        // Calculate expected outcomes
        recommendations.expectedOutcomes = this.calculateExpectedOutcomes(analysis, recommendations.primary);

        return recommendations;
    }

    analyzeInvestorProfile(profile) {
        return {
            riskProfile: this.determineRiskProfile(profile),
            capitalAvailable: profile.capital || 10000,
            accountTier: profile.accountType || 'starter',
            experienceLevel: profile.experienceLevel || 'beginner',
            timeHorizon: profile.timeHorizon || 'medium',
            performanceGoal: profile.performanceGoal || 'balanced_growth',
            riskTolerance: profile.riskTolerance || 2,
            diversificationPreference: profile.diversificationPreference || 'medium'
        };
    }

    determineRiskProfile(profile) {
        const capital = profile.capital || 10000;
        const riskTolerance = profile.riskTolerance || 2;
        const accountType = profile.accountType || 'starter';

        // Auto-determine risk profile based on multiple factors
        if (capital < 50000 && riskTolerance <= 2) {
            return 'conservative';
        } else if (capital >= 50000 && capital < 250000 && riskTolerance <= 3) {
            return 'balanced';
        } else if (capital >= 250000 || riskTolerance > 3) {
            return 'aggressive';
        }

        return 'balanced'; // Default
    }

    selectPrimaryPackage(analysis) {
        const packages = Object.values(this.algorithmPackages);
        
        // Filter packages by minimum capital requirement
        const affordablePackages = packages.filter(pkg => 
            analysis.capitalAvailable >= pkg.minCapital
        );

        // Score each package based on investor profile
        const scoredPackages = affordablePackages.map(pkg => ({
            ...pkg,
            score: this.calculatePackageScore(pkg, analysis)
        }));

        // Sort by score and return the best match
        scoredPackages.sort((a, b) => b.score - a.score);
        
        return scoredPackages[0] || null;
    }

    calculatePackageScore(package_, analysis) {
        let score = 0;
        
        // Risk profile match (40% weight)
        const riskProfileMatch = this.getRiskProfileMatch(package_, analysis);
        score += riskProfileMatch * 40;
        
        // Performance target match (30% weight)
        const performanceMatch = this.getPerformanceMatch(package_, analysis);
        score += performanceMatch * 30;
        
        // Capital efficiency (20% weight)
        const capitalEfficiency = this.getCapitalEfficiency(package_, analysis);
        score += capitalEfficiency * 20;
        
        // Diversification score (10% weight) 
        const diversificationScore = package_.algorithms.length * 2;
        score += Math.min(diversificationScore, 10);

        return score;
    }

    getRiskProfileMatch(package_, analysis) {
        const targetProfile = this.riskProfiles[analysis.riskProfile];
        
        if (package_.expectedAPY >= targetProfile.targetAPY.min && 
            package_.expectedAPY <= targetProfile.targetAPY.max &&
            package_.maxDrawdown <= targetProfile.maxDrawdown) {
            return 10; // Perfect match
        }
        
        // Calculate partial match based on distance from target
        const apyDistance = Math.abs(package_.expectedAPY - 
            (targetProfile.targetAPY.min + targetProfile.targetAPY.max) / 2);
        const ddDistance = Math.abs(package_.maxDrawdown - targetProfile.maxDrawdown);
        
        return Math.max(0, 10 - (apyDistance * 20) - (ddDistance * 50));
    }

    getPerformanceMatch(package_, analysis) {
        const performanceTarget = this.performanceTargets[analysis.performanceGoal];
        
        if (!performanceTarget) return 5; // Neutral score
        
        const apyDistance = Math.abs(package_.expectedAPY - performanceTarget.targetAPY);
        const ddDistance = Math.abs(package_.maxDrawdown - performanceTarget.maxDrawdown);
        
        return Math.max(0, 10 - (apyDistance * 15) - (ddDistance * 30));
    }

    getCapitalEfficiency(package_, analysis) {
        const capitalRatio = package_.minCapital / analysis.capitalAvailable;
        
        if (capitalRatio <= 0.1) return 10;  // Uses < 10% of capital
        if (capitalRatio <= 0.2) return 8;   // Uses < 20% of capital  
        if (capitalRatio <= 0.5) return 6;   // Uses < 50% of capital
        if (capitalRatio <= 0.8) return 4;   // Uses < 80% of capital
        if (capitalRatio <= 1.0) return 2;   // Uses all available capital
        
        return 0; // Requires more capital than available
    }

    selectAlternativePackages(analysis) {
        const packages = Object.values(this.algorithmPackages);
        
        // Get all affordable packages
        const affordablePackages = packages.filter(pkg => 
            analysis.capitalAvailable >= pkg.minCapital
        );

        // Score and sort
        const scoredPackages = affordablePackages.map(pkg => ({
            ...pkg,
            score: this.calculatePackageScore(pkg, analysis)
        })).sort((a, b) => b.score - a.score);

        // Return top 3 alternatives (excluding the primary recommendation)
        return scoredPackages.slice(1, 4);
    }

    createCustomBundle(analysis) {
        // Create a custom algorithm bundle based on specific investor needs
        const availableAlgorithms = window.algorithmicMarketplace ? 
            Array.from(window.algorithmicMarketplace.algorithms.values()) : [];
        
        if (availableAlgorithms.length === 0) return null;

        // Filter algorithms based on risk profile
        const suitableAlgorithms = availableAlgorithms.filter(algo => {
            const riskProfile = this.riskProfiles[analysis.riskProfile];
            return algo.performance.sharpeRatio >= riskProfile.minSharpe &&
                   algo.performance.maxDrawdown <= riskProfile.maxDrawdown;
        });

        if (suitableAlgorithms.length < 2) return null;

        // Select optimal combination
        const selectedAlgorithms = this.optimizeAlgorithmCombination(
            suitableAlgorithms, 
            analysis
        );

        // Calculate bundle pricing
        const totalPrice = selectedAlgorithms.reduce((sum, algo) => sum + algo.price, 0);
        const totalMonthlyFee = selectedAlgorithms.reduce((sum, algo) => sum + algo.monthlyFee, 0);
        
        // Apply bundle discount (10-20% based on number of algorithms)
        const discountRate = Math.min(0.20, selectedAlgorithms.length * 0.03);
        const bundlePrice = Math.floor(totalPrice * (1 - discountRate));
        const bundleMonthlyFee = Math.floor(totalMonthlyFee * (1 - discountRate * 0.5));

        return {
            id: 'custom-bundle',
            name: 'Custom Optimized Bundle',
            description: `Personalized algorithm selection for ${analysis.riskProfile} investors`,
            algorithms: selectedAlgorithms.map(a => a.id),
            price: bundlePrice,
            monthlyFee: bundleMonthlyFee,
            originalPrice: totalPrice,
            savings: totalPrice - bundlePrice,
            expectedAPY: this.calculateBundleAPY(selectedAlgorithms),
            maxDrawdown: this.calculateBundleMaxDrawdown(selectedAlgorithms),
            features: [
                `${selectedAlgorithms.length} optimized algorithms`,
                `${Math.round(discountRate * 100)}% bundle discount`,
                'Personalized risk management',
                'Custom performance targets',
                'Adaptive allocation'
            ]
        };
    }

    optimizeAlgorithmCombination(algorithms, analysis) {
        const maxAlgorithms = this.getMaxAlgorithmsForProfile(analysis);
        
        // Sort by Sharpe ratio and performance fit
        const sortedAlgorithms = algorithms.sort((a, b) => {
            const aScore = a.performance.sharpeRatio * 
                (1 - Math.abs(a.performance.dailyReturn * 365 - 
                    this.riskProfiles[analysis.riskProfile].targetAPY.min));
            const bScore = b.performance.sharpeRatio * 
                (1 - Math.abs(b.performance.dailyReturn * 365 - 
                    this.riskProfiles[analysis.riskProfile].targetAPY.min));
            return bScore - aScore;
        });

        // Select top algorithms with diversification
        const selected = [];
        const usedCategories = new Set();
        
        for (const algorithm of sortedAlgorithms) {
            if (selected.length >= maxAlgorithms) break;
            
            // Prefer diversification across categories
            if (!usedCategories.has(algorithm.category) || usedCategories.size >= 3) {
                selected.push(algorithm);
                usedCategories.add(algorithm.category);
            }
        }

        return selected;
    }

    getMaxAlgorithmsForProfile(analysis) {
        if (analysis.capitalAvailable >= 1000000) return 6;
        if (analysis.capitalAvailable >= 250000) return 4;
        if (analysis.capitalAvailable >= 100000) return 3;
        return 2;
    }

    calculateBundleAPY(algorithms) {
        // Weighted average of algorithm returns
        const totalWeight = algorithms.length;
        const weightedReturn = algorithms.reduce((sum, algo) => 
            sum + (algo.performance.dailyReturn * 365), 0) / totalWeight;
        return weightedReturn;
    }

    calculateBundleMaxDrawdown(algorithms) {
        // Conservative estimate - take the highest individual drawdown
        return Math.max(...algorithms.map(algo => algo.performance.maxDrawdown));
    }

    generateReasoning(analysis, recommendations) {
        const reasoning = [];
        const primary = recommendations.primary;
        
        if (!primary) {
            reasoning.push('Insufficient capital for recommended packages. Consider increasing investment capital.');
            return reasoning;
        }

        // Risk profile reasoning
        reasoning.push(
            `Selected ${primary.name} based on your ${analysis.riskProfile} risk profile and $${analysis.capitalAvailable.toLocaleString()} capital.`
        );

        // Performance reasoning
        reasoning.push(
            `This package targets ${(primary.expectedAPY * 100).toFixed(1)}% annual returns with maximum ${(primary.maxDrawdown * 100).toFixed(1)}% drawdown, matching your risk tolerance.`
        );

        // Algorithm diversity reasoning
        reasoning.push(
            `Includes ${primary.algorithms.length} complementary algorithms providing diversification across ${new Set(primary.algorithms.map(id => {
                const algo = window.algorithmicMarketplace?.algorithms.get(id);
                return algo?.category || 'strategies';
            })).size} different strategy types.`
        );

        // Capital efficiency reasoning
        const capitalUsage = (primary.price / analysis.capitalAvailable) * 100;
        reasoning.push(
            `Uses ${capitalUsage.toFixed(1)}% of available capital, leaving sufficient reserves for risk management.`
        );

        return reasoning;
    }

    calculateExpectedOutcomes(analysis, primaryPackage) {
        if (!primaryPackage) return {};

        const annualReturn = primaryPackage.expectedAPY * analysis.capitalAvailable;
        const monthlyReturn = annualReturn / 12;
        const maxLoss = primaryPackage.maxDrawdown * analysis.capitalAvailable;

        return {
            expectedAnnualReturn: annualReturn,
            expectedMonthlyReturn: monthlyReturn,
            potentialMaxLoss: maxLoss,
            breakevenTime: primaryPackage.price / monthlyReturn, // months to break even
            riskAdjustedReturn: annualReturn / Math.max(primaryPackage.maxDrawdown, 0.01),
            sharpeEstimate: primaryPackage.expectedAPY / Math.max(primaryPackage.maxDrawdown, 0.01)
        };
    }

    // PUBLIC API METHODS
    getRecommendationForInvestor(investorProfile) {
        return this.generateRecommendations(investorProfile);
    }

    getAllPackages() {
        return this.algorithmPackages;
    }

    getPackageDetails(packageId) {
        return this.algorithmPackages[packageId] || null;
    }

    updateInvestorRecommendation(investorId, recommendation) {
        this.recommendations.set(investorId, recommendation);
        return recommendation;
    }

    getStoredRecommendation(investorId) {
        return this.recommendations.get(investorId) || null;
    }
}

// AUTOMATED RECOMMENDATION UI INTEGRATION
class RecommendationUI {
    constructor(recommendationEngine) {
        this.engine = recommendationEngine;
        this.currentRecommendations = null;
    }

    displayRecommendations(investorProfile) {
        this.currentRecommendations = this.engine.getRecommendationForInvestor(investorProfile);
        
        // Create recommendation display
        this.createRecommendationModal();
        this.populateRecommendationModal(this.currentRecommendations);
        this.showRecommendationModal();
    }

    createRecommendationModal() {
        // Remove existing modal if present
        const existingModal = document.getElementById('recommendation-modal');
        if (existingModal) {
            existingModal.remove();
        }

        const modalHTML = `
            <div id="recommendation-modal" class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
                <div class="bg-gradient-to-br from-cream-50 to-cream-100 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                    <div class="p-8">
                        <div class="text-center mb-6">
                            <h2 class="text-3xl font-bold text-gray-900 mb-2">Personalized Algorithm Recommendations</h2>
                            <p class="text-lg text-gray-600">AI-powered analysis of your investment profile</p>
                        </div>
                        
                        <div id="recommendation-content">
                            <!-- Content will be populated by JavaScript -->
                        </div>
                        
                        <div class="flex justify-center space-x-4 mt-8">
                            <button id="accept-primary-recommendation" class="px-8 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg font-semibold hover:from-green-700 hover:to-green-800 transition-all shadow-lg">
                                Purchase Recommended Package
                            </button>
                            <button id="view-alternatives" class="px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-blue-800 transition-all shadow-lg">
                                View Alternatives
                            </button>
                            <button id="close-recommendations" class="px-8 py-3 bg-gray-500 text-white rounded-lg font-semibold hover:bg-gray-600 transition-all">
                                Close
                            </button>
                        </div>
                        
                        <button id="close-recommendation-modal" class="absolute top-4 right-4 text-gray-500 hover:text-gray-700">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.setupRecommendationEventListeners();
    }

    populateRecommendationModal(recommendations) {
        const content = document.getElementById('recommendation-content');
        if (!content || !recommendations.primary) return;

        const primary = recommendations.primary;
        const outcomes = recommendations.expectedOutcomes;

        content.innerHTML = `
            <!-- Primary Recommendation -->
            <div class="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-xl border-2 border-green-300 mb-6">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h3 class="text-2xl font-bold text-gray-900">${primary.name}</h3>
                        <p class="text-gray-600">${primary.description}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-3xl font-bold text-green-600">$${primary.price.toLocaleString()}</div>
                        <div class="text-sm text-gray-500">+$${primary.monthlyFee}/month</div>
                        ${primary.savings ? `<div class="text-sm text-green-600 font-bold">Save $${primary.savings.toLocaleString()}</div>` : ''}
                    </div>
                </div>
                
                <!-- Performance Metrics -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div class="text-center p-3 bg-white rounded-lg">
                        <div class="text-lg font-bold text-green-600">${(primary.expectedAPY * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">Expected APY</div>
                    </div>
                    <div class="text-center p-3 bg-white rounded-lg">
                        <div class="text-lg font-bold text-blue-600">${(primary.maxDrawdown * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">Max Drawdown</div>
                    </div>
                    <div class="text-center p-3 bg-white rounded-lg">
                        <div class="text-lg font-bold text-purple-600">${primary.algorithms.length}</div>
                        <div class="text-xs text-gray-500">Algorithms</div>
                    </div>
                    <div class="text-center p-3 bg-white rounded-lg">
                        <div class="text-lg font-bold text-orange-600">${outcomes.sharpeEstimate ? outcomes.sharpeEstimate.toFixed(2) : 'N/A'}</div>
                        <div class="text-xs text-gray-500">Est. Sharpe</div>
                    </div>
                </div>

                <!-- Expected Outcomes -->
                <div class="bg-white p-4 rounded-lg mb-4">
                    <h4 class="font-semibold text-gray-900 mb-3">Expected Financial Outcomes</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Expected Annual Return:</span>
                            <span class="font-bold text-green-600">$${outcomes.expectedAnnualReturn?.toLocaleString() || '0'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Expected Monthly Return:</span>
                            <span class="font-bold text-green-600">$${outcomes.expectedMonthlyReturn?.toLocaleString() || '0'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Potential Max Loss:</span>
                            <span class="font-bold text-red-600">$${outcomes.potentialMaxLoss?.toLocaleString() || '0'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Breakeven Time:</span>
                            <span class="font-bold text-blue-600">${outcomes.breakevenTime ? Math.ceil(outcomes.breakevenTime) : 'N/A'} months</span>
                        </div>
                    </div>
                </div>

                <!-- Features -->
                <div class="bg-white p-4 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-3">Package Features</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                        ${primary.features.map(feature => `
                            <div class="flex items-center space-x-2">
                                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                                <span class="text-sm text-gray-700">${feature}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            <!-- AI Reasoning -->
            <div class="bg-blue-50 p-4 rounded-lg mb-6">
                <h4 class="font-semibold text-blue-900 mb-3">Why This Package Was Selected</h4>
                <ul class="space-y-2">
                    ${recommendations.reasoning.map(reason => `
                        <li class="flex items-start space-x-2">
                            <div class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                            <span class="text-sm text-blue-800">${reason}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>

            <!-- Alternative Options -->
            ${recommendations.alternatives.length > 0 ? `
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-3">Alternative Options</h4>
                    <div class="grid grid-cols-1 md:grid-cols-${Math.min(recommendations.alternatives.length, 2)} gap-4">
                        ${recommendations.alternatives.slice(0, 2).map(alt => `
                            <div class="bg-white p-4 rounded-lg border">
                                <h5 class="font-semibold text-gray-900">${alt.name}</h5>
                                <p class="text-sm text-gray-600 mb-2">${alt.description}</p>
                                <div class="flex justify-between text-sm">
                                    <span class="text-gray-600">Price: $${alt.price.toLocaleString()}</span>
                                    <span class="text-green-600">${(alt.expectedAPY * 100).toFixed(1)}% APY</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    setupRecommendationEventListeners() {
        // Accept primary recommendation
        document.getElementById('accept-primary-recommendation')?.addEventListener('click', () => {
            this.acceptPrimaryRecommendation();
        });

        // View alternatives
        document.getElementById('view-alternatives')?.addEventListener('click', () => {
            this.showAlternatives();
        });

        // Close modal
        document.getElementById('close-recommendations')?.addEventListener('click', () => {
            this.closeRecommendationModal();
        });
        
        document.getElementById('close-recommendation-modal')?.addEventListener('click', () => {
            this.closeRecommendationModal();
        });

        // Close on backdrop click
        document.getElementById('recommendation-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'recommendation-modal') {
                this.closeRecommendationModal();
            }
        });
    }

    acceptPrimaryRecommendation() {
        if (!this.currentRecommendations?.primary) return;

        const package_ = this.currentRecommendations.primary;
        
        // Simulate package purchase
        if (window.investorSystem && window.investorSystem.isLoggedIn()) {
            const portfolio = window.algorithmicMarketplace?.getPortfolio();
            if (portfolio && portfolio.balance >= package_.price) {
                // Process purchase
                portfolio.balance -= package_.price;
                package_.algorithms.forEach(algorithmId => {
                    portfolio.ownedAlgorithms.add(algorithmId);
                });
                
                showNotification(`Successfully purchased ${package_.name}!`, 'success');
                this.closeRecommendationModal();
                
                // Update UI displays
                if (window.updateUIDisplays) {
                    window.updateUIDisplays();
                }
            } else {
                showNotification('Insufficient balance to purchase this package', 'error');
            }
        } else {
            showNotification('Please log in to purchase packages', 'info');
        }
    }

    showAlternatives() {
        // Display detailed alternatives view
        const content = document.getElementById('recommendation-content');
        if (!content || !this.currentRecommendations?.alternatives) return;

        content.innerHTML = `
            <div class="text-center mb-6">
                <h3 class="text-2xl font-bold text-gray-900">Alternative Package Options</h3>
                <p class="text-gray-600">Compare different packages based on your profile</p>
            </div>
            
            <div class="grid grid-cols-1 gap-6">
                ${this.currentRecommendations.alternatives.map(alt => `
                    <div class="bg-white p-6 rounded-xl border shadow-sm">
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <h4 class="text-xl font-bold text-gray-900">${alt.name}</h4>
                                <p class="text-gray-600">${alt.description}</p>
                            </div>
                            <div class="text-right">
                                <div class="text-2xl font-bold text-blue-600">$${alt.price.toLocaleString()}</div>
                                <div class="text-sm text-gray-500">+$${alt.monthlyFee}/month</div>
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-3 gap-4 mb-4">
                            <div class="text-center p-2 bg-gray-50 rounded">
                                <div class="font-bold text-green-600">${(alt.expectedAPY * 100).toFixed(1)}%</div>
                                <div class="text-xs text-gray-500">Expected APY</div>
                            </div>
                            <div class="text-center p-2 bg-gray-50 rounded">
                                <div class="font-bold text-red-600">${(alt.maxDrawdown * 100).toFixed(1)}%</div>
                                <div class="text-xs text-gray-500">Max Drawdown</div>
                            </div>
                            <div class="text-center p-2 bg-gray-50 rounded">
                                <div class="font-bold text-purple-600">${alt.algorithms.length}</div>
                                <div class="text-xs text-gray-500">Algorithms</div>
                            </div>
                        </div>
                        
                        <button class="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors" 
                                onclick="selectAlternativePackage('${alt.id}')">
                            Select This Package
                        </button>
                    </div>
                `).join('')}
            </div>
            
            <div class="text-center mt-6">
                <button onclick="window.recommendationUI.displayRecommendations(window.currentInvestorProfile)" 
                        class="px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors">
                    Back to Primary Recommendation
                </button>
            </div>
        `;
    }

    showRecommendationModal() {
        const modal = document.getElementById('recommendation-modal');
        if (modal) {
            modal.classList.remove('hidden');
        }
    }

    closeRecommendationModal() {
        const modal = document.getElementById('recommendation-modal');
        if (modal) {
            modal.classList.add('hidden');
            setTimeout(() => modal.remove(), 300);
        }
    }
}

// GLOBAL FUNCTIONS
function selectAlternativePackage(packageId) {
    const engine = window.recommendationEngine;
    const package_ = Object.values(engine.algorithmPackages).find(pkg => pkg.id === packageId);
    
    if (package_) {
        // Simulate purchase of alternative package
        if (window.investorSystem && window.investorSystem.isLoggedIn()) {
            const portfolio = window.algorithmicMarketplace?.getPortfolio();
            if (portfolio && portfolio.balance >= package_.price) {
                portfolio.balance -= package_.price;
                package_.algorithms.forEach(algorithmId => {
                    portfolio.ownedAlgorithms.add(algorithmId);
                });
                
                showNotification(`Successfully purchased ${package_.name}!`, 'success');
                window.recommendationUI.closeRecommendationModal();
                
                if (window.updateUIDisplays) {
                    window.updateUIDisplays();
                }
            } else {
                showNotification('Insufficient balance to purchase this package', 'error');
            }
        }
    }
}

// INITIALIZE RECOMMENDATION SYSTEM
function initializeRecommendationSystem() {
    window.recommendationEngine = new AlgorithmRecommendationEngine();
    window.recommendationUI = new RecommendationUI(window.recommendationEngine);
    
    console.log('Automated Algorithm Recommendation System initialized');
    
    // Add recommendation trigger to registration process
    enhanceRegistrationWithRecommendations();
}

function enhanceRegistrationWithRecommendations() {
    // Override the existing registration handler to include recommendations
    const originalRegistrationHandler = window.handleTradingRegistration;
    
    window.handleTradingRegistration = function(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const userData = Object.fromEntries(formData);
        
        // Store user profile for recommendations
        window.currentInvestorProfile = userData;
        
        // Process registration
        if (window.investorSystem) {
            const success = window.investorSystem.register(userData);
            if (success) {
                showNotification('Account created successfully!', 'success');
                
                // Close registration modal
                const regModal = document.getElementById('registration-modal');
                if (regModal) regModal.classList.add('hidden');
                
                // Show automated recommendations after 1 second
                setTimeout(() => {
                    window.recommendationUI.displayRecommendations(userData);
                }, 1000);
                
                if (window.updateUIDisplays) {
                    window.updateUIDisplays();
                }
            } else {
                showNotification('Registration failed. Please try again.', 'error');
            }
        }
    };
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeRecommendationSystem);
} else {
    initializeRecommendationSystem();
}

// Export for external use
window.initializeRecommendationSystem = initializeRecommendationSystem;