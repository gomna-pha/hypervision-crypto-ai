/**
 * Image Agent - Visual Pattern Recognition for Market Microstructure
 * Analyzes orderbook heatmaps, chart patterns, and social sentiment images
 * Converts visual market data into embeddings and pattern recognition signals
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import { createCanvas, Canvas } from 'canvas';

export interface ImageData {
  image_type: 'orderbook_heatmap' | 'chart_pattern' | 'social_image';
  image_source: string;
  image_url?: string;
  image_dimensions: { width: number; height: number };
  processing_timestamp: number;
}

export interface VisualPattern {
  pattern_type: string;           // e.g., 'asymmetric_left_liquidity_drop', 'triangle_breakout'
  confidence: number;             // Pattern recognition confidence [0, 1]
  coordinates?: number[];         // Pattern location in image
  strength: number;               // Pattern strength/clarity [0, 1]
}

export interface ImageFeatures {
  visual_embedding_id: string;    // Hash/ID of visual embedding
  visual_pattern: string;         // Primary detected pattern
  visual_confidence: number;      // Overall visual analysis confidence [0, 1]
  visual_sentiment_score: number; // Sentiment extracted from visual [-1, 1]
  pattern_urgency: number;       // How urgent the visual signal is [0, 1]
  liquidity_asymmetry: number;   // Detected liquidity imbalance [-1, 1]
  support_resistance_strength: number; // Technical level strength [0, 1]
}

interface OrderBookSnapshot {
  bids: Array<{ price: number; size: number }>;
  asks: Array<{ price: number; size: number }>;
  timestamp: number;
  exchange: string;
  pair: string;
}

interface ChartData {
  timestamps: number[];
  prices: number[];
  volumes: number[];
  timeframe: string; // '1m', '5m', '15m', '1h'
}

export class ImageAgent extends BaseAgent {
  private imageHistory: Map<string, ImageData> = new Map();
  private patternHistory: VisualPattern[] = [];
  private embeddingCache: Map<string, number[]> = new Map();
  
  // Visual pattern recognition parameters
  private readonly HEATMAP_RESOLUTION = { width: 200, height: 150 };
  private readonly PATTERN_CONFIDENCE_THRESHOLD = 0.6;
  private readonly MAX_IMAGE_AGE_SEC = 60;
  
  // Pattern definitions for recognition
  private readonly KNOWN_PATTERNS = {
    'asymmetric_left_liquidity_drop': {
      description: 'Liquidity drops significantly on left side of orderbook',
      bullish_signal: 0.7
    },
    'asymmetric_right_liquidity_drop': {
      description: 'Liquidity drops significantly on right side of orderbook',
      bullish_signal: -0.7
    },
    'liquidity_wall_support': {
      description: 'Large bid wall creating support level',
      bullish_signal: 0.5
    },
    'liquidity_wall_resistance': {
      description: 'Large ask wall creating resistance level',
      bullish_signal: -0.5
    },
    'triangular_convergence': {
      description: 'Bid/ask spreads converging in triangle pattern',
      bullish_signal: 0.0
    },
    'flash_crash_signature': {
      description: 'Characteristic pattern before flash crashes',
      bullish_signal: -0.9
    }
  };

  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Generate orderbook heatmaps from current market data
      const heatmapData = await this.generateOrderbookHeatmaps();
      
      // Analyze chart patterns from price data
      const chartPatternData = await this.analyzeChartPatterns();
      
      // Process any social/meme images (if available)
      const socialImageData = await this.processSocialImages();

      // Combine all visual analysis
      const combinedFeatures = this.combineVisualFeatures(
        heatmapData, 
        chartPatternData, 
        socialImageData
      );

      // Calculate key signal from visual patterns
      const keySignal = this.calculateVisualSignal(combinedFeatures);

      // Calculate confidence based on pattern clarity and data quality
      const confidence = this.calculateVisualConfidence(combinedFeatures);

      return {
        agent_name: 'ImageAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...combinedFeatures,
          detected_patterns: this.patternHistory.slice(-5), // Last 5 patterns
          heatmap_generated: heatmapData.length > 0,
          chart_analysis_performed: chartPatternData.length > 0
        },
        metadata: {
          images_processed: heatmapData.length + chartPatternData.length + socialImageData.length,
          pattern_count: this.patternHistory.length,
          embedding_cache_size: this.embeddingCache.size
        }
      };

    } catch (error) {
      console.error('ImageAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Generate orderbook heatmaps for visual analysis
   */
  private async generateOrderbookHeatmaps(): Promise<ImageFeatures[]> {
    const features: ImageFeatures[] = [];
    
    try {
      // Simulate orderbook data (in production, get from Price Agent)
      const orderbooks = await this.getOrderbookSnapshots();
      
      for (const orderbook of orderbooks) {
        // Generate heatmap visualization
        const canvas = this.createOrderbookHeatmap(orderbook);
        
        // Analyze the generated heatmap
        const analysis = this.analyzeHeatmapPattern(canvas, orderbook);
        
        if (analysis) {
          features.push(analysis);
          
          // Store image data
          const imageData: ImageData = {
            image_type: 'orderbook_heatmap',
            image_source: `${orderbook.exchange}_${orderbook.pair}`,
            image_dimensions: this.HEATMAP_RESOLUTION,
            processing_timestamp: Date.now()
          };
          
          this.imageHistory.set(`heatmap_${orderbook.exchange}_${Date.now()}`, imageData);
        }
      }
    } catch (error) {
      console.warn('Heatmap generation failed:', error.message);
    }
    
    return features;
  }

  /**
   * Get orderbook snapshots (simulated for demo)
   */
  private async getOrderbookSnapshots(): Promise<OrderBookSnapshot[]> {
    // In production, this would fetch real orderbook data from exchanges
    // For demo, generate realistic orderbook data
    
    const exchanges = ['binance', 'coinbase'];
    const pairs = ['BTC-USDT', 'ETH-USDT'];
    const snapshots: OrderBookSnapshot[] = [];
    
    for (const exchange of exchanges) {
      for (const pair of pairs) {
        const basePrice = pair.includes('BTC') ? 42000 : 2500;
        const snapshot = this.generateMockOrderbook(exchange, pair, basePrice);
        snapshots.push(snapshot);
      }
    }
    
    return snapshots;
  }

  /**
   * Generate mock orderbook data for testing
   */
  private generateMockOrderbook(exchange: string, pair: string, basePrice: number): OrderBookSnapshot {
    const bids: Array<{ price: number; size: number }> = [];
    const asks: Array<{ price: number; size: number }> = [];
    
    // Generate realistic bid/ask levels
    for (let i = 0; i < 20; i++) {
      const bidPrice = basePrice - (i + 1) * (basePrice * 0.0001); // 0.01% increments
      const askPrice = basePrice + (i + 1) * (basePrice * 0.0001);
      
      // Simulate liquidity distribution with some randomness
      const baseLiquidity = 10 / (i + 1); // Decreasing liquidity with distance
      const bidSize = baseLiquidity * (0.5 + Math.random());
      const askSize = baseLiquidity * (0.5 + Math.random());
      
      bids.push({ price: bidPrice, size: bidSize });
      asks.push({ price: askPrice, size: askSize });
    }
    
    return {
      bids,
      asks,
      timestamp: Date.now(),
      exchange,
      pair
    };
  }

  /**
   * Create orderbook heatmap visualization
   */
  private createOrderbookHeatmap(orderbook: OrderBookSnapshot): Canvas {
    const canvas = createCanvas(this.HEATMAP_RESOLUTION.width, this.HEATMAP_RESOLUTION.height);
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Find price range
    const allPrices = [
      ...orderbook.bids.map(b => b.price),
      ...orderbook.asks.map(a => a.price)
    ];
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const priceRange = maxPrice - minPrice;
    
    // Find max size for normalization
    const allSizes = [
      ...orderbook.bids.map(b => b.size),
      ...orderbook.asks.map(a => a.size)
    ];
    const maxSize = Math.max(...allSizes);
    
    // Draw bids (green, left side)
    orderbook.bids.forEach((bid, index) => {
      const y = ((bid.price - minPrice) / priceRange) * canvas.height;
      const intensity = bid.size / maxSize;
      const width = (canvas.width / 2) * intensity;
      
      ctx.fillStyle = `rgba(0, 255, 0, ${intensity})`;
      ctx.fillRect(0, canvas.height - y - 2, width, 4);
    });
    
    // Draw asks (red, right side)
    orderbook.asks.forEach((ask, index) => {
      const y = ((ask.price - minPrice) / priceRange) * canvas.height;
      const intensity = ask.size / maxSize;
      const width = (canvas.width / 2) * intensity;
      
      ctx.fillStyle = `rgba(255, 0, 0, ${intensity})`;
      ctx.fillRect(canvas.width - width, canvas.height - y - 2, width, 4);
    });
    
    // Draw mid-price line
    const midPrice = (orderbook.bids[0].price + orderbook.asks[0].price) / 2;
    const midY = ((midPrice - minPrice) / priceRange) * canvas.height;
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height - midY);
    ctx.lineTo(canvas.width, canvas.height - midY);
    ctx.stroke();
    
    return canvas;
  }

  /**
   * Analyze heatmap pattern using computer vision techniques
   */
  private analyzeHeatmapPattern(canvas: Canvas, orderbook: OrderBookSnapshot): ImageFeatures | null {
    try {
      const ctx = canvas.getContext('2d');
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;
      
      // Analyze left (bid) vs right (ask) liquidity distribution
      const liquidityAnalysis = this.analyzeLiquidityAsymmetry(pixels, canvas.width, canvas.height);
      
      // Detect specific patterns
      const detectedPattern = this.detectVisualPatterns(liquidityAnalysis, orderbook);
      
      // Calculate visual sentiment from pattern
      const visualSentiment = this.calculateVisualSentiment(detectedPattern);
      
      // Generate embedding
      const embeddingId = this.generateVisualEmbedding(pixels);
      
      return {
        visual_embedding_id: embeddingId,
        visual_pattern: detectedPattern.pattern_type,
        visual_confidence: detectedPattern.confidence,
        visual_sentiment_score: visualSentiment,
        pattern_urgency: this.calculatePatternUrgency(detectedPattern),
        liquidity_asymmetry: liquidityAnalysis.asymmetry_score,
        support_resistance_strength: liquidityAnalysis.support_resistance_strength
      };
      
    } catch (error) {
      console.warn('Heatmap analysis failed:', error.message);
      return null;
    }
  }

  /**
   * Analyze liquidity asymmetry in the orderbook heatmap
   */
  private analyzeLiquidityAsymmetry(pixels: Uint8ClampedArray, width: number, height: number): {
    asymmetry_score: number;
    support_resistance_strength: number;
    liquidity_concentration: number;
  } {
    let leftLiquidity = 0;
    let rightLiquidity = 0;
    let totalLiquidity = 0;
    let maxRowLiquidity = 0;
    
    // Analyze each row (price level)
    for (let y = 0; y < height; y++) {
      let leftRowLiquidity = 0;
      let rightRowLiquidity = 0;
      
      for (let x = 0; x < width; x++) {
        const pixelIndex = (y * width + x) * 4;
        const red = pixels[pixelIndex];     // Ask side
        const green = pixels[pixelIndex + 1]; // Bid side
        
        const liquidity = (red + green) / 255; // Normalize to 0-1
        
        if (x < width / 2) {
          leftRowLiquidity += liquidity;
          leftLiquidity += liquidity;
        } else {
          rightRowLiquidity += liquidity;
          rightLiquidity += liquidity;
        }
        
        totalLiquidity += liquidity;
      }
      
      maxRowLiquidity = Math.max(maxRowLiquidity, leftRowLiquidity + rightRowLiquidity);
    }
    
    // Calculate asymmetry score (-1 = left heavy, +1 = right heavy)
    const asymmetryScore = totalLiquidity > 0 ? (rightLiquidity - leftLiquidity) / totalLiquidity : 0;
    
    // Calculate support/resistance strength based on liquidity concentration
    const supportResistanceStrength = maxRowLiquidity / (totalLiquidity / height);
    
    // Calculate overall liquidity concentration
    const liquidityConcentration = maxRowLiquidity / totalLiquidity;
    
    return {
      asymmetry_score: Math.max(-1, Math.min(1, asymmetryScore)),
      support_resistance_strength: Math.max(0, Math.min(1, supportResistanceStrength - 1)),
      liquidity_concentration: Math.max(0, Math.min(1, liquidityConcentration))
    };
  }

  /**
   * Detect specific visual patterns in the analysis
   */
  private detectVisualPatterns(liquidityAnalysis: any, orderbook: OrderBookSnapshot): VisualPattern {
    const patterns: VisualPattern[] = [];
    
    // Check for asymmetric liquidity patterns
    if (Math.abs(liquidityAnalysis.asymmetry_score) > 0.3) {
      if (liquidityAnalysis.asymmetry_score < -0.3) {
        patterns.push({
          pattern_type: 'asymmetric_left_liquidity_drop',
          confidence: Math.abs(liquidityAnalysis.asymmetry_score),
          strength: liquidityAnalysis.liquidity_concentration
        });
      } else {
        patterns.push({
          pattern_type: 'asymmetric_right_liquidity_drop',
          confidence: Math.abs(liquidityAnalysis.asymmetry_score),
          strength: liquidityAnalysis.liquidity_concentration
        });
      }
    }
    
    // Check for liquidity walls (strong support/resistance)
    if (liquidityAnalysis.support_resistance_strength > 0.7) {
      const topBid = orderbook.bids[0];
      const topAsk = orderbook.asks[0];
      
      if (topBid.size > topAsk.size * 2) {
        patterns.push({
          pattern_type: 'liquidity_wall_support',
          confidence: Math.min(1, topBid.size / topAsk.size / 3),
          strength: liquidityAnalysis.support_resistance_strength
        });
      } else if (topAsk.size > topBid.size * 2) {
        patterns.push({
          pattern_type: 'liquidity_wall_resistance',
          confidence: Math.min(1, topAsk.size / topBid.size / 3),
          strength: liquidityAnalysis.support_resistance_strength
        });
      }
    }
    
    // Check for triangular convergence
    const spread = orderbook.asks[0].price - orderbook.bids[0].price;
    const averagePrice = (orderbook.asks[0].price + orderbook.bids[0].price) / 2;
    const spreadPct = (spread / averagePrice) * 100;
    
    if (spreadPct < 0.05) { // Very tight spread
      patterns.push({
        pattern_type: 'triangular_convergence',
        confidence: Math.max(0.1, 1 - spreadPct * 20),
        strength: 1 - spreadPct * 10
      });
    }
    
    // Return the most confident pattern, or default
    if (patterns.length > 0) {
      patterns.sort((a, b) => b.confidence - a.confidence);
      return patterns[0];
    }
    
    return {
      pattern_type: 'normal_distribution',
      confidence: 0.5,
      strength: 0.5
    };
  }

  /**
   * Calculate visual sentiment from detected patterns
   */
  private calculateVisualSentiment(pattern: VisualPattern): number {
    const patternInfo = this.KNOWN_PATTERNS[pattern.pattern_type as keyof typeof this.KNOWN_PATTERNS];
    
    if (patternInfo) {
      return patternInfo.bullish_signal * pattern.confidence;
    }
    
    return 0; // Neutral for unknown patterns
  }

  /**
   * Calculate pattern urgency (how actionable the signal is)
   */
  private calculatePatternUrgency(pattern: VisualPattern): number {
    const urgentPatterns = ['flash_crash_signature', 'liquidity_wall_support', 'liquidity_wall_resistance'];
    
    if (urgentPatterns.includes(pattern.pattern_type)) {
      return pattern.confidence * pattern.strength;
    }
    
    return pattern.confidence * 0.5; // Lower urgency for other patterns
  }

  /**
   * Generate visual embedding hash/ID
   */
  private generateVisualEmbedding(pixels: Uint8ClampedArray): string {
    // Simple hash of pixel data for embedding ID
    let hash = 0;
    const step = Math.floor(pixels.length / 100); // Sample every 100th pixel for speed
    
    for (let i = 0; i < pixels.length; i += step) {
      hash = ((hash << 5) - hash + pixels[i]) & 0xffffffff;
    }
    
    return `embed_${Math.abs(hash).toString(16)}`;
  }

  /**
   * Analyze chart patterns (simplified implementation)
   */
  private async analyzeChartPatterns(): Promise<ImageFeatures[]> {
    const features: ImageFeatures[] = [];
    
    try {
      // In production, this would analyze actual chart data
      // For demo, create a simple chart pattern analysis
      
      const chartData = this.generateMockChartData();
      const pattern = this.detectChartPattern(chartData);
      
      if (pattern) {
        const embeddingId = this.generateChartEmbedding(chartData);
        
        features.push({
          visual_embedding_id: embeddingId,
          visual_pattern: pattern.pattern_type,
          visual_confidence: pattern.confidence,
          visual_sentiment_score: this.calculateVisualSentiment(pattern),
          pattern_urgency: this.calculatePatternUrgency(pattern),
          liquidity_asymmetry: 0, // Not applicable for chart patterns
          support_resistance_strength: pattern.strength
        });
      }
    } catch (error) {
      console.warn('Chart pattern analysis failed:', error.message);
    }
    
    return features;
  }

  /**
   * Generate mock chart data for testing
   */
  private generateMockChartData(): ChartData {
    const timestamps: number[] = [];
    const prices: number[] = [];
    const volumes: number[] = [];
    
    const now = Date.now();
    const basePrice = 42000;
    
    // Generate 24 hours of hourly data
    for (let i = 0; i < 24; i++) {
      timestamps.push(now - (23 - i) * 60 * 60 * 1000);
      
      // Simple trending price with noise
      const trend = Math.sin(i * 0.3) * 500;
      const noise = (Math.random() - 0.5) * 200;
      prices.push(basePrice + trend + noise);
      
      volumes.push(Math.random() * 1000 + 500);
    }
    
    return {
      timestamps,
      prices,
      volumes,
      timeframe: '1h'
    };
  }

  /**
   * Detect chart patterns from price data
   */
  private detectChartPattern(chartData: ChartData): VisualPattern | null {
    const prices = chartData.prices;
    
    if (prices.length < 10) return null;
    
    // Simple trend analysis
    const start = prices.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
    const end = prices.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const trendStrength = Math.abs(end - start) / start;
    
    if (trendStrength > 0.02) { // 2% trend
      const pattern_type = end > start ? 'bullish_trend' : 'bearish_trend';
      
      return {
        pattern_type,
        confidence: Math.min(1, trendStrength * 10),
        strength: Math.min(1, trendStrength * 5)
      };
    }
    
    return {
      pattern_type: 'sideways_consolidation',
      confidence: 0.6,
      strength: 0.4
    };
  }

  /**
   * Generate chart embedding
   */
  private generateChartEmbedding(chartData: ChartData): string {
    // Simple hash of price data
    const priceSum = chartData.prices.reduce((a, b) => a + b, 0);
    const volumeSum = chartData.volumes.reduce((a, b) => a + b, 0);
    const hash = Math.abs(priceSum + volumeSum * 1000);
    
    return `chart_${hash.toString(16)}`;
  }

  /**
   * Process social/meme images (placeholder)
   */
  private async processSocialImages(): Promise<ImageFeatures[]> {
    // In production, this would analyze social media images
    // For demo, return empty array
    return [];
  }

  /**
   * Combine features from all visual sources
   */
  private combineVisualFeatures(
    heatmapFeatures: ImageFeatures[],
    chartFeatures: ImageFeatures[],
    socialFeatures: ImageFeatures[]
  ): ImageFeatures {
    const allFeatures = [...heatmapFeatures, ...chartFeatures, ...socialFeatures];
    
    if (allFeatures.length === 0) {
      return this.getDefaultVisualFeatures();
    }
    
    // Weight heatmap analysis more heavily than chart analysis
    const weights = {
      heatmap: 0.6,
      chart: 0.3,
      social: 0.1
    };
    
    let weightedConfidence = 0;
    let weightedSentiment = 0;
    let weightedUrgency = 0;
    let totalWeight = 0;
    
    // Combine heatmap features
    if (heatmapFeatures.length > 0) {
      const avgHeatmapConfidence = heatmapFeatures.reduce((sum, f) => sum + f.visual_confidence, 0) / heatmapFeatures.length;
      const avgHeatmapSentiment = heatmapFeatures.reduce((sum, f) => sum + f.visual_sentiment_score, 0) / heatmapFeatures.length;
      const avgHeatmapUrgency = heatmapFeatures.reduce((sum, f) => sum + f.pattern_urgency, 0) / heatmapFeatures.length;
      
      weightedConfidence += avgHeatmapConfidence * weights.heatmap;
      weightedSentiment += avgHeatmapSentiment * weights.heatmap;
      weightedUrgency += avgHeatmapUrgency * weights.heatmap;
      totalWeight += weights.heatmap;
    }
    
    // Combine chart features
    if (chartFeatures.length > 0) {
      const avgChartConfidence = chartFeatures.reduce((sum, f) => sum + f.visual_confidence, 0) / chartFeatures.length;
      const avgChartSentiment = chartFeatures.reduce((sum, f) => sum + f.visual_sentiment_score, 0) / chartFeatures.length;
      const avgChartUrgency = chartFeatures.reduce((sum, f) => sum + f.pattern_urgency, 0) / chartFeatures.length;
      
      weightedConfidence += avgChartConfidence * weights.chart;
      weightedSentiment += avgChartSentiment * weights.chart;
      weightedUrgency += avgChartUrgency * weights.chart;
      totalWeight += weights.chart;
    }
    
    // Get the most confident pattern
    const bestPattern = allFeatures.reduce((best, current) => 
      current.visual_confidence > best.visual_confidence ? current : best
    );
    
    return {
      visual_embedding_id: bestPattern.visual_embedding_id,
      visual_pattern: bestPattern.visual_pattern,
      visual_confidence: totalWeight > 0 ? weightedConfidence / totalWeight : 0.5,
      visual_sentiment_score: totalWeight > 0 ? weightedSentiment / totalWeight : 0,
      pattern_urgency: totalWeight > 0 ? weightedUrgency / totalWeight : 0.5,
      liquidity_asymmetry: heatmapFeatures.length > 0 ? heatmapFeatures[0].liquidity_asymmetry : 0,
      support_resistance_strength: bestPattern.support_resistance_strength
    };
  }

  /**
   * Calculate overall visual signal strength
   */
  private calculateVisualSignal(features: ImageFeatures): number {
    // Combine sentiment and urgency with confidence weighting
    const signal = features.visual_sentiment_score * features.pattern_urgency;
    return signal * features.visual_confidence;
  }

  /**
   * Calculate overall visual confidence
   */
  private calculateVisualConfidence(features: ImageFeatures): number {
    let confidence = features.visual_confidence;
    
    // Adjust based on pattern type
    if (features.visual_pattern.includes('asymmetric') || features.visual_pattern.includes('wall')) {
      confidence *= 1.1; // Boost confidence for clear orderbook patterns
    }
    
    // Adjust based on support/resistance strength
    confidence *= (0.8 + features.support_resistance_strength * 0.2);
    
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Get default visual features when no analysis is available
   */
  private getDefaultVisualFeatures(): ImageFeatures {
    return {
      visual_embedding_id: 'default_embed',
      visual_pattern: 'normal_distribution',
      visual_confidence: 0.3,
      visual_sentiment_score: 0,
      pattern_urgency: 0.1,
      liquidity_asymmetry: 0,
      support_resistance_strength: 0.5
    };
  }

  /**
   * Get visual summary for debugging
   */
  getVisualSummary(): string {
    const recentPatterns = this.patternHistory.slice(-3);
    if (recentPatterns.length === 0) {
      return 'No visual patterns detected';
    }
    
    const summaries = recentPatterns.map(pattern => 
      `${pattern.pattern_type} (conf: ${pattern.confidence.toFixed(2)})`
    );
    
    return summaries.join(', ');
  }
}

/**
 * Factory function to create ImageAgent with config
 */
export function createImageAgent(): ImageAgent {
  const config: AgentConfig = {
    name: 'image',
    enabled: true,
    polling_interval_ms: 60 * 1000, // 1 minute
    confidence_min: 0.6,
    data_age_max_ms: 60 * 1000, // 1 minute max age
    retry_attempts: 2,
    retry_backoff_ms: 3000
  };

  return new ImageAgent(config);
}

// Export for testing
export { ImageAgent as default };