/**
 * Image Agent - Visual Pattern Recognition for Market Microstructure
 * Generates and analyzes orderbook heatmaps, chart patterns, and visual sentiment
 * Converts visual market microstructure into embeddings and pattern flags
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import { createCanvas, CanvasRenderingContext2D, Canvas } from 'canvas';
import axios from 'axios';

export interface ImageData {
  image_type: 'orderbook_heatmap' | 'chart_pattern' | 'social_image';
  pair: string;
  exchange?: string;
  visual_pattern: string;
  visual_confidence: number;
  visual_sentiment_score: number;
  embedding_id: string;
  generated_at: number;
  image_data?: string; // Base64 encoded image data
}

export interface VisualFeatures {
  orderbook_asymmetry: number;       // Visual asymmetry in orderbook
  liquidity_concentration: number;   // How concentrated liquidity appears
  price_action_strength: number;    // Strength of visual price patterns
  volatility_visualization: number; // Visual volatility indicators
  pattern_consistency: number;      // Consistency across multiple patterns
}

export interface OrderbookHeatmap {
  canvas: Canvas;
  width: number;
  height: number;
  price_levels: number[];
  bid_volumes: number[];
  ask_volumes: number[];
  max_volume: number;
  asymmetry_score: number;
}

export interface PatternRecognition {
  pattern_type: 'bull_flag' | 'bear_flag' | 'triangle' | 'channel' | 'breakout' | 'reversal' | 'unknown';
  confidence: number;
  strength: number;
  direction: 'bullish' | 'bearish' | 'neutral';
  time_frame: string;
}

export class ImageAgent extends BaseAgent {
  private canvasCache: Map<string, Canvas> = new Map();
  private patternCache: Map<string, PatternRecognition> = new Map();
  private embeddingCache: Map<string, number[]> = new Map();
  private lastHeatmapGeneration: Map<string, number> = new Map();
  
  // Configuration
  private readonly CANVAS_WIDTH = 400;
  private readonly CANVAS_HEIGHT = 300;
  private readonly HEATMAP_COLORS = {
    bid: { r: 0, g: 255, b: 0 },    // Green for bids
    ask: { r: 255, g: 0, b: 0 },   // Red for asks
    background: { r: 20, g: 20, b: 30 } // Dark background
  };
  private readonly PATTERN_CONFIDENCE_THRESHOLD = 0.6;
  private readonly MAX_IMAGE_AGE_MS = 60000; // 1 minute

  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Generate orderbook heatmaps for all exchange-pairs
      const orderbookImages = await this.generateOrderbookHeatmaps();
      
      // Analyze visual patterns from generated images
      const patternAnalyses = await this.analyzeVisualPatterns(orderbookImages);
      
      // Process social/chart images if available
      const socialImageAnalysis = await this.processSocialImages();
      
      // Combine all image data
      const allImageData = [...orderbookImages, ...socialImageAnalysis];
      
      // Calculate visual features
      const features = this.calculateFeatures(allImageData, patternAnalyses);
      
      // Calculate key signal (visual pattern strength and reliability)
      const keySignal = this.calculateKeySignal(features, patternAnalyses);
      
      // Calculate confidence based on image quality and pattern recognition
      const confidence = this.calculateDataConfidence(allImageData, patternAnalyses);
      
      return {
        agent_name: 'ImageAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          images_generated: allImageData.length,
          patterns_detected: patternAnalyses.length
        },
        metadata: {
          canvas_dimensions: `${this.CANVAS_WIDTH}x${this.CANVAS_HEIGHT}`,
          supported_patterns: ['orderbook_heatmap', 'chart_pattern', 'social_image'],
          pattern_types: ['bull_flag', 'bear_flag', 'triangle', 'channel', 'breakout', 'reversal']
        }
      };

    } catch (error) {
      console.error('ImageAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Generate orderbook heatmaps for visualization
   */
  private async generateOrderbookHeatmaps(): Promise<ImageData[]> {
    const imageData: ImageData[] = [];
    const exchanges = ['binance', 'coinbase', 'kraken'];
    const pairs = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];

    for (const exchange of exchanges) {
      for (const pair of pairs) {
        try {
          // Check if we need to regenerate (rate limiting)
          const key = `${exchange}_${pair}`;
          const lastGeneration = this.lastHeatmapGeneration.get(key) || 0;
          const now = Date.now();
          
          if (now - lastGeneration < 30000) { // 30 seconds minimum interval
            continue;
          }

          const orderbook = await this.fetchOrderbook(exchange, pair);
          if (orderbook) {
            const heatmap = this.generateOrderbookHeatmap(orderbook, exchange, pair);
            const imageInfo = await this.processHeatmapImage(heatmap, exchange, pair);
            
            if (imageInfo) {
              imageData.push(imageInfo);
              this.lastHeatmapGeneration.set(key, now);
            }
          }
        } catch (error) {
          console.warn(`Heatmap generation error for ${exchange} ${pair}:`, error.message);
        }
      }
    }

    return imageData;
  }

  /**
   * Fetch orderbook data from exchange
   */
  private async fetchOrderbook(exchange: string, pair: string): Promise<any> {
    try {
      switch (exchange) {
        case 'binance':
          return await this.fetchBinanceOrderbook(this.denormalizePair(pair, 'binance'));
        case 'coinbase':
          return await this.fetchCoinbaseOrderbook(this.denormalizePair(pair, 'coinbase'));
        case 'kraken':
          return await this.fetchKrakenOrderbook(this.denormalizePair(pair, 'kraken'));
        default:
          return null;
      }
    } catch (error) {
      console.warn(`Failed to fetch orderbook for ${exchange} ${pair}:`, error.message);
      return null;
    }
  }

  /**
   * Fetch Binance orderbook
   */
  private async fetchBinanceOrderbook(symbol: string): Promise<any> {
    const response = await axios.get('https://api.binance.com/api/v3/depth', {
      params: { symbol, limit: 100 },
      timeout: 5000
    });
    return response.data;
  }

  /**
   * Fetch Coinbase orderbook
   */
  private async fetchCoinbaseOrderbook(productId: string): Promise<any> {
    const response = await axios.get(`https://api.exchange.coinbase.com/products/${productId}/book`, {
      params: { level: 2 },
      timeout: 5000
    });
    return response.data;
  }

  /**
   * Fetch Kraken orderbook
   */
  private async fetchKrakenOrderbook(pair: string): Promise<any> {
    const response = await axios.get('https://api.kraken.com/0/public/Depth', {
      params: { pair, count: 100 },
      timeout: 5000
    });
    return response.data?.result?.[Object.keys(response.data.result)[0]];
  }

  /**
   * Generate orderbook heatmap visualization
   */
  private generateOrderbookHeatmap(orderbook: any, exchange: string, pair: string): OrderbookHeatmap {
    const canvas = createCanvas(this.CANVAS_WIDTH, this.CANVAS_HEIGHT);
    const ctx = canvas.getContext('2d');
    
    // Extract bids and asks
    const bids = orderbook.bids || orderbook.b || [];
    const asks = orderbook.asks || orderbook.a || [];
    
    if (bids.length === 0 || asks.length === 0) {
      // Return empty heatmap
      return {
        canvas,
        width: this.CANVAS_WIDTH,
        height: this.CANVAS_HEIGHT,
        price_levels: [],
        bid_volumes: [],
        ask_volumes: [],
        max_volume: 0,
        asymmetry_score: 0
      };
    }
    
    // Process orderbook data
    const bidData = bids.slice(0, 50).map((bid: any) => ({
      price: parseFloat(bid[0]),
      volume: parseFloat(bid[1])
    }));
    
    const askData = asks.slice(0, 50).map((ask: any) => ({
      price: parseFloat(ask[0]),
      volume: parseFloat(ask[1])
    }));
    
    // Combine and sort by price
    const allData = [...bidData, ...askData].sort((a, b) => a.price - b.price);
    const maxVolume = Math.max(...allData.map(d => d.volume));
    
    // Find mid price
    const bestBid = bidData[0]?.price || 0;
    const bestAsk = askData[0]?.price || 0;
    const midPrice = (bestBid + bestAsk) / 2;
    
    // Draw heatmap
    this.drawOrderbookHeatmap(ctx, bidData, askData, midPrice, maxVolume);
    
    // Calculate asymmetry score
    const asymmetryScore = this.calculateOrderbookAsymmetry(bidData, askData, maxVolume);
    
    return {
      canvas,
      width: this.CANVAS_WIDTH,
      height: this.CANVAS_HEIGHT,
      price_levels: allData.map(d => d.price),
      bid_volumes: bidData.map(d => d.volume),
      ask_volumes: askData.map(d => d.volume),
      max_volume: maxVolume,
      asymmetry_score: asymmetryScore
    };
  }

  /**
   * Draw orderbook heatmap on canvas
   */
  private drawOrderbookHeatmap(
    ctx: CanvasRenderingContext2D,
    bids: any[],
    asks: any[],
    midPrice: number,
    maxVolume: number
  ): void {
    const width = this.CANVAS_WIDTH;
    const height = this.CANVAS_HEIGHT;
    
    // Clear canvas with dark background
    ctx.fillStyle = `rgb(${this.HEATMAP_COLORS.background.r}, ${this.HEATMAP_COLORS.background.g}, ${this.HEATMAP_COLORS.background.b})`;
    ctx.fillRect(0, 0, width, height);
    
    // Calculate price range for visualization
    const allPrices = [...bids.map(b => b.price), ...asks.map(a => a.price)];
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const priceRange = maxPrice - minPrice;
    
    if (priceRange === 0) return;
    
    // Draw bids (green, left side)
    for (let i = 0; i < bids.length; i++) {
      const bid = bids[i];
      const intensity = bid.volume / maxVolume;
      const y = height - ((bid.price - minPrice) / priceRange) * height;
      const barWidth = (intensity * width) / 2; // Half width for bids
      
      const alpha = Math.max(0.1, intensity);
      ctx.fillStyle = `rgba(${this.HEATMAP_COLORS.bid.r}, ${this.HEATMAP_COLORS.bid.g}, ${this.HEATMAP_COLORS.bid.b}, ${alpha})`;
      ctx.fillRect(0, y - 2, barWidth, 4);
    }
    
    // Draw asks (red, right side)
    for (let i = 0; i < asks.length; i++) {
      const ask = asks[i];
      const intensity = ask.volume / maxVolume;
      const y = height - ((ask.price - minPrice) / priceRange) * height;
      const barWidth = (intensity * width) / 2; // Half width for asks
      
      const alpha = Math.max(0.1, intensity);
      ctx.fillStyle = `rgba(${this.HEATMAP_COLORS.ask.r}, ${this.HEATMAP_COLORS.ask.g}, ${this.HEATMAP_COLORS.ask.b}, ${alpha})`;
      ctx.fillRect(width / 2, y - 2, barWidth, 4);
    }
    
    // Draw mid price line
    const midY = height - ((midPrice - minPrice) / priceRange) * height;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Calculate orderbook asymmetry score
   */
  private calculateOrderbookAsymmetry(bids: any[], asks: any[], maxVolume: number): number {
    if (bids.length === 0 || asks.length === 0) return 0;
    
    // Calculate total volume on each side (weighted by distance from mid)
    const totalBidVolume = bids.reduce((sum, bid) => sum + bid.volume, 0);
    const totalAskVolume = asks.reduce((sum, ask) => sum + ask.volume, 0);
    
    if (totalBidVolume + totalAskVolume === 0) return 0;
    
    // Calculate asymmetry as difference in volume distribution
    const volumeImbalance = Math.abs(totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume);
    
    // Calculate depth asymmetry (how far liquidity extends)
    const bidDepth = bids.length;
    const askDepth = asks.length;
    const depthImbalance = Math.abs(bidDepth - askDepth) / Math.max(bidDepth, askDepth);
    
    // Composite asymmetry score
    return (volumeImbalance + depthImbalance) / 2;
  }

  /**
   * Process heatmap image and extract features
   */
  private async processHeatmapImage(heatmap: OrderbookHeatmap, exchange: string, pair: string): Promise<ImageData | null> {
    try {
      // Generate embedding from heatmap
      const embedding = this.generateImageEmbedding(heatmap.canvas);
      const embeddingId = this.generateEmbeddingId(embedding);
      
      // Analyze visual patterns
      const visualPattern = this.analyzeOrderbookPattern(heatmap);
      
      // Calculate visual confidence based on data quality
      const visualConfidence = this.calculateVisualConfidence(heatmap);
      
      // Calculate sentiment score from visual asymmetry
      const visualSentimentScore = this.calculateVisualSentiment(heatmap);
      
      // Cache the canvas and embedding
      const cacheKey = `${exchange}_${pair}`;
      this.canvasCache.set(cacheKey, heatmap.canvas);
      this.embeddingCache.set(embeddingId, embedding);
      
      return {
        image_type: 'orderbook_heatmap',
        pair,
        exchange,
        visual_pattern: visualPattern,
        visual_confidence: visualConfidence,
        visual_sentiment_score: visualSentimentScore,
        embedding_id: embeddingId,
        generated_at: Date.now(),
        image_data: this.canvasToBase64(heatmap.canvas)
      };
      
    } catch (error) {
      console.error('Failed to process heatmap image:', error);
      return null;
    }
  }

  /**
   * Generate image embedding using simple feature extraction
   */
  private generateImageEmbedding(canvas: Canvas): number[] {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Simple feature extraction: color histograms and spatial features
    const features: number[] = [];
    
    // Color histogram (RGB channels)
    const rHist = new Array(8).fill(0);
    const gHist = new Array(8).fill(0);
    const bHist = new Array(8).fill(0);
    
    // Spatial features (quadrants)
    const quadrants = new Array(4).fill(0);
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      
      if (a > 0) { // Only process non-transparent pixels
        // Color histogram
        rHist[Math.floor(r / 32)]++;
        gHist[Math.floor(g / 32)]++;
        bHist[Math.floor(b / 32)]++;
        
        // Spatial features
        const pixelIndex = i / 4;
        const x = pixelIndex % canvas.width;
        const y = Math.floor(pixelIndex / canvas.width);
        
        const quadrant = (x < canvas.width / 2 ? 0 : 1) + (y < canvas.height / 2 ? 0 : 2);
        quadrants[quadrant] += (r + g + b) / 3; // Average intensity
      }
    }
    
    // Normalize histograms
    const totalPixels = canvas.width * canvas.height;
    features.push(...rHist.map(h => h / totalPixels));
    features.push(...gHist.map(h => h / totalPixels));
    features.push(...bHist.map(h => h / totalPixels));
    
    // Normalize quadrant features
    const maxQuadrant = Math.max(...quadrants);
    if (maxQuadrant > 0) {
      features.push(...quadrants.map(q => q / maxQuadrant));
    } else {
      features.push(0, 0, 0, 0);
    }
    
    // Edge detection features (simplified)
    const edgeFeatures = this.extractEdgeFeatures(canvas);
    features.push(...edgeFeatures);
    
    return features;
  }

  /**
   * Extract edge features from canvas
   */
  private extractEdgeFeatures(canvas: Canvas): number[] {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let totalEdgeStrength = 0;
    let edgeCount = 0;
    
    // Simple Sobel edge detection
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const idx = (y * canvas.width + x) * 4;
        
        // Get grayscale values for 3x3 neighborhood
        const pixels = [];
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nIdx = ((y + dy) * canvas.width + (x + dx)) * 4;
            const gray = (data[nIdx] + data[nIdx + 1] + data[nIdx + 2]) / 3;
            pixels.push(gray);
          }
        }
        
        // Sobel operator
        const gx = (-pixels[0] - 2*pixels[3] - pixels[6]) + (pixels[2] + 2*pixels[5] + pixels[8]);
        const gy = (-pixels[0] - 2*pixels[1] - pixels[2]) + (pixels[6] + 2*pixels[7] + pixels[8]);
        const edgeStrength = Math.sqrt(gx*gx + gy*gy);
        
        if (edgeStrength > 30) { // Threshold for edge detection
          totalEdgeStrength += edgeStrength;
          edgeCount++;
        }
      }
    }
    
    const avgEdgeStrength = edgeCount > 0 ? totalEdgeStrength / edgeCount : 0;
    const edgeDensity = edgeCount / (canvas.width * canvas.height);
    
    return [avgEdgeStrength / 255, edgeDensity];
  }

  /**
   * Analyze orderbook visual patterns
   */
  private analyzeOrderbookPattern(heatmap: OrderbookHeatmap): string {
    const asymmetry = heatmap.asymmetry_score;
    const bidVolume = heatmap.bid_volumes.reduce((sum, v) => sum + v, 0);
    const askVolume = heatmap.ask_volumes.reduce((sum, v) => sum + v, 0);
    const totalVolume = bidVolume + askVolume;
    
    if (totalVolume === 0) return 'no_liquidity';
    
    const bidRatio = bidVolume / totalVolume;
    
    // Pattern classification based on visual features
    if (asymmetry > 0.7) {
      if (bidRatio > 0.6) {
        return 'strong_bid_wall';
      } else {
        return 'strong_ask_wall';
      }
    } else if (asymmetry > 0.4) {
      if (bidRatio > 0.6) {
        return 'bid_heavy';
      } else if (bidRatio < 0.4) {
        return 'ask_heavy';
      } else {
        return 'moderate_imbalance';
      }
    } else if (asymmetry < 0.2) {
      return 'symmetric_liquidity';
    } else {
      return 'balanced_orderbook';
    }
  }

  /**
   * Calculate visual confidence
   */
  private calculateVisualConfidence(heatmap: OrderbookHeatmap): number {
    // Base confidence on data quality
    const volumeScore = Math.min(1, heatmap.max_volume / 100); // Normalize to reasonable volume
    const depthScore = Math.min(1, (heatmap.bid_volumes.length + heatmap.ask_volumes.length) / 100);
    const asymmetryScore = 1 - Math.min(1, heatmap.asymmetry_score); // Less asymmetry = higher confidence
    
    return (volumeScore + depthScore + asymmetryScore) / 3;
  }

  /**
   * Calculate visual sentiment score
   */
  private calculateVisualSentiment(heatmap: OrderbookHeatmap): number {
    const bidVolume = heatmap.bid_volumes.reduce((sum, v) => sum + v, 0);
    const askVolume = heatmap.ask_volumes.reduce((sum, v) => sum + v, 0);
    const totalVolume = bidVolume + askVolume;
    
    if (totalVolume === 0) return 0;
    
    // Sentiment based on bid/ask volume ratio
    const bidRatio = bidVolume / totalVolume;
    
    // Convert to sentiment score [-1, 1]
    return (bidRatio - 0.5) * 2;
  }

  /**
   * Generate embedding ID
   */
  private generateEmbeddingId(embedding: number[]): string {
    // Simple hash of embedding vector
    const hash = embedding.reduce((acc, val, idx) => {
      return acc + (val * (idx + 1)) * 1000;
    }, 0);
    
    return `embed_${Math.abs(Math.floor(hash)).toString(36)}`;
  }

  /**
   * Convert canvas to base64 string
   */
  private canvasToBase64(canvas: Canvas): string {
    return canvas.toDataURL('image/png');
  }

  /**
   * Analyze visual patterns from generated images
   */
  private async analyzeVisualPatterns(imageData: ImageData[]): Promise<PatternRecognition[]> {
    const patterns: PatternRecognition[] = [];
    
    for (const image of imageData) {
      try {
        const pattern = this.recognizePattern(image);
        if (pattern && pattern.confidence >= this.PATTERN_CONFIDENCE_THRESHOLD) {
          patterns.push(pattern);
          
          // Cache pattern recognition
          this.patternCache.set(image.embedding_id, pattern);
        }
      } catch (error) {
        console.warn('Pattern recognition error:', error.message);
      }
    }
    
    return patterns;
  }

  /**
   * Recognize patterns from image data
   */
  private recognizePattern(image: ImageData): PatternRecognition | null {
    if (image.image_type === 'orderbook_heatmap') {
      return this.recognizeOrderbookPattern(image);
    } else if (image.image_type === 'chart_pattern') {
      return this.recognizeChartPattern(image);
    }
    
    return null;
  }

  /**
   * Recognize orderbook patterns
   */
  private recognizeOrderbookPattern(image: ImageData): PatternRecognition {
    const pattern = image.visual_pattern;
    const sentiment = image.visual_sentiment_score;
    
    let patternType: PatternRecognition['pattern_type'] = 'unknown';
    let confidence = image.visual_confidence;
    let strength = Math.abs(sentiment);
    let direction: PatternRecognition['direction'] = 'neutral';
    
    // Map visual patterns to recognized patterns
    switch (pattern) {
      case 'strong_bid_wall':
        patternType = 'bull_flag';
        direction = 'bullish';
        confidence = Math.min(0.9, confidence + 0.2);
        break;
      case 'strong_ask_wall':
        patternType = 'bear_flag';
        direction = 'bearish';
        confidence = Math.min(0.9, confidence + 0.2);
        break;
      case 'bid_heavy':
        patternType = 'breakout';
        direction = 'bullish';
        break;
      case 'ask_heavy':
        patternType = 'breakout';
        direction = 'bearish';
        break;
      case 'symmetric_liquidity':
        patternType = 'channel';
        direction = 'neutral';
        break;
      default:
        patternType = 'unknown';
    }
    
    return {
      pattern_type: patternType,
      confidence: Math.max(0.1, Math.min(1, confidence)),
      strength: Math.max(0, Math.min(1, strength)),
      direction,
      time_frame: '1m' // Orderbook patterns are short-term
    };
  }

  /**
   * Recognize chart patterns (placeholder for future implementation)
   */
  private recognizeChartPattern(image: ImageData): PatternRecognition {
    // This would implement technical analysis pattern recognition
    // For now, return a default pattern
    return {
      pattern_type: 'unknown',
      confidence: 0.5,
      strength: 0.5,
      direction: 'neutral',
      time_frame: '5m'
    };
  }

  /**
   * Process social images (placeholder for future implementation)
   */
  private async processSocialImages(): Promise<ImageData[]> {
    // This would process social media images, memes, etc.
    // For now, return empty array
    return [];
  }

  /**
   * Calculate visual features
   */
  private calculateFeatures(imageData: ImageData[], patterns: PatternRecognition[]): VisualFeatures {
    if (imageData.length === 0) {
      return {
        orderbook_asymmetry: 0,
        liquidity_concentration: 0,
        price_action_strength: 0,
        volatility_visualization: 0,
        pattern_consistency: 0
      };
    }
    
    // Orderbook asymmetry (average across all orderbook images)
    const orderbookImages = imageData.filter(img => img.image_type === 'orderbook_heatmap');
    const orderbookAsymmetry = orderbookImages.length > 0
      ? orderbookImages.reduce((sum, img) => sum + Math.abs(img.visual_sentiment_score), 0) / orderbookImages.length
      : 0;
    
    // Liquidity concentration (based on visual confidence)
    const liquidityConcentration = imageData.reduce((sum, img) => sum + img.visual_confidence, 0) / imageData.length;
    
    // Price action strength (from pattern recognition)
    const priceActionStrength = patterns.length > 0
      ? patterns.reduce((sum, p) => sum + p.strength, 0) / patterns.length
      : 0;
    
    // Volatility visualization (variance in visual patterns)
    const sentimentScores = imageData.map(img => img.visual_sentiment_score);
    const volatilityVisualization = this.calculateVariance(sentimentScores);
    
    // Pattern consistency (how consistent patterns are across images)
    const patternConsistency = this.calculatePatternConsistency(patterns);
    
    return {
      orderbook_asymmetry: Math.max(0, Math.min(1, orderbookAsymmetry)),
      liquidity_concentration: Math.max(0, Math.min(1, liquidityConcentration)),
      price_action_strength: Math.max(0, Math.min(1, priceActionStrength)),
      volatility_visualization: Math.max(0, Math.min(1, volatilityVisualization)),
      pattern_consistency: Math.max(0, Math.min(1, patternConsistency))
    };
  }

  /**
   * Calculate variance of an array
   */
  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance); // Return standard deviation
  }

  /**
   * Calculate pattern consistency
   */
  private calculatePatternConsistency(patterns: PatternRecognition[]): number {
    if (patterns.length === 0) return 0;
    
    // Group patterns by type
    const patternCounts = new Map<string, number>();
    let totalConfidence = 0;
    
    for (const pattern of patterns) {
      patternCounts.set(pattern.pattern_type, (patternCounts.get(pattern.pattern_type) || 0) + 1);
      totalConfidence += pattern.confidence;
    }
    
    // Calculate consistency as the dominance of the most common pattern
    const maxCount = Math.max(...Array.from(patternCounts.values()));
    const consistency = maxCount / patterns.length;
    
    // Weight by average confidence
    const avgConfidence = totalConfidence / patterns.length;
    
    return consistency * avgConfidence;
  }

  /**
   * Calculate key signal (visual pattern strength and reliability)
   */
  private calculateKeySignal(features: VisualFeatures, patterns: PatternRecognition[]): number {
    // Weight different visual factors
    const weights = {
      pattern_strength: 0.3,       // Strength of detected patterns
      pattern_consistency: 0.25,   // Consistency across patterns
      liquidity_visualization: 0.2, // Quality of liquidity visualization
      asymmetry_signal: 0.15,     // Orderbook asymmetry strength
      volatility_indicator: 0.1   // Volatility visualization
    };
    
    // Pattern strength signal
    const avgPatternStrength = patterns.length > 0
      ? patterns.reduce((sum, p) => sum + p.strength, 0) / patterns.length
      : 0;
    
    // Liquidity visualization signal
    const liquiditySignal = features.liquidity_concentration;
    
    // Asymmetry signal (we want some asymmetry for arbitrage opportunities)
    const asymmetrySignal = Math.min(1, features.orderbook_asymmetry * 2); // Up to 50% asymmetry is good
    
    // Volatility indicator (moderate volatility is good for trading)
    const volatilitySignal = features.volatility_visualization > 0 
      ? Math.min(1, 1 / (1 + Math.pow(features.volatility_visualization - 0.5, 2))) // Optimal around 0.5
      : 0;
    
    // Weighted composite
    const signal = (
      avgPatternStrength * weights.pattern_strength +
      features.pattern_consistency * weights.pattern_consistency +
      liquiditySignal * weights.liquidity_visualization +
      asymmetrySignal * weights.asymmetry_signal +
      volatilitySignal * weights.volatility_indicator
    );
    
    return Math.max(0, Math.min(1, signal));
  }

  /**
   * Calculate confidence based on image quality and pattern recognition
   */
  private calculateDataConfidence(imageData: ImageData[], patterns: PatternRecognition[]): number {
    if (imageData.length === 0) return 0;
    
    // Image quality factor (average visual confidence)
    const avgVisualConfidence = imageData.reduce((sum, img) => sum + img.visual_confidence, 0) / imageData.length;
    
    // Pattern recognition factor (average pattern confidence)
    const avgPatternConfidence = patterns.length > 0
      ? patterns.reduce((sum, p) => sum + p.confidence, 0) / patterns.length
      : 0.5;
    
    // Data freshness factor (how recent are the images)
    const now = Date.now();
    const avgAge = imageData.reduce((sum, img) => sum + (now - img.generated_at), 0) / imageData.length;
    const freshnessFactor = Math.max(0.1, 1 - (avgAge / this.MAX_IMAGE_AGE_MS));
    
    // Coverage factor (how many exchanges/pairs we have data for)
    const uniquePairs = new Set(imageData.map(img => `${img.exchange}_${img.pair}`)).size;
    const expectedCoverage = 3 * 4; // 3 exchanges * 4 pairs
    const coverageFactor = Math.min(1, uniquePairs / expectedCoverage);
    
    return Math.max(0.1, avgVisualConfidence * 0.4 + avgPatternConfidence * 0.3 + freshnessFactor * 0.2 + coverageFactor * 0.1);
  }

  /**
   * Denormalize pair for exchange APIs
   */
  private denormalizePair(pair: string, exchange: string): string {
    switch (exchange) {
      case 'binance':
        return pair.replace('-', '');
      case 'coinbase':
        return pair;
      case 'kraken':
        return pair.replace('BTC-', 'XBT').replace('-', '');
      default:
        return pair;
    }
  }

  /**
   * Get visual analysis summary for debugging
   */
  getVisualSummary(): string {
    const totalImages = this.canvasCache.size;
    const totalEmbeddings = this.embeddingCache.size;
    const totalPatterns = this.patternCache.size;
    
    return `${totalImages} images cached, ${totalEmbeddings} embeddings, ${totalPatterns} patterns recognized`;
  }

  /**
   * Cleanup old cached data
   */
  async cleanup(): Promise<void> {
    const now = Date.now();
    
    // Clean up old heatmap generation times
    for (const [key, timestamp] of this.lastHeatmapGeneration) {
      if (now - timestamp > this.MAX_IMAGE_AGE_MS * 2) {
        this.lastHeatmapGeneration.delete(key);
      }
    }
    
    // Limit cache sizes
    if (this.canvasCache.size > 50) {
      const keys = Array.from(this.canvasCache.keys());
      const keysToDelete = keys.slice(0, keys.length - 50);
      keysToDelete.forEach(key => this.canvasCache.delete(key));
    }
    
    if (this.embeddingCache.size > 100) {
      const keys = Array.from(this.embeddingCache.keys());
      const keysToDelete = keys.slice(0, keys.length - 100);
      keysToDelete.forEach(key => this.embeddingCache.delete(key));
    }
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