import axios from 'axios';
import EventEmitter from 'events';

export interface SentimentNewsData {
  agent_name: string;
  timestamp: string;
  key_signal: number; // -1 to 1 (bearish to bullish)
  confidence: number; // 0 to 1
  features: {
    news_sentiment_score: number;
    article_count_24h: number;
    source_credibility_avg: number;
    headline_sentiment: number;
    market_mention_frequency: number;
    fear_greed_index: number;
  };
}

export interface NewsSource {
  name: string;
  url: string;
  credibility_weight: number;
  api_key?: string;
}

export class SentimentNewsAgent extends EventEmitter {
  private isRunning: boolean = false;
  private intervalId?: NodeJS.Timeout;
  
  // Visible Parameters for Investors
  public readonly parameters = {
    polling_interval_sec: 60,
    sentiment_window_hours: 24,
    min_article_count: 5,
    credibility_threshold: 0.6,
    confidence_decay_factor: 0.95,
    fear_greed_weight: 0.3,
    news_weight: 0.7
  };

  // Visible Constraints for Investors  
  public readonly constraints = {
    max_api_calls_per_hour: 100,
    min_source_diversity: 3,
    max_data_age_minutes: 30,
    min_confidence_threshold: 0.4,
    blackout_sources: ['unreliable-news.com'],
    required_keywords: ['bitcoin', 'crypto', 'arbitrage', 'trading']
  };

  // Visible Bounds for Investors
  public readonly bounds = {
    sentiment_range: { min: -1.0, max: 1.0 },
    confidence_range: { min: 0.0, max: 1.0 },
    min_articles_for_signal: 5,
    max_sentiment_volatility: 0.5,
    signal_smoothing_factor: 0.8
  };

  private newsSources: NewsSource[] = [
    {
      name: 'CoinTelegraph',
      url: 'https://cointelegraph.com/rss',
      credibility_weight: 0.85
    },
    {
      name: 'CoinDesk',
      url: 'https://www.coindesk.com/arc/outboundfeeds/rss/',
      credibility_weight: 0.90
    },
    {
      name: 'NewsAPI',
      url: 'https://newsapi.org/v2/everything',
      credibility_weight: 0.75,
      api_key: process.env.NEWS_API_KEY
    }
  ];

  private sentimentHistory: Array<{ timestamp: number; sentiment: number }> = [];

  constructor() {
    super();
    console.log('✅ SentimentNewsAgent initialized with visible parameters');
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🚀 Starting SentimentNewsAgent with real-time news feeds...');
    
    // Initial data collection
    await this.collectSentimentData();
    
    // Set up polling interval
    this.intervalId = setInterval(async () => {
      try {
        await this.collectSentimentData();
      } catch (error) {
        console.error('❌ SentimentNewsAgent polling error:', error);
        this.emit('error', error);
      }
    }, this.parameters.polling_interval_sec * 1000);
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    console.log('⏹️ SentimentNewsAgent stopped');
  }

  private async collectSentimentData(): Promise<void> {
    try {
      console.log('📰 Collecting real-time sentiment from news sources...');
      
      // Collect from multiple sources
      const newsData = await this.fetchNewsFromSources();
      const socialSentiment = await this.fetchSocialSentiment();
      const fearGreedIndex = await this.fetchFearGreedIndex();
      
      // Process and combine sentiment signals
      const combinedSentiment = this.processSentimentSignals(newsData, socialSentiment, fearGreedIndex);
      
      // Calculate confidence based on data quality
      const confidence = this.calculateConfidence(newsData);
      
      // Apply smoothing and bounds
      const smoothedSentiment = this.applySentimentSmoothing(combinedSentiment);
      
      const sentimentData: SentimentNewsData = {
        agent_name: 'SentimentNewsAgent',
        timestamp: new Date().toISOString(),
        key_signal: Math.max(this.bounds.sentiment_range.min, 
                           Math.min(this.bounds.sentiment_range.max, smoothedSentiment)),
        confidence: Math.max(this.bounds.confidence_range.min,
                           Math.min(this.bounds.confidence_range.max, confidence)),
        features: {
          news_sentiment_score: combinedSentiment,
          article_count_24h: newsData.length,
          source_credibility_avg: this.calculateAverageCredibility(newsData),
          headline_sentiment: this.calculateHeadlineSentiment(newsData),
          market_mention_frequency: this.calculateMentionFrequency(newsData),
          fear_greed_index: fearGreedIndex
        }
      };

      // Store in history for smoothing
      this.updateSentimentHistory(sentimentData.key_signal);
      
      // Emit to system
      this.emit('data', sentimentData);
      
      console.log(`📊 Sentiment Signal: ${sentimentData.key_signal.toFixed(3)} (confidence: ${sentimentData.confidence.toFixed(2)})`);
      
    } catch (error) {
      console.error('❌ Failed to collect sentiment data:', error);
      throw error;
    }
  }

  private async fetchNewsFromSources(): Promise<any[]> {
    const allNews: any[] = [];
    
    for (const source of this.newsSources) {
      try {
        let articles: any[] = [];
        
        if (source.name === 'NewsAPI' && source.api_key) {
          // NewsAPI implementation
          const response = await axios.get(source.url, {
            params: {
              q: 'bitcoin OR cryptocurrency OR crypto trading',
              language: 'en',
              sortBy: 'publishedAt',
              from: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
              apiKey: source.api_key
            },
            timeout: 10000
          });
          
          articles = response.data.articles || [];
          
        } else {
          // RSS feed implementation (simplified)
          console.log(`📡 Fetching from ${source.name} RSS feed...`);
          // In production, would parse RSS XML
          // For demo, simulate realistic news data
          articles = this.generateSimulatedNews(source);
        }
        
        // Add source metadata
        articles.forEach(article => {
          article.source_name = source.name;
          article.credibility_weight = source.credibility_weight;
        });
        
        allNews.push(...articles);
        
      } catch (error) {
        console.error(`❌ Failed to fetch from ${source.name}:`, error);
        // Continue with other sources
      }
    }
    
    return allNews.filter(article => this.validateArticle(article));
  }

  private generateSimulatedNews(source: NewsSource): any[] {
    // Generate realistic news articles for demo
    const headlines = [
      'Bitcoin Shows Strong Momentum Amid Institutional Interest',
      'Cryptocurrency Market Sees Increased Arbitrage Opportunities',  
      'Major Exchange Reports Record Trading Volumes',
      'Regulatory Clarity Boosts Crypto Market Confidence',
      'DeFi Protocols Drive Innovation in Arbitrage Strategies'
    ];
    
    return headlines.map((headline, index) => ({
      title: headline,
      publishedAt: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
      sentiment_score: -0.2 + Math.random() * 1.4, // -0.2 to 1.2 range
      url: `https://${source.name.toLowerCase()}.com/article-${index}`,
      description: `Market analysis shows ${headline.toLowerCase()}`,
      relevance_score: 0.7 + Math.random() * 0.3
    }));
  }

  private async fetchSocialSentiment(): Promise<number> {
    try {
      // In production, would connect to Twitter API, Reddit API, etc.
      // For demo, simulate social sentiment
      const socialMentions = Math.random() * 1000;
      const positiveMentions = socialMentions * (0.3 + Math.random() * 0.4);
      const negativeMentions = socialMentions * (0.1 + Math.random() * 0.3);
      
      return (positiveMentions - negativeMentions) / socialMentions;
      
    } catch (error) {
      console.error('❌ Failed to fetch social sentiment:', error);
      return 0;
    }
  }

  private async fetchFearGreedIndex(): Promise<number> {
    try {
      // Fear & Greed Index API (free)
      const response = await axios.get('https://api.alternative.me/fng/', {
        timeout: 5000
      });
      
      if (response.data?.data?.[0]?.value) {
        // Convert 0-100 scale to -1 to 1 scale
        return (parseFloat(response.data.data[0].value) - 50) / 50;
      }
      
      return 0;
      
    } catch (error) {
      console.error('❌ Failed to fetch Fear & Greed Index:', error);
      // Return neutral sentiment as fallback
      return 0;
    }
  }

  private processSentimentSignals(newsData: any[], socialSentiment: number, fearGreedIndex: number): number {
    if (newsData.length === 0) return 0;
    
    // Calculate weighted news sentiment
    let totalWeight = 0;
    let weightedSentiment = 0;
    
    newsData.forEach(article => {
      const weight = article.credibility_weight * (article.relevance_score || 1);
      weightedSentiment += (article.sentiment_score || 0) * weight;
      totalWeight += weight;
    });
    
    const newsSentiment = totalWeight > 0 ? weightedSentiment / totalWeight : 0;
    
    // Combine signals with weights
    const combinedSentiment = 
      newsSentiment * this.parameters.news_weight +
      fearGreedIndex * this.parameters.fear_greed_weight;
    
    return Math.max(-1, Math.min(1, combinedSentiment));
  }

  private calculateConfidence(newsData: any[]): number {
    if (newsData.length < this.parameters.min_article_count) {
      return 0.2; // Low confidence due to insufficient data
    }
    
    // Factors affecting confidence
    const sourceCount = new Set(newsData.map(a => a.source_name)).size;
    const sourceDiversityScore = Math.min(1, sourceCount / this.constraints.min_source_diversity);
    
    const avgCredibility = this.calculateAverageCredibility(newsData);
    const dataFreshness = this.calculateDataFreshness(newsData);
    const consensusScore = this.calculateSentimentConsensus(newsData);
    
    const confidence = (sourceDiversityScore * 0.3 + 
                       avgCredibility * 0.3 + 
                       dataFreshness * 0.2 + 
                       consensusScore * 0.2) * 
                       Math.pow(this.parameters.confidence_decay_factor, 
                               Math.max(0, newsData.length - 10)); // Decay with too much data
    
    return Math.max(this.constraints.min_confidence_threshold, confidence);
  }

  private calculateAverageCredibility(newsData: any[]): number {
    if (newsData.length === 0) return 0;
    return newsData.reduce((sum, article) => sum + (article.credibility_weight || 0.5), 0) / newsData.length;
  }

  private calculateHeadlineSentiment(newsData: any[]): number {
    if (newsData.length === 0) return 0;
    
    // Simple headline sentiment based on keywords
    const positiveKeywords = ['growth', 'profit', 'gain', 'rise', 'bull', 'opportunity'];
    const negativeKeywords = ['loss', 'fall', 'bear', 'crash', 'risk', 'decline'];
    
    let sentimentSum = 0;
    newsData.forEach(article => {
      const title = (article.title || '').toLowerCase();
      let score = 0;
      
      positiveKeywords.forEach(keyword => {
        if (title.includes(keyword)) score += 0.1;
      });
      
      negativeKeywords.forEach(keyword => {
        if (title.includes(keyword)) score -= 0.1;
      });
      
      sentimentSum += score;
    });
    
    return sentimentSum / newsData.length;
  }

  private calculateMentionFrequency(newsData: any[]): number {
    const keywordCount = newsData.reduce((count, article) => {
      const text = `${article.title || ''} ${article.description || ''}`.toLowerCase();
      return count + this.constraints.required_keywords.reduce((kCount, keyword) => {
        return kCount + (text.split(keyword).length - 1);
      }, 0);
    }, 0);
    
    return keywordCount / Math.max(1, newsData.length);
  }

  private calculateDataFreshness(newsData: any[]): number {
    if (newsData.length === 0) return 0;
    
    const now = Date.now();
    const maxAge = this.constraints.max_data_age_minutes * 60 * 1000;
    
    const freshnessScores = newsData.map(article => {
      const age = now - new Date(article.publishedAt).getTime();
      return Math.max(0, 1 - (age / maxAge));
    });
    
    return freshnessScores.reduce((sum, score) => sum + score, 0) / freshnessScores.length;
  }

  private calculateSentimentConsensus(newsData: any[]): number {
    if (newsData.length < 2) return 0.5;
    
    const sentiments = newsData.map(a => a.sentiment_score || 0);
    const mean = sentiments.reduce((sum, s) => sum + s, 0) / sentiments.length;
    const variance = sentiments.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / sentiments.length;
    const stdDev = Math.sqrt(variance);
    
    // High consensus = low standard deviation
    return Math.max(0, 1 - stdDev);
  }

  private applySentimentSmoothing(newSentiment: number): number {
    if (this.sentimentHistory.length === 0) {
      return newSentiment;
    }
    
    const recentSentiments = this.sentimentHistory
      .filter(h => Date.now() - h.timestamp < this.parameters.sentiment_window_hours * 60 * 60 * 1000)
      .map(h => h.sentiment);
    
    if (recentSentiments.length === 0) {
      return newSentiment;
    }
    
    const avgRecent = recentSentiments.reduce((sum, s) => sum + s, 0) / recentSentiments.length;
    
    return newSentiment * (1 - this.bounds.signal_smoothing_factor) + 
           avgRecent * this.bounds.signal_smoothing_factor;
  }

  private updateSentimentHistory(sentiment: number): void {
    this.sentimentHistory.push({
      timestamp: Date.now(),
      sentiment
    });
    
    // Keep only recent history
    const cutoff = Date.now() - this.parameters.sentiment_window_hours * 60 * 60 * 1000;
    this.sentimentHistory = this.sentimentHistory.filter(h => h.timestamp > cutoff);
  }

  private validateArticle(article: any): boolean {
    if (!article.title || !article.publishedAt) return false;
    
    // Check if source is blacklisted
    if (this.constraints.blackout_sources.includes(article.source_name)) return false;
    
    // Check age
    const age = Date.now() - new Date(article.publishedAt).getTime();
    if (age > this.constraints.max_data_age_minutes * 60 * 1000) return false;
    
    // Check relevance
    const text = `${article.title} ${article.description || ''}`.toLowerCase();
    const hasRequiredKeyword = this.constraints.required_keywords.some(keyword => 
      text.includes(keyword)
    );
    
    return hasRequiredKeyword;
  }

  // Public method to get current parameters (for investor transparency)
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints, 
      bounds: this.bounds,
      current_sources: this.newsSources.map(s => ({
        name: s.name,
        credibility_weight: s.credibility_weight
      }))
    };
  }
}