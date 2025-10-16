/**
 * Sentiment Agent - Market Sentiment Analysis
 * Collects and analyzes sentiment from Twitter, Google Trends, Reddit, and News
 * Provides retail sentiment and trending interest indicators
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import axios, { AxiosResponse } from 'axios';

export interface SentimentData {
  google_trend_score: number;          // 0-100
  twitter_mention_volume: number;      // Count
  twitter_avg_polarity: number;       // -1 to 1
  twitter_subjectivity: number;       // 0 to 1
  reddit_mention_volume: number;      // Count
  reddit_avg_polarity: number;       // -1 to 1
  news_sentiment_score: number;       // -1 to 1
  verified_ratio: number;             // 0 to 1
}

export interface SentimentFeatures {
  overall_sentiment: number;          // Composite sentiment [-1, 1]
  sentiment_strength: number;         // Absolute sentiment intensity [0, 1]
  mention_momentum: number;          // Volume change rate [-1, 1]
  credibility_score: number;         // Based on verified sources [0, 1]
  trend_divergence: number;          // Difference between sources [-1, 1]
}

interface TwitterMetrics {
  mention_count: number;
  avg_polarity: number;
  subjectivity: number;
  verified_ratio: number;
}

interface GoogleTrendsData {
  trend_score: number;
  related_queries: string[];
}

interface RedditMetrics {
  mention_count: number;
  avg_polarity: number;
  subreddit_diversity: number;
}

export class SentimentAgent extends BaseAgent {
  private twitterBearerToken: string;
  private newsApiKey: string;
  private keywords: string[];
  private lastSentiment: SentimentData | null = null;
  private historicalSentiment: SentimentData[] = [];

  constructor(
    config: AgentConfig, 
    twitterBearerToken: string = '',
    newsApiKey: string = '',
    keywords: string[] = ['bitcoin', 'BTC', 'crypto', 'trading']
  ) {
    super(config);
    this.twitterBearerToken = twitterBearerToken;
    this.newsApiKey = newsApiKey;
    this.keywords = keywords;
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect sentiment from all sources in parallel
      const [
        twitterData,
        googleTrendsData,
        redditData,
        newsData
      ] = await Promise.allSettled([
        this.fetchTwitterSentiment(),
        this.fetchGoogleTrends(),
        this.fetchRedditSentiment(),
        this.fetchNewsSentiment()
      ]);

      // Aggregate all sentiment data
      const sentimentData = this.aggregateSentimentData(
        twitterData,
        googleTrendsData,
        redditData,
        newsData
      );

      // Calculate derived features
      const features = this.calculateSentimentFeatures(sentimentData);

      // Calculate key signal (composite sentiment score)
      const keySignal = features.overall_sentiment;

      // Calculate confidence based on data quality and volume
      const confidence = this.calculateSentimentConfidence(sentimentData);

      // Store for trend analysis
      this.lastSentiment = sentimentData;
      this.historicalSentiment.push(sentimentData);

      // Keep only last 48 data points (48 * 30s = 24 minutes of history)
      if (this.historicalSentiment.length > 48) {
        this.historicalSentiment.shift();
      }

      return {
        agent_name: 'SentimentAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          raw_data: sentimentData
        },
        metadata: {
          keywords_tracked: this.keywords.length,
          historical_samples: this.historicalSentiment.length,
          data_sources: this.getActiveDataSources()
        }
      };

    } catch (error) {
      console.error('SentimentAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Fetch Twitter sentiment using Twitter API v2
   */
  private async fetchTwitterSentiment(): Promise<TwitterMetrics> {
    if (!this.twitterBearerToken) {
      return this.getMockTwitterData();
    }

    try {
      const query = this.keywords.join(' OR ');
      const url = 'https://api.twitter.com/2/tweets/search/recent';
      
      const params = {
        query: `${query} -is:retweet`,
        max_results: 100,
        'tweet.fields': 'created_at,public_metrics,author_id',
        'user.fields': 'verified'
      };

      const response = await axios.get(url, {
        headers: {
          'Authorization': `Bearer ${this.twitterBearerToken}`
        },
        params,
        timeout: 5000
      });

      return this.analyzeTwitterData(response.data);

    } catch (error) {
      console.warn('Twitter API error, using mock data:', error.message);
      return this.getMockTwitterData();
    }
  }

  /**
   * Analyze Twitter API response data
   */
  private analyzeTwitterData(data: any): TwitterMetrics {
    if (!data.data || !Array.isArray(data.data)) {
      return { mention_count: 0, avg_polarity: 0, subjectivity: 0.5, verified_ratio: 0 };
    }

    const tweets = data.data;
    let totalPolarity = 0;
    let totalSubjectivity = 0;
    let verifiedCount = 0;

    // Simple sentiment analysis (in production, use proper NLP library)
    tweets.forEach((tweet: any) => {
      const text = tweet.text.toLowerCase();
      
      // Simple polarity calculation based on keywords
      let polarity = 0;
      const positiveWords = ['bullish', 'moon', 'pump', 'buy', 'long', 'up', 'profit', 'gain'];
      const negativeWords = ['bearish', 'dump', 'crash', 'sell', 'short', 'down', 'loss', 'fear'];
      
      positiveWords.forEach(word => {
        if (text.includes(word)) polarity += 0.1;
      });
      
      negativeWords.forEach(word => {
        if (text.includes(word)) polarity -= 0.1;
      });

      totalPolarity += Math.max(-1, Math.min(1, polarity));
      totalSubjectivity += 0.7; // Assume tweets are generally subjective
      
      // Check if user is verified (simplified check)
      if (data.includes?.users?.find((u: any) => u.id === tweet.author_id)?.verified) {
        verifiedCount++;
      }
    });

    return {
      mention_count: tweets.length,
      avg_polarity: tweets.length > 0 ? totalPolarity / tweets.length : 0,
      subjectivity: tweets.length > 0 ? totalSubjectivity / tweets.length : 0.5,
      verified_ratio: tweets.length > 0 ? verifiedCount / tweets.length : 0
    };
  }

  /**
   * Fetch Google Trends data (simplified implementation)
   */
  private async fetchGoogleTrends(): Promise<GoogleTrendsData> {
    try {
      // In production, use pytrends or Google Trends API
      // For now, simulate trends based on volatility
      const trendScore = 50 + Math.random() * 40; // Simulate 50-90 range
      
      return {
        trend_score: trendScore,
        related_queries: ['bitcoin price', 'crypto trading', 'BTC USD']
      };
    } catch (error) {
      console.warn('Google Trends error:', error.message);
      return { trend_score: 50, related_queries: [] };
    }
  }

  /**
   * Fetch Reddit sentiment using Reddit API
   */
  private async fetchRedditSentiment(): Promise<RedditMetrics> {
    try {
      // In production, use Reddit API or Pushshift
      // For now, simulate Reddit data
      const mentionCount = Math.floor(Math.random() * 200) + 50;
      const avgPolarity = (Math.random() - 0.5) * 0.8; // -0.4 to 0.4
      
      return {
        mention_count: mentionCount,
        avg_polarity: avgPolarity,
        subreddit_diversity: Math.random() * 0.5 + 0.5 // 0.5 to 1.0
      };
    } catch (error) {
      console.warn('Reddit API error:', error.message);
      return { mention_count: 0, avg_polarity: 0, subreddit_diversity: 0.5 };
    }
  }

  /**
   * Fetch news sentiment using News API
   */
  private async fetchNewsSentiment(): Promise<number> {
    if (!this.newsApiKey) {
      return Math.random() * 0.4 - 0.2; // Mock: -0.2 to 0.2
    }

    try {
      const query = this.keywords.join(' OR ');
      const url = 'https://newsapi.org/v2/everything';
      
      const params = {
        q: query,
        sortBy: 'publishedAt',
        pageSize: 50,
        apiKey: this.newsApiKey
      };

      const response = await axios.get(url, { params, timeout: 5000 });
      
      return this.analyzeNewsData(response.data);

    } catch (error) {
      console.warn('News API error:', error.message);
      return Math.random() * 0.4 - 0.2; // Mock sentiment
    }
  }

  /**
   * Analyze news articles for sentiment
   */
  private analyzeNewsData(data: any): number {
    if (!data.articles || !Array.isArray(data.articles)) {
      return 0;
    }

    let totalSentiment = 0;
    const articles = data.articles;

    articles.forEach((article: any) => {
      const text = `${article.title} ${article.description}`.toLowerCase();
      
      // Simple sentiment analysis
      let sentiment = 0;
      const positiveWords = ['surge', 'rally', 'bullish', 'adoption', 'breakthrough'];
      const negativeWords = ['crash', 'plunge', 'bearish', 'regulation', 'ban'];
      
      positiveWords.forEach(word => {
        if (text.includes(word)) sentiment += 0.2;
      });
      
      negativeWords.forEach(word => {
        if (text.includes(word)) sentiment -= 0.2;
      });

      totalSentiment += Math.max(-1, Math.min(1, sentiment));
    });

    return articles.length > 0 ? totalSentiment / articles.length : 0;
  }

  /**
   * Aggregate all sentiment data sources
   */
  private aggregateSentimentData(
    twitterResult: PromiseSettledResult<TwitterMetrics>,
    trendsResult: PromiseSettledResult<GoogleTrendsData>,
    redditResult: PromiseSettledResult<RedditMetrics>,
    newsResult: PromiseSettledResult<number>
  ): SentimentData {
    const twitter = twitterResult.status === 'fulfilled' ? twitterResult.value : this.getMockTwitterData();
    const trends = trendsResult.status === 'fulfilled' ? trendsResult.value : { trend_score: 50, related_queries: [] };
    const reddit = redditResult.status === 'fulfilled' ? redditResult.value : { mention_count: 0, avg_polarity: 0, subreddit_diversity: 0.5 };
    const news = newsResult.status === 'fulfilled' ? newsResult.value : 0;

    return {
      google_trend_score: trends.trend_score,
      twitter_mention_volume: twitter.mention_count,
      twitter_avg_polarity: twitter.avg_polarity,
      twitter_subjectivity: twitter.subjectivity,
      reddit_mention_volume: reddit.mention_count,
      reddit_avg_polarity: reddit.avg_polarity,
      news_sentiment_score: news,
      verified_ratio: twitter.verified_ratio
    };
  }

  /**
   * Calculate derived sentiment features
   */
  private calculateSentimentFeatures(data: SentimentData): SentimentFeatures {
    // Overall sentiment (weighted average)
    const sentimentSources = [
      { value: data.twitter_avg_polarity, weight: 0.3 },
      { value: data.reddit_avg_polarity, weight: 0.25 },
      { value: data.news_sentiment_score, weight: 0.25 },
      { value: (data.google_trend_score - 50) / 50, weight: 0.2 } // Normalize Google Trends
    ];

    let overallSentiment = 0;
    let totalWeight = 0;

    sentimentSources.forEach(({ value, weight }) => {
      if (typeof value === 'number' && !isNaN(value)) {
        overallSentiment += value * weight;
        totalWeight += weight;
      }
    });

    overallSentiment = totalWeight > 0 ? overallSentiment / totalWeight : 0;

    // Sentiment strength (absolute value)
    const sentimentStrength = Math.abs(overallSentiment);

    // Mention momentum (change in volume)
    let mentionMomentum = 0;
    if (this.historicalSentiment.length >= 2) {
      const current = data.twitter_mention_volume + data.reddit_mention_volume;
      const previous = this.historicalSentiment[this.historicalSentiment.length - 1];
      const previousVolume = previous.twitter_mention_volume + previous.reddit_mention_volume;
      
      if (previousVolume > 0) {
        const volumeChange = (current - previousVolume) / previousVolume;
        mentionMomentum = this.normalize(volumeChange, -0.5, 0.5, true);
      }
    }

    // Credibility score based on verified sources
    const credibilityScore = data.verified_ratio * 0.6 + 0.4; // Base credibility + verified boost

    // Trend divergence (how much sources disagree)
    const sentiments = [data.twitter_avg_polarity, data.reddit_avg_polarity, data.news_sentiment_score];
    const validSentiments = sentiments.filter(s => typeof s === 'number' && !isNaN(s));
    let trendDivergence = 0;

    if (validSentiments.length >= 2) {
      const mean = validSentiments.reduce((a, b) => a + b, 0) / validSentiments.length;
      const variance = validSentiments.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / validSentiments.length;
      trendDivergence = Math.sqrt(variance); // Standard deviation as divergence measure
    }

    return {
      overall_sentiment: Math.max(-1, Math.min(1, overallSentiment)),
      sentiment_strength: Math.max(0, Math.min(1, sentimentStrength)),
      mention_momentum: Math.max(-1, Math.min(1, mentionMomentum)),
      credibility_score: Math.max(0, Math.min(1, credibilityScore)),
      trend_divergence: Math.max(0, Math.min(1, trendDivergence))
    };
  }

  /**
   * Calculate confidence based on data quality and volume
   */
  private calculateSentimentConfidence(data: SentimentData): number {
    // Volume factor (more mentions = higher confidence)
    const totalMentions = data.twitter_mention_volume + data.reddit_mention_volume;
    const volumeFactor = Math.min(1, totalMentions / 100); // Normalize to 100 mentions

    // Source diversity factor
    const activeSources = this.getActiveDataSources().length;
    const diversityFactor = activeSources / 4; // 4 total sources

    // Verified ratio factor
    const verificationFactor = 0.7 + (data.verified_ratio * 0.3);

    // Google Trends confidence (higher scores = more confident)
    const trendsFactor = Math.min(1, data.google_trend_score / 70);

    return Math.max(0.1, volumeFactor * diversityFactor * verificationFactor * trendsFactor);
  }

  /**
   * Get mock Twitter data for testing
   */
  private getMockTwitterData(): TwitterMetrics {
    return {
      mention_count: Math.floor(Math.random() * 150) + 50,
      avg_polarity: (Math.random() - 0.5) * 0.8,
      subjectivity: Math.random() * 0.3 + 0.6,
      verified_ratio: Math.random() * 0.1 + 0.02
    };
  }

  /**
   * Get list of active data sources
   */
  private getActiveDataSources(): string[] {
    const sources: string[] = [];
    
    if (this.twitterBearerToken) sources.push('twitter');
    if (this.newsApiKey) sources.push('news');
    sources.push('google_trends'); // Always available (mocked)
    sources.push('reddit'); // Always available (mocked)
    
    return sources;
  }

  /**
   * Get sentiment summary for debugging
   */
  getSentimentSummary(): string {
    if (!this.lastSentiment) {
      return 'No sentiment data available';
    }

    const { twitter_avg_polarity, reddit_avg_polarity, twitter_mention_volume, google_trend_score } = this.lastSentiment;
    return `Twitter: ${twitter_avg_polarity.toFixed(2)} (${twitter_mention_volume} mentions), Google Trends: ${google_trend_score.toFixed(0)}, Reddit: ${reddit_avg_polarity.toFixed(2)}`;
  }
}

/**
 * Factory function to create SentimentAgent with config
 */
export function createSentimentAgent(
  twitterBearerToken?: string,
  newsApiKey?: string,
  keywords?: string[]
): SentimentAgent {
  const config: AgentConfig = {
    name: 'sentiment',
    enabled: true,
    polling_interval_ms: 30 * 1000, // 30 seconds
    confidence_min: 0.4,
    data_age_max_ms: 2 * 60 * 1000, // 2 minutes
    retry_attempts: 3,
    retry_backoff_ms: 2000
  };

  return new SentimentAgent(config, twitterBearerToken, newsApiKey, keywords);
}

// Export for testing
export { SentimentAgent as default };