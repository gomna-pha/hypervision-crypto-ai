/**
 * Sentiment Agent - Social Media & News Sentiment Analysis
 * Collects and analyzes sentiment from Twitter, Reddit, Google Trends, and News APIs
 * Provides market sentiment scoring and trending analysis
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import axios from 'axios';

export interface SentimentData {
  google_trend_score: number;    // Google Trends relative interest (0-100)
  mention_volume: number;        // Total mentions across platforms
  avg_polarity: number;          // Average sentiment polarity (-1 to 1)
  subjectivity: number;          // Subjectivity score (0 to 1)
  verified_ratio: number;        // Ratio of verified source mentions
  top_keywords: string[];        // Trending keywords
  engagement_weighted_polarity: number; // Polarity weighted by engagement
}

export interface SentimentFeatures {
  overall_sentiment: number;     // Composite sentiment score (-1 to 1)
  sentiment_momentum: number;    // Rate of change in sentiment
  news_sentiment: number;        // News-specific sentiment
  social_sentiment: number;      // Social media sentiment
  trend_strength: number;        // Google Trends momentum
  credibility_score: number;     // Source credibility weighting
}

export interface PlatformSentiment {
  platform: string;
  mentions: number;
  avg_polarity: number;
  engagement_score: number;
  top_mentions: Array<{
    text: string;
    polarity: number;
    engagement: number;
    verified: boolean;
  }>;
}

export class SentimentAgent extends BaseAgent {
  private twitterBearerToken: string;
  private newsApiKey: string;
  private redditUserAgent: string;
  private historicalSentiment: SentimentData[] = [];
  private lastTrendsFetch: number = 0;
  private keywordCache: Map<string, any> = new Map();
  
  // Configuration
  private readonly KEYWORDS = ['bitcoin', 'BTC', 'crypto', 'trading', 'arbitrage', 'ethereum', 'ETH'];
  private readonly PLATFORMS = ['twitter', 'reddit', 'news', 'trends'];
  private readonly MIN_MENTION_VOLUME = 50;
  private readonly CONFIDENCE_THRESHOLD = 0.4;

  constructor(
    config: AgentConfig,
    twitterBearerToken: string = 'demo_token',
    newsApiKey: string = 'demo_key',
    redditUserAgent: string = 'ArbitrageBot/1.0'
  ) {
    super(config);
    this.twitterBearerToken = twitterBearerToken;
    this.newsApiKey = newsApiKey;
    this.redditUserAgent = redditUserAgent;
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect sentiment from all platforms in parallel
      const [twitterSentiment, redditSentiment, newsSentiment, trendData] = await Promise.allSettled([
        this.collectTwitterSentiment(),
        this.collectRedditSentiment(),
        this.collectNewsSentiment(),
        this.collectGoogleTrends()
      ]);

      // Aggregate sentiment data
      const sentimentData = this.aggregateSentimentData([
        this.extractResult(twitterSentiment),
        this.extractResult(redditSentiment),
        this.extractResult(newsSentiment),
        this.extractResult(trendData)
      ]);

      // Calculate derived features
      const features = this.calculateFeatures(sentimentData);
      
      // Calculate key signal (sentiment strength and direction)
      const keySignal = this.calculateKeySignal(features);
      
      // Calculate confidence based on data quality and volume
      const confidence = this.calculateDataConfidence(sentimentData);
      
      // Store for trend analysis
      this.historicalSentiment.push(sentimentData);
      
      // Keep only last 48 data points (48 * 30s = 24 minutes of data)
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
          raw_sentiment_data: sentimentData,
          platforms_active: this.PLATFORMS.length,
          keywords_tracked: this.KEYWORDS.length
        },
        metadata: {
          data_sources: this.PLATFORMS,
          keywords: this.KEYWORDS,
          historical_samples: this.historicalSentiment.length,
          last_trends_fetch: this.lastTrendsFetch
        }
      };

    } catch (error) {
      console.error('SentimentAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Collect Twitter sentiment using Twitter API v2
   */
  private async collectTwitterSentiment(): Promise<PlatformSentiment> {
    try {
      // Construct search query
      const query = this.KEYWORDS.map(k => `"${k}"`).join(' OR ');
      const url = 'https://api.twitter.com/2/tweets/search/recent';
      
      const params = {
        query: `(${query}) -is:retweet lang:en`,
        max_results: 100,
        'tweet.fields': 'created_at,public_metrics,author_id,context_annotations',
        'user.fields': 'verified,public_metrics',
        expansions: 'author_id'
      };

      const response = await axios.get(url, {
        headers: {
          'Authorization': `Bearer ${this.twitterBearerToken}`,
          'User-Agent': 'ArbitrageBot/1.0'
        },
        params,
        timeout: 10000
      });

      if (response.data?.data) {
        const tweets = response.data.data;
        const users = response.data.includes?.users || [];
        
        return this.processTweets(tweets, users);
      } else {
        return this.getEmptyPlatformSentiment('twitter');
      }
      
    } catch (error) {
      console.warn('Twitter API error:', error.message);
      return this.getEmptyPlatformSentiment('twitter');
    }
  }

  /**
   * Process Twitter data and calculate sentiment
   */
  private processTweets(tweets: any[], users: any[]): PlatformSentiment {
    const userMap = new Map(users.map(u => [u.id, u]));
    let totalPolarity = 0;
    let totalEngagement = 0;
    let verifiedCount = 0;
    const topMentions: any[] = [];

    for (const tweet of tweets) {
      const user = userMap.get(tweet.author_id);
      const polarity = this.calculateTextSentiment(tweet.text);
      const engagement = this.calculateEngagementScore(tweet.public_metrics);
      const verified = user?.verified || false;
      
      totalPolarity += polarity * engagement; // Weight by engagement
      totalEngagement += engagement;
      
      if (verified) verifiedCount++;
      
      topMentions.push({
        text: tweet.text.substring(0, 100) + '...',
        polarity,
        engagement,
        verified
      });
    }

    // Sort by engagement and take top mentions
    topMentions.sort((a, b) => b.engagement - a.engagement);
    
    return {
      platform: 'twitter',
      mentions: tweets.length,
      avg_polarity: totalEngagement > 0 ? totalPolarity / totalEngagement : 0,
      engagement_score: totalEngagement / tweets.length,
      top_mentions: topMentions.slice(0, 5)
    };
  }

  /**
   * Collect Reddit sentiment using Reddit API
   */
  private async collectRedditSentiment(): Promise<PlatformSentiment> {
    try {
      const subreddits = ['cryptocurrency', 'Bitcoin', 'ethereum', 'trading'];
      const allPosts: any[] = [];
      
      for (const subreddit of subreddits) {
        try {
          const url = `https://www.reddit.com/r/${subreddit}/hot.json`;
          const response = await axios.get(url, {
            headers: {
              'User-Agent': this.redditUserAgent
            },
            params: { limit: 25 },
            timeout: 5000
          });
          
          if (response.data?.data?.children) {
            const posts = response.data.data.children
              .map((child: any) => child.data)
              .filter((post: any) => this.containsKeywords(post.title + ' ' + (post.selftext || '')));
            
            allPosts.push(...posts);
          }
        } catch (error) {
          console.warn(`Reddit ${subreddit} error:`, error.message);
        }
      }
      
      return this.processRedditPosts(allPosts);
      
    } catch (error) {
      console.warn('Reddit API error:', error.message);
      return this.getEmptyPlatformSentiment('reddit');
    }
  }

  /**
   * Process Reddit posts and calculate sentiment
   */
  private processRedditPosts(posts: any[]): PlatformSentiment {
    if (posts.length === 0) {
      return this.getEmptyPlatformSentiment('reddit');
    }

    let totalPolarity = 0;
    let totalEngagement = 0;
    const topMentions: any[] = [];

    for (const post of posts) {
      const text = post.title + ' ' + (post.selftext || '');
      const polarity = this.calculateTextSentiment(text);
      const engagement = post.score + post.num_comments;
      
      totalPolarity += polarity * engagement;
      totalEngagement += engagement;
      
      topMentions.push({
        text: post.title.substring(0, 100) + '...',
        polarity,
        engagement,
        verified: false // Reddit doesn't have verification
      });
    }

    topMentions.sort((a, b) => b.engagement - a.engagement);
    
    return {
      platform: 'reddit',
      mentions: posts.length,
      avg_polarity: totalEngagement > 0 ? totalPolarity / totalEngagement : 0,
      engagement_score: totalEngagement / posts.length,
      top_mentions: topMentions.slice(0, 5)
    };
  }

  /**
   * Collect news sentiment using NewsAPI
   */
  private async collectNewsSentiment(): Promise<PlatformSentiment> {
    try {
      const query = this.KEYWORDS.join(' OR ');
      const url = 'https://newsapi.org/v2/everything';
      
      const params = {
        q: query,
        language: 'en',
        sortBy: 'publishedAt',
        pageSize: 50,
        from: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().split('T')[0] // Last 24 hours
      };

      const response = await axios.get(url, {
        headers: {
          'X-API-Key': this.newsApiKey
        },
        params,
        timeout: 10000
      });

      if (response.data?.articles) {
        return this.processNewsArticles(response.data.articles);
      } else {
        return this.getEmptyPlatformSentiment('news');
      }
      
    } catch (error) {
      console.warn('NewsAPI error:', error.message);
      return this.getEmptyPlatformSentiment('news');
    }
  }

  /**
   * Process news articles and calculate sentiment
   */
  private processNewsArticles(articles: any[]): PlatformSentiment {
    if (articles.length === 0) {
      return this.getEmptyPlatformSentiment('news');
    }

    let totalPolarity = 0;
    const topMentions: any[] = [];

    for (const article of articles) {
      const text = (article.title || '') + ' ' + (article.description || '');
      const polarity = this.calculateTextSentiment(text);
      
      // News articles have higher base credibility
      const engagement = 10; // Base engagement score for news
      
      totalPolarity += polarity;
      
      topMentions.push({
        text: article.title?.substring(0, 100) + '...' || 'No title',
        polarity,
        engagement,
        verified: true // News sources are generally verified
      });
    }

    topMentions.sort((a, b) => Math.abs(b.polarity) - Math.abs(a.polarity));
    
    return {
      platform: 'news',
      mentions: articles.length,
      avg_polarity: totalPolarity / articles.length,
      engagement_score: 10, // Standard news engagement
      top_mentions: topMentions.slice(0, 5)
    };
  }

  /**
   * Collect Google Trends data
   */
  private async collectGoogleTrends(): Promise<{ google_trend_score: number; trending_keywords: string[] }> {
    try {
      // Rate limit Google Trends requests (max once per 5 minutes)
      const now = Date.now();
      if (now - this.lastTrendsFetch < 5 * 60 * 1000) {
        // Return cached data if available
        const cached = this.keywordCache.get('trends');
        if (cached) {
          return cached;
        }
      }

      // Simplified Google Trends simulation (would use pytrends or Google Trends API in production)
      const trendScore = await this.simulateGoogleTrends();
      
      this.lastTrendsFetch = now;
      const result = {
        google_trend_score: trendScore,
        trending_keywords: this.KEYWORDS.slice(0, 3) // Top 3 keywords
      };
      
      this.keywordCache.set('trends', result);
      return result;
      
    } catch (error) {
      console.warn('Google Trends error:', error.message);
      return {
        google_trend_score: 50, // Neutral baseline
        trending_keywords: []
      };
    }
  }

  /**
   * Simulate Google Trends data (placeholder for actual implementation)
   */
  private async simulateGoogleTrends(): Promise<number> {
    // In production, this would use pytrends or Google Trends API
    // For demo, simulate based on historical patterns
    const baseScore = 50 + (Math.random() - 0.5) * 30; // 35-65 range
    const volatility = Math.sin(Date.now() / (24 * 60 * 60 * 1000)) * 10; // Daily cycle
    
    return Math.max(0, Math.min(100, baseScore + volatility));
  }

  /**
   * Calculate sentiment polarity for text using basic NLP
   */
  private calculateTextSentiment(text: string): number {
    if (!text) return 0;
    
    const words = text.toLowerCase().split(/\s+/);
    
    // Basic sentiment lexicon
    const positiveWords = [
      'bullish', 'moon', 'pump', 'gain', 'profit', 'bull', 'buy', 'long',
      'positive', 'good', 'great', 'excellent', 'amazing', 'fantastic',
      'surge', 'rise', 'increase', 'up', 'growth', 'opportunity'
    ];
    
    const negativeWords = [
      'bearish', 'dump', 'crash', 'loss', 'bear', 'sell', 'short',
      'negative', 'bad', 'terrible', 'awful', 'disaster', 'panic',
      'drop', 'fall', 'decrease', 'down', 'decline', 'risk'
    ];
    
    let score = 0;
    let wordCount = 0;
    
    for (const word of words) {
      if (positiveWords.includes(word)) {
        score += 1;
        wordCount++;
      } else if (negativeWords.includes(word)) {
        score -= 1;
        wordCount++;
      }
    }
    
    // Normalize to [-1, 1] range
    if (wordCount === 0) return 0;
    
    const normalizedScore = score / Math.max(wordCount, words.length * 0.1);
    return Math.max(-1, Math.min(1, normalizedScore));
  }

  /**
   * Calculate engagement score from Twitter metrics
   */
  private calculateEngagementScore(metrics: any): number {
    if (!metrics) return 1;
    
    const likes = metrics.like_count || 0;
    const retweets = metrics.retweet_count || 0;
    const replies = metrics.reply_count || 0;
    const quotes = metrics.quote_count || 0;
    
    // Weighted engagement score
    return 1 + (likes * 0.1) + (retweets * 0.5) + (replies * 0.3) + (quotes * 0.4);
  }

  /**
   * Check if text contains any of our keywords
   */
  private containsKeywords(text: string): boolean {
    if (!text) return false;
    
    const lowerText = text.toLowerCase();
    return this.KEYWORDS.some(keyword => 
      lowerText.includes(keyword.toLowerCase())
    );
  }

  /**
   * Get empty platform sentiment for fallback
   */
  private getEmptyPlatformSentiment(platform: string): PlatformSentiment {
    return {
      platform,
      mentions: 0,
      avg_polarity: 0,
      engagement_score: 0,
      top_mentions: []
    };
  }

  /**
   * Extract result from Promise.allSettled
   */
  private extractResult(result: PromiseSettledResult<any>): any {
    return result.status === 'fulfilled' ? result.value : null;
  }

  /**
   * Aggregate sentiment data from all platforms
   */
  private aggregateSentimentData(platformResults: any[]): SentimentData {
    const validResults = platformResults.filter(r => r !== null);
    
    if (validResults.length === 0) {
      return {
        google_trend_score: 50,
        mention_volume: 0,
        avg_polarity: 0,
        subjectivity: 0.5,
        verified_ratio: 0,
        top_keywords: [],
        engagement_weighted_polarity: 0
      };
    }

    let totalMentions = 0;
    let totalPolarity = 0;
    let totalEngagement = 0;
    let verifiedMentions = 0;
    let trendScore = 50;
    const allKeywords: string[] = [];

    for (const result of validResults) {
      if (result.platform) {
        // Platform sentiment result
        totalMentions += result.mentions;
        totalPolarity += result.avg_polarity * result.mentions;
        totalEngagement += result.engagement_score * result.mentions;
        
        // Count verified mentions
        verifiedMentions += result.top_mentions.filter((m: any) => m.verified).length;
      } else if (result.google_trend_score !== undefined) {
        // Google Trends result
        trendScore = result.google_trend_score;
        allKeywords.push(...(result.trending_keywords || []));
      }
    }

    const avgPolarity = totalMentions > 0 ? totalPolarity / totalMentions : 0;
    const engagementWeightedPolarity = totalEngagement > 0 ? totalPolarity / totalEngagement : 0;
    const verifiedRatio = totalMentions > 0 ? verifiedMentions / totalMentions : 0;

    return {
      google_trend_score: trendScore,
      mention_volume: totalMentions,
      avg_polarity: avgPolarity,
      subjectivity: Math.abs(avgPolarity), // Higher absolute polarity = more subjective
      verified_ratio: verifiedRatio,
      top_keywords: [...new Set(allKeywords)], // Remove duplicates
      engagement_weighted_polarity: engagementWeightedPolarity
    };
  }

  /**
   * Calculate derived sentiment features
   */
  private calculateFeatures(data: SentimentData): SentimentFeatures {
    // Overall sentiment (composite of polarity and trend strength)
    const trendNormalized = (data.google_trend_score - 50) / 50; // Normalize to [-1, 1]
    const overallSentiment = (data.avg_polarity * 0.7) + (trendNormalized * 0.3);

    // Sentiment momentum (rate of change)
    let sentimentMomentum = 0;
    if (this.historicalSentiment.length >= 2) {
      const current = data.avg_polarity;
      const previous = this.historicalSentiment[this.historicalSentiment.length - 1].avg_polarity;
      sentimentMomentum = current - previous;
    }

    // News vs social sentiment (would require platform-specific data)
    const newsSentiment = data.avg_polarity * (data.verified_ratio || 0.5);
    const socialSentiment = data.avg_polarity * (1 - (data.verified_ratio || 0.5));

    // Trend strength based on Google Trends deviation from baseline
    const trendStrength = Math.abs(data.google_trend_score - 50) / 50;

    // Credibility score based on verified sources and engagement
    const credibilityScore = (data.verified_ratio * 0.6) + 
                            (Math.min(1, data.mention_volume / 100) * 0.4);

    return {
      overall_sentiment: Math.max(-1, Math.min(1, overallSentiment)),
      sentiment_momentum: Math.max(-1, Math.min(1, sentimentMomentum)),
      news_sentiment: Math.max(-1, Math.min(1, newsSentiment)),
      social_sentiment: Math.max(-1, Math.min(1, socialSentiment)),
      trend_strength: Math.max(0, Math.min(1, trendStrength)),
      credibility_score: Math.max(0, Math.min(1, credibilityScore))
    };
  }

  /**
   * Calculate key signal (sentiment strength and reliability)
   */
  private calculateKeySignal(features: SentimentFeatures): number {
    // Weight different sentiment factors
    const weights = {
      overall_sentiment: 0.3,      // Overall sentiment direction
      sentiment_momentum: 0.2,     // Rate of sentiment change
      trend_strength: 0.2,         // Google Trends strength
      credibility_score: 0.3       // Source credibility
    };

    // Calculate absolute sentiment strength (regardless of direction)
    const sentimentStrength = Math.abs(features.overall_sentiment);
    
    // Calculate momentum strength
    const momentumStrength = Math.abs(features.sentiment_momentum);
    
    // Composite signal
    const signal = (
      sentimentStrength * weights.overall_sentiment +
      momentumStrength * weights.sentiment_momentum +
      features.trend_strength * weights.trend_strength +
      features.credibility_score * weights.credibility_score
    );

    return Math.max(0, Math.min(1, signal));
  }

  /**
   * Calculate confidence based on data quality and volume
   */
  private calculateDataConfidence(data: SentimentData): number {
    // Volume factor (more mentions = higher confidence)
    const volumeFactor = Math.min(1, data.mention_volume / this.MIN_MENTION_VOLUME);
    
    // Source diversity factor (multiple sources = higher confidence)
    const platformsActive = this.PLATFORMS.length;
    const diversityFactor = Math.min(1, platformsActive / 4); // Max 4 platforms
    
    // Credibility factor (verified sources = higher confidence)
    const credibilityFactor = Math.max(0.3, data.verified_ratio);
    
    // Data recency factor (always fresh for social media)
    const recencyFactor = 1.0;
    
    // Subjectivity factor (less subjective = more reliable)
    const objectivityFactor = 1 - (data.subjectivity || 0.5);
    
    const baseConfidence = (
      volumeFactor * 0.3 +
      diversityFactor * 0.2 +
      credibilityFactor * 0.2 +
      recencyFactor * 0.2 +
      objectivityFactor * 0.1
    );

    return Math.max(0.1, Math.min(1, baseConfidence));
  }

  /**
   * Get sentiment summary for debugging
   */
  getSentimentSummary(): string {
    if (this.historicalSentiment.length === 0) {
      return 'No sentiment data available';
    }

    const latest = this.historicalSentiment[this.historicalSentiment.length - 1];
    const polarity = latest.avg_polarity >= 0 ? 'Positive' : 'Negative';
    const strength = Math.abs(latest.avg_polarity);
    const trend = latest.google_trend_score;

    return `${polarity} sentiment (${strength.toFixed(2)}), ${latest.mention_volume} mentions, ${trend.toFixed(0)} trend score`;
  }
}

/**
 * Factory function to create SentimentAgent with config
 */
export function createSentimentAgent(
  twitterBearerToken?: string,
  newsApiKey?: string,
  redditUserAgent?: string
): SentimentAgent {
  const config: AgentConfig = {
    name: 'sentiment',
    enabled: true,
    polling_interval_ms: 30 * 1000, // 30 seconds
    confidence_min: 0.4,
    data_age_max_ms: 5 * 60 * 1000, // 5 minutes max age
    retry_attempts: 3,
    retry_backoff_ms: 2000
  };

  return new SentimentAgent(config, twitterBearerToken, newsApiKey, redditUserAgent);
}

/**
 * Demo API keys (replace with real keys for production)
 */
export const DEMO_SENTIMENT_KEYS = {
  twitter_bearer_token: 'demo_twitter_bearer_token_replace_with_real',
  news_api_key: 'demo_news_api_key_replace_with_real',
  reddit_user_agent: 'ArbitrageBot/1.0'
};

// Export for testing
export { SentimentAgent as default };