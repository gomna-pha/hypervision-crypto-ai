import axios from 'axios';
import { BaseAgent } from './BaseAgent';
import { SentimentAgentOutput } from '../types';
import config from '../utils/ConfigLoader';

interface SocialMetrics {
  mentions: number;
  sentiment: number; // -1 to 1
  volume: number;
  engagement: number;
  verified_ratio: number;
}

interface MarketMood {
  fear_greed_index: number; // 0-100
  social_volume: number;
  dominant_emotion: string;
  trending_topics: string[];
}

export class SentimentAgent extends BaseAgent {
  private fearGreedIndex: number = 50;
  private socialVolume: number = 0;
  private trendingTopics: string[] = [];
  private cryptoPairs: string[] = ['BTC', 'ETH', 'SOL'];

  constructor(port: number = 3003) {
    super('SentimentAgent', port);
  }

  protected async initialize(): Promise<void> {
    const sentimentConfig = config.get('agents.sentiment');
    
    this.logger.info('SentimentAgent initialized', {
      window: sentimentConfig?.window_sec,
      minVolume: sentimentConfig?.min_mention_volume
    });
  }

  protected async update(): Promise<void> {
    try {
      // Fetch data from multiple sources
      const [socialMetrics, marketMood, googleTrends] = await Promise.all([
        this.fetchSocialMetrics(),
        this.fetchMarketMood(),
        this.fetchGoogleTrends()
      ]);

      // Calculate features
      const features = this.calculateFeatures(socialMetrics, marketMood, googleTrends);
      
      // Calculate key signal
      const keySignal = this.calculateKeySignal(socialMetrics, marketMood);
      
      // Calculate confidence
      const confidence = this.calculateConfidence(socialMetrics);

      const output: SentimentAgentOutput = {
        agent_name: 'SentimentAgent',
        timestamp: new Date().toISOString(),
        key_signal: keySignal,
        confidence: confidence,
        features: features
      };

      await this.publishOutput(output);
      
      this.logger.info('Sentiment data updated', {
        keySignal,
        confidence,
        socialVolume: socialMetrics.volume,
        fearGreed: marketMood.fear_greed_index
      });
    } catch (error) {
      this.logger.error('Failed to update sentiment data', error);
      throw error;
    }
  }

  /**
   * Fetch social media metrics
   */
  private async fetchSocialMetrics(): Promise<SocialMetrics> {
    // Check if APIs are configured
    const twitterBearer = process.env.TWITTER_BEARER_TOKEN;
    const redditClientId = process.env.REDDIT_CLIENT_ID;

    if (!twitterBearer && !redditClientId) {
      // Return mock data for development
      return this.getMockSocialMetrics();
    }

    const metrics: SocialMetrics = {
      mentions: 0,
      sentiment: 0,
      volume: 0,
      engagement: 0,
      verified_ratio: 0
    };

    // Fetch Twitter data
    if (twitterBearer) {
      try {
        const twitterData = await this.fetchTwitterData(twitterBearer);
        metrics.mentions += twitterData.mentions;
        metrics.sentiment = (metrics.sentiment + twitterData.sentiment) / 2;
        metrics.volume += twitterData.volume;
        metrics.engagement += twitterData.engagement;
        metrics.verified_ratio = twitterData.verified_ratio;
      } catch (error) {
        this.logger.error('Twitter API error', error);
      }
    }

    // Fetch Reddit data
    if (redditClientId) {
      try {
        const redditData = await this.fetchRedditData();
        metrics.mentions += redditData.mentions;
        metrics.sentiment = (metrics.sentiment + redditData.sentiment) / 2;
        metrics.volume += redditData.volume;
        metrics.engagement += redditData.engagement;
      } catch (error) {
        this.logger.error('Reddit API error', error);
      }
    }

    return metrics;
  }

  /**
   * Fetch Twitter data
   */
  private async fetchTwitterData(bearerToken: string): Promise<SocialMetrics> {
    try {
      const queries = this.cryptoPairs.map(pair => `${pair} OR #${pair}`).join(' OR ');
      
      const response = await axios.get('https://api.twitter.com/2/tweets/search/recent', {
        headers: {
          'Authorization': `Bearer ${bearerToken}`
        },
        params: {
          query: queries,
          'tweet.fields': 'created_at,public_metrics,author_id',
          'user.fields': 'verified',
          max_results: 100
        },
        timeout: 5000
      });

      const tweets = response.data.data || [];
      const users = response.data.includes?.users || [];

      // Calculate metrics
      const mentions = tweets.length;
      const totalEngagement = tweets.reduce((sum: number, tweet: any) => {
        const metrics = tweet.public_metrics || {};
        return sum + (metrics.like_count || 0) + (metrics.retweet_count || 0) + (metrics.reply_count || 0);
      }, 0);

      const verifiedCount = users.filter((user: any) => user.verified).length;
      const verified_ratio = users.length > 0 ? verifiedCount / users.length : 0;

      // Simple sentiment analysis (would use NLP in production)
      const sentiment = this.analyzeSentiment(tweets.map((t: any) => t.text || ''));

      return {
        mentions,
        sentiment,
        volume: mentions,
        engagement: totalEngagement,
        verified_ratio
      };
    } catch (error) {
      this.logger.error('Twitter API request failed', error);
      return this.getMockSocialMetrics();
    }
  }

  /**
   * Fetch Reddit data
   */
  private async fetchRedditData(): Promise<SocialMetrics> {
    // Simplified Reddit data fetching
    // In production, would use proper Reddit API with OAuth
    try {
      const subreddits = ['cryptocurrency', 'bitcoin', 'ethereum'];
      let totalMentions = 0;
      let totalEngagement = 0;
      let sentimentSum = 0;

      for (const subreddit of subreddits) {
        const response = await axios.get(`https://www.reddit.com/r/${subreddit}/hot.json`, {
          params: { limit: 25 },
          timeout: 5000
        });

        const posts = response.data?.data?.children || [];
        totalMentions += posts.length;
        
        for (const post of posts) {
          const data = post.data;
          totalEngagement += (data.score || 0) + (data.num_comments || 0);
          
          // Simple sentiment based on upvote ratio
          const upvoteRatio = data.upvote_ratio || 0.5;
          sentimentSum += (upvoteRatio - 0.5) * 2; // Convert to -1 to 1 range
        }
      }

      return {
        mentions: totalMentions,
        sentiment: totalMentions > 0 ? sentimentSum / totalMentions : 0,
        volume: totalMentions,
        engagement: totalEngagement,
        verified_ratio: 0 // Reddit doesn't have verified concept
      };
    } catch (error) {
      this.logger.error('Reddit API request failed', error);
      return this.getMockSocialMetrics();
    }
  }

  /**
   * Fetch market mood indicators
   */
  private async fetchMarketMood(): Promise<MarketMood> {
    try {
      // Fetch Fear & Greed Index from alternative.me
      const response = await axios.get('https://api.alternative.me/fng/', {
        timeout: 5000
      });

      const data = response.data?.data?.[0];
      
      return {
        fear_greed_index: parseInt(data?.value || '50'),
        social_volume: this.socialVolume,
        dominant_emotion: data?.value_classification || 'neutral',
        trending_topics: this.trendingTopics
      };
    } catch (error) {
      // Return mock data if API fails
      return {
        fear_greed_index: 48 + Math.random() * 10,
        social_volume: 150000 + Math.random() * 20000,
        dominant_emotion: Math.random() > 0.5 ? 'fear' : 'greed',
        trending_topics: ['bitcoin', 'ethereum', 'defi']
      };
    }
  }

  /**
   * Fetch Google Trends data
   */
  private async fetchGoogleTrends(): Promise<number> {
    // Google Trends doesn't have a direct API
    // In production, would use pytrends or similar
    // For now, return mock trend score
    return 75 + Math.random() * 25;
  }

  /**
   * Get mock social metrics for development
   */
  private getMockSocialMetrics(): Promise<SocialMetrics> {
    const baseVolume = 10000;
    const variation = Math.random() * 5000;
    
    return Promise.resolve({
      mentions: Math.floor(baseVolume + variation),
      sentiment: (Math.random() - 0.5) * 0.8, // -0.4 to 0.4
      volume: Math.floor(baseVolume + variation),
      engagement: Math.floor((baseVolume + variation) * 10),
      verified_ratio: 0.02 + Math.random() * 0.03
    });
  }

  /**
   * Simple sentiment analysis
   */
  private analyzeSentiment(texts: string[]): number {
    if (texts.length === 0) return 0;

    const positiveWords = ['bullish', 'moon', 'pump', 'buy', 'long', 'up', 'green', 'profit', 'gain', 'rally'];
    const negativeWords = ['bearish', 'dump', 'sell', 'short', 'down', 'red', 'loss', 'crash', 'fear', 'panic'];

    let positiveCount = 0;
    let negativeCount = 0;

    for (const text of texts) {
      const lowerText = text.toLowerCase();
      
      for (const word of positiveWords) {
        if (lowerText.includes(word)) positiveCount++;
      }
      
      for (const word of negativeWords) {
        if (lowerText.includes(word)) negativeCount++;
      }
    }

    const total = positiveCount + negativeCount;
    if (total === 0) return 0;

    // Return sentiment between -1 and 1
    return (positiveCount - negativeCount) / total;
  }

  /**
   * Calculate features for output
   */
  private calculateFeatures(
    social: SocialMetrics,
    mood: MarketMood,
    googleTrend: number
  ): Record<string, any> {
    return {
      google_trend: googleTrend,
      avg_polarity: social.sentiment,
      mention_volume: social.volume,
      verified_ratio: social.verified_ratio,
      engagement_weighted_polarity: social.sentiment * Math.log(1 + social.engagement),
      fear_greed_index: mood.fear_greed_index,
      social_volume: mood.social_volume,
      dominant_emotion: mood.dominant_emotion,
      trending_topics: mood.trending_topics
    };
  }

  /**
   * Calculate key signal
   */
  private calculateKeySignal(social: SocialMetrics, mood: MarketMood): number {
    // Combine multiple factors into a single signal
    // Positive = bullish sentiment, Negative = bearish
    
    // Normalize fear & greed to -1 to 1
    const fgNormalized = (mood.fear_greed_index - 50) / 50;
    
    // Volume signal (high volume = stronger signal)
    const volumeWeight = Math.min(social.volume / 50000, 1);
    
    // Combine signals
    const signal = (
      social.sentiment * 0.4 +           // 40% weight on sentiment
      fgNormalized * 0.3 +               // 30% weight on fear & greed
      (social.verified_ratio - 0.02) * 10 * 0.2 + // 20% weight on verified ratio
      volumeWeight * 0.1                 // 10% weight on volume
    );

    // Clamp to [-1, 1]
    return Math.max(-1, Math.min(1, signal));
  }

  /**
   * Calculate confidence
   */
  private calculateConfidence(social: SocialMetrics): number {
    // Confidence based on data quality and volume
    const volumeConfidence = Math.min(social.volume / 10000, 1);
    const engagementConfidence = Math.min(social.engagement / 100000, 1);
    const verifiedConfidence = Math.min(social.verified_ratio * 20, 1);

    return (volumeConfidence * 0.4 + engagementConfidence * 0.4 + verifiedConfidence * 0.2);
  }

  protected async cleanup(): Promise<void> {
    this.logger.info('SentimentAgent cleanup completed');
  }

  protected getPollingInterval(): number {
    const windowSec = config.get<number>('agents.sentiment.window_sec') || 30;
    return windowSec * 1000; // Convert to milliseconds
  }
}

// Export for standalone execution
if (require.main === module) {
  const agent = new SentimentAgent();
  agent.start().catch(error => {
    console.error('Failed to start SentimentAgent:', error);
    process.exit(1);
  });

  process.on('SIGINT', async () => {
    await agent.stop();
    process.exit(0);
  });
}