#!/usr/bin/env python3
"""
Sentiment Arbitrage Engine with FinBERT Integration
Advanced sentiment-based trading using social media, news, and financial sentiment analysis
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import aiohttp
import json
import re
from collections import deque, defaultdict
import hashlib
from enum import Enum
import tweepy
import praw  # Reddit API
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Sentiment Data Models ====================

class SentimentSource(Enum):
    TWITTER = "TWITTER"
    REDDIT = "REDDIT"
    NEWS = "NEWS"
    BLOOMBERG = "BLOOMBERG"
    REUTERS = "REUTERS"
    FINVIZ = "FINVIZ"
    STOCKTWITS = "STOCKTWITS"
    TELEGRAM = "TELEGRAM"

class SentimentSignal(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

@dataclass
class SentimentData:
    """Sentiment data from various sources"""
    timestamp: datetime
    source: SentimentSource
    symbol: str
    raw_text: str
    finbert_score: float  # -1 to 1
    finbert_confidence: float
    vader_score: float
    textblob_polarity: float
    textblob_subjectivity: float
    volume: int  # Number of mentions/engagement
    reach: int  # Potential audience size
    credibility_score: float  # Source credibility
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentimentArbitrageOpportunity:
    """Sentiment-based arbitrage opportunity"""
    opportunity_id: str
    symbol: str
    sentiment_signal: SentimentSignal
    price_action_divergence: float  # Sentiment vs price divergence
    expected_move: float  # Expected price movement %
    confidence: float
    time_horizon_minutes: int
    entry_price: float
    target_price: float
    stop_loss: float
    social_volume_spike: float  # Relative volume increase
    news_catalyst: Optional[str]
    risk_reward_ratio: float
    timestamp: datetime

# ==================== FinBERT Model Manager ====================

class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer"""
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize FinBERT model"""
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            # Fallback to mock mode
            self.sentiment_pipeline = None
    
    def analyze(self, text: str) -> Tuple[float, float, str]:
        """
        Analyze sentiment using FinBERT
        Returns: (score, confidence, label)
        """
        if not self.sentiment_pipeline:
            # Mock mode for testing
            return self._mock_analyze(text)
        
        try:
            # Clean and truncate text
            text = self._preprocess_text(text)
            
            # Get FinBERT predictions
            results = self.sentiment_pipeline(text, truncation=True, max_length=512)
            
            if results:
                result = results[0]
                label = result['label']
                confidence = result['score']
                
                # Convert to normalized score
                if label == 'positive':
                    score = confidence
                elif label == 'negative':
                    score = -confidence
                else:  # neutral
                    score = 0
                
                return score, confidence, label
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
        
        return 0.0, 0.0, "neutral"
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text[:512]  # Truncate to model max length
    
    def _mock_analyze(self, text: str) -> Tuple[float, float, str]:
        """Mock analysis for testing"""
        # Simple keyword-based sentiment
        positive_words = ['bullish', 'buy', 'long', 'moon', 'pump', 'growth', 'profit']
        negative_words = ['bearish', 'sell', 'short', 'dump', 'crash', 'loss', 'decline']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            score = min(pos_count * 0.2, 1.0)
            return score, 0.7 + np.random.random() * 0.3, "positive"
        elif neg_count > pos_count:
            score = max(-neg_count * 0.2, -1.0)
            return abs(score), 0.7 + np.random.random() * 0.3, "negative"
        else:
            return 0.0, 0.5, "neutral"

# ==================== Social Media Data Collectors ====================

class SocialMediaCollector:
    """Collects data from various social media platforms"""
    
    def __init__(self):
        self.twitter_api = None
        self.reddit_api = None
        self.stocktwits_session = aiohttp.ClientSession()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize social media APIs"""
        # In production, these would use real API keys
        try:
            # Twitter API (would use real credentials)
            # self.twitter_api = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")
            
            # Reddit API (would use real credentials)
            # self.reddit_api = praw.Reddit(
            #     client_id="YOUR_CLIENT_ID",
            #     client_secret="YOUR_CLIENT_SECRET",
            #     user_agent="sentiment_bot"
            # )
            pass
        except Exception as e:
            logger.error(f"Failed to initialize social media APIs: {e}")
    
    async def collect_twitter_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect Twitter sentiment data"""
        sentiments = []
        
        # Mock implementation - in production, would use Twitter API v2
        tweets = self._mock_twitter_data(symbol)
        
        for tweet in tweets:
            sentiment = SentimentData(
                timestamp=datetime.now(),
                source=SentimentSource.TWITTER,
                symbol=symbol,
                raw_text=tweet['text'],
                finbert_score=0,  # Will be filled by FinBERT
                finbert_confidence=0,
                vader_score=self._calculate_vader_score(tweet['text']),
                textblob_polarity=self._calculate_textblob_sentiment(tweet['text'])[0],
                textblob_subjectivity=self._calculate_textblob_sentiment(tweet['text'])[1],
                volume=tweet.get('retweet_count', 0) + tweet.get('like_count', 0),
                reach=tweet.get('author_followers', 1000),
                credibility_score=self._calculate_credibility(tweet),
                metadata=tweet
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    async def collect_reddit_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect Reddit sentiment data"""
        sentiments = []
        
        # Mock implementation - in production, would use Reddit API
        posts = self._mock_reddit_data(symbol)
        
        for post in posts:
            sentiment = SentimentData(
                timestamp=datetime.now(),
                source=SentimentSource.REDDIT,
                symbol=symbol,
                raw_text=post['text'],
                finbert_score=0,
                finbert_confidence=0,
                vader_score=self._calculate_vader_score(post['text']),
                textblob_polarity=self._calculate_textblob_sentiment(post['text'])[0],
                textblob_subjectivity=self._calculate_textblob_sentiment(post['text'])[1],
                volume=post.get('score', 0) + post.get('num_comments', 0),
                reach=post.get('subreddit_subscribers', 10000),
                credibility_score=self._calculate_credibility(post),
                metadata=post
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    async def collect_stocktwits_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect StockTwits sentiment data"""
        sentiments = []
        
        try:
            # StockTwits API
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            
            # Mock response for demonstration
            messages = self._mock_stocktwits_data(symbol)
            
            for msg in messages:
                sentiment = SentimentData(
                    timestamp=datetime.now(),
                    source=SentimentSource.STOCKTWITS,
                    symbol=symbol,
                    raw_text=msg['body'],
                    finbert_score=0,
                    finbert_confidence=0,
                    vader_score=self._calculate_vader_score(msg['body']),
                    textblob_polarity=self._calculate_textblob_sentiment(msg['body'])[0],
                    textblob_subjectivity=self._calculate_textblob_sentiment(msg['body'])[1],
                    volume=msg.get('likes', 0),
                    reach=msg.get('user_followers', 100),
                    credibility_score=self._calculate_credibility(msg),
                    metadata=msg
                )
                sentiments.append(sentiment)
                
        except Exception as e:
            logger.error(f"StockTwits collection error: {e}")
        
        return sentiments
    
    def _calculate_vader_score(self, text: str) -> float:
        """Calculate VADER sentiment score"""
        scores = self.vader_analyzer.polarity_scores(text)
        return scores['compound']
    
    def _calculate_textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """Calculate TextBlob sentiment"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.5
    
    def _calculate_credibility(self, post: Dict) -> float:
        """Calculate source credibility score"""
        credibility = 0.5
        
        # Factors that increase credibility
        if post.get('verified', False):
            credibility += 0.2
        
        followers = post.get('author_followers', 0)
        if followers > 10000:
            credibility += 0.1
        if followers > 100000:
            credibility += 0.1
        
        # Account age (mock)
        if post.get('account_age_days', 0) > 365:
            credibility += 0.1
        
        return min(credibility, 1.0)
    
    def _mock_twitter_data(self, symbol: str) -> List[Dict]:
        """Mock Twitter data for testing"""
        templates = [
            f"${symbol} is looking bullish! Breaking out of resistance 🚀",
            f"Massive volume on ${symbol} today. Something big coming?",
            f"Just went long on ${symbol}. Technical setup is perfect!",
            f"${symbol} earnings beat expectations! To the moon! 🌙",
            f"Warning: ${symbol} showing bearish divergence on the daily",
            f"Selling my ${symbol} position. Taking profits here.",
            f"${symbol} partnership announced with major tech company!",
            f"Institutional buying detected on ${symbol} 📈"
        ]
        
        return [
            {
                'text': templates[i % len(templates)],
                'retweet_count': np.random.randint(10, 1000),
                'like_count': np.random.randint(50, 5000),
                'author_followers': np.random.randint(100, 100000),
                'verified': np.random.random() > 0.7,
                'account_age_days': np.random.randint(30, 3000)
            }
            for i in range(5)
        ]
    
    def _mock_reddit_data(self, symbol: str) -> List[Dict]:
        """Mock Reddit data for testing"""
        templates = [
            f"DD on {symbol}: Undervalued with huge potential",
            f"{symbol} YOLO update: Up 200% this month!",
            f"Why I'm bearish on {symbol} (Unpopular Opinion)",
            f"{symbol} technical analysis - Cup and handle forming",
            f"News: {symbol} announces record quarterly revenue"
        ]
        
        return [
            {
                'text': templates[i % len(templates)],
                'score': np.random.randint(10, 5000),
                'num_comments': np.random.randint(5, 500),
                'subreddit_subscribers': np.random.randint(10000, 5000000),
                'author_karma': np.random.randint(100, 100000)
            }
            for i in range(3)
        ]
    
    def _mock_stocktwits_data(self, symbol: str) -> List[Dict]:
        """Mock StockTwits data for testing"""
        sentiments = ['Bullish', 'Bearish', 'Neutral']
        templates = [
            f"${symbol} breaking out! Next target $X",
            f"Loading up on ${symbol} dips",
            f"${symbol} overbought here, taking profits",
            f"${symbol} consolidating nicely for next leg up"
        ]
        
        return [
            {
                'body': templates[i % len(templates)],
                'sentiment': sentiments[i % len(sentiments)],
                'likes': np.random.randint(0, 100),
                'user_followers': np.random.randint(10, 10000)
            }
            for i in range(4)
        ]

# ==================== News Sentiment Analyzer ====================

class NewsAnalyzer:
    """Analyzes news sentiment from various sources"""
    
    def __init__(self):
        self.news_sources = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://feeds.finance.yahoo.com/rss/2.0/headline'
        ]
        self.session = aiohttp.ClientSession()
    
    async def collect_news_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect news sentiment for a symbol"""
        sentiments = []
        
        # Mock news data for demonstration
        news_items = self._mock_news_data(symbol)
        
        for item in news_items:
            sentiment = SentimentData(
                timestamp=datetime.now(),
                source=SentimentSource.NEWS,
                symbol=symbol,
                raw_text=item['title'] + ' ' + item['description'],
                finbert_score=0,
                finbert_confidence=0,
                vader_score=0,
                textblob_polarity=0,
                textblob_subjectivity=0,
                volume=1,
                reach=item.get('readership', 100000),
                credibility_score=0.8,  # News sources generally credible
                metadata=item
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    def _mock_news_data(self, symbol: str) -> List[Dict]:
        """Mock news data for testing"""
        templates = [
            {
                'title': f'{symbol} Surges on Strong Earnings Report',
                'description': f'Shares of {symbol} jumped 5% in pre-market trading following better-than-expected quarterly results.',
                'source': 'Bloomberg',
                'readership': 1000000
            },
            {
                'title': f'Analysts Upgrade {symbol} to Buy',
                'description': f'Major investment banks raise price targets for {symbol} citing growth potential.',
                'source': 'Reuters',
                'readership': 800000
            },
            {
                'title': f'{symbol} Faces Regulatory Scrutiny',
                'description': f'Government officials announce investigation into {symbol} business practices.',
                'source': 'CNBC',
                'readership': 600000
            }
        ]
        
        return templates[:2]  # Return subset

# ==================== Sentiment Arbitrage Detection ====================

class SentimentArbitrageDetector:
    """Detects arbitrage opportunities based on sentiment analysis"""
    
    def __init__(self, finbert_analyzer: FinBERTAnalyzer):
        self.finbert = finbert_analyzer
        self.social_collector = SocialMediaCollector()
        self.news_analyzer = NewsAnalyzer()
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    async def detect_opportunities(
        self,
        symbols: List[str],
        market_data: Dict[str, Any]
    ) -> List[SentimentArbitrageOpportunity]:
        """Detect sentiment-based arbitrage opportunities"""
        opportunities = []
        
        for symbol in symbols:
            try:
                # Collect sentiment from all sources
                all_sentiments = await self._collect_all_sentiments(symbol)
                
                # Analyze with FinBERT
                enhanced_sentiments = self._enhance_with_finbert(all_sentiments)
                
                # Calculate aggregate sentiment
                agg_sentiment = self._calculate_aggregate_sentiment(enhanced_sentiments)
                
                # Get current market data
                current_price = market_data.get(symbol, {}).get('price', 0)
                
                # Detect divergence
                divergence = self._detect_sentiment_price_divergence(
                    symbol, agg_sentiment, current_price
                )
                
                # Check for opportunities
                if abs(divergence) > 0.3:  # Significant divergence threshold
                    opportunity = self._create_opportunity(
                        symbol, agg_sentiment, divergence, current_price, enhanced_sentiments
                    )
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.error(f"Error detecting sentiment arbitrage for {symbol}: {e}")
        
        return opportunities
    
    async def _collect_all_sentiments(self, symbol: str) -> List[SentimentData]:
        """Collect sentiments from all sources"""
        all_sentiments = []
        
        # Collect from each source
        twitter_sentiments = await self.social_collector.collect_twitter_sentiment(symbol)
        reddit_sentiments = await self.social_collector.collect_reddit_sentiment(symbol)
        stocktwits_sentiments = await self.social_collector.collect_stocktwits_sentiment(symbol)
        news_sentiments = await self.news_analyzer.collect_news_sentiment(symbol)
        
        all_sentiments.extend(twitter_sentiments)
        all_sentiments.extend(reddit_sentiments)
        all_sentiments.extend(stocktwits_sentiments)
        all_sentiments.extend(news_sentiments)
        
        return all_sentiments
    
    def _enhance_with_finbert(self, sentiments: List[SentimentData]) -> List[SentimentData]:
        """Enhance sentiments with FinBERT analysis"""
        for sentiment in sentiments:
            score, confidence, label = self.finbert.analyze(sentiment.raw_text)
            sentiment.finbert_score = score
            sentiment.finbert_confidence = confidence
        
        return sentiments
    
    def _calculate_aggregate_sentiment(self, sentiments: List[SentimentData]) -> Dict[str, float]:
        """Calculate weighted aggregate sentiment"""
        if not sentiments:
            return {'score': 0, 'confidence': 0, 'volume': 0}
        
        total_weight = 0
        weighted_score = 0
        total_volume = 0
        
        for sentiment in sentiments:
            # Weight by credibility, confidence, and reach
            weight = (
                sentiment.credibility_score *
                sentiment.finbert_confidence *
                np.log1p(sentiment.reach)
            )
            
            # Combine different sentiment scores
            combined_score = (
                sentiment.finbert_score * 0.5 +
                sentiment.vader_score * 0.3 +
                sentiment.textblob_polarity * 0.2
            )
            
            weighted_score += combined_score * weight
            total_weight += weight
            total_volume += sentiment.volume
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0
        
        # Calculate sentiment signal
        if final_score > 0.5:
            signal = SentimentSignal.STRONG_BULLISH
        elif final_score > 0.2:
            signal = SentimentSignal.BULLISH
        elif final_score < -0.5:
            signal = SentimentSignal.STRONG_BEARISH
        elif final_score < -0.2:
            signal = SentimentSignal.BEARISH
        else:
            signal = SentimentSignal.NEUTRAL
        
        return {
            'score': final_score,
            'confidence': total_weight / len(sentiments) if sentiments else 0,
            'volume': total_volume,
            'signal': signal
        }
    
    def _detect_sentiment_price_divergence(
        self,
        symbol: str,
        agg_sentiment: Dict,
        current_price: float
    ) -> float:
        """Detect divergence between sentiment and price action"""
        
        # Store current data
        self.sentiment_history[symbol].append(agg_sentiment['score'])
        self.price_history[symbol].append(current_price)
        
        if len(self.sentiment_history[symbol]) < 10:
            return 0  # Not enough history
        
        # Calculate recent trends
        recent_sentiments = list(self.sentiment_history[symbol])[-10:]
        recent_prices = list(self.price_history[symbol])[-10:]
        
        # Normalize
        sentiment_change = (recent_sentiments[-1] - np.mean(recent_sentiments[:-1])) / (np.std(recent_sentiments[:-1]) + 1e-6)
        price_change = (recent_prices[-1] - np.mean(recent_prices[:-1])) / (np.std(recent_prices[:-1]) + 1e-6)
        
        # Divergence: sentiment moving opposite to price
        divergence = sentiment_change - price_change
        
        return divergence
    
    def _create_opportunity(
        self,
        symbol: str,
        agg_sentiment: Dict,
        divergence: float,
        current_price: float,
        sentiments: List[SentimentData]
    ) -> SentimentArbitrageOpportunity:
        """Create sentiment arbitrage opportunity"""
        
        # Calculate expected move based on sentiment strength
        expected_move_pct = abs(agg_sentiment['score']) * 2.5 * np.sign(divergence)
        
        # Time horizon based on sentiment velocity
        if abs(divergence) > 0.5:
            time_horizon = 60  # 1 hour for strong divergence
        else:
            time_horizon = 240  # 4 hours for moderate divergence
        
        # Calculate targets
        target_price = current_price * (1 + expected_move_pct / 100)
        stop_loss = current_price * (1 - abs(expected_move_pct) / 200)  # Half of expected move
        
        # Risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Find news catalyst if any
        news_sentiments = [s for s in sentiments if s.source == SentimentSource.NEWS]
        news_catalyst = news_sentiments[0].raw_text[:100] if news_sentiments else None
        
        # Calculate social volume spike
        current_volume = agg_sentiment['volume']
        avg_volume = np.mean([s.volume for s in sentiments]) if sentiments else 1
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
        
        return SentimentArbitrageOpportunity(
            opportunity_id=hashlib.md5(f"{symbol}_{datetime.now()}".encode()).hexdigest(),
            symbol=symbol,
            sentiment_signal=agg_sentiment['signal'],
            price_action_divergence=divergence,
            expected_move=expected_move_pct,
            confidence=agg_sentiment['confidence'],
            time_horizon_minutes=time_horizon,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            social_volume_spike=volume_spike,
            news_catalyst=news_catalyst,
            risk_reward_ratio=risk_reward,
            timestamp=datetime.now()
        )

# ==================== Sentiment Execution Strategy ====================

class SentimentExecutor:
    """Executes sentiment-based trading strategies"""
    
    def __init__(self):
        self.active_positions = {}
        self.execution_history = []
        
    async def execute_sentiment_arbitrage(
        self,
        opportunity: SentimentArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute sentiment arbitrage opportunity"""
        
        # Validate opportunity
        if opportunity.risk_reward_ratio < 2:
            return {'status': 'rejected', 'reason': 'Risk-reward ratio too low'}
        
        if opportunity.confidence < 0.6:
            return {'status': 'rejected', 'reason': 'Confidence too low'}
        
        # Calculate position size based on Kelly Criterion
        position_size = self._calculate_position_size(opportunity)
        
        # Create order
        order = {
            'opportunity_id': opportunity.opportunity_id,
            'symbol': opportunity.symbol,
            'side': 'BUY' if opportunity.expected_move > 0 else 'SELL',
            'quantity': position_size,
            'entry_price': opportunity.entry_price,
            'target_price': opportunity.target_price,
            'stop_loss': opportunity.stop_loss,
            'time_horizon': opportunity.time_horizon_minutes,
            'sentiment_signal': opportunity.sentiment_signal.value,
            'timestamp': datetime.now()
        }
        
        # Store position
        self.active_positions[opportunity.opportunity_id] = order
        self.execution_history.append(order)
        
        # Set up monitoring
        asyncio.create_task(self._monitor_position(opportunity.opportunity_id))
        
        return {
            'status': 'executed',
            'order': order,
            'expected_profit': position_size * abs(opportunity.target_price - opportunity.entry_price)
        }
    
    def _calculate_position_size(self, opportunity: SentimentArbitrageOpportunity) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly: f = (p*b - q) / b
        # where p = probability of win, q = probability of loss, b = odds
        
        p = opportunity.confidence
        q = 1 - p
        b = opportunity.risk_reward_ratio
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply Kelly scaling (usually 0.25 Kelly for safety)
        safe_kelly = kelly_fraction * 0.25
        
        # Cap at maximum position size
        max_position = 10000  # $10k max per position
        base_position = 1000   # $1k base position
        
        position_size = min(max_position, base_position * (1 + safe_kelly * 10))
        
        return position_size
    
    async def _monitor_position(self, opportunity_id: str):
        """Monitor and manage active position"""
        position = self.active_positions.get(opportunity_id)
        if not position:
            return
        
        start_time = datetime.now()
        time_horizon = timedelta(minutes=position['time_horizon'])
        
        while datetime.now() - start_time < time_horizon:
            await asyncio.sleep(60)  # Check every minute
            
            # In production, would check actual price and close if targets hit
            # Mock closing logic
            if np.random.random() > 0.8:  # 20% chance to close each check
                self._close_position(opportunity_id, 'target_hit')
                break
        
        # Close if time horizon reached
        if opportunity_id in self.active_positions:
            self._close_position(opportunity_id, 'time_expired')
    
    def _close_position(self, opportunity_id: str, reason: str):
        """Close an active position"""
        if opportunity_id in self.active_positions:
            position = self.active_positions[opportunity_id]
            position['close_reason'] = reason
            position['close_time'] = datetime.now()
            
            # Calculate mock P&L
            if reason == 'target_hit':
                position['pnl'] = position['quantity'] * abs(position['target_price'] - position['entry_price'])
            else:
                position['pnl'] = -position['quantity'] * abs(position['stop_loss'] - position['entry_price']) * 0.5
            
            del self.active_positions[opportunity_id]
            logger.info(f"Closed position {opportunity_id}: {reason}, P&L: ${position['pnl']:.2f}")

# ==================== Main Sentiment Arbitrage System ====================

class SentimentArbitrageSystem:
    """Main sentiment arbitrage system integrator"""
    
    def __init__(self):
        self.finbert = FinBERTAnalyzer()
        self.detector = SentimentArbitrageDetector(self.finbert)
        self.executor = SentimentExecutor()
        self.opportunities_history = deque(maxlen=100)
        
        logger.info("Sentiment Arbitrage System initialized")
    
    async def run(self, symbols: List[str], market_data: Dict[str, Any]):
        """Run sentiment arbitrage detection and execution"""
        
        # Detect opportunities
        opportunities = await self.detector.detect_opportunities(symbols, market_data)
        
        # Store in history
        self.opportunities_history.extend(opportunities)
        
        # Execute profitable opportunities
        execution_results = []
        for opportunity in opportunities:
            if opportunity.expected_move > 1.0 and opportunity.confidence > 0.7:
                result = await self.executor.execute_sentiment_arbitrage(opportunity)
                execution_results.append(result)
                logger.info(f"Sentiment arbitrage: {opportunity.symbol} - {result['status']}")
        
        return {
            'opportunities': opportunities,
            'executions': execution_results,
            'active_positions': len(self.executor.active_positions)
        }
    
    def get_opportunities(self, limit: int = 10) -> List[SentimentArbitrageOpportunity]:
        """Get recent sentiment arbitrage opportunities"""
        return list(self.opportunities_history)[-limit:]
    
    def get_active_positions(self) -> Dict[str, Any]:
        """Get active sentiment-based positions"""
        return self.executor.active_positions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        total_trades = len(self.executor.execution_history)
        profitable_trades = sum(1 for trade in self.executor.execution_history if trade.get('pnl', 0) > 0)
        total_pnl = sum(trade.get('pnl', 0) for trade in self.executor.execution_history)
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'active_positions': len(self.executor.active_positions),
            'opportunities_detected': len(self.opportunities_history)
        }

# ==================== Integration with Main Platform ====================

async def integrate_sentiment_arbitrage(platform_instance):
    """Integrate sentiment arbitrage with main HyperVision platform"""
    
    sentiment_system = SentimentArbitrageSystem()
    
    # Add to platform's arbitrage detection
    platform_instance.sentiment_arbitrage = sentiment_system
    
    # Run sentiment analysis in parallel with other arbitrage detection
    while True:
        try:
            # Get current market data from platform
            market_data = {}
            for symbol in platform_instance.symbols:
                # Get latest price from market buffer
                latest_data = [d for d in platform_instance.data_pipeline.market_buffer if d.symbol == symbol]
                if latest_data:
                    market_data[symbol] = {'price': latest_data[-1].mid_price}
            
            # Run sentiment arbitrage
            results = await sentiment_system.run(platform_instance.symbols, market_data)
            
            # Log results
            if results['opportunities']:
                logger.info(f"Found {len(results['opportunities'])} sentiment arbitrage opportunities")
            
            await asyncio.sleep(60)  # Run every minute
            
        except Exception as e:
            logger.error(f"Sentiment arbitrage error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Test the sentiment arbitrage system
    async def test():
        system = SentimentArbitrageSystem()
        
        symbols = ["BTC", "ETH", "AAPL", "TSLA"]
        market_data = {
            "BTC": {"price": 50000},
            "ETH": {"price": 3000},
            "AAPL": {"price": 150},
            "TSLA": {"price": 800}
        }
        
        results = await system.run(symbols, market_data)
        
        print(f"Found {len(results['opportunities'])} opportunities")
        for opp in results['opportunities']:
            print(f"  {opp.symbol}: {opp.sentiment_signal.value} - Expected move: {opp.expected_move:.2f}%")
        
        print(f"\nExecuted {len(results['executions'])} trades")
        print(f"Active positions: {results['active_positions']}")
    
    asyncio.run(test())