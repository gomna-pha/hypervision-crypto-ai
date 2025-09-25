"""
HyperVision AI - FinBERT Sentiment Analysis Engine
Low-latency news and social media sentiment analysis for HFT arbitrage

Integrates with:
- Twitter/X API for real-time sentiment
- News feeds (Reuters, Bloomberg, etc.)
- Hyperbolic embeddings for hierarchical impact analysis
- Sub-100ms inference for HFT requirements
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict, deque
import threading
import hashlib

# For FinBERT inference (in production, use optimized ONNX Runtime)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logging.warning("FinBERT dependencies not available. Using mock sentiment analysis.")

logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    """Represents a sentiment-driven trading signal"""
    symbol: str
    sentiment_score: float  # -1 (very negative) to +1 (very positive)
    confidence: float       # 0 to 1
    impact_magnitude: float # Expected price impact
    time_decay_factor: float # How long signal is valid
    source_type: str        # 'news', 'twitter', 'reddit', etc.
    entity_mentions: List[str]
    hierarchical_impact: Dict[str, float]  # Parent/child entity impacts
    timestamp: datetime
    raw_text: str
    metadata: Dict[str, Any]

@dataclass
class NewsEvent:
    """Represents a financial news event"""
    title: str
    content: str
    source: str
    entities_mentioned: List[str]
    timestamp: datetime
    url: Optional[str]
    sentiment_signals: List[SentimentSignal]

class FinBERTEngine:
    """Optimized FinBERT engine for real-time sentiment analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for consistent low latency
        
        # Performance optimization
        self.max_sequence_length = 256  # Shorter sequences for speed
        self.batch_size = config.get('batch_size', 8)
        self.inference_cache = {}
        self.cache_ttl = config.get('cache_ttl_seconds', 60)
        
        # Entity recognition patterns
        self.entity_patterns = {
            'stock_symbols': r'\b[A-Z]{1,5}\b',  # Basic stock symbol pattern
            'crypto_symbols': r'\b(BTC|ETH|BNB|SOL|ADA|DOT|MATIC|AVAX|ATOM|LINK)\b',
            'company_names': r'\b(Apple|Microsoft|Google|Amazon|Tesla|Meta|Netflix|Nvidia)\b',
            'indices': r'\b(S&P\s?500|NASDAQ|DOW|FTSE|DAX|NIKKEI)\b'
        }
        
        # Load model if available
        if FINBERT_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FinBERT model for inference"""
        try:
            model_name = self.config.get('finbert_model', 'ProsusAI/finbert')
            
            logger.info(f"Loading FinBERT model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Optimize for inference
            self.model.eval()
            if hasattr(self.model, 'half'):
                self.model.half()  # Use FP16 for speed
            
            logger.info("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    async def analyze_sentiment(self, text: str, entities: List[str] = None) -> SentimentSignal:
        """Analyze sentiment of financial text"""
        
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.inference_cache:
            cached_result, timestamp = self.inference_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_result
        
        start_time = time.time()
        
        if self.model and self.tokenizer:
            sentiment_signal = await self._finbert_inference(text, entities)
        else:
            sentiment_signal = self._mock_sentiment_analysis(text, entities)
        
        inference_time = (time.time() - start_time) * 1000
        sentiment_signal.metadata['inference_time_ms'] = inference_time
        
        # Cache result
        self.inference_cache[cache_key] = (sentiment_signal, datetime.now())
        
        # Clean old cache entries periodically
        if len(self.inference_cache) > 1000:
            self._clean_cache()
        
        return sentiment_signal
    
    async def _finbert_inference(self, text: str, entities: List[str] = None) -> SentimentSignal:
        """Run FinBERT inference on text"""
        
        # Preprocess text for financial context
        processed_text = self._preprocess_financial_text(text)
        
        # Tokenize with truncation for speed
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_sequence_length,
            padding=True
        )
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to sentiment score (-1 to +1)
        # FinBERT typically outputs [negative, neutral, positive]
        probs = predictions[0].cpu().numpy()
        
        if len(probs) >= 3:
            negative_prob = probs[0]
            neutral_prob = probs[1] 
            positive_prob = probs[2]
            
            # Calculate weighted sentiment score
            sentiment_score = positive_prob - negative_prob
            confidence = max(positive_prob, negative_prob)  # Confidence in non-neutral prediction
        else:
            sentiment_score = 0.0
            confidence = 0.5
        
        # Extract entities mentioned in text
        if entities is None:
            entities = self._extract_entities(text)
        
        # Calculate hierarchical impact (simplified)
        hierarchical_impact = self._calculate_hierarchical_impact(entities, sentiment_score)
        
        return SentimentSignal(
            symbol=entities[0] if entities else "UNKNOWN",
            sentiment_score=float(sentiment_score),
            confidence=float(confidence),
            impact_magnitude=abs(sentiment_score) * confidence,
            time_decay_factor=self._calculate_time_decay(text),
            source_type="news",
            entity_mentions=entities,
            hierarchical_impact=hierarchical_impact,
            timestamp=datetime.now(),
            raw_text=text[:500],  # Truncate for storage
            metadata={
                'model_used': 'finbert',
                'text_length': len(text),
                'entities_found': len(entities)
            }
        )
    
    def _mock_sentiment_analysis(self, text: str, entities: List[str] = None) -> SentimentSignal:
        """Mock sentiment analysis when FinBERT is unavailable"""
        
        # Simple keyword-based sentiment for demo
        positive_keywords = ['bullish', 'growth', 'profit', 'beat', 'surge', 'rally', 'upgrade', 'strong']
        negative_keywords = ['bearish', 'loss', 'decline', 'crash', 'downgrade', 'weak', 'miss', 'fall']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_keywords if word in text_lower)
        negative_score = sum(1 for word in negative_keywords if word in text_lower)
        
        total_score = positive_score + negative_score
        if total_score > 0:
            sentiment_score = (positive_score - negative_score) / total_score
            confidence = min(total_score / 10, 1.0)
        else:
            sentiment_score = 0.0
            confidence = 0.1
        
        entities = entities or self._extract_entities(text)
        hierarchical_impact = self._calculate_hierarchical_impact(entities, sentiment_score)
        
        return SentimentSignal(
            symbol=entities[0] if entities else "UNKNOWN",
            sentiment_score=sentiment_score,
            confidence=confidence,
            impact_magnitude=abs(sentiment_score) * confidence,
            time_decay_factor=self._calculate_time_decay(text),
            source_type="mock",
            entity_mentions=entities,
            hierarchical_impact=hierarchical_impact,
            timestamp=datetime.now(),
            raw_text=text[:500],
            metadata={
                'model_used': 'mock',
                'positive_keywords': positive_score,
                'negative_keywords': negative_score
            }
        )
    
    def _preprocess_financial_text(self, text: str) -> str:
        """Preprocess text for financial sentiment analysis"""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize financial terms
        financial_replacements = {
            'Q1': 'first quarter',
            'Q2': 'second quarter', 
            'Q3': 'third quarter',
            'Q4': 'fourth quarter',
            'YoY': 'year over year',
            'MoM': 'month over month',
            'EPS': 'earnings per share',
            'P/E': 'price to earnings',
            'IPO': 'initial public offering'
        }
        
        for abbrev, full_form in financial_replacements.items():
            text = text.replace(abbrev, full_form)
        
        return text
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract financial entities from text"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def _calculate_hierarchical_impact(self, entities: List[str], sentiment_score: float) -> Dict[str, float]:
        """Calculate hierarchical impact across related entities"""
        
        # Simplified hierarchical relationships
        hierarchies = {
            'AAPL': ['NASDAQ', 'TECH', 'SP500'],
            'MSFT': ['NASDAQ', 'TECH', 'SP500'],
            'GOOGL': ['NASDAQ', 'TECH', 'SP500'],
            'TSLA': ['NASDAQ', 'AUTO', 'SP500'],
            'BTC': ['CRYPTO', 'ALTCOINS'],
            'ETH': ['CRYPTO', 'ALTCOINS', 'DeFi']
        }
        
        impact_map = {}
        
        for entity in entities:
            entity_upper = entity.upper()
            
            # Direct impact
            impact_map[entity_upper] = sentiment_score
            
            # Hierarchical propagation with decay
            if entity_upper in hierarchies:
                for i, parent_entity in enumerate(hierarchies[entity_upper]):
                    decay_factor = 0.5 ** (i + 1)  # Exponential decay up hierarchy
                    impact_map[parent_entity] = sentiment_score * decay_factor
        
        return impact_map
    
    def _calculate_time_decay(self, text: str) -> float:
        """Calculate how quickly sentiment signal should decay"""
        
        # Keywords that suggest urgency/immediacy
        urgent_keywords = ['breaking', 'urgent', 'alert', 'now', 'just', 'immediate']
        long_term_keywords = ['outlook', 'forecast', 'guidance', 'strategy', 'plan']
        
        text_lower = text.lower()
        urgent_count = sum(1 for word in urgent_keywords if word in text_lower)
        long_term_count = sum(1 for word in long_term_keywords if word in text_lower)
        
        # Base decay: 0.5 = moderate (hours), 0.9 = slow (days), 0.1 = fast (minutes)
        if urgent_count > long_term_count:
            return 0.1  # Fast decay for urgent news
        elif long_term_count > urgent_count:
            return 0.9  # Slow decay for strategic news
        else:
            return 0.5  # Moderate decay for general news
    
    def _clean_cache(self):
        """Clean expired entries from inference cache"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.inference_cache.items()
            if (current_time - timestamp).total_seconds() > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.inference_cache[key]

class SentimentDataStreamer:
    """Streams real-time sentiment data from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.finbert_engine = FinBERTEngine(config.get('finbert', {}))
        
        # Data sources configuration
        self.news_sources = config.get('news_sources', {
            'reuters': {'enabled': True, 'priority': 1},
            'bloomberg': {'enabled': True, 'priority': 1},
            'cnbc': {'enabled': True, 'priority': 2}
        })
        
        self.social_sources = config.get('social_sources', {
            'twitter': {'enabled': True, 'priority': 1},
            'reddit': {'enabled': True, 'priority': 3}
        })
        
        # Signal processing
        self.signal_queue = asyncio.Queue(maxsize=1000)
        self.processed_signals = deque(maxlen=10000)
        self.signal_aggregator = SentimentSignalAggregator()
        
        # Rate limiting
        self.api_rate_limits = {
            'twitter': {'calls_per_minute': 100, 'last_reset': datetime.now(), 'current_count': 0},
            'news_api': {'calls_per_minute': 60, 'last_reset': datetime.now(), 'current_count': 0}
        }
    
    async def start_streaming(self):
        """Start all sentiment data streams"""
        logger.info("🚀 Starting FinBERT Sentiment Streaming Engine")
        
        # Start individual stream tasks
        tasks = [
            asyncio.create_task(self._stream_news()),
            asyncio.create_task(self._stream_twitter()),
            asyncio.create_task(self._process_sentiment_signals()),
            asyncio.create_task(self._aggregate_signals())
        ]
        
        logger.info("✅ All sentiment streams started")
        
        # Keep tasks running
        await asyncio.gather(*tasks)
    
    async def _stream_news(self):
        """Stream financial news for sentiment analysis"""
        while True:
            try:
                # In production, integrate with real news APIs
                # For demo, simulate news events
                mock_news = await self._fetch_mock_news()
                
                for news_item in mock_news:
                    # Analyze sentiment
                    sentiment_signal = await self.finbert_engine.analyze_sentiment(
                        news_item['content'],
                        news_item.get('entities', [])
                    )
                    
                    # Add news-specific metadata
                    sentiment_signal.source_type = 'news'
                    sentiment_signal.metadata.update({
                        'news_source': news_item['source'],
                        'news_title': news_item['title'],
                        'news_url': news_item.get('url', '')
                    })
                    
                    await self.signal_queue.put(sentiment_signal)
                
                await asyncio.sleep(10)  # Fetch news every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in news streaming: {e}")
                await asyncio.sleep(5)
    
    async def _stream_twitter(self):
        """Stream Twitter/X sentiment for financial entities"""
        while True:
            try:
                # Rate limit check
                if not self._check_rate_limit('twitter'):
                    await asyncio.sleep(1)
                    continue
                
                # In production, integrate with Twitter API v2
                mock_tweets = await self._fetch_mock_tweets()
                
                for tweet in mock_tweets:
                    sentiment_signal = await self.finbert_engine.analyze_sentiment(
                        tweet['text'],
                        tweet.get('entities', [])
                    )
                    
                    sentiment_signal.source_type = 'twitter'
                    sentiment_signal.metadata.update({
                        'twitter_author': tweet['author'],
                        'twitter_followers': tweet.get('followers', 0),
                        'retweet_count': tweet.get('retweets', 0)
                    })
                    
                    # Weight by follower count and engagement
                    follower_weight = min(tweet.get('followers', 0) / 10000, 2.0)
                    engagement_weight = min(tweet.get('retweets', 0) / 100, 1.5)
                    sentiment_signal.confidence *= (1 + follower_weight + engagement_weight) / 3
                    
                    await self.signal_queue.put(sentiment_signal)
                
                await asyncio.sleep(2)  # High frequency Twitter monitoring
                
            except Exception as e:
                logger.error(f"Error in Twitter streaming: {e}")
                await asyncio.sleep(5)
    
    async def _process_sentiment_signals(self):
        """Process incoming sentiment signals"""
        while True:
            try:
                signal = await self.signal_queue.get()
                
                # Apply time decay
                age_seconds = (datetime.now() - signal.timestamp).total_seconds()
                decay_factor = np.exp(-age_seconds * signal.time_decay_factor)
                signal.impact_magnitude *= decay_factor
                
                # Store processed signal
                self.processed_signals.append(signal)
                
                # Log significant signals
                if signal.impact_magnitude > 0.5:
                    logger.info(f"📊 Strong sentiment signal: {signal.symbol} = {signal.sentiment_score:.3f} (confidence: {signal.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing sentiment signal: {e}")
    
    async def _aggregate_signals(self):
        """Aggregate sentiment signals by entity and time window"""
        while True:
            try:
                await asyncio.sleep(5)  # Aggregate every 5 seconds
                
                current_time = datetime.now()
                
                # Get recent signals (last 5 minutes)
                recent_signals = [
                    signal for signal in self.processed_signals
                    if (current_time - signal.timestamp).total_seconds() <= 300
                ]
                
                # Aggregate by symbol
                aggregated = self.signal_aggregator.aggregate_by_symbol(recent_signals)
                
                # Generate trading signals for strong sentiment shifts
                for symbol, agg_signal in aggregated.items():
                    if agg_signal['impact_magnitude'] > 0.7:  # Strong signal threshold
                        logger.info(f"🎯 Trading signal generated: {symbol} sentiment = {agg_signal['weighted_sentiment']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in signal aggregation: {e}")
    
    def _check_rate_limit(self, service: str) -> bool:
        """Check if we can make API call within rate limits"""
        now = datetime.now()
        rate_info = self.api_rate_limits.get(service, {})
        
        # Reset counter if minute has passed
        if (now - rate_info['last_reset']).total_seconds() >= 60:
            rate_info['current_count'] = 0
            rate_info['last_reset'] = now
        
        # Check if under limit
        if rate_info['current_count'] < rate_info.get('calls_per_minute', 100):
            rate_info['current_count'] += 1
            return True
        
        return False
    
    async def _fetch_mock_news(self) -> List[Dict[str, Any]]:
        """Mock news fetching for demonstration"""
        mock_articles = [
            {
                'title': 'Apple Reports Strong Q4 Earnings Beat',
                'content': 'Apple Inc. reported quarterly earnings that beat analyst expectations, driven by strong iPhone sales and services revenue growth.',
                'source': 'reuters',
                'entities': ['AAPL', 'Apple'],
                'url': 'https://reuters.com/mock-article'
            },
            {
                'title': 'Bitcoin Surges on Institutional Adoption',
                'content': 'Bitcoin prices rally as major financial institutions announce increased cryptocurrency adoption and investment strategies.',
                'source': 'bloomberg',
                'entities': ['BTC', 'Bitcoin'],
                'url': 'https://bloomberg.com/mock-article'
            }
        ]
        
        return mock_articles
    
    async def _fetch_mock_tweets(self) -> List[Dict[str, Any]]:
        """Mock Twitter data for demonstration"""
        mock_tweets = [
            {
                'text': '$AAPL looking bullish after earnings beat. Strong fundamentals and growing services revenue.',
                'author': 'financial_analyst',
                'followers': 50000,
                'retweets': 125,
                'entities': ['AAPL']
            },
            {
                'text': 'Tesla delivery numbers disappointing. Production challenges continue to weigh on $TSLA',
                'author': 'auto_expert',
                'followers': 25000,
                'retweets': 80,
                'entities': ['TSLA', 'Tesla']
            }
        ]
        
        return mock_tweets
    
    def get_latest_sentiment(self, symbol: str, lookback_minutes: int = 5) -> Dict[str, Any]:
        """Get latest aggregated sentiment for a symbol"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        relevant_signals = [
            signal for signal in self.processed_signals
            if signal.timestamp >= cutoff_time and symbol in signal.entity_mentions
        ]
        
        if not relevant_signals:
            return {'sentiment': 0.0, 'confidence': 0.0, 'signal_count': 0}
        
        # Weighted average by confidence and recency
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for signal in relevant_signals:
            recency_weight = 1.0 - (datetime.now() - signal.timestamp).total_seconds() / (lookback_minutes * 60)
            weight = signal.confidence * recency_weight
            weighted_sentiment += signal.sentiment_score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_sentiment /= total_weight
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': total_weight / len(relevant_signals) if relevant_signals else 0.0,
            'signal_count': len(relevant_signals),
            'latest_timestamp': max(signal.timestamp for signal in relevant_signals) if relevant_signals else None
        }

class SentimentSignalAggregator:
    """Aggregates sentiment signals for trading decisions"""
    
    def aggregate_by_symbol(self, signals: List[SentimentSignal]) -> Dict[str, Dict[str, Any]]:
        """Aggregate sentiment signals by symbol"""
        symbol_groups = defaultdict(list)
        
        # Group signals by symbol
        for signal in signals:
            for entity in signal.entity_mentions:
                symbol_groups[entity].append(signal)
        
        aggregated = {}
        
        for symbol, symbol_signals in symbol_groups.items():
            if not symbol_signals:
                continue
            
            # Calculate weighted sentiment
            total_weight = 0.0
            weighted_sentiment = 0.0
            
            for signal in symbol_signals:
                weight = signal.confidence * signal.impact_magnitude
                weighted_sentiment += signal.sentiment_score * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_sentiment /= total_weight
            
            # Calculate aggregate metrics
            aggregated[symbol] = {
                'weighted_sentiment': weighted_sentiment,
                'signal_count': len(symbol_signals),
                'avg_confidence': np.mean([s.confidence for s in symbol_signals]),
                'impact_magnitude': np.sum([s.impact_magnitude for s in symbol_signals]),
                'sources': list(set(s.source_type for s in symbol_signals)),
                'latest_timestamp': max(s.timestamp for s in symbol_signals)
            }
        
        return aggregated

# Default configuration
DEFAULT_SENTIMENT_CONFIG = {
    'finbert': {
        'finbert_model': 'ProsusAI/finbert',
        'batch_size': 8,
        'cache_ttl_seconds': 60
    },
    'news_sources': {
        'reuters': {'enabled': True, 'priority': 1},
        'bloomberg': {'enabled': True, 'priority': 1},
        'cnbc': {'enabled': True, 'priority': 2}
    },
    'social_sources': {
        'twitter': {'enabled': True, 'priority': 1}
    }
}

async def main():
    """Main function to run sentiment analysis engine"""
    streamer = SentimentDataStreamer(DEFAULT_SENTIMENT_CONFIG)
    await streamer.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())