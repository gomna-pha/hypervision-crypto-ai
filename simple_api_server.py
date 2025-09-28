#!/usr/bin/env python3
"""
Simplified HyperVision API Server with Mock Data
Ensures all features are working for demonstration
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import datetime
import random
import threading
import time
import asyncio
import uuid
from urllib.parse import urlparse, parse_qs

class MockDataGenerator:
    """Generates realistic mock data for all platform features"""
    
    def __init__(self):
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'MATIC/USD']
        self.exchanges = ['Binance', 'Coinbase', 'Kraken', 'FTX', 'Gemini']
        self.arbitrage_types = ['TRIANGULAR', 'STATISTICAL', 'CROSS_EXCHANGE', 'SPATIAL', 'SENTIMENT']
        
    def generate_market_data(self):
        """Generate realistic market data"""
        data = []
        for symbol in self.symbols:
            base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
            for exchange in self.exchanges[:3]:
                price = base_price * (1 + random.uniform(-0.005, 0.005))
                data.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'bid': price - random.uniform(1, 5),
                    'ask': price + random.uniform(1, 5),
                    'bid_volume': random.uniform(10, 100),
                    'ask_volume': random.uniform(10, 100),
                    'last_price': price,
                    'volume_24h': random.uniform(1000000, 10000000),
                    'vwap': price * 0.999,
                    'spread': random.uniform(0.1, 1),
                    'order_book_imbalance': random.uniform(-0.5, 0.5)
                })
        return data
    
    def generate_opportunities(self):
        """Generate arbitrage opportunities"""
        opportunities = []
        for i in range(5):
            opp_type = random.choice(self.arbitrage_types)
            symbols = random.sample(self.symbols, 2 if opp_type == 'CROSS_EXCHANGE' else 3)
            exchanges = random.sample(self.exchanges, 2)
            
            profit = random.uniform(50, 5000)
            probability = random.uniform(0.6, 0.95)
            
            opportunities.append({
                'opportunity_id': str(uuid.uuid4()),
                'type': opp_type,
                'symbols': symbols,
                'exchanges': exchanges,
                'expected_profit': profit,
                'probability': probability,
                'risk_score': random.uniform(0.1, 0.5),
                'latency_window_ms': random.randint(10, 100),
                'hyperbolic_distance': random.uniform(0.1, 0.9),
                'timestamp': datetime.datetime.now().isoformat(),
                'entry_prices': [random.uniform(100, 50000) for _ in symbols],
                'exit_prices': [random.uniform(100, 50000) for _ in symbols],
                'volume_limits': [random.uniform(1, 100) for _ in symbols],
                'sentiment_score': random.uniform(-1, 1) if opp_type == 'SENTIMENT' else 0
            })
        return opportunities
    
    def generate_embeddings(self):
        """Generate Poincaré disk embeddings"""
        import math
        embeddings = []
        for i in range(30):
            # Generate points within Poincaré disk
            r = random.random() * 0.95  # Keep within disk
            theta = random.random() * 2 * math.pi
            
            embeddings.append({
                'id': i,
                'x': r * math.cos(theta),
                'y': r * math.sin(theta),
                'symbol': self.symbols[i % len(self.symbols)],
                'cluster': i % 4,
                'risk_score': random.random(),
                'distance_to_origin': r,
                'angle': theta
            })
        return embeddings
    
    def generate_metrics(self):
        """Generate platform metrics"""
        return {
            'performance': {
                'avg_latency': random.uniform(0.1, 1.0),
                'throughput': random.randint(8000, 12000),
                'orders_per_second': random.randint(100, 500),
                'success_rate': random.uniform(0.95, 0.99)
            },
            'execution': {
                'total_orders': random.randint(1000, 5000),
                'successful_orders': random.randint(950, 4900),
                'failed_orders': random.randint(10, 100),
                'pending_orders': random.randint(0, 10)
            },
            'risk': {
                'current_daily_pnl': random.uniform(-1000, 5000),
                'position_limits': {
                    'BTC/USD': 100,
                    'ETH/USD': 500
                },
                'max_position_size': 1000000,
                'max_daily_loss': 50000,
                'var_95': random.uniform(1000, 5000),
                'sharpe_ratio': random.uniform(1.5, 3.0)
            },
            'data_pipeline': {
                'market_buffer_size': random.randint(100, 1000),
                'social_buffer_size': random.randint(50, 500),
                'macro_buffer_size': random.randint(10, 100),
                'news_items_processed': random.randint(100, 1000),
                'tweets_analyzed': random.randint(1000, 10000)
            },
            'sentiment': {
                'overall_market': random.uniform(-0.5, 0.5),
                'btc_sentiment': random.uniform(-0.3, 0.7),
                'eth_sentiment': random.uniform(-0.4, 0.6),
                'news_sentiment': random.uniform(-0.2, 0.4),
                'social_sentiment': random.uniform(-0.3, 0.5)
            }
        }

class APIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for API endpoints"""
    
    def __init__(self, *args, **kwargs):
        self.generator = MockDataGenerator()
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Route handlers
        routes = {
            '/api/status': self.handle_status,
            '/api/metrics': self.handle_metrics,
            '/api/opportunities': self.handle_opportunities,
            '/api/market-data': self.handle_market_data,
            '/api/embeddings': self.handle_embeddings,
            '/api/orders': self.handle_orders,
            '/health': self.handle_health
        }
        
        handler = routes.get(path, self.handle_not_found)
        handler()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/execute':
            self.handle_execute()
        else:
            self.handle_not_found()
    
    def send_json_response(self, data, status=200):
        """Send JSON response with proper headers"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def handle_status(self):
        """System status endpoint"""
        data = {
            'status': 'online',
            'timestamp': datetime.datetime.now().isoformat(),
            'uptime_seconds': random.randint(1000, 10000),
            'connections': random.randint(5, 20),
            'platform_metrics': {
                'initialized': True,
                'services': {
                    'hyperbolic_engine': 'active',
                    'sentiment_analyzer': 'active',
                    'arbitrage_detector': 'active',
                    'execution_engine': 'active'
                }
            }
        }
        self.send_json_response(data)
    
    def handle_metrics(self):
        """Platform metrics endpoint"""
        self.send_json_response(self.generator.generate_metrics())
    
    def handle_opportunities(self):
        """Arbitrage opportunities endpoint"""
        opportunities = self.generator.generate_opportunities()
        self.send_json_response({'opportunities': opportunities})
    
    def handle_market_data(self):
        """Market data endpoint"""
        market_data = self.generator.generate_market_data()
        self.send_json_response({'market_data': market_data})
    
    def handle_embeddings(self):
        """Hyperbolic embeddings endpoint"""
        embeddings = self.generator.generate_embeddings()
        self.send_json_response({'embeddings': embeddings})
    
    def handle_orders(self):
        """Orders history endpoint"""
        orders = []
        for i in range(10):
            orders.append({
                'order_id': str(uuid.uuid4()),
                'symbol': random.choice(self.generator.symbols),
                'exchange': random.choice(self.generator.exchanges),
                'side': random.choice(['BUY', 'SELL']),
                'type': 'LIMIT',
                'quantity': random.uniform(0.1, 10),
                'price': random.uniform(100, 50000),
                'status': random.choice(['FILLED', 'PARTIALLY_FILLED', 'PENDING']),
                'filled_quantity': random.uniform(0.05, 10),
                'average_fill_price': random.uniform(100, 50000),
                'fees': random.uniform(1, 50),
                'timestamp': datetime.datetime.now().isoformat()
            })
        self.send_json_response({'orders': orders})
    
    def handle_execute(self):
        """Execute arbitrage opportunity"""
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            request_data = json.loads(body)
            opportunity_id = request_data.get('opportunity_id')
            
            # Generate mock execution response
            orders = []
            for i in range(random.randint(2, 4)):
                orders.append({
                    'order_id': str(uuid.uuid4()),
                    'symbol': random.choice(self.generator.symbols),
                    'exchange': random.choice(self.generator.exchanges),
                    'side': random.choice(['BUY', 'SELL']),
                    'type': 'MARKET',
                    'quantity': random.uniform(0.1, 10),
                    'price': random.uniform(100, 50000),
                    'status': 'FILLED',
                    'filled_quantity': random.uniform(0.1, 10),
                    'average_fill_price': random.uniform(100, 50000),
                    'fees': random.uniform(1, 50),
                    'timestamp': datetime.datetime.now().isoformat()
                })
            
            self.send_json_response({
                'success': True,
                'opportunity_id': opportunity_id,
                'orders': orders,
                'total_profit': random.uniform(50, 500),
                'execution_time_ms': random.uniform(10, 100)
            })
        except Exception as e:
            self.send_json_response({'error': str(e)}, 400)
    
    def handle_health(self):
        """Health check endpoint"""
        self.send_json_response({
            'status': 'healthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'platform': 'initialized'
        })
    
    def handle_not_found(self):
        """404 handler"""
        self.send_json_response({'error': 'Not found'}, 404)
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def run_api_server(port=8000):
    """Run the API server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, APIHandler)
    print(f"API Server running on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_api_server(port)