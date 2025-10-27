-- Trading Intelligence Platform Database Schema

-- Market Data Table
CREATE TABLE IF NOT EXISTS market_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  exchange TEXT NOT NULL,
  price REAL NOT NULL,
  volume REAL,
  timestamp INTEGER NOT NULL,
  data_type TEXT NOT NULL, -- 'spot', 'futures', 'index'
  metadata TEXT, -- JSON string for additional data
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Economic Indicators Table
CREATE TABLE IF NOT EXISTS economic_indicators (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  indicator_name TEXT NOT NULL,
  indicator_code TEXT NOT NULL, -- e.g., 'GDP', 'CPI', 'FEDFUNDS'
  value REAL NOT NULL,
  period TEXT NOT NULL, -- e.g., '2024-Q1', '2024-01'
  source TEXT NOT NULL, -- 'FRED', 'IMF', 'BLS'
  timestamp INTEGER NOT NULL,
  metadata TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment Signals Table
CREATE TABLE IF NOT EXISTS sentiment_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL, -- 'twitter', 'reddit', 'google_trends', 'news'
  symbol TEXT,
  sentiment_score REAL NOT NULL, -- -1 to 1
  volume INTEGER,
  confidence REAL, -- 0 to 1
  timestamp INTEGER NOT NULL,
  raw_data TEXT, -- JSON string
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Trading Strategies Table
CREATE TABLE IF NOT EXISTS trading_strategies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy_name TEXT NOT NULL UNIQUE,
  strategy_type TEXT NOT NULL, -- 'momentum', 'mean_reversion', 'arbitrage', 'sentiment', 'factor'
  description TEXT,
  parameters TEXT, -- JSON string
  is_active INTEGER DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Strategy Signals Table
CREATE TABLE IF NOT EXISTS strategy_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy_id INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  signal_type TEXT NOT NULL, -- 'buy', 'sell', 'hold'
  signal_strength REAL NOT NULL, -- 0 to 1
  confidence REAL, -- 0 to 1
  price_target REAL,
  stop_loss REAL,
  timestamp INTEGER NOT NULL,
  metadata TEXT, -- JSON string
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (strategy_id) REFERENCES trading_strategies(id)
);

-- Backtest Results Table
CREATE TABLE IF NOT EXISTS backtest_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy_id INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  start_date INTEGER NOT NULL,
  end_date INTEGER NOT NULL,
  initial_capital REAL NOT NULL,
  final_capital REAL NOT NULL,
  total_return REAL NOT NULL,
  sharpe_ratio REAL,
  max_drawdown REAL,
  win_rate REAL,
  total_trades INTEGER,
  avg_trade_return REAL,
  metadata TEXT, -- JSON string with detailed metrics
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (strategy_id) REFERENCES trading_strategies(id)
);

-- LLM Analysis Table
CREATE TABLE IF NOT EXISTS llm_analysis (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_type TEXT NOT NULL, -- 'market_commentary', 'strategy_recommendation', 'risk_assessment'
  symbol TEXT,
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  confidence REAL,
  context_data TEXT, -- JSON string with input data
  timestamp INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Feature Engineering Cache Table
CREATE TABLE IF NOT EXISTS feature_cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  feature_value REAL NOT NULL,
  calculation_window TEXT, -- e.g., '1d', '1w', '1m'
  timestamp INTEGER NOT NULL,
  metadata TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Market Regime Table
CREATE TABLE IF NOT EXISTS market_regime (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  regime_type TEXT NOT NULL, -- 'bull', 'bear', 'sideways', 'high_volatility', 'low_volatility'
  confidence REAL NOT NULL, -- 0 to 1
  indicators TEXT, -- JSON string with indicator values
  timestamp INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_economic_indicators_code_timestamp ON economic_indicators(indicator_code, timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment_signals_symbol_timestamp ON sentiment_signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_strategy_timestamp ON strategy_signals(strategy_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_type_timestamp ON llm_analysis(analysis_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_cache_name_symbol ON feature_cache(feature_name, symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_regime_timestamp ON market_regime(timestamp);

-- Insert Default Trading Strategies
INSERT INTO trading_strategies (strategy_name, strategy_type, description, parameters) VALUES
  ('Momentum Breakout', 'momentum', 'Buy on price breakouts with volume confirmation', '{"window": 20, "threshold": 2.0, "volume_factor": 1.5}'),
  ('Mean Reversion RSI', 'mean_reversion', 'Buy oversold, sell overbought using RSI', '{"rsi_period": 14, "oversold": 30, "overbought": 70}'),
  ('Statistical Arbitrage', 'arbitrage', 'Cross-exchange and pairs trading opportunities', '{"zscore_threshold": 2.0, "cointegration_pvalue": 0.05}'),
  ('Sentiment-Driven', 'sentiment', 'Trade based on social sentiment and news', '{"sentiment_threshold": 0.6, "volume_threshold": 1000}'),
  ('Multi-Factor', 'factor', 'Combined momentum, value, and quality factors', '{"factors": ["momentum", "value", "quality"], "weights": [0.4, 0.3, 0.3]}');
