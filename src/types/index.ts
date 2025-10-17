/**
 * Core Type Definitions for Agent-Based LLM Arbitrage Platform
 */

// ========== Agent Output Types ==========

export interface BaseAgentOutput {
  agent_name: string;
  timestamp: string; // ISO-8601 UTC
  key_signal: number; // Normalized [-1..1] or [0..1]
  confidence: number; // [0..1]
  features: Record<string, number | string>;
}

export interface EconomicAgentOutput extends BaseAgentOutput {
  agent_name: 'EconomicAgent';
  signals: {
    CPI?: number;
    FEDFUNDS?: number;
    UNRATE?: number;
    M2SL?: number;
    GDP?: number;
    VIX?: number;
  };
  features: {
    inflation_trend: number;
    real_rate: number;
    liquidity_bias: number;
  };
}

export interface SentimentAgentOutput extends BaseAgentOutput {
  agent_name: 'SentimentAgent';
  features: {
    google_trend: number;
    avg_polarity: number;
    mention_volume: number;
    verified_ratio: number;
    engagement_weighted_polarity?: number;
    top_keywords?: string[];
  };
}

export interface PriceAgentOutput extends BaseAgentOutput {
  agent_name: 'PriceAgent';
  pair: string;
  exchange: string;
  best_bid: number;
  best_ask: number;
  mid_price: number;
  last_trade_price?: number;
  volume_1m: number;
  vwap_1m?: number;
  orderbook_depth_usd: number;
  open_interest?: number;
  funding_rate?: number;
}

export interface VolumeAgentOutput extends BaseAgentOutput {
  agent_name: 'VolumeAgent';
  pair: string;
  features: {
    volume_1m: number;
    quote_volume_1m: number;
    liquidity_index: number;
    buy_sell_ratio: number;
    volume_spike_flag?: boolean;
    trade_count_1m?: number;
  };
}

export interface TradeAgentOutput extends BaseAgentOutput {
  agent_name: 'TradeAgent';
  pair: string;
  features: {
    taker_buy_volume_1m: number;
    taker_sell_volume_1m: number;
    vwap_1m: number;
    slippage_estimate?: number;
    trade_latency_ms_dist?: number[];
    abnormal_trade_pattern_flag?: boolean;
  };
}

export interface ImageAgentOutput extends BaseAgentOutput {
  agent_name: 'ImageAgent';
  pair: string;
  visual_pattern: string;
  visual_confidence: number;
  embedding_id: string;
  visual_sentiment_score?: number;
}

export type AgentOutput = 
  | EconomicAgentOutput 
  | SentimentAgentOutput 
  | PriceAgentOutput 
  | VolumeAgentOutput 
  | TradeAgentOutput 
  | ImageAgentOutput;

// ========== Fusion Brain Types ==========

export interface FusionInput {
  economic_agent?: EconomicAgentOutput;
  sentiment_agent?: SentimentAgentOutput;
  price_agent?: PriceAgentOutput;
  volume_agent?: VolumeAgentOutput;
  trade_agent?: TradeAgentOutput;
  image_agent?: ImageAgentOutput;
  hyperbolic_neighbors?: HyperbolicNeighbor[];
}

export interface FusionPrediction {
  predicted_spread_pct: number;
  confidence: number;
  direction: 'converge' | 'diverge';
  expected_time_s: number;
  arbitrage_plan: {
    buy: string;
    sell: string;
    notional_usd: number;
  };
  rationale: string;
  risk_flags: string[];
}

export interface HyperbolicNeighbor {
  agent_id: string;
  distance: number;
  timestamp: string;
  features: Record<string, number>;
}

// ========== Decision Engine Types ==========

export interface DecisionConstraints {
  max_open_exposure_pct_of_NAV: number;
  api_health_pause_threshold: number;
  event_blackout_sec: number;
  data_freshness_max_sec: number;
  authorized_exchanges: string[];
}

export interface DecisionBounds {
  min_spread_pct_for_execution: number;
  llm_confidence_threshold: number;
  min_expected_net_profit_usd: number;
  max_hold_time_sec: number;
  max_simultaneous_trades: number;
  max_slippage_pct_estimate: number;
  z_threshold_k: number;
}

export interface ExecutionPlan {
  id: string;
  timestamp: string;
  buy_exchange: string;
  sell_exchange: string;
  pair: string;
  notional_usd: number;
  predicted_spread_pct: number;
  expected_profit_usd: number;
  max_hold_time_sec: number;
  slippage_estimate_pct: number;
  aos_score: number;
  risk_flags: string[];
  status: 'pending' | 'approved' | 'rejected' | 'executing' | 'completed' | 'failed';
}

// ========== Execution Types ==========

export interface Order {
  id: string;
  exchange: string;
  pair: string;
  side: 'buy' | 'sell';
  type: 'limit' | 'market';
  price?: number;
  quantity: number;
  status: 'pending' | 'placed' | 'filled' | 'cancelled' | 'failed';
  placed_at?: string;
  filled_at?: string;
  filled_price?: number;
  filled_quantity?: number;
  fees?: number;
}

export interface ExecutionResult {
  execution_plan_id: string;
  buy_order: Order;
  sell_order: Order;
  realized_spread_pct: number;
  realized_profit_usd: number;
  total_fees_usd: number;
  execution_time_ms: number;
  slippage_pct: number;
}

// ========== Exchange Types ==========

export interface ExchangeConfig {
  name: string;
  ws_url: string;
  rest_url: string;
  api_key: string;
  api_secret: string;
  passphrase?: string;
  testnet?: boolean;
}

export interface OrderbookUpdate {
  exchange: string;
  pair: string;
  timestamp: number;
  bids: [number, number][]; // [price, quantity]
  asks: [number, number][];
  sequence?: number;
}

export interface TradeUpdate {
  exchange: string;
  pair: string;
  timestamp: number;
  price: number;
  quantity: number;
  side: 'buy' | 'sell';
  trade_id: string;
}

// ========== Validation Types ==========

export interface ValidationOutcome {
  execution_plan_id: string;
  predicted_spread_pct: number;
  realized_spread_pct: number;
  prediction_error: number;
  predicted_profit_usd: number;
  realized_profit_usd: number;
  profit_error: number;
  execution_time_s: number;
  timestamp: string;
}

// ========== Configuration Types ==========

export interface AOSWeights {
  price: number;
  sentiment: number;
  volume: number;
  image: number;
  risk: number;
}

export interface HyperbolicConfig {
  model: 'poincare' | 'lorentz';
  curvature: number;
  embedding_dim: number;
  neighbor_k: number;
  update_interval_sec: number;
}

export interface SystemConfig {
  log_level: 'debug' | 'info' | 'warn' | 'error';
  log_dir: string;
  data_dir: string;
  checkpoint_interval_sec: number;
  health_check_port: number;
}

// ========== Kafka Message Types ==========

export interface KafkaMessage<T> {
  key?: string;
  value: T;
  timestamp: string;
  partition?: number;
  offset?: string;
}

// ========== Backtest Types ==========

export interface BacktestConfig {
  start_date: string;
  end_date: string;
  initial_capital_usd: number;
  fee_model: {
    maker_fee_pct: number;
    taker_fee_pct: number;
  };
  slippage_model: {
    base_slippage_pct: number;
    volume_impact_factor: number;
  };
}

export interface BacktestResult {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl_usd: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  avg_trade_duration_s: number;
  avg_profit_per_trade_usd: number;
  best_trade: ExecutionResult;
  worst_trade: ExecutionResult;
}

// ========== Monitoring Types ==========

export interface SystemHealth {
  timestamp: string;
  agents: Record<string, {
    status: 'healthy' | 'degraded' | 'unhealthy';
    last_update: string;
    error_rate: number;
    latency_ms: number;
  }>;
  exchanges: Record<string, {
    connected: boolean;
    latency_ms: number;
    error_count: number;
  }>;
  kafka: {
    connected: boolean;
    lag: number;
  };
  database: {
    connected: boolean;
    query_time_ms: number;
  };
  llm: {
    available: boolean;
    response_time_ms: number;
    tokens_used: number;
  };
}

export interface PerformanceMetrics {
  timestamp: string;
  total_pnl_usd: number;
  open_positions: number;
  total_exposure_usd: number;
  sharpe_ratio_30d: number;
  win_rate_24h: number;
  avg_spread_captured_pct: number;
  prediction_accuracy: number;
}