# Agent-Based LLM Arbitrage Platform

## Project Overview
- **Name**: LLM Arbitrage Platform
- **Goal**: Real-time, investor-ready LLM-assisted arbitrage platform using agent-based architecture
- **Features**: Multi-agent data collection, hyperbolic embeddings, LLM fusion brain, deterministic decision engine

## Architecture

```
[Exchange WS/REST] → [Price Agent] ──┐
[Economic APIs] → [Economic Agent] ───┼→ [Fusion Brain (LLM)] → [Decision Engine] → [Execution Agent]
[Social APIs] → [Sentiment Agent] ────┘         ↑
                                                 │
                                    [Hyperbolic Embedding Layer]
```

## Currently Completed Features ✅
1. **Core Infrastructure**
   - TypeScript project structure with modular architecture
   - Configuration management with YAML and environment variables
   - Comprehensive logging system with Winston
   - Base Agent class with Kafka integration

2. **Economic Agent**
   - FRED API integration for economic indicators (CPI, Fed Funds, Unemployment, M2, GDP)
   - VIX data fetching capability
   - Feature calculation (inflation trend, real rate, liquidity bias)
   - Normalized key signal generation
   - HTTP REST API endpoints

3. **Price Agent**
   - WebSocket connections to multiple exchanges (Binance, Coinbase, Kraken)
   - Real-time orderbook management
   - Trade data streaming
   - Spread and depth calculations
   - Volume-weighted average price (VWAP)

4. **Hyperbolic Embedding Layer**
   - Poincaré ball model implementation
   - TensorFlow.js neural network for feature encoding
   - Exponential map for Euclidean to hyperbolic projection
   - K-nearest neighbor search in hyperbolic space
   - Distance calculations and centroid computation

5. **Fusion Brain**
   - Multi-agent data aggregation
   - Anthropic Claude LLM integration
   - Strict JSON schema validation
   - Hyperbolic context enrichment
   - Kafka message publishing

## API Endpoints

### Economic Agent (Port 3001)
- `GET /health` - Health check
- `GET /agents/economicagent/latest` - Latest economic data
- `GET /agents/economicagent/config` - Agent configuration
- `POST /agents/economicagent/update` - Trigger manual update

### Price Agent (Port 3002)
- `GET /health` - Health check
- `GET /agents/priceagent/latest` - Latest price data
- `GET /agents/priceagent/config` - Agent configuration
- `POST /agents/priceagent/update` - Trigger manual update

## Features Not Yet Implemented 🚧
1. **Remaining Agents**
   - Sentiment Agent (Twitter, Reddit, Google Trends integration)
   - Volume Agent (liquidity analysis)
   - Trade Agent (trade flow analysis)
   - Image Agent (visual pattern recognition)
   - Execution Agent (order placement)

2. **Decision Engine**
   - Constraint validation system
   - Bounds checking
   - AOS (Arbitrage Opportunity Score) calculation
   - Risk management rules

3. **Execution System**
   - Exchange API authentication
   - Order placement and management
   - Pre-trade simulation
   - Position tracking

4. **Backtesting Framework**
   - Historical data replay
   - Performance metrics calculation
   - Sharpe ratio, drawdown analysis

5. **Monitoring Dashboard**
   - Real-time metrics visualization
   - System health monitoring
   - P&L tracking
   - Audit logs

## Recommended Next Steps 🎯

### Immediate (High Priority)
1. **Complete Kafka Setup**
   ```bash
   docker-compose up -d kafka zookeeper redis postgres
   ```

2. **Implement Remaining Core Agents**
   - Complete SentimentAgent.ts with Twitter/Reddit APIs
   - Implement VolumeAgent.ts and TradeAgent.ts
   - Add ExecutionAgent.ts with CCXT integration

3. **Build Decision Engine**
   - Implement constraint validation
   - Add bounds checking logic
   - Create execution plan generator

### Short-term (1-2 weeks)
1. **Add Exchange Authentication**
   - Configure API keys in .env
   - Implement signature generation
   - Add rate limiting

2. **Create Monitoring Dashboard**
   - Express server with Socket.io
   - React frontend
   - Grafana integration

3. **Implement Backtesting**
   - Historical data loader
   - Simulation engine
   - Performance analytics

### Medium-term (2-4 weeks)
1. **Production Hardening**
   - Add comprehensive error handling
   - Implement circuit breakers
   - Add health checks
   - Set up monitoring alerts

2. **Performance Optimization**
   - Optimize hyperbolic computations
   - Add caching layers
   - Implement connection pooling

3. **Testing Suite**
   - Unit tests for agents
   - Integration tests
   - Load testing

## Installation & Setup

### Prerequisites
- Node.js 18+
- Docker & Docker Compose
- API Keys (FRED, Twitter, Exchange APIs, Anthropic)

### Quick Start
```bash
# Install dependencies
npm install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure services
docker-compose up -d

# Run tests
npm run test

# Start individual agents
npm run agent:economic
npm run agent:price
npm run fusion

# Or start all with PM2
npm run start:pm2
```

## Configuration

Edit `config.yaml` to customize:
- Agent polling intervals
- Confidence thresholds
- Exchange connections
- LLM parameters
- Hyperbolic embedding dimensions

## Data Architecture
- **Data Models**: Agent outputs, fusion predictions, execution plans
- **Storage Services**: 
  - PostgreSQL (transactional data)
  - Redis (caching)
  - Kafka (message streaming)
- **Data Flow**: Agents → Kafka → Fusion Brain → Decision Engine → Execution

## Tech Stack
- **Runtime**: Node.js with TypeScript
- **ML**: TensorFlow.js
- **LLM**: Anthropic Claude
- **Messaging**: Apache Kafka
- **Database**: PostgreSQL + Redis
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Docker Compose / PM2

## Development Status
- **Platform**: Development
- **Status**: 🟡 In Progress
- **Last Updated**: 2025-10-17

## User Guide

### For Developers
1. Run agents individually for debugging
2. Use PM2 for production-like environment
3. Monitor logs in `./logs` directory
4. Check agent health via HTTP endpoints

### For Investors/Demo
1. Start full system with `npm run start:pm2`
2. Access monitoring dashboard at `http://localhost:8080`
3. View real-time predictions in logs
4. Check system metrics in Grafana at `http://localhost:3001`

## License
MIT

## Contact
For questions or contributions, please open an issue on GitHub.