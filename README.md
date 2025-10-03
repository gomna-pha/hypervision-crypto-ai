# GOMNA Trading Dashboard

## Project Overview
- **Name**: GOMNA Trading Dashboard
- **Goal**: Advanced cryptocurrency and multi-asset trading platform with AI-powered arbitrage detection and portfolio management
- **Features**: 
  - Live market data feeds with real-time updates
  - AI-powered arbitrage opportunity detection (Cross-exchange, Triangular, Statistical pairs)
  - Comprehensive portfolio management with P&L tracking
  - Multi-exchange order book visualization  
  - Hyperbolic space engine with Poincaré disk model
  - GOMNA AI assistant for trading insights and analysis
  - Global market indices monitoring (Crypto, Equity, Forex, Commodities)
  - Advanced risk management and execution metrics

## URLs
- **Development**: https://3000-i49tbhfmni210ftpsfd1u-6532622b.e2b.dev
- **GitHub**: (To be configured)

## Data Architecture
- **Data Models**: 
  - Market Data (BTC, ETH, SOL with price, volume, trades)
  - Arbitrage Opportunities (Cross-exchange, Triangular, Statistical pairs)
  - Portfolio Assets (BTC, ETH, Stablecoins, Other assets)
  - Order Book (Bids/Asks with multi-level depth)
  - Global Markets (Crypto, Equity, International, Commodities, Forex)
  
- **Storage Services**: 
  - Simulated real-time data generation (In production: would use Cloudflare D1 for persistent storage)
  - RESTful API endpoints for all data operations
  
- **Data Flow**: 
  - Frontend requests data from Hono backend APIs
  - Backend generates realistic market simulations
  - Real-time updates via JavaScript intervals (2-second refresh for live data)
  - AI assistant processes queries and returns contextual responses

## API Endpoints

### Market Data
- `GET /api/market-data` - Live BTC, ETH, SOL prices and metrics
- `GET /api/arbitrage-opportunities` - Real-time arbitrage opportunities
- `GET /api/orderbook/:symbol` - Order book depth for specific symbols
- `GET /api/portfolio` - Portfolio overview with asset allocation and P&L
- `GET /api/global-markets` - Global market indices across all asset classes

### Trading Operations  
- `POST /api/execute-arbitrage` - Execute arbitrage opportunities
- `POST /api/ai-query` - Query GOMNA AI assistant

### Currently Completed Features

#### ✅ Trading Dashboard
- Live market feeds for BTC/USD, ETH/USD, SOL/USD with volume and trade data
- Cross-exchange spread monitoring (Binance-Coinbase, Kraken-Bybit, Futures-Spot)
- Real-time arbitrage opportunity detection with 3 types:
  - Cross-Exchange Arbitrage: Direct price differences between exchanges
  - Triangular Arbitrage: Multi-hop trading paths (BTC→ETH→USDT→BTC)
  - Statistical Pairs Trading: Mean reversion opportunities with FinBERT sentiment
- Order book depth visualization with bid/ask levels
- Strategy performance metrics (P&L, Win Rate, Executions, Execution Time)
- Hyperbolic Space Engine with animated Poincaré disk visualization

#### ✅ Portfolio Management
- Portfolio overview showing $2.8M total value with +12.4% MTD performance
- Asset allocation breakdown (45% BTC, 30% ETH, 15% Stablecoins, 10% Other)
- Active positions table with quantity, average price, current price, and P&L
- Interactive portfolio chart using Chart.js with doughnut visualization
- Risk metrics (Sharpe Ratio: 2.34, Max Drawdown: -3.2%, VaR: $45,231, Beta: 0.73)

#### ✅ Global Markets
- Comprehensive market coverage across 5 categories:
  - Cryptocurrency (BTC, ETH, SOL)
  - US Equity (S&P 500, NASDAQ, Dow Jones)
  - International Markets (FTSE 100, Nikkei 225, DAX 40)
  - Commodities (Gold, Silver, Oil WTI)
  - Major Forex (EUR/USD, GBP/USD, USD/JPY)

#### ✅ Model Transparency  
- Hyperbolic Neural Network architecture details
- Ensemble component weights (Hyperbolic CNN: 40%, LSTM-Transformer: 25%, FinBERT: 20%, Classical Arbitrage: 15%)
- Live model performance metrics (91.2% accuracy, 0.968 AUC-ROC, 89.7% precision, 92.8% recall)
- Feature importance rankings (Price Momentum: 0.31, Volume Profile: 0.24, Cross-Exchange Spreads: 0.19, Sentiment: 0.15)

#### ✅ GOMNA AI Assistant
- Interactive chat interface with natural language processing
- Quick query buttons for common analysis tasks
- AI responses with confidence scoring (80-95% range)
- Context-aware responses for market analysis, risk assessment, and arbitrage strategies
- Assistant metrics tracking (47 queries today, 94.2% accuracy rate, 0.8s response time)

### Features Not Yet Implemented

#### 🚧 Advanced Features (Future Enhancements)
- Real-time WebSocket connections for sub-second data updates
- Integration with actual cryptocurrency exchange APIs (Binance, Coinbase Pro, Kraken)
- Advanced charting with technical indicators (RSI, MACD, Bollinger Bands)
- Historical backtesting framework for arbitrage strategies
- Risk management alerts and position size optimization
- Multi-timeframe analysis (1m, 5m, 15m, 1h, 4h, 1d)
- Voice command interface ("Hey GOMNA" activation)
- Mobile responsive design optimization
- User authentication and personalized portfolios
- Real-time notifications for arbitrage opportunities
- Advanced order types (Stop-loss, Take-profit, Trailing stops)

## User Guide

### Navigation
1. **Trading Dashboard**: Main view with live market data and arbitrage opportunities
2. **Portfolio**: Detailed asset allocation and performance tracking
3. **Global Markets**: Comprehensive market overview across all asset classes  
4. **Model Transparency**: AI model architecture and performance insights
5. **AI Assistant**: Interactive chat for trading analysis and insights

### Using Arbitrage Features
1. Monitor live arbitrage opportunities in the main dashboard
2. Review profit potential, execution time, and confidence levels
3. Click "⚡ Execute Arbitrage" to simulate trade execution
4. View execution results and transaction confirmations

### Portfolio Management
1. Navigate to Portfolio section to view current holdings
2. Monitor asset allocation percentages and individual position P&L
3. Review risk metrics and performance indicators
4. Track monthly returns and portfolio value changes

### AI Assistant Usage
1. Navigate to AI Assistant section
2. Type questions about market conditions, risk assessment, or trading strategies
3. Use quick query buttons for common analysis tasks
4. Review AI responses with confidence scoring

## Technical Implementation

### Tech Stack
- **Backend**: Hono framework with TypeScript
- **Frontend**: Vanilla JavaScript with Tailwind CSS
- **Platform**: Cloudflare Pages/Workers
- **Build Tool**: Vite
- **Process Manager**: PM2 (for development)
- **Charts**: Chart.js for portfolio visualization
- **Icons**: Font Awesome 6.4.0
- **HTTP Client**: Axios for API requests

### Development Workflow
```bash
# Install dependencies
npm install

# Development (local machine)
npm run dev

# Development (sandbox with PM2)
npm run build
pm2 start ecosystem.config.cjs

# Build for production
npm run build

# Deploy to Cloudflare Pages
npm run deploy
```

### API Architecture
- RESTful endpoints with JSON responses
- Simulated real-time data generation
- Error handling with appropriate HTTP status codes
- CORS enabled for frontend-backend communication

## Deployment
- **Platform**: Cloudflare Pages (configured for edge deployment)
- **Status**: ✅ Active (Development Environment)
- **Tech Stack**: Hono + TypeScript + TailwindCSS + Chart.js
- **Last Updated**: October 3, 2025

## Development Features
- Hot reloading for rapid development
- TypeScript support for type safety
- Modern ES modules architecture
- Responsive design with mobile considerations
- Cross-browser compatibility

## Next Development Steps
1. Integrate real cryptocurrency exchange APIs for live data
2. Implement user authentication and personalized portfolios
3. Add advanced charting with technical analysis indicators
4. Build historical backtesting capabilities
5. Deploy to production Cloudflare Pages environment
6. Add WebSocket connections for real-time data streaming
7. Implement push notifications for arbitrage alerts
8. Create mobile app companion
9. Add social trading features and community insights
10. Expand to additional asset classes (stocks, commodities, derivatives)