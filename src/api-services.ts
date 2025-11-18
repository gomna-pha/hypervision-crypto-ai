// API Services - Replace Simulated Data with Real Free APIs
// All APIs used are FREE and do NOT require API keys

// Cache to respect rate limits
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 60000; // 1 minute cache

function getCached(key: string) {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  return null;
}

function setCache(key: string, data: any) {
  cache.set(key, { data, timestamp: Date.now() });
}

// ===================================================================
// 1. CROSS-EXCHANGE AGENT - Real BTC/ETH Prices
// ===================================================================

export async function getCrossExchangePrices() {
  const cacheKey = 'cross_exchange_prices';
  const cached = getCached(cacheKey);
  if (cached) return cached;

  try {
    // Get prices from multiple sources
    const [coingecko, binance, coinbase] = await Promise.all([
      // CoinGecko - Multi-exchange aggregated prices
      fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true')
        .then(r => r.json())
        .catch(() => null),
      
      // Binance - Real-time ticker
      fetch('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT')
        .then(r => r.json())
        .catch(() => null),
      
      // Coinbase - Spot price
      fetch('https://api.coinbase.com/v2/prices/BTC-USD/spot')
        .then(r => r.json())
        .catch(() => null)
    ]);

    let btcPrice = 95000; // Fallback
    let ethPrice = 3500; // Fallback
    let volume24h = 30000000000; // Fallback
    let change24h = 0;

    // Use CoinGecko as primary source
    if (coingecko && coingecko.bitcoin) {
      btcPrice = coingecko.bitcoin.usd;
      ethPrice = coingecko.ethereum?.usd || 3500;
      volume24h = coingecko.bitcoin.usd_24h_vol || 30000000000;
      change24h = coingecko.bitcoin.usd_24h_change || 0;
    }

    // Use Binance as backup
    if (!coingecko && binance && binance.lastPrice) {
      btcPrice = parseFloat(binance.lastPrice);
      volume24h = parseFloat(binance.volume) * btcPrice;
      change24h = parseFloat(binance.priceChangePercent || '0');
    }

    // Use Coinbase as tertiary backup
    if (!coingecko && !binance && coinbase && coinbase.data) {
      btcPrice = parseFloat(coinbase.data.amount);
    }

    // Calculate realistic spread based on exchanges
    const binancePrice = binance?.lastPrice ? parseFloat(binance.lastPrice) : btcPrice;
    const coinbasePrice = coinbase?.data?.amount ? parseFloat(coinbase.data.amount) : btcPrice;
    const spread = Math.abs((coinbasePrice - binancePrice) / binancePrice) * 100;

    const result = {
      btcPrice,
      ethPrice,
      volume24h,
      change24h,
      spread: Math.max(spread, 0.05), // Minimum 0.05% spread
      binancePrice,
      coinbasePrice,
      source: coingecko ? 'coingecko' : (binance ? 'binance' : 'coinbase'),
      timestamp: Date.now()
    };

    setCache(cacheKey, result);
    return result;

  } catch (error) {
    console.error('Cross-exchange API error:', error);
    return null; // Will fallback to simulated
  }
}

// ===================================================================
// 2. SENTIMENT AGENT - Real Fear & Greed Index
// ===================================================================

export async function getFearGreedIndex() {
  const cacheKey = 'fear_greed_index';
  const cached = getCached(cacheKey);
  if (cached) return cached;

  try {
    // Alternative.me Crypto Fear & Greed Index - FREE, NO KEY
    const response = await fetch('https://api.alternative.me/fng/?limit=1');
    const data = await response.json();

    if (data && data.data && data.data[0]) {
      const fearGreed = parseInt(data.data[0].value);
      const classification = data.data[0].value_classification; // Extreme Fear, Fear, Neutral, Greed, Extreme Greed
      
      const result = {
        fearGreed,
        classification,
        timestamp: parseInt(data.data[0].timestamp) * 1000,
        source: 'alternative.me'
      };

      setCache(cacheKey, result);
      return result;
    }

    return null;

  } catch (error) {
    console.error('Fear & Greed API error:', error);
    return null; // Will fallback to simulated
  }
}

// ===================================================================
// 3. ON-CHAIN AGENT - Basic Blockchain Stats
// ===================================================================

export async function getOnChainData() {
  const cacheKey = 'onchain_data';
  const cached = getCached(cacheKey);
  if (cached) return cached;

  try {
    // Blockchain.info Stats - FREE, NO KEY
    const response = await fetch('https://blockchain.info/stats?format=json');
    const data = await response.json();

    if (data) {
      const result = {
        marketCap: data.market_price_usd * data.totalbc / 100000000,
        transactions24h: data.n_tx,
        hashRate: data.hash_rate,
        difficulty: data.difficulty,
        totalBTC: data.totalbc / 100000000,
        blockchainSize: data.blocks_size,
        avgBlockTime: data.minutes_between_blocks,
        source: 'blockchain.info',
        timestamp: Date.now()
      };

      setCache(cacheKey, result);
      return result;
    }

    return null;

  } catch (error) {
    console.error('On-chain API error:', error);
    return null; // Will fallback to simulated
  }
}

// ===================================================================
// 4. MARKET DATA - CoinGecko Global Stats
// ===================================================================

export async function getGlobalMarketData() {
  const cacheKey = 'global_market';
  const cached = getCached(cacheKey);
  if (cached) return cached;

  try {
    // CoinGecko Global Stats - FREE, NO KEY
    const response = await fetch('https://api.coingecko.com/api/v3/global');
    const data = await response.json();

    if (data && data.data) {
      const result = {
        totalMarketCap: data.data.total_market_cap?.usd || 0,
        total24hVolume: data.data.total_volume?.usd || 0,
        btcDominance: data.data.market_cap_percentage?.btc || 0,
        ethDominance: data.data.market_cap_percentage?.eth || 0,
        marketCapChange24h: data.data.market_cap_change_percentage_24h_usd || 0,
        activeExchanges: data.data.markets || 0,
        source: 'coingecko',
        timestamp: Date.now()
      };

      setCache(cacheKey, result);
      return result;
    }

    return null;

  } catch (error) {
    console.error('Global market API error:', error);
    return null; // Will fallback to simulated
  }
}

// ===================================================================
// 5. MULTI-EXCHANGE ARBITRAGE OPPORTUNITIES
// ===================================================================

export async function calculateArbitrageOpportunities() {
  const cacheKey = 'arbitrage_opps';
  const cached = getCached(cacheKey);
  if (cached) return cached;

  try {
    // Get prices from multiple exchanges
    const [binance, coinbase, kraken] = await Promise.all([
      fetch('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
        .then(r => r.json())
        .catch(() => null),
      fetch('https://api.coinbase.com/v2/prices/BTC-USD/spot')
        .then(r => r.json())
        .catch(() => null),
      fetch('https://api.kraken.com/0/public/Ticker?pair=XBTUSD')
        .then(r => r.json())
        .catch(() => null)
    ]);

    const opportunities = [];

    const binancePrice = binance?.price ? parseFloat(binance.price) : null;
    const coinbasePrice = coinbase?.data?.amount ? parseFloat(coinbase.data.amount) : null;
    const krakenPrice = kraken?.result?.XXBTZUSD?.c?.[0] ? parseFloat(kraken.result.XXBTZUSD.c[0]) : null;

    // Calculate all possible arbitrage pairs
    if (binancePrice && coinbasePrice) {
      const spread = Math.abs((coinbasePrice - binancePrice) / binancePrice) * 100;
      if (spread > 0.1) { // Minimum 0.1% spread
        opportunities.push({
          buyExchange: binancePrice < coinbasePrice ? 'Binance' : 'Coinbase',
          sellExchange: binancePrice < coinbasePrice ? 'Coinbase' : 'Binance',
          spread,
          buyPrice: Math.min(binancePrice, coinbasePrice),
          sellPrice: Math.max(binancePrice, coinbasePrice)
        });
      }
    }

    if (binancePrice && krakenPrice) {
      const spread = Math.abs((krakenPrice - binancePrice) / binancePrice) * 100;
      if (spread > 0.1) {
        opportunities.push({
          buyExchange: binancePrice < krakenPrice ? 'Binance' : 'Kraken',
          sellExchange: binancePrice < krakenPrice ? 'Kraken' : 'Binance',
          spread,
          buyPrice: Math.min(binancePrice, krakenPrice),
          sellPrice: Math.max(binancePrice, krakenPrice)
        });
      }
    }

    if (coinbasePrice && krakenPrice) {
      const spread = Math.abs((krakenPrice - coinbasePrice) / coinbasePrice) * 100;
      if (spread > 0.1) {
        opportunities.push({
          buyExchange: coinbasePrice < krakenPrice ? 'Coinbase' : 'Kraken',
          sellExchange: coinbasePrice < krakenPrice ? 'Kraken' : 'Coinbase',
          spread,
          buyPrice: Math.min(coinbasePrice, krakenPrice),
          sellPrice: Math.max(coinbasePrice, krakenPrice)
        });
      }
    }

    const result = {
      opportunities,
      timestamp: Date.now()
    };

    setCache(cacheKey, result);
    return result;

  } catch (error) {
    console.error('Arbitrage calculation error:', error);
    return null; // Will fallback to simulated
  }
}

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

export function clearCache() {
  cache.clear();
}

export function getCacheStats() {
  return {
    size: cache.size,
    keys: Array.from(cache.keys()),
    oldestEntry: Math.min(...Array.from(cache.values()).map(v => v.timestamp))
  };
}
