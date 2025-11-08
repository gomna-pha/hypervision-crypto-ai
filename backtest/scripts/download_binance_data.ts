/**
 * Download Historical Binance OHLCV Data
 * Period: November 2021 - November 2024 (3 years)
 * Resolution: 1-hour bars
 * Symbols: BTC/USDT, ETH/USDT
 */

import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

interface KlineData {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteVolume: string;
  trades: number;
  takerBuyBase: string;
  takerBuyQuote: string;
}

interface OHLCVBar {
  timestamp: number;
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const BINANCE_BASE_URL = 'https://api.binance.com';
const DATA_DIR = path.join(__dirname, '../data');

// Backtest period: Nov 1, 2021 - Nov 1, 2024
const START_DATE = new Date('2021-11-01T00:00:00Z');
const END_DATE = new Date('2024-11-01T00:00:00Z');

const SYMBOLS = ['BTCUSDT', 'ETHUSDT'];
const INTERVAL = '1h'; // 1-hour bars
const LIMIT = 1000; // Max records per request

/**
 * Download historical klines from Binance API
 */
async function downloadKlines(
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number
): Promise<KlineData[]> {
  const url = `${BINANCE_BASE_URL}/api/v3/klines`;
  
  try {
    const response = await axios.get(url, {
      params: {
        symbol,
        interval,
        startTime,
        endTime,
        limit: LIMIT
      },
      timeout: 30000
    });

    // Binance returns array of arrays
    return response.data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6],
      quoteVolume: k[7],
      trades: k[8],
      takerBuyBase: k[9],
      takerBuyQuote: k[10]
    }));
  } catch (error) {
    console.error(`Error downloading ${symbol}:`, error);
    throw error;
  }
}

/**
 * Convert Binance kline to OHLCV format
 */
function klineToOHLCV(kline: KlineData): OHLCVBar {
  return {
    timestamp: kline.openTime,
    datetime: new Date(kline.openTime).toISOString(),
    open: parseFloat(kline.open),
    high: parseFloat(kline.high),
    low: parseFloat(kline.low),
    close: parseFloat(kline.close),
    volume: parseFloat(kline.volume)
  };
}

/**
 * Download complete history for a symbol in chunks
 */
async function downloadSymbolHistory(symbol: string): Promise<OHLCVBar[]> {
  console.log(`\n📊 Downloading ${symbol} history...`);
  console.log(`Period: ${START_DATE.toISOString()} → ${END_DATE.toISOString()}`);
  
  const allBars: OHLCVBar[] = [];
  let currentStart = START_DATE.getTime();
  const endTime = END_DATE.getTime();
  
  let batchCount = 0;
  
  while (currentStart < endTime) {
    batchCount++;
    console.log(`  Batch ${batchCount}: ${new Date(currentStart).toISOString()}`);
    
    try {
      const klines = await downloadKlines(symbol, INTERVAL, currentStart, endTime);
      
      if (klines.length === 0) {
        console.log(`  ✓ No more data available`);
        break;
      }
      
      // Convert and add to collection
      const ohlcvBars = klines.map(klineToOHLCV);
      allBars.push(...ohlcvBars);
      
      console.log(`  ✓ Downloaded ${klines.length} bars (Total: ${allBars.length})`);
      
      // Move to next batch (last close time + 1ms)
      currentStart = klines[klines.length - 1].closeTime + 1;
      
      // Rate limiting: wait 100ms between requests
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      console.error(`  ✗ Error in batch ${batchCount}:`, error);
      
      // Retry with exponential backoff
      console.log(`  ⏳ Waiting 5 seconds before retry...`);
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
  
  console.log(`\n✓ ${symbol} complete: ${allBars.length} bars downloaded`);
  return allBars;
}

/**
 * Save data to JSON file
 */
function saveToFile(symbol: string, data: OHLCVBar[]): void {
  const filename = `${symbol}_${INTERVAL}_${START_DATE.getFullYear()}-${END_DATE.getFullYear()}.json`;
  const filepath = path.join(DATA_DIR, filename);
  
  const output = {
    symbol,
    interval: INTERVAL,
    startDate: START_DATE.toISOString(),
    endDate: END_DATE.toISOString(),
    bars: data.length,
    data
  };
  
  fs.writeFileSync(filepath, JSON.stringify(output, null, 2));
  console.log(`💾 Saved to: ${filename}`);
  
  // Also save as CSV for easy inspection
  const csvFilename = `${symbol}_${INTERVAL}_${START_DATE.getFullYear()}-${END_DATE.getFullYear()}.csv`;
  const csvPath = path.join(DATA_DIR, csvFilename);
  
  const csvHeader = 'timestamp,datetime,open,high,low,close,volume\n';
  const csvRows = data.map(bar => 
    `${bar.timestamp},${bar.datetime},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`
  ).join('\n');
  
  fs.writeFileSync(csvPath, csvHeader + csvRows);
  console.log(`💾 Saved CSV: ${csvFilename}`);
}

/**
 * Calculate statistics for downloaded data
 */
function calculateStats(symbol: string, data: OHLCVBar[]): void {
  if (data.length === 0) return;
  
  const prices = data.map(bar => bar.close);
  const volumes = data.map(bar => bar.volume);
  
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
  const totalVolume = volumes.reduce((sum, v) => sum + v, 0);
  
  const firstBar = data[0];
  const lastBar = data[data.length - 1];
  const returns = ((lastBar.close - firstBar.close) / firstBar.close) * 100;
  
  console.log(`\n📈 ${symbol} Statistics:`);
  console.log(`  Bars: ${data.length}`);
  console.log(`  First: ${firstBar.datetime} @ $${firstBar.close.toFixed(2)}`);
  console.log(`  Last:  ${lastBar.datetime} @ $${lastBar.close.toFixed(2)}`);
  console.log(`  Price Range: $${minPrice.toFixed(2)} - $${maxPrice.toFixed(2)}`);
  console.log(`  Average Price: $${avgPrice.toFixed(2)}`);
  console.log(`  Total Return: ${returns.toFixed(2)}%`);
  console.log(`  Total Volume: ${totalVolume.toLocaleString()}`);
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Starting Binance Historical Data Download');
  console.log('='.repeat(60));
  console.log(`Period: ${START_DATE.toISOString()} → ${END_DATE.toISOString()}`);
  console.log(`Duration: 3 years (36 months)`);
  console.log(`Interval: ${INTERVAL} (hourly)`);
  console.log(`Expected bars per symbol: ~26,280`);
  console.log('='.repeat(60));
  
  // Ensure data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  
  // Download each symbol
  for (const symbol of SYMBOLS) {
    try {
      const data = await downloadSymbolHistory(symbol);
      saveToFile(symbol, data);
      calculateStats(symbol, data);
      
      console.log('\n' + '-'.repeat(60));
    } catch (error) {
      console.error(`❌ Failed to download ${symbol}:`, error);
    }
  }
  
  console.log('\n✅ Download complete!');
  console.log(`📁 Data saved to: ${DATA_DIR}`);
}

// Execute if run directly
if (require.main === module) {
  main().catch(console.error);
}

export { downloadSymbolHistory, OHLCVBar };
