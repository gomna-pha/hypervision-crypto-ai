/**
 * Generate Realistic Synthetic Historical Data for Backtesting
 * Period: November 2021 - November 2024 (3 years)
 * 
 * This generates data based on actual historical crypto market patterns:
 * - Nov 2021: Peak bull market (BTC ~$69k)
 * - 2022: Bear market (BTC declined to ~$16k)
 * - 2023: Recovery year (BTC recovered to ~$44k)
 * - 2024: Bull run (BTC reached ~$73k)
 */

import * as fs from 'fs';
import * as path from 'path';

const DATA_DIR = path.join(__dirname, '../data');

// Historical reference points (actual market data)
const BTC_PRICE_POINTS = [
  { date: '2021-11-01', price: 61000 },   // Pre-ATH
  { date: '2021-11-10', price: 69000 },   // ATH
  { date: '2022-01-01', price: 47000 },   // Crash beginning
  { date: '2022-06-01', price: 30000 },   // Bear market
  { date: '2022-11-01', price: 20500 },   // FTX collapse
  { date: '2023-01-01', price: 16500 },   // Bottom
  { date: '2023-06-01', price: 26800 },   // Recovery
  { date: '2023-12-01', price: 42000 },   // Pre-halving pump
  { date: '2024-03-01', price: 62000 },   // ETF approval
  { date: '2024-06-01', price: 70000 },   // New ATH
  { date: '2024-11-01', price: 73000 }    // Election pump
];

const ETH_PRICE_POINTS = [
  { date: '2021-11-01', price: 4300 },
  { date: '2021-11-10', price: 4850 },
  { date: '2022-01-01', price: 3700 },
  { date: '2022-06-01', price: 1800 },
  { date: '2022-11-01', price: 1550 },
  { date: '2023-01-01', price: 1200 },
  { date: '2023-06-01', price: 1900 },
  { date: '2023-12-01', price: 2300 },
  { date: '2024-03-01', price: 3500 },
  { date: '2024-06-01', price: 3800 },
  { date: '2024-11-01', price: 3950 }
];

interface OHLCVBar {
  timestamp: number;
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Interpolate price between two reference points
 */
function interpolatePrice(
  startDate: Date,
  endDate: Date,
  startPrice: number,
  endPrice: number,
  currentDate: Date
): number {
  const totalDuration = endDate.getTime() - startDate.getTime();
  const currentDuration = currentDate.getTime() - startDate.getTime();
  const progress = currentDuration / totalDuration;
  
  // Add some realistic volatility (±5% random walk)
  const basePrice = startPrice + (endPrice - startPrice) * progress;
  const volatility = basePrice * 0.05 * (Math.random() - 0.5);
  
  return basePrice + volatility;
}

/**
 * Find the surrounding reference points for interpolation
 */
function findSurroundingPoints(
  date: Date,
  pricePoints: { date: string; price: number }[]
): { before: { date: Date; price: number }; after: { date: Date; price: number } } {
  for (let i = 0; i < pricePoints.length - 1; i++) {
    const beforeDate = new Date(pricePoints[i].date);
    const afterDate = new Date(pricePoints[i + 1].date);
    
    if (date >= beforeDate && date <= afterDate) {
      return {
        before: { date: beforeDate, price: pricePoints[i].price },
        after: { date: afterDate, price: pricePoints[i + 1].price }
      };
    }
  }
  
  // If we're past the last point, use last two points for extrapolation
  const lastIdx = pricePoints.length - 1;
  return {
    before: {
      date: new Date(pricePoints[lastIdx - 1].date),
      price: pricePoints[lastIdx - 1].price
    },
    after: {
      date: new Date(pricePoints[lastIdx].date),
      price: pricePoints[lastIdx].price
    }
  };
}

/**
 * Generate OHLCV bar for a specific hour
 */
function generateOHLCVBar(
  date: Date,
  basePrice: number,
  symbol: string
): OHLCVBar {
  // Realistic intrabar volatility (0.5-2% range)
  const volatilityRange = basePrice * 0.01;
  
  const open = basePrice + (Math.random() - 0.5) * volatilityRange;
  const close = basePrice + (Math.random() - 0.5) * volatilityRange;
  
  const maxVariation = Math.max(open, close) + Math.abs(Math.random() * volatilityRange);
  const minVariation = Math.min(open, close) - Math.abs(Math.random() * volatilityRange);
  
  const high = maxVariation;
  const low = minVariation;
  
  // Volume: higher during volatile periods
  const priceChange = Math.abs(close - open) / open;
  const baseVolume = symbol === 'BTCUSDT' ? 1000 : 10000; // ETH has higher volume
  const volume = baseVolume * (1 + priceChange * 50) * (0.5 + Math.random());
  
  return {
    timestamp: date.getTime(),
    datetime: date.toISOString(),
    open: parseFloat(open.toFixed(2)),
    high: parseFloat(high.toFixed(2)),
    low: parseFloat(low.toFixed(2)),
    close: parseFloat(close.toFixed(2)),
    volume: parseFloat(volume.toFixed(4))
  };
}

/**
 * Generate complete historical data for a symbol
 */
function generateHistoricalData(
  symbol: string,
  pricePoints: { date: string; price: number }[]
): OHLCVBar[] {
  console.log(`\n📊 Generating ${symbol} data...`);
  
  const startDate = new Date('2021-11-01T00:00:00Z');
  const endDate = new Date('2024-11-01T00:00:00Z');
  const bars: OHLCVBar[] = [];
  
  let currentDate = new Date(startDate);
  let barCount = 0;
  
  while (currentDate <= endDate) {
    const { before, after } = findSurroundingPoints(currentDate, pricePoints);
    const basePrice = interpolatePrice(
      before.date,
      after.date,
      before.price,
      after.price,
      currentDate
    );
    
    const bar = generateOHLCVBar(currentDate, basePrice, symbol);
    bars.push(bar);
    
    // Move to next hour
    currentDate = new Date(currentDate.getTime() + 60 * 60 * 1000);
    barCount++;
    
    // Progress indicator
    if (barCount % 1000 === 0) {
      console.log(`  Generated ${barCount} bars...`);
    }
  }
  
  console.log(`✓ ${symbol} complete: ${bars.length} bars`);
  return bars;
}

/**
 * Generate synthetic Fear & Greed Index data
 */
function generateFearGreedData(): any[] {
  console.log('\n😨 Generating Fear & Greed Index data...');
  
  const startDate = new Date('2021-11-01T00:00:00Z');
  const endDate = new Date('2024-11-01T00:00:00Z');
  const data: any[] = [];
  
  let currentDate = new Date(startDate);
  let dayCount = 0;
  
  while (currentDate <= endDate) {
    // Correlate with BTC price trends
    const { before, after } = findSurroundingPoints(currentDate, BTC_PRICE_POINTS);
    const btcPrice = interpolatePrice(
      before.date,
      after.date,
      before.price,
      after.price,
      currentDate
    );
    
    // Fear/Greed correlates with price trend
    const priceChangePercent = ((btcPrice - before.price) / before.price) * 100;
    
    // Base fear/greed on trend (50 = neutral)
    let value = 50 + priceChangePercent * 2;
    
    // Add some noise
    value += (Math.random() - 0.5) * 10;
    
    // Clamp to 0-100
    value = Math.max(0, Math.min(100, value));
    
    let classification: string;
    if (value <= 25) classification = 'Extreme Fear';
    else if (value <= 45) classification = 'Fear';
    else if (value <= 55) classification = 'Neutral';
    else if (value <= 75) classification = 'Greed';
    else classification = 'Extreme Greed';
    
    data.push({
      date: currentDate.toISOString().split('T')[0],
      timestamp: currentDate.getTime(),
      value: Math.round(value),
      classification
    });
    
    // Move to next day
    currentDate = new Date(currentDate.getTime() + 24 * 60 * 60 * 1000);
    dayCount++;
    
    if (dayCount % 100 === 0) {
      console.log(`  Generated ${dayCount} days...`);
    }
  }
  
  console.log(`✓ Fear & Greed complete: ${data.length} days`);
  return data;
}

/**
 * Generate synthetic FRED economic data
 */
function generateEconomicData(): any {
  console.log('\n📈 Generating FRED economic data...');
  
  const startDate = new Date('2021-11-01');
  const endDate = new Date('2024-11-01');
  
  const datasets: Array<{
    series: string;
    name: string;
    unit: string;
    frequency: string;
    startValue: number;
    endValue: number;
    data: Array<{ date: string; timestamp: number; value: number }>;
  }> = [
    {
      series: 'FEDFUNDS',
      name: 'Federal Funds Rate',
      unit: 'Percent',
      frequency: 'monthly',
      // Actual Fed policy: 0.25% (2021) → 5.5% (2023-2024)
      startValue: 0.25,
      endValue: 5.5,
      data: []
    },
    {
      series: 'CPIAUCSL',
      name: 'Consumer Price Index',
      unit: 'Index 1982-1984=100',
      frequency: 'monthly',
      startValue: 276,
      endValue: 310,
      data: []
    },
    {
      series: 'GDP',
      name: 'Gross Domestic Product',
      unit: 'Billions of Dollars',
      frequency: 'quarterly',
      startValue: 23700,
      endValue: 27800,
      data: []
    },
    {
      series: 'UNRATE',
      name: 'Unemployment Rate',
      unit: 'Percent',
      frequency: 'monthly',
      startValue: 4.2,
      endValue: 3.8,
      data: []
    },
    {
      series: 'MANEMP',
      name: 'Manufacturing PMI',
      unit: 'Index',
      frequency: 'monthly',
      startValue: 53,
      endValue: 48,
      data: []
    }
  ];
  
  for (const dataset of datasets) {
    let currentDate = new Date(startDate);
    const totalMonths = (endDate.getFullYear() - startDate.getFullYear()) * 12 +
                       (endDate.getMonth() - startDate.getMonth());
    
    let dataPoint = 0;
    while (currentDate <= endDate) {
      const progress = dataPoint / totalMonths;
      const value = dataset.startValue + 
                   (dataset.endValue - dataset.startValue) * progress +
                   (Math.random() - 0.5) * 2; // Add noise
      
      dataset.data.push({
        date: currentDate.toISOString().split('T')[0],
        timestamp: currentDate.getTime(),
        value: parseFloat(value.toFixed(2))
      });
      
      // Move to next month (or quarter for GDP)
      if (dataset.frequency === 'quarterly') {
        currentDate = new Date(currentDate.setMonth(currentDate.getMonth() + 3));
      } else {
        currentDate = new Date(currentDate.setMonth(currentDate.getMonth() + 1));
      }
      dataPoint++;
    }
    
    console.log(`  ✓ ${dataset.series}: ${dataset.data.length} points`);
  }
  
  return {
    source: 'Synthetic (based on historical trends)',
    period: { start: '2021-11-01', end: '2024-11-01', years: 3 },
    indicators: datasets.length,
    datasets
  };
}

/**
 * Save data to files
 */
function saveData(symbol: string, data: OHLCVBar[]): void {
  // Save JSON
  const jsonFile = `${symbol}_1h_2021-2024.json`;
  const jsonPath = path.join(DATA_DIR, jsonFile);
  fs.writeFileSync(jsonPath, JSON.stringify({
    symbol,
    interval: '1h',
    startDate: '2021-11-01',
    endDate: '2024-11-01',
    bars: data.length,
    source: 'Synthetic (based on historical market trends)',
    data
  }, null, 2));
  
  // Save CSV
  const csvFile = `${symbol}_1h_2021-2024.csv`;
  const csvPath = path.join(DATA_DIR, csvFile);
  const csvHeader = 'timestamp,datetime,open,high,low,close,volume\n';
  const csvRows = data.map(bar =>
    `${bar.timestamp},${bar.datetime},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`
  ).join('\n');
  fs.writeFileSync(csvPath, csvHeader + csvRows);
  
  console.log(`💾 Saved ${symbol} to ${jsonFile} and ${csvFile}`);
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 SYNTHETIC DATA GENERATION');
  console.log('='.repeat(80));
  console.log('Period: November 2021 - November 2024 (3 years)');
  console.log('Based on: Actual historical crypto market cycles');
  console.log('Purpose: Institutional-grade backtesting validation');
  console.log('='.repeat(80));
  
  const startTime = Date.now();
  
  // Ensure data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  
  // Generate BTC data
  const btcData = generateHistoricalData('BTCUSDT', BTC_PRICE_POINTS);
  saveData('BTCUSDT', btcData);
  
  // Generate ETH data
  const ethData = generateHistoricalData('ETHUSDT', ETH_PRICE_POINTS);
  saveData('ETHUSDT', ethData);
  
  // Generate Fear & Greed
  const fearGreedData = generateFearGreedData();
  const fgPath = path.join(DATA_DIR, 'feargreed_2021-2024.json');
  fs.writeFileSync(fgPath, JSON.stringify({
    source: 'Synthetic Fear & Greed Index',
    period: { start: '2021-11-01', end: '2024-11-01', years: 3 },
    dataPoints: fearGreedData.length,
    data: fearGreedData
  }, null, 2));
  console.log(`💾 Saved Fear & Greed to feargreed_2021-2024.json`);
  
  // Generate Economic data
  const economicData = generateEconomicData();
  const econPath = path.join(DATA_DIR, 'economic_data_2021-2024.json');
  fs.writeFileSync(econPath, JSON.stringify(economicData, null, 2));
  console.log(`💾 Saved economic data to economic_data_2021-2024.json`);
  
  const endTime = Date.now();
  const duration = (endTime - startTime) / 1000;
  
  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 GENERATION SUMMARY');
  console.log('='.repeat(80));
  console.log(`Duration: ${duration.toFixed(1)} seconds`);
  console.log(`\nGenerated Data:`);
  console.log(`  BTC OHLCV: ${btcData.length} hourly bars`);
  console.log(`  ETH OHLCV: ${ethData.length} hourly bars`);
  console.log(`  Fear & Greed: ${fearGreedData.length} daily readings`);
  console.log(`  Economic: ${economicData.datasets.length} indicators`);
  console.log(`\nTotal Data Points: ${btcData.length + ethData.length + fearGreedData.length + economicData.datasets.reduce((sum: number, d: any) => sum + d.data.length, 0)}`);
  console.log(`\n✅ ALL DATA GENERATED SUCCESSFULLY`);
  console.log(`📁 Data location: ${DATA_DIR}`);
  console.log('='.repeat(80));
}

// Execute
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Generation failed:', error);
    process.exit(1);
  });
}

export { main as generateSyntheticData };
