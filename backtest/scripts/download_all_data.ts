/**
 * Master Data Download Script
 * Downloads all historical data needed for backtesting:
 * 1. Binance OHLCV (BTC/ETH, 1-hour bars, 3 years)
 * 2. FRED Economic Data (Fed Funds, CPI, GDP, Unemployment, PMI)
 * 3. Fear & Greed Index (Daily, 3 years)
 */

import { downloadSymbolHistory } from './download_binance_data';
import { downloadAllIndicators } from './download_fred_data';
import { downloadFearGreedData } from './download_feargreed_data';
import * as fs from 'fs';
import * as path from 'path';

const DATA_DIR = path.join(__dirname, '../data');

interface DownloadSummary {
  timestamp: string;
  duration: number;
  success: boolean;
  downloads: {
    binance: {
      success: boolean;
      symbols: string[];
      totalBars: number;
      error?: string;
    };
    fred: {
      success: boolean;
      indicators: number;
      totalPoints: number;
      error?: string;
    };
    fearGreed: {
      success: boolean;
      dataPoints: number;
      error?: string;
    };
  };
}

/**
 * Download all Binance data
 */
async function downloadBinanceAll(): Promise<{ success: boolean; symbols: string[]; totalBars: number; error?: string }> {
  console.log('\n' + '='.repeat(80));
  console.log('📊 PHASE 1: BINANCE OHLCV DATA');
  console.log('='.repeat(80));
  
  try {
    const symbols = ['BTCUSDT', 'ETHUSDT'];
    let totalBars = 0;
    
    for (const symbol of symbols) {
      const data = await downloadSymbolHistory(symbol);
      totalBars += data.length;
      
      // Save to file
      const filename = `${symbol}_1h_2021-2024.json`;
      const filepath = path.join(DATA_DIR, filename);
      fs.writeFileSync(filepath, JSON.stringify({
        symbol,
        interval: '1h',
        startDate: '2021-11-01',
        endDate: '2024-11-01',
        bars: data.length,
        data
      }, null, 2));
      
      // Save CSV
      const csvFilename = `${symbol}_1h_2021-2024.csv`;
      const csvPath = path.join(DATA_DIR, csvFilename);
      const csvHeader = 'timestamp,datetime,open,high,low,close,volume\n';
      const csvRows = data.map(bar => 
        `${bar.timestamp},${bar.datetime},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`
      ).join('\n');
      fs.writeFileSync(csvPath, csvHeader + csvRows);
      
      console.log(`✓ Saved ${symbol}: ${data.length} bars`);
    }
    
    return {
      success: true,
      symbols,
      totalBars
    };
    
  } catch (error) {
    console.error('❌ Binance download failed:', error);
    return {
      success: false,
      symbols: [],
      totalBars: 0,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Download all FRED data
 */
async function downloadFREDAll(): Promise<{ success: boolean; indicators: number; totalPoints: number; error?: string }> {
  console.log('\n' + '='.repeat(80));
  console.log('📈 PHASE 2: FRED ECONOMIC DATA');
  console.log('='.repeat(80));
  
  try {
    const datasets = await downloadAllIndicators();
    
    const totalPoints = datasets.reduce((sum, ds) => sum + ds.dataPoints, 0);
    
    // Save combined file
    const filename = 'economic_data_2021-2024.json';
    const filepath = path.join(DATA_DIR, filename);
    fs.writeFileSync(filepath, JSON.stringify({
      source: 'FRED',
      period: { start: '2021-11-01', end: '2024-11-01', years: 3 },
      indicators: datasets.length,
      datasets
    }, null, 2));
    
    // Save individual CSVs
    for (const dataset of datasets) {
      const csvFilename = `${dataset.series}_2021-2024.csv`;
      const csvPath = path.join(DATA_DIR, csvFilename);
      const csvHeader = 'date,timestamp,value\n';
      const csvRows = dataset.data.map(point => 
        `${point.date},${point.timestamp},${point.value}`
      ).join('\n');
      fs.writeFileSync(csvPath, csvHeader + csvRows);
    }
    
    console.log(`✓ Saved ${datasets.length} economic indicators (${totalPoints} total points)`);
    
    return {
      success: true,
      indicators: datasets.length,
      totalPoints
    };
    
  } catch (error) {
    console.error('❌ FRED download failed:', error);
    return {
      success: false,
      indicators: 0,
      totalPoints: 0,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Download Fear & Greed data
 */
async function downloadFearGreedAll(): Promise<{ success: boolean; dataPoints: number; error?: string }> {
  console.log('\n' + '='.repeat(80));
  console.log('😨 PHASE 3: FEAR & GREED INDEX');
  console.log('='.repeat(80));
  
  try {
    const rawData = await downloadFearGreedData();
    
    // Process and save
    const processedData = rawData.map(point => ({
      date: new Date(parseInt(point.timestamp) * 1000).toISOString().split('T')[0],
      timestamp: parseInt(point.timestamp) * 1000,
      value: parseInt(point.value),
      classification: point.value_classification
    }));
    
    const filename = 'feargreed_2021-2024.json';
    const filepath = path.join(DATA_DIR, filename);
    fs.writeFileSync(filepath, JSON.stringify({
      source: 'Alternative.me',
      period: { start: '2021-11-01', end: '2024-11-01', years: 3 },
      dataPoints: processedData.length,
      data: processedData
    }, null, 2));
    
    // Save CSV
    const csvFilename = 'feargreed_2021-2024.csv';
    const csvPath = path.join(DATA_DIR, csvFilename);
    const csvHeader = 'date,timestamp,value,classification\n';
    const csvRows = processedData.map(point => 
      `${point.date},${point.timestamp},${point.value},${point.classification}`
    ).join('\n');
    fs.writeFileSync(csvPath, csvHeader + csvRows);
    
    console.log(`✓ Saved Fear & Greed data: ${processedData.length} days`);
    
    return {
      success: true,
      dataPoints: processedData.length
    };
    
  } catch (error) {
    console.error('❌ Fear & Greed download failed:', error);
    return {
      success: false,
      dataPoints: 0,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Main execution
 */
async function main() {
  const startTime = Date.now();
  
  console.log('🚀 COMPREHENSIVE DATA DOWNLOAD');
  console.log('='.repeat(80));
  console.log('Period: November 2021 - November 2024 (3 years)');
  console.log('Purpose: Backtesting validation for LLM Agent System');
  console.log('='.repeat(80));
  
  // Ensure data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    console.log(`📁 Created data directory: ${DATA_DIR}`);
  }
  
  // Download all data sources
  const binanceResult = await downloadBinanceAll();
  const fredResult = await downloadFREDAll();
  const fearGreedResult = await downloadFearGreedAll();
  
  const endTime = Date.now();
  const duration = (endTime - startTime) / 1000;
  
  // Create summary
  const summary: DownloadSummary = {
    timestamp: new Date().toISOString(),
    duration,
    success: binanceResult.success && fredResult.success && fearGreedResult.success,
    downloads: {
      binance: binanceResult,
      fred: fredResult,
      fearGreed: fearGreedResult
    }
  };
  
  // Save summary
  const summaryPath = path.join(DATA_DIR, 'download_summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  
  // Print final report
  console.log('\n' + '='.repeat(80));
  console.log('📊 DOWNLOAD SUMMARY');
  console.log('='.repeat(80));
  console.log(`Duration: ${duration.toFixed(1)} seconds`);
  console.log(`\nBinance OHLCV:`);
  console.log(`  Status: ${binanceResult.success ? '✅ Success' : '❌ Failed'}`);
  console.log(`  Symbols: ${binanceResult.symbols.join(', ')}`);
  console.log(`  Total Bars: ${binanceResult.totalBars.toLocaleString()}`);
  
  console.log(`\nFRED Economic:`);
  console.log(`  Status: ${fredResult.success ? '✅ Success' : '❌ Failed'}`);
  console.log(`  Indicators: ${fredResult.indicators}`);
  console.log(`  Total Points: ${fredResult.totalPoints.toLocaleString()}`);
  
  console.log(`\nFear & Greed:`);
  console.log(`  Status: ${fearGreedResult.success ? '✅ Success' : '❌ Failed'}`);
  console.log(`  Data Points: ${fearGreedResult.dataPoints.toLocaleString()}`);
  
  console.log('\n' + '='.repeat(80));
  console.log(`Overall: ${summary.success ? '✅ ALL DOWNLOADS COMPLETE' : '⚠️  SOME DOWNLOADS FAILED'}`);
  console.log(`📁 Data location: ${DATA_DIR}`);
  console.log(`📄 Summary saved: ${summaryPath}`);
  console.log('='.repeat(80));
  
  if (!summary.success) {
    process.exit(1);
  }
}

// Execute if run directly
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { main as downloadAllData };
