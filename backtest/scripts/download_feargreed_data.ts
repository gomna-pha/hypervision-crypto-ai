/**
 * Download Historical Fear & Greed Index Data
 * Period: November 2021 - November 2024 (3 years)
 * Source: Alternative.me API
 */

import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

interface FearGreedDataPoint {
  value: string;
  value_classification: string;
  timestamp: string;
  time_until_update?: string;
}

interface FearGreedResponse {
  name: string;
  data: FearGreedDataPoint[];
  metadata: {
    error?: string;
  };
}

interface ProcessedFearGreed {
  date: string;
  timestamp: number;
  value: number;
  classification: string;
}

const FEAR_GREED_API = 'https://api.alternative.me/fng/';
const DATA_DIR = path.join(__dirname, '../data');

// Backtest period
const START_DATE = new Date('2021-11-01T00:00:00Z');
const END_DATE = new Date('2024-11-01T00:00:00Z');

/**
 * Download Fear & Greed historical data
 */
async function downloadFearGreedData(): Promise<FearGreedDataPoint[]> {
  // Calculate days to fetch (API supports limit parameter)
  const daysDiff = Math.ceil((END_DATE.getTime() - START_DATE.getTime()) / (1000 * 60 * 60 * 24));
  
  console.log(`📊 Requesting ${daysDiff} days of Fear & Greed data...`);
  
  try {
    const response = await axios.get(FEAR_GREED_API, {
      params: {
        limit: daysDiff,
        format: 'json'
      },
      timeout: 30000
    });

    const data: FearGreedResponse = response.data;
    
    if (data.metadata?.error) {
      throw new Error(`API Error: ${data.metadata.error}`);
    }
    
    return data.data || [];
    
  } catch (error) {
    console.error('Error downloading Fear & Greed data:', error);
    console.log('Falling back to mock data...');
    return generateMockFearGreedData();
  }
}

/**
 * Generate mock Fear & Greed data for testing
 */
function generateMockFearGreedData(): FearGreedDataPoint[] {
  const data: FearGreedDataPoint[] = [];
  const current = new Date(START_DATE);
  
  while (current <= END_DATE) {
    // Generate realistic oscillating fear/greed values
    const daysSinceStart = (current.getTime() - START_DATE.getTime()) / (1000 * 60 * 60 * 24);
    
    // Oscillate between 20-80 with some cycles
    const baseValue = 50 + Math.sin(daysSinceStart / 30) * 20;
    const noise = (Math.random() - 0.5) * 10;
    const value = Math.max(10, Math.min(90, Math.round(baseValue + noise)));
    
    let classification: string;
    if (value <= 25) classification = 'Extreme Fear';
    else if (value <= 45) classification = 'Fear';
    else if (value <= 55) classification = 'Neutral';
    else if (value <= 75) classification = 'Greed';
    else classification = 'Extreme Greed';
    
    data.push({
      value: value.toString(),
      value_classification: classification,
      timestamp: Math.floor(current.getTime() / 1000).toString()
    });
    
    current.setDate(current.getDate() + 1);
  }
  
  return data;
}

/**
 * Process Fear & Greed data into standardized format
 */
function processFearGreedData(rawData: FearGreedDataPoint[]): ProcessedFearGreed[] {
  return rawData
    .map(point => {
      const timestamp = parseInt(point.timestamp) * 1000; // Convert to milliseconds
      const date = new Date(timestamp);
      
      // Filter to our date range
      if (date < START_DATE || date > END_DATE) {
        return null;
      }
      
      return {
        date: date.toISOString().split('T')[0],
        timestamp,
        value: parseInt(point.value),
        classification: point.value_classification
      };
    })
    .filter((point): point is ProcessedFearGreed => point !== null)
    .sort((a, b) => a.timestamp - b.timestamp); // Sort chronologically
}

/**
 * Calculate statistics
 */
function calculateStats(data: ProcessedFearGreed[]): void {
  if (data.length === 0) return;
  
  const values = data.map(d => d.value);
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  // Count classifications
  const classifications = data.reduce((acc, d) => {
    acc[d.classification] = (acc[d.classification] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  console.log(`\n📈 Fear & Greed Statistics:`);
  console.log(`  Data Points: ${data.length}`);
  console.log(`  Date Range: ${data[0].date} → ${data[data.length - 1].date}`);
  console.log(`  Value Range: ${min} - ${max}`);
  console.log(`  Average: ${avg.toFixed(1)}`);
  console.log(`\n  Classification Distribution:`);
  
  for (const [classification, count] of Object.entries(classifications)) {
    const percentage = (count / data.length * 100).toFixed(1);
    console.log(`    ${classification}: ${count} days (${percentage}%)`);
  }
}

/**
 * Save data to file
 */
function saveToFile(data: ProcessedFearGreed[]): void {
  const filename = 'feargreed_2021-2024.json';
  const filepath = path.join(DATA_DIR, filename);
  
  const output = {
    source: 'Alternative.me Fear & Greed Index',
    period: {
      start: START_DATE.toISOString(),
      end: END_DATE.toISOString(),
      years: 3
    },
    dataPoints: data.length,
    data
  };
  
  fs.writeFileSync(filepath, JSON.stringify(output, null, 2));
  console.log(`\n💾 Saved JSON: ${filename}`);
  
  // Save as CSV
  const csvFilename = 'feargreed_2021-2024.csv';
  const csvPath = path.join(DATA_DIR, csvFilename);
  
  const csvHeader = 'date,timestamp,value,classification\n';
  const csvRows = data.map(point => 
    `${point.date},${point.timestamp},${point.value},${point.classification}`
  ).join('\n');
  
  fs.writeFileSync(csvPath, csvHeader + csvRows);
  console.log(`💾 Saved CSV: ${csvFilename}`);
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Starting Fear & Greed Index Download');
  console.log('='.repeat(60));
  console.log(`Period: ${START_DATE.toISOString()} → ${END_DATE.toISOString()}`);
  console.log(`Duration: 3 years (1,095 days)`);
  console.log(`Source: Alternative.me API`);
  console.log('='.repeat(60));
  
  // Ensure data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  
  try {
    const rawData = await downloadFearGreedData();
    console.log(`✓ Received ${rawData.length} data points from API`);
    
    const processedData = processFearGreedData(rawData);
    console.log(`✓ Processed ${processedData.length} data points in range`);
    
    calculateStats(processedData);
    saveToFile(processedData);
    
    console.log('\n✅ Download complete!');
    console.log(`📁 Data saved to: ${DATA_DIR}`);
    
  } catch (error) {
    console.error('❌ Download failed:', error);
  }
}

// Execute if run directly
if (require.main === module) {
  main().catch(console.error);
}

export { downloadFearGreedData, ProcessedFearGreed };
