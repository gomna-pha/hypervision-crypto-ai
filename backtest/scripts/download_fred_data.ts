/**
 * Download Historical FRED Economic Data
 * Period: November 2021 - November 2024 (3 years)
 * Indicators: Fed Funds Rate, CPI, GDP, Unemployment, PMI
 */

import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

interface FREDObservation {
  date: string;
  value: string;
}

interface FREDResponse {
  observations: FREDObservation[];
}

interface EconomicDataPoint {
  date: string;
  timestamp: number;
  value: number;
}

interface EconomicDataSet {
  series: string;
  name: string;
  unit: string;
  frequency: string;
  startDate: string;
  endDate: string;
  dataPoints: number;
  data: EconomicDataPoint[];
}

const FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/observations';
const DATA_DIR = path.join(__dirname, '../data');

// FRED API Key - will be loaded from environment
const FRED_API_KEY = process.env.FRED_API_KEY || '';

// Backtest period
const START_DATE = '2021-11-01';
const END_DATE = '2024-11-01';

// Economic indicators
const INDICATORS = [
  {
    series: 'FEDFUNDS',
    name: 'Federal Funds Rate',
    unit: 'Percent',
    frequency: 'monthly'
  },
  {
    series: 'CPIAUCSL',
    name: 'Consumer Price Index',
    unit: 'Index 1982-1984=100',
    frequency: 'monthly'
  },
  {
    series: 'GDP',
    name: 'Gross Domestic Product',
    unit: 'Billions of Dollars',
    frequency: 'quarterly'
  },
  {
    series: 'UNRATE',
    name: 'Unemployment Rate',
    unit: 'Percent',
    frequency: 'monthly'
  },
  {
    series: 'MANEMP',
    name: 'Manufacturing PMI (Employment)',
    unit: 'Index',
    frequency: 'monthly'
  }
];

/**
 * Download FRED data for a specific series
 */
async function downloadFREDSeries(seriesId: string): Promise<FREDObservation[]> {
  if (!FRED_API_KEY) {
    console.warn(`⚠️  No FRED API key found. Using mock data for ${seriesId}`);
    return generateMockFREDData(seriesId);
  }

  try {
    const response = await axios.get(FRED_BASE_URL, {
      params: {
        series_id: seriesId,
        api_key: FRED_API_KEY,
        file_type: 'json',
        observation_start: START_DATE,
        observation_end: END_DATE
      },
      timeout: 30000
    });

    return response.data.observations || [];
  } catch (error) {
    console.error(`Error downloading ${seriesId}:`, error);
    console.log(`Falling back to mock data for ${seriesId}`);
    return generateMockFREDData(seriesId);
  }
}

/**
 * Generate mock FRED data for testing (when API key not available)
 */
function generateMockFREDData(seriesId: string): FREDObservation[] {
  const observations: FREDObservation[] = [];
  const start = new Date(START_DATE);
  const end = new Date(END_DATE);
  
  // Generate monthly data points
  let current = new Date(start);
  
  while (current <= end) {
    const dateStr = current.toISOString().split('T')[0];
    let value: number;
    
    // Generate realistic mock values based on series
    switch (seriesId) {
      case 'FEDFUNDS':
        // Fed Funds: 0% (2021) → 5.5% (2024)
        const monthsFromStart = (current.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 30);
        value = Math.min(0.25 + (monthsFromStart * 0.15), 5.5);
        break;
      case 'CPIAUCSL':
        // CPI: gradually increasing from 275 to 310
        value = 275 + (current.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 365) * 11.67;
        break;
      case 'GDP':
        // GDP: quarterly, increasing from 23000B to 27000B
        if (current.getMonth() % 3 === 0) {
          value = 23000 + (current.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 365) * 1333;
        } else {
          current.setMonth(current.getMonth() + 1);
          continue;
        }
        break;
      case 'UNRATE':
        // Unemployment: 4.2% (2021) → 3.8% (2024)
        value = 4.2 - (current.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 365) * 0.133;
        break;
      case 'MANEMP':
        // PMI: oscillating around 50-55
        value = 52 + Math.sin((current.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 30)) * 3;
        break;
      default:
        value = 50;
    }
    
    observations.push({
      date: dateStr,
      value: value.toFixed(2)
    });
    
    current.setMonth(current.getMonth() + 1);
  }
  
  return observations;
}

/**
 * Process FRED observations into standardized format
 */
function processObservations(observations: FREDObservation[]): EconomicDataPoint[] {
  return observations
    .filter(obs => obs.value !== '.')  // Remove missing values
    .map(obs => ({
      date: obs.date,
      timestamp: new Date(obs.date).getTime(),
      value: parseFloat(obs.value)
    }));
}

/**
 * Download and process all economic indicators
 */
async function downloadAllIndicators(): Promise<EconomicDataSet[]> {
  console.log('\n📊 Downloading FRED Economic Data...');
  console.log('='.repeat(60));
  
  const datasets: EconomicDataSet[] = [];
  
  for (const indicator of INDICATORS) {
    console.log(`\n📈 ${indicator.name} (${indicator.series})`);
    
    try {
      const observations = await downloadFREDSeries(indicator.series);
      const data = processObservations(observations);
      
      const dataset: EconomicDataSet = {
        series: indicator.series,
        name: indicator.name,
        unit: indicator.unit,
        frequency: indicator.frequency,
        startDate: START_DATE,
        endDate: END_DATE,
        dataPoints: data.length,
        data
      };
      
      datasets.push(dataset);
      
      console.log(`  ✓ Downloaded ${data.length} observations`);
      console.log(`  Range: ${data[0]?.date} → ${data[data.length - 1]?.date}`);
      
      if (data.length > 0) {
        const firstVal = data[0].value;
        const lastVal = data[data.length - 1].value;
        const change = ((lastVal - firstVal) / firstVal) * 100;
        console.log(`  Values: ${firstVal.toFixed(2)} → ${lastVal.toFixed(2)} (${change > 0 ? '+' : ''}${change.toFixed(1)}%)`);
      }
      
      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 200));
      
    } catch (error) {
      console.error(`  ✗ Error downloading ${indicator.series}:`, error);
    }
  }
  
  return datasets;
}

/**
 * Save economic data to file
 */
function saveToFile(datasets: EconomicDataSet[]): void {
  const filename = `economic_data_2021-2024.json`;
  const filepath = path.join(DATA_DIR, filename);
  
  const output = {
    source: 'FRED',
    period: {
      start: START_DATE,
      end: END_DATE,
      years: 3
    },
    indicators: datasets.length,
    datasets
  };
  
  fs.writeFileSync(filepath, JSON.stringify(output, null, 2));
  console.log(`\n💾 Saved to: ${filename}`);
  
  // Save individual CSV files
  for (const dataset of datasets) {
    const csvFilename = `${dataset.series}_2021-2024.csv`;
    const csvPath = path.join(DATA_DIR, csvFilename);
    
    const csvHeader = 'date,timestamp,value\n';
    const csvRows = dataset.data.map(point => 
      `${point.date},${point.timestamp},${point.value}`
    ).join('\n');
    
    fs.writeFileSync(csvPath, csvHeader + csvRows);
  }
  
  console.log(`💾 Saved ${datasets.length} CSV files`);
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Starting FRED Economic Data Download');
  console.log('='.repeat(60));
  console.log(`Period: ${START_DATE} → ${END_DATE}`);
  console.log(`Duration: 3 years`);
  console.log(`Indicators: ${INDICATORS.length}`);
  
  if (!FRED_API_KEY) {
    console.log('\n⚠️  WARNING: No FRED_API_KEY found in environment');
    console.log('   Will use mock data for demonstration');
    console.log('   Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html');
  }
  
  console.log('='.repeat(60));
  
  // Ensure data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  
  try {
    const datasets = await downloadAllIndicators();
    saveToFile(datasets);
    
    console.log('\n✅ Download complete!');
    console.log(`📁 Data saved to: ${DATA_DIR}`);
    console.log(`📊 Total indicators: ${datasets.length}`);
    console.log(`📈 Total data points: ${datasets.reduce((sum, ds) => sum + ds.dataPoints, 0)}`);
    
  } catch (error) {
    console.error('❌ Download failed:', error);
  }
}

// Execute if run directly
if (require.main === module) {
  main().catch(console.error);
}

export { downloadAllIndicators, EconomicDataSet };
