import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, BarChart3, Brain, Database, Layers, Activity, Target, Award, Upload, Image, Eye, Camera, Zap, Wifi, WifiOff, RefreshCw, ShieldCheck, Globe, Wand2, ListChecks, FileCode, Link, Server, Search, Terminal, Sparkles, Shield } from 'lucide-react';

const CryptoPredictionApp = () => {
  const [selectedAsset, setSelectedAsset] = useState('BTC-USD');
  const [predictionHorizon, setPredictionHorizon] = useState('1h');
  const [userMode, setUserMode] = useState('retail');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modelStatus, setModelStatus] = useState('active');
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [liveData, setLiveData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [realTimePrice, setRealTimePrice] = useState(null);
  const [newsData, setNewsData] = useState([]);
  const [sentimentData, setSentimentData] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [hyperbolicFeatures, setHyperbolicFeatures] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const fileInputRef = useRef(null);
  const wsRef = useRef(null);

  const cryptoAssets = [
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 
    'MATIC-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD'
  ];

  // Real-time WebSocket connection simulation
  const connectWebSocket = useCallback(() => {
    // In production, this would connect to actual crypto WebSocket APIs
    // like Binance, Coinbase Pro, or Alpha Vantage
    setConnectionStatus('connecting');
    
    setTimeout(() => {
      setConnectionStatus('connected');
      // Simulate real-time price updates
      const interval = setInterval(() => {
        const now = new Date();
        const basePrice = selectedAsset === 'BTC-USD' ? 45000 : 
                         selectedAsset === 'ETH-USD' ? 2800 : 
                         selectedAsset === 'ADA-USD' ? 0.45 : 1500;
        
        const priceChange = (Math.random() - 0.5) * (basePrice * 0.02);
        const currentPrice = basePrice + priceChange;
        
        setRealTimePrice({
          symbol: selectedAsset,
          price: currentPrice,
          change: priceChange,
          changePercent: (priceChange / basePrice) * 100,
          volume: Math.random() * 1000000,
          timestamp: now,
          bid: currentPrice - (Math.random() * 10),
          ask: currentPrice + (Math.random() * 10)
        });
        
        // Update live data array
        setLiveData(prev => {
          const newData = [...prev.slice(-23), {
            timestamp: now.toLocaleTimeString(),
            price: currentPrice,
            volume: Math.random() * 1000000,
            predicted: currentPrice + (Math.random() - 0.5) * (basePrice * 0.01),
            confidence: 0.75 + Math.random() * 0.2,
            sentiment: Math.random() * 2 - 1,
            volatility: 0.02 + Math.random() * 0.03,
            imageFeature: Math.random() * 0.8 + 0.1,
            hyperbolicDist: Math.random() * 3.14
          }];
          return newData;
        });
        
        setLastUpdate(now);
      }, 2000); // Update every 2 seconds for demo
      
      wsRef.current = interval;
    }, 1000);
  }, [selectedAsset]);

  // Fetch real-time news and sentiment
  const fetchLiveNews = useCallback(async () => {
    try {
      // Simulating news API call - in production use NewsAPI, Alpha Vantage, or similar
      const mockNews = [
        {
          title: `${selectedAsset.split('-')[0]} Technical Analysis Shows Bullish Pattern`,
          sentiment: 0.7,
          source: 'CryptoNews',
          timestamp: new Date(),
          impact: 'high'
        },
        {
          title: `Market Update: ${selectedAsset.split('-')[0]} Trading Volume Surges`,
          sentiment: 0.3,
          source: 'CoinDesk',
          timestamp: new Date(Date.now() - 300000),
          impact: 'medium'
        },
        {
          title: `${selectedAsset.split('-')[0]} Network Activity Increases 15%`,
          sentiment: 0.5,
          source: 'Blockchain.com',
          timestamp: new Date(Date.now() - 600000),
          impact: 'low'
        }
      ];
      
      setNewsData(mockNews);
      
      // Calculate overall sentiment
      const avgSentiment = mockNews.reduce((acc, news) => acc + news.sentiment, 0) / mockNews.length;
      setSentimentData({
        overall: avgSentiment,
        bullish: mockNews.filter(n => n.sentiment > 0.3).length,
        bearish: mockNews.filter(n => n.sentiment < -0.3).length,
        neutral: mockNews.filter(n => Math.abs(n.sentiment) <= 0.3).length,
        lastUpdate: new Date()
      });
      
    } catch (error) {
      console.error('Error fetching news:', error);
    }
  }, [selectedAsset]);

  // Generate hyperbolic predictions
  const generateHyperbolicPrediction = useCallback(async () => {
    if (!liveData.length) return;
    
    try {
      // Fallback to mock prediction for demo
      const mockPrediction = {
        prediction: Math.random() > 0.5 ? 'bullish' : 'bearish',
        confidence: 0.75 + Math.random() * 0.2,
        hyperbolic_features: ['price_momentum', 'volume_pattern', 'sentiment_shift'],
        risk_score: Math.random() * 0.5,
        time_horizon: predictionHorizon
      };
      
      setPredictions(prev => [...prev.slice(-23), {
        ...liveData[liveData.length - 1],
        aiPrediction: mockPrediction,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (error) {
      console.error('Error generating prediction:', error);
    }
  }, [liveData, selectedAsset, sentimentData, predictionHorizon]);

  // Initialize real-time connections
  useEffect(() => {
    connectWebSocket();
    fetchLiveNews();
    
    const newsInterval = setInterval(fetchLiveNews, 30000); // Update news every 30 seconds
    const predictionInterval = setInterval(generateHyperbolicPrediction, 10000); // Generate predictions every 10 seconds
    
    return () => {
      if (wsRef.current) {
        clearInterval(wsRef.current);
      }
      clearInterval(newsInterval);
      clearInterval(predictionInterval);
    };
  }, [selectedAsset]); // Fixed: Removed function dependencies that cause infinite loops

  // Generate hyperbolic features
  useEffect(() => {
    if (liveData.length > 0) {
      const features = Array.from({ length: 8 }, (_, i) => ({
        feature: `H-Feature ${i + 1}`,
        numerical: Math.random() * 100,
        visual: Math.random() * 100,
        combined: Math.random() * 100,
        importance: Math.random()
      }));
      setHyperbolicFeatures(features);
    }
  }, [liveData]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        // Simulate hyperbolic CNN analysis
        setTimeout(() => {
          setImageAnalysis({
            patternType: 'Ascending Triangle',
            confidence: 0.892,
            supportLevel: realTimePrice?.price ? realTimePrice.price * 0.95 : 44200,
            resistanceLevel: realTimePrice?.price ? realTimePrice.price * 1.05 : 46800,
            hyperbolicCurvature: 1.67,
            geometricComplexity: 0.84,
            fractalDimension: 1.58,
            embedDimension: 'H³',
            prediction: 'Strong Bullish Breakout Expected',
            timeFrame: predictionHorizon,
            riskScore: 0.28
          });
        }, 1500);
      };
      reader.readAsDataURL(file);
    }
  };

  // Enhanced Performance Metrics with Real-time Updates
  const [performanceMetrics, setPerformanceMetrics] = useState({
    // Core AI Model Performance
    accuracy: 0.914,
    precision: 0.897,
    recall: 0.931,
    f1Score: 0.913,
    
    // Risk-Adjusted Returns
    sharpe: 2.89,
    sortino: 4.12,
    calmar: 5.24,
    treynor: 0.847,
    jensen: 0.139,
    information: 1.94,
    
    // Drawdown & Risk Metrics
    maxDrawdown: -0.048,
    avgDrawdown: -0.012,
    drawdownDuration: 3.2,
    ulcerIndex: 0.021,
    var95: -0.024,
    var99: -0.037,
    expectedShortfall: -0.032,
    conditionalVaR: -0.041,
    
    // Volatility Measures
    volatility: 0.1876,
    downsideVolatility: 0.069,
    upVolatility: 0.134,
    volatilitySkew: -0.23,
    kurtosis: 2.67,
    
    // Market Correlation
    betaToMarket: 0.73,
    correlationToBTC: 0.68,
    correlationToETH: 0.74,
    trackingError: 0.071,
    
    // Alpha Generation
    alphaGeneration: 0.139,
    excessReturn: 0.847,
    activeReturn: 0.139,
    
    // Trading Performance
    winRate: 0.873,
    avgWin: 0.0234,
    avgLoss: -0.0087,
    profitFactor: 2.69,
    expectancy: 0.0187,
    
    // Returns Breakdown
    dailyReturn: 0.0287,
    weeklyReturn: 0.1342,
    monthlyReturn: 0.4891,
    quarterlyReturn: 1.847,
    yearlyReturn: 0.914,
    annualizedReturn: 0.914,
    cumulativeReturn: 2.847,
    
    // Portfolio Value & P&L
    portfolioValue: 2400000,
    totalPnL: 487632,
    unrealizedPnL: 23456,
    realizedPnL: 464176,
    totalReturn: 0.2537,
    
    // Trading Statistics
    totalTrades: 1847,
    winningTrades: 1611,
    losingTrades: 236,
    avgTradeDuration: 4.7,
    bestTrade: 0.127,
    worstTrade: -0.043,
    consecutiveWins: 12,
    consecutiveLosses: 3,
    
    // Model-Specific Metrics
    imageContribution: 0.267,
    sentimentAccuracy: 0.834,
    technicalAccuracy: 0.921,
    fundamentalAccuracy: 0.789,
    hybridAccuracy: 0.941,
    
    // Live Performance
    liveAccuracy: 0.847,
    realtimeAlpha: 0.0423,
    currentDrawdown: -0.012,
    todaysPnL: 12847,
    
    // Advanced Risk Metrics
    tailRisk: 0.089,
    skewness: -0.23,
    excessKurtosis: 0.67,
    conditionalDrawdown: -0.067,
    maxRunup: 0.237,
    recoveryFactor: 19.6,
    profitabilityIndex: 3.42,
    
    // Multi-Asset Performance
    btcAlpha: 0.156,
    ethAlpha: 0.142,
    altcoinAlpha: 0.167,
    correlationStability: 0.89,
    
    // Model Confidence & Reliability
    avgConfidence: 0.847,
    predictionReliability: 0.892,
    modelStability: 0.934,
    backtestConsistency: 0.918
  });
  
  // Update performance metrics in real-time
  useEffect(() => {
    if (realTimePrice && liveData.length > 0) {
      setPerformanceMetrics(prev => ({
        ...prev,
        portfolioValue: 2400000 + (realTimePrice.changePercent || 0) * 24000,
        unrealizedPnL: 23456 + (realTimePrice.changePercent || 0) * 1000,
        todaysPnL: 12847 + (realTimePrice.changePercent || 0) * 500,
        currentDrawdown: Math.min(-0.001, (realTimePrice.changePercent || 0) / 100),
        liveAccuracy: predictions.length > 0 ? 
          predictions.filter(p => p.aiPrediction?.confidence > 0.8).length / predictions.length : 0.847,
        realtimeAlpha: 0.0423 + (Math.random() - 0.5) * 0.01
      }));
    }
  }, [realTimePrice, liveData, predictions]);

  const dataSourceHealth = {
    priceFeeds: { status: connectionStatus, latency: 125, coverage: 0.99, lastUpdate: lastUpdate },
    sentiment: { status: 'active', latency: 340, coverage: 0.96, lastUpdate: sentimentData?.lastUpdate },
    news: { status: newsData.length > 0 ? 'active' : 'degraded', latency: 230, coverage: 0.94, lastUpdate: new Date() },
    onChain: { status: 'active', latency: 580, coverage: 0.92, lastUpdate: new Date() },
    imageData: { status: uploadedImage ? 'active' : 'idle', latency: 340, coverage: 0.88, lastUpdate: new Date() },
    aiModel: { status: 'active', latency: 890, coverage: 0.97, lastUpdate: lastUpdate }
  };

  const StatusIndicator = ({ status, pulse = false }) => (
    <div className={`w-3 h-3 rounded-full ${
      status === 'active' || status === 'connected' ? 'bg-[#FFF7E8]' : 
      status === 'connecting' ? 'bg-[#BDB6A8] animate-pulse' :
      status === 'degraded' ? 'bg-[#BDB6A8]' : 
      status === 'idle' ? 'bg-[#A9A297]' :
      'bg-[#A9A297]'
    } ${pulse ? 'animate-pulse' : ''}`} />
  );

  const MetricCard = ({ title, value, icon: Icon, trend = null, live = false }) => (
    <div className="bg-[#0F0F0F] rounded-xl p-5 border transition-all duration-300 hover:bg-[#0F0F0F]/90" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
            <Icon className="w-4 h-4 text-[#EDE8DF]" strokeWidth="1.5" />
          </div>
          {live && <StatusIndicator status="active" pulse={true} />}
        </div>
        {trend && (
          <span className={`text-sm font-semibold tracking-tight ${trend > 0 ? 'text-[#FFF7E8]' : 'text-[#BDB6A8]'}`}>
            {trend > 0 ? '↗' : '↘'} {Math.abs(trend).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="text-2xl font-semibold text-[#FFF7E8] tracking-tight">{value}</div>
      <div className="text-sm text-[#CFC8B8] font-medium">{title}</div>
      {live && (
        <div className="text-xs text-[#BDB6A8] mt-2 font-medium">● LIVE • {lastUpdate.toLocaleTimeString()}</div>
      )}
    </div>
  );

  const LivePricePanel = () => (
    <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
            <Wifi className="w-5 h-5 text-[#EDE8DF]" strokeWidth="1.5" />
          </div>
          <h3 className="text-lg font-semibold text-[#FFF7E8] tracking-tight">Live Market Feed</h3>
          <StatusIndicator status={connectionStatus} pulse={connectionStatus === 'connecting'} />
        </div>
        <div className="text-sm text-[#BDB6A8] font-medium">
          Last update: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>
      
      {realTimePrice && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-3xl font-semibold text-[#FFF7E8] tracking-tight">
                ${realTimePrice.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className="text-sm text-[#CFC8B8] font-medium">{realTimePrice.symbol}</div>
            </div>
            <div className={`text-right ${realTimePrice.changePercent >= 0 ? 'text-[#FFF7E8]' : 'text-[#BDB6A8]'}`}>
              <div className="text-xl font-semibold tracking-tight">
                {realTimePrice.changePercent >= 0 ? '+' : ''}{realTimePrice.changePercent.toFixed(2)}%
              </div>
              <div className="text-sm font-medium">
                {realTimePrice.change >= 0 ? '+' : ''}${realTimePrice.change.toFixed(2)}
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-white/5 p-3 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
              <span className="text-[#CFC8B8]">Bid:</span>
              <span className="font-semibold ml-2 text-[#FFF7E8]">${realTimePrice.bid.toFixed(2)}</span>
            </div>
            <div className="bg-white/5 p-3 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
              <span className="text-[#CFC8B8]">Ask:</span>
              <span className="font-semibold ml-2 text-[#FFF7E8]">${realTimePrice.ask.toFixed(2)}</span>
            </div>
            <div className="bg-white/5 p-3 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
              <span className="text-[#CFC8B8]">Volume:</span>
              <span className="font-semibold ml-2 text-[#FFF7E8]">{(realTimePrice.volume / 1000000).toFixed(1)}M</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const LiveNewsPanel = () => (
    <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
          <RefreshCw className="w-5 h-5 text-[#EDE8DF]" strokeWidth="1.5" />
        </div>
        <h3 className="text-lg font-semibold text-[#FFF7E8] tracking-tight">Market Intelligence</h3>
        <StatusIndicator status={newsData.length > 0 ? 'active' : 'idle'} />
      </div>
      
      {sentimentData && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-[#FFF7E8] tracking-tight">Sentiment Analysis</span>
            <span className={`font-semibold text-lg tracking-tight ${
              sentimentData.overall > 0.3 ? 'text-[#FFF7E8]' : 
              sentimentData.overall < -0.3 ? 'text-[#BDB6A8]' : 'text-[#D8D2C8]'
            }`}>
              {sentimentData.overall > 0.3 ? 'Bullish' : 
               sentimentData.overall < -0.3 ? 'Bearish' : 'Neutral'}
            </span>
          </div>
          <div className="flex gap-4 text-sm">
            <span className="text-[#FFF7E8] font-medium">↗ {sentimentData.bullish} Bullish</span>
            <span className="text-[#BDB6A8] font-medium">↘ {sentimentData.bearish} Bearish</span>
            <span className="text-[#D8D2C8] font-medium">— {sentimentData.neutral} Neutral</span>
          </div>
        </div>
      )}
      
      <div className="space-y-3 max-h-60 overflow-y-auto">
        {newsData.map((news, index) => (
          <div key={index} className="p-3 bg-white/5 rounded-lg border hover:bg-white/10 transition-colors" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
            <div className="flex items-start justify-between mb-1">
              <div className="font-medium text-sm text-[#FFF7E8] tracking-tight">{news.title}</div>
              <div className={`w-3 h-3 rounded-full ${
                news.sentiment > 0.3 ? 'bg-[#FFF7E8]' : 
                news.sentiment < -0.3 ? 'bg-[#BDB6A8]' : 'bg-[#D8D2C8]'
              }`} />
            </div>
            <div className="flex items-center gap-3 text-xs text-[#CFC8B8] font-medium">
              <span>{news.source}</span>
              <span>{news.timestamp.toLocaleTimeString()}</span>
              <span className={`px-2 py-1 rounded text-[10px] font-semibold ${
                news.impact === 'high' ? 'bg-[#FFF7E8] text-[#0B0B0B]' :
                news.impact === 'medium' ? 'bg-[#D8D2C8] text-[#0B0B0B]' :
                'bg-[#BDB6A8] text-[#0B0B0B]'
              }`}>
                {news.impact.toUpperCase()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const DataSourcePanel = () => (
    <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
          <Database className="w-5 h-5 text-[#EDE8DF]" strokeWidth="1.5" />
        </div>
        <h3 className="text-lg font-semibold text-[#FFF7E8] tracking-tight">Data Feeds</h3>
      </div>
      
      <div className="space-y-3">
        {Object.entries(dataSourceHealth).map(([source, health]) => (
          <div key={source} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border hover:bg-white/10 transition-colors" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
            <div className="flex items-center gap-3">
              <StatusIndicator status={health.status} />
              <div>
                <span className="font-semibold capitalize text-[#FFF7E8] tracking-tight">{source.replace(/([A-Z])/g, ' $1').trim()}</span>
                {source === 'priceFeeds' && (
                  <div className="text-xs text-[#FFF7E8] font-semibold">● INSTITUTIONAL FEED</div>
                )}
                {source === 'aiModel' && (
                  <div className="text-xs text-[#D8D2C8] font-semibold">⚡ HYPERBOLIC CNN</div>
                )}
              </div>
            </div>
            <div className="text-sm text-[#CFC8B8] font-medium">
              {health.latency}ms • {(health.coverage * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const LivePredictionChart = () => (
    <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-[#FFF7E8] tracking-tight">Live Predictions vs Reality</h3>
        <div className="flex gap-2">
          <span className="px-3 py-1 bg-white/5 text-[#FFF7E8] text-sm rounded-md font-medium border" style={{ borderColor: 'rgba(237,232,223,0.14)' }}>
            ● LIVE
          </span>
          {predictions.length > 0 && (
            <span className="px-3 py-1 bg-white/5 text-[#D8D2C8] text-sm rounded-md font-medium border" style={{ borderColor: 'rgba(237,232,223,0.14)' }}>
              Confidence: {((predictions[predictions.length - 1]?.aiPrediction?.confidence || 0) * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={liveData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(237,232,223,0.08)" />
          <XAxis dataKey="timestamp" stroke="#D8D2C8" />
          <YAxis stroke="#D8D2C8" />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0F0F0F', 
              border: '1px solid rgba(237,232,223,0.14)',
              borderRadius: '8px',
              color: '#EDE8DF'
            }} 
          />
          <Legend />
          <Line type="monotone" dataKey="price" stroke="#FFF7E8" strokeWidth={2} name="Live Price" />
          <Line type="monotone" dataKey="predicted" stroke="#D8D2C8" strokeWidth={2} name="AI Prediction" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="min-h-screen antialiased bg-[#0A0A0A] text-[#EDE8DF]" style={{fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji"'}}>
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-black/50">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-md bg-[#0F0F0F] border flex items-center justify-center" style={{ borderColor: 'rgba(237,232,223,0.14)' }}>
              <span className="text-sm font-semibold tracking-tight text-[#EDE8DF]">HV</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-[#BDB6A8]">
              <Brain className="h-4 w-4" strokeWidth="1.5" />
              <span>HyperVision AI • Professional Trading Platform</span>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm text-[#D8D2C8]">
            <a href="#dashboard" className="hover:text-[#FFF7E8] transition-colors">Dashboard</a>
            <a href="#analytics" className="hover:text-[#FFF7E8] transition-colors">Analytics</a>
            <a href="#performance" className="hover:text-[#FFF7E8] transition-colors">Performance</a>
            <a href="#portfolio" className="hover:text-[#FFF7E8] transition-colors">Portfolio</a>
          </nav>
          <div className="flex items-center gap-2">
            <div className="inline-flex items-center gap-2 text-xs text-[#D8D2C8] hover:text-[#FFF7E8] transition-colors">
              <StatusIndicator status="active" pulse={true} />
              <span>Alpha Generation: +{(performanceMetrics.alphaGeneration * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
        <div className="border-t" style={{ borderColor: 'rgba(237,232,223,0.08)' }}></div>
      </header>

      {/* Hero Section */}
      <section className="relative">
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
          <div className="absolute -top-28 left-1/2 -translate-x-1/2 w-[90%] h-64 blur-3xl opacity-20" style={{background: 'radial-gradient(40% 60% at 50% 40%, rgba(255,247,232,0.18) 0%, rgba(255,247,232,0.06) 45%, transparent 60%)'}}></div>
        </div>
        <div className="mx-auto max-w-7xl px-6 pt-14 pb-10">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="space-y-5">
              <h1 className="text-3xl md:text-5xl tracking-tight font-semibold text-[#FFF7E8] leading-tight">
                Revolutionary AI Trading with Hyperbolic CNN Technology
              </h1>
              <p className="text-[#CFC8B8] text-base md:text-lg">
                Advanced cryptocurrency prediction using real-time multi-modal data fusion, professional portfolio management, and institutional-grade analytics.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <button 
                  onClick={() => setActiveTab('dashboard')}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-[#FFF7E8] hover:bg-[#F4ECDD] text-[#0B0B0B] border text-sm transition-colors" 
                  style={{ borderColor: 'rgba(237,232,223,0.18)' }}
                >
                  <Activity className="h-4 w-4" strokeWidth="1.5" />
                  View Dashboard
                </button>
                <button 
                  onClick={() => setActiveTab('performance')}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-transparent hover:bg-white/5 border text-[#EDE8DF]" 
                  style={{ borderColor: 'rgba(237,232,223,0.18)' }}
                >
                  <TrendingUp className="h-4 w-4" strokeWidth="1.5" />
                  Performance
                </button>
              </div>
              <div className="flex items-center gap-5 text-xs text-[#BDB6A8] pt-1">
                <div className="inline-flex items-center gap-1.5">
                  <ShieldCheck className="h-4 w-4" strokeWidth="1.5" />
                  Real-time Security
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <Globe className="h-4 w-4" strokeWidth="1.5" />
                  Multi-asset Support
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <Wand2 className="h-4 w-4" strokeWidth="1.5" />
                  AI Predictions
                </div>
              </div>
            </div>
            <div className="relative">
              <div className="aspect-[16/10] rounded-xl overflow-hidden border bg-[#0F0F0F]" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
                <img src="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1600&auto=format&fit=crop" alt="Modern trading platform" className="w-full h-full object-cover opacity-90 grayscale" />
              </div>
              <div className="absolute -bottom-4 -right-4 hidden md:block">
                <div className="px-3 py-2 rounded-lg bg-black/50 border text-[#CFC8B8] text-xs" style={{ borderColor: 'rgba(237,232,223,0.14)', backdropFilter: 'blur(6px)' }}>
                  Live: {(performanceMetrics.liveAccuracy * 100).toFixed(1)}% prediction accuracy
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="border-t" style={{ borderColor: 'rgba(237,232,223,0.08)' }}></div>
      </section>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex gap-2 mb-6 flex-wrap">
          {['dashboard', 'live', 'vision', 'analytics', 'performance', 'portfolio'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-medium capitalize transition-all duration-200 flex items-center gap-2 tracking-tight ${
                activeTab === tab 
                  ? 'bg-[#FFF7E8] text-[#0B0B0B]' 
                  : 'bg-[#0F0F0F] text-[#D8D2C8] hover:bg-white/5 border'
              }`}
              style={activeTab !== tab ? { borderColor: 'rgba(237,232,223,0.14)' } : {}}
            >
              {tab === 'live' && <Wifi className="w-4 h-4" strokeWidth="1.5" />}
              {tab === 'vision' && <Image className="w-4 h-4" strokeWidth="1.5" />}
              {tab === 'performance' && <TrendingUp className="w-4 h-4" strokeWidth="1.5" />}
              {tab === 'portfolio' && <BarChart3 className="w-4 h-4" strokeWidth="1.5" />}
              {tab === 'dashboard' && <Activity className="w-4 h-4" strokeWidth="1.5" />}
              {tab === 'analytics' && <Brain className="w-4 h-4" strokeWidth="1.5" />}
              {tab}
            </button>
          ))}
        </div>

        {/* Controls */}
        <div className="flex gap-4 mb-6 flex-wrap">
          <select 
            value={selectedAsset} 
            onChange={(e) => setSelectedAsset(e.target.value)}
            className="px-4 py-3 bg-[#0F0F0F] border rounded-lg text-[#FFF7E8] focus:outline-none focus:ring-2 focus:ring-[#FFF7E8]/20 hover:border-[#FFF7E8]/30 transition-colors font-medium"
            style={{ borderColor: 'rgba(237,232,223,0.14)' }}
          >
            {cryptoAssets.map(asset => (
              <option key={asset} value={asset} className="bg-[#0F0F0F] text-[#FFF7E8]">{asset}</option>
            ))}
          </select>
          
          <select 
            value={predictionHorizon} 
            onChange={(e) => setPredictionHorizon(e.target.value)}
            className="px-4 py-3 bg-[#0F0F0F] border rounded-lg text-[#FFF7E8] focus:outline-none focus:ring-2 focus:ring-[#FFF7E8]/20 hover:border-[#FFF7E8]/30 transition-colors font-medium"
            style={{ borderColor: 'rgba(237,232,223,0.14)' }}
          >
            <option value="1h" className="bg-[#0F0F0F] text-[#FFF7E8]">1 Hour</option>
            <option value="4h" className="bg-[#0F0F0F] text-[#FFF7E8]">4 Hours</option>
            <option value="1d" className="bg-[#0F0F0F] text-[#FFF7E8]">1 Day</option>
            <option value="1w" className="bg-[#0F0F0F] text-[#FFF7E8]">1 Week</option>
          </select>

          <select 
            value={userMode} 
            onChange={(e) => setUserMode(e.target.value)}
            className="px-4 py-3 bg-[#0F0F0F] border rounded-lg text-[#FFF7E8] focus:outline-none focus:ring-2 focus:ring-[#FFF7E8]/20 hover:border-[#FFF7E8]/30 transition-colors font-medium"
            style={{ borderColor: 'rgba(237,232,223,0.14)' }}
          >
            <option value="retail" className="bg-[#0F0F0F] text-[#FFF7E8]">Professional Trader</option>
            <option value="institutional" className="bg-[#0F0F0F] text-[#FFF7E8]">Institutional</option>
            <option value="research" className="bg-[#0F0F0F] text-[#FFF7E8]">Quantitative Research</option>
          </select>
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MetricCard 
                title="Live Accuracy" 
                value={`${(performanceMetrics.liveAccuracy * 100).toFixed(1)}%`} 
                icon={Target} 
                trend={2.8} 
                live={true}
              />
              <MetricCard 
                title="Current Price" 
                value={realTimePrice ? `$${realTimePrice.price.toFixed(2)}` : 'Loading...'} 
                icon={TrendingUp} 
                trend={realTimePrice?.changePercent} 
                live={true}
              />
              <MetricCard title="AI Predictions" value={predictions.length.toString()} icon={Brain} live={true} />
              <MetricCard title="Data Sources" value="6 Active" icon={Database} live={true} />
            </div>

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <LivePricePanel />
              <div className="lg:col-span-2">
                <LivePredictionChart />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LiveNewsPanel />
              <DataSourcePanel />
            </div>
          </div>
        )}

        {/* Live Tab */}
        {activeTab === 'live' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <LivePricePanel />
              <LiveNewsPanel />
              <DataSourcePanel />
            </div>
            
            <LivePredictionChart />
          </div>
        )}

        {/* Vision Tab */}
        {activeTab === 'vision' && (
          <div className="space-y-6">
            <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
              <div className="flex items-center gap-3 mb-4">
                <Camera className="w-6 h-6 text-[#D8D2C8]" strokeWidth="1.5" />
                <h3 className="text-lg font-semibold text-[#FFF7E8] tracking-tight">Chart Analysis</h3>
              </div>
              
              <div 
                className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-[#FFF7E8]/30 transition-colors"
                style={{ borderColor: 'rgba(237,232,223,0.18)' }}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-8 h-8 text-[#D8D2C8] mx-auto mb-2" strokeWidth="1.5" />
                <p className="text-[#CFC8B8]">Upload candlestick chart or trading pattern</p>
                <p className="text-sm text-[#BDB6A8] mt-1">Real-time analysis with live data context</p>
              </div>
              
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                className="hidden"
              />
              
              {imageAnalysis && (
                <div className="mt-4 bg-white/5 p-4 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>Pattern: <span className="font-medium text-[#FFF7E8]">{imageAnalysis.patternType}</span></div>
                    <div>Confidence: <span className="font-medium text-[#FFF7E8]">{(imageAnalysis.confidence * 100).toFixed(1)}%</span></div>
                    <div>Prediction: <span className="font-medium text-[#FFF7E8]">{imageAnalysis.prediction}</span></div>
                    <div>Risk Score: <span className="font-medium text-[#D8D2C8]">{imageAnalysis.riskScore}</span></div>
                  </div>
                </div>
              )}
            </div>

            <LivePredictionChart />
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <LivePredictionChart />
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LiveNewsPanel />
              <DataSourcePanel />
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            {/* Performance Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard 
                title="Daily Returns" 
                value={`${(performanceMetrics.dailyReturn * 100).toFixed(2)}%`} 
                icon={TrendingUp} 
                trend={performanceMetrics.dailyReturn * 100}
              />
              <MetricCard 
                title="Monthly Returns" 
                value={`${(performanceMetrics.monthlyReturn * 100).toFixed(1)}%`} 
                icon={Target} 
                trend={performanceMetrics.monthlyReturn * 100}
              />
              <MetricCard 
                title="Yearly Returns" 
                value={`${(performanceMetrics.yearlyReturn * 100).toFixed(1)}%`} 
                icon={Award} 
                trend={performanceMetrics.yearlyReturn * 100}
              />
              <MetricCard 
                title="Sharpe Ratio" 
                value={performanceMetrics.sharpe.toFixed(2)} 
                icon={BarChart3}
              />
            </div>

            {/* Detailed Performance Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <TrendingUp className="w-4 h-4 text-[#EDE8DF]" strokeWidth="1.5" />
                  </div>
                  <span className="text-[#FFF7E8] tracking-tight">AI Model Performance</span>
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <div className="text-[#CFC8B8] font-semibold">Win Rate</div>
                    <div className="text-2xl font-semibold text-[#FFF7E8] tracking-tight">{(performanceMetrics.winRate * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <div className="text-[#CFC8B8] font-semibold">Accuracy</div>
                    <div className="text-2xl font-semibold text-[#FFF7E8] tracking-tight">{(performanceMetrics.accuracy * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <div className="text-[#CFC8B8] font-semibold">Profit Factor</div>
                    <div className="text-2xl font-semibold text-[#FFF7E8] tracking-tight">{performanceMetrics.profitFactor.toFixed(2)}</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <div className="text-[#CFC8B8] font-semibold">Calmar Ratio</div>
                    <div className="text-2xl font-semibold text-[#FFF7E8] tracking-tight">{performanceMetrics.calmar.toFixed(2)}</div>
                  </div>
                </div>
              </div>

              <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <div className="p-2 bg-white/5 rounded-lg border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <Shield className="w-4 h-4 text-[#EDE8DF]" strokeWidth="1.5" />
                  </div>
                  <span className="text-[#FFF7E8] tracking-tight">Risk Management</span>
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-2 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <span className="text-[#CFC8B8] font-medium">Max Drawdown</span>
                    <span className="text-[#BDB6A8] font-semibold">{(performanceMetrics.maxDrawdown * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <span className="text-[#CFC8B8] font-medium">Volatility</span>
                    <span className="text-[#D8D2C8] font-semibold">{(performanceMetrics.volatility * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <span className="text-[#CFC8B8] font-medium">VaR (95%)</span>
                    <span className="text-[#D8D2C8] font-semibold">{(performanceMetrics.var95 * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <span className="text-[#CFC8B8] font-medium">Sortino Ratio</span>
                    <span className="text-[#D8D2C8] font-semibold">{performanceMetrics.sortino.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                    <span className="text-[#CFC8B8] font-medium">Beta to Market</span>
                    <span className="text-[#D8D2C8] font-semibold">{performanceMetrics.betaToMarket.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Trading Statistics */}
            <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-[#D8D2C8]" strokeWidth="1.5" />
                <span className="text-[#FFF7E8] tracking-tight">Trading Performance Statistics</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 text-sm">
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Total Trades</div>
                  <div className="text-xl font-semibold text-[#FFF7E8] tracking-tight">{performanceMetrics.totalTrades.toLocaleString()}</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Winning Trades</div>
                  <div className="text-xl font-semibold text-[#FFF7E8] tracking-tight">{performanceMetrics.winningTrades.toLocaleString()}</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Losing Trades</div>
                  <div className="text-xl font-semibold text-[#BDB6A8] tracking-tight">{performanceMetrics.losingTrades.toLocaleString()}</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Avg Win</div>
                  <div className="text-xl font-semibold text-[#FFF7E8] tracking-tight">{(performanceMetrics.avgWin * 100).toFixed(2)}%</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Avg Loss</div>
                  <div className="text-xl font-semibold text-[#BDB6A8] tracking-tight">{(performanceMetrics.avgLoss * 100).toFixed(2)}%</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded border" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                  <div className="text-[#CFC8B8] font-medium">Alpha Generation</div>
                  <div className="text-xl font-semibold text-[#FFF7E8] tracking-tight">+{(performanceMetrics.alphaGeneration * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Returns Chart */}
            <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
              <h3 className="text-lg font-semibold mb-4 text-[#FFF7E8] tracking-tight">Historical Returns Performance</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={[
                  { period: 'Daily', return: performanceMetrics.dailyReturn * 100, benchmark: 0.12 },
                  { period: 'Weekly', return: performanceMetrics.weeklyReturn * 100, benchmark: 0.84 },
                  { period: 'Monthly', return: performanceMetrics.monthlyReturn * 100, benchmark: 3.6 },
                  { period: 'Yearly', return: performanceMetrics.yearlyReturn * 100, benchmark: 43.2 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(237,232,223,0.08)" />
                  <XAxis dataKey="period" stroke="#D8D2C8" />
                  <YAxis stroke="#D8D2C8" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0F0F0F', 
                      border: '1px solid rgba(237,232,223,0.14)',
                      borderRadius: '8px',
                      color: '#EDE8DF'
                    }} 
                    formatter={(value, name) => [`${value.toFixed(2)}%`, name === 'return' ? 'HyperVision AI' : 'Market Benchmark']}
                  />
                  <Legend />
                  <Bar dataKey="return" fill="#FFF7E8" name="HyperVision AI Returns" />
                  <Bar dataKey="benchmark" fill="#BDB6A8" name="Market Benchmark" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Portfolio Tab */}
        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            {/* Portfolio Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard 
                title="Portfolio Value" 
                value={`$${(performanceMetrics.portfolioValue / 1000000).toFixed(2)}M`} 
                icon={Award} 
                trend={5.7}
                live={true}
              />
              <MetricCard 
                title="Total P&L" 
                value={`$${(performanceMetrics.totalPnL / 1000).toFixed(0)}K`} 
                icon={TrendingUp} 
                trend={12.3}
                live={true}
              />
              <MetricCard 
                title="Unrealized P&L" 
                value={`$${(performanceMetrics.unrealizedPnL / 1000).toFixed(0)}K`} 
                icon={Activity} 
                trend={2.1}
                live={true}
              />
              <MetricCard 
                title="Realized P&L" 
                value={`$${(performanceMetrics.realizedPnL / 1000).toFixed(0)}K`} 
                icon={Target} 
                trend={11.8}
              />
            </div>

            {/* Asset Allocation */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#D8D2C8]" strokeWidth="1.5" />
                  <span className="text-[#FFF7E8] tracking-tight">Current Asset Allocation</span>
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={[
                    { asset: 'BTC', allocation: 35, performance: 28 },
                    { asset: 'ETH', allocation: 25, performance: 42 },
                    { asset: 'ADA', allocation: 15, performance: 18 },
                    { asset: 'SOL', allocation: 12, performance: 65 },
                    { asset: 'MATIC', allocation: 8, performance: 22 },
                    { asset: 'DOT', allocation: 5, performance: 15 }
                  ]}>
                    <PolarGrid stroke="rgba(237,232,223,0.12)" />
                    <PolarAngleAxis dataKey="asset" tick={{ fill: '#D8D2C8' }} />
                    <PolarRadiusAxis angle={90} domain={[0, 70]} tick={{ fill: '#CFC8B8' }} />
                    <Radar name="Allocation %" dataKey="allocation" stroke="#FFF7E8" fill="#FFF7E8" fillOpacity={0.1} />
                    <Radar name="Performance %" dataKey="performance" stroke="#D8D2C8" fill="#D8D2C8" fillOpacity={0.1} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-[#D8D2C8]" strokeWidth="1.5" />
                  <span className="text-[#FFF7E8] tracking-tight">Position Details</span>
                </h3>
                <div className="space-y-3 max-h-72 overflow-y-auto">
                  {[
                    { symbol: 'BTC-USD', size: 35.2, value: 840000, pnl: 87432, pnlPercent: 11.6 },
                    { symbol: 'ETH-USD', size: 428.6, value: 600000, pnl: 126890, pnlPercent: 26.8 },
                    { symbol: 'ADA-USD', size: 800000, value: 360000, pnl: 45670, pnlPercent: 14.5 },
                    { symbol: 'SOL-USD', size: 1920, value: 288000, pnl: 87234, pnlPercent: 43.4 },
                    { symbol: 'MATIC-USD', size: 320000, value: 192000, pnl: 23456, pnlPercent: 13.9 },
                    { symbol: 'DOT-USD', size: 24000, value: 120000, pnl: 8450, pnlPercent: 7.6 }
                  ].map((position, index) => (
                    <div key={index} className="p-3 bg-white/5 rounded-lg border hover:bg-white/10 transition-colors" style={{ borderColor: 'rgba(237,232,223,0.12)' }}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-semibold text-[#FFF7E8] tracking-tight">{position.symbol}</div>
                        <div className={`font-semibold tracking-tight ${position.pnlPercent > 0 ? 'text-[#FFF7E8]' : 'text-[#BDB6A8]'}`}>
                          {position.pnlPercent > 0 ? '+' : ''}{position.pnlPercent}%
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-sm text-[#CFC8B8]">
                        <div>Size: {position.size.toLocaleString()}</div>
                        <div>Value: ${(position.value / 1000).toFixed(0)}K</div>
                        <div className={position.pnl > 0 ? 'text-[#FFF7E8]' : 'text-[#BDB6A8]'}>
                          P&L: {position.pnl > 0 ? '+' : ''}${(position.pnl / 1000).toFixed(1)}K
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Portfolio Performance Chart */}
            <div className="bg-[#0F0F0F] rounded-xl p-6 border" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
              <h3 className="text-lg font-semibold mb-4 text-[#FFF7E8] tracking-tight">Portfolio Value Over Time</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={Array.from({ length: 30 }, (_, i) => ({
                  date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
                  portfolioValue: 2000000 + (Math.random() * 400000) + (i * 13333),
                  benchmark: 2000000 + (i * 8000) + (Math.random() * 100000)
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(237,232,223,0.08)" />
                  <XAxis dataKey="date" stroke="#D8D2C8" />
                  <YAxis stroke="#D8D2C8" tickFormatter={(value) => `$${(value / 1000000).toFixed(1)}M`} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0F0F0F', 
                      border: '1px solid rgba(237,232,223,0.14)',
                      borderRadius: '8px',
                      color: '#EDE8DF'
                    }}
                    formatter={(value, name) => [`$${(value / 1000000).toFixed(2)}M`, name === 'portfolioValue' ? 'Portfolio' : 'Benchmark']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="portfolioValue" stroke="#FFF7E8" strokeWidth={2} name="HyperVision Portfolio" />
                  <Line type="monotone" dataKey="benchmark" stroke="#BDB6A8" strokeWidth={2} name="Market Benchmark" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t" style={{ borderColor: 'rgba(237,232,223,0.08)' }}>
        <div className="mx-auto max-w-7xl px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-[#BDB6A8] text-sm">
            <div className="h-6 w-6 rounded-md bg-[#0F0F0F] border flex items-center justify-center" style={{ borderColor: 'rgba(237,232,223,0.10)' }}>
              <span className="text-[10px] font-semibold tracking-tight text-[#EDE8DF]">HV</span>
            </div>
            <span>HyperVision AI • Professional Trading Platform</span>
          </div>
          <div className="text-xs text-[#A9A297]">
            Disclaimer: Trading involves substantial risk. Past performance is not indicative of future results.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default CryptoPredictionApp;