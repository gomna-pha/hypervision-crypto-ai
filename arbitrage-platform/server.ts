import express from 'express';
import cors from 'cors';
import path from 'path';
import { PlatformOrchestrator } from './core/platform-orchestrator';

const app = express();
const PORT = process.env.PORT || 3001;

// Initialize the arbitrage platform
const platform = new PlatformOrchestrator();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Serve the dashboard
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'dashboard.html'));
});

// API Routes for dashboard data
app.get('/api/dashboard', (req, res) => {
  try {
    const dashboardData = platform.getDashboardData();
    res.json(dashboardData);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/status', (req, res) => {
  try {
    const platformState = platform.getPlatformState();
    res.json(platformState);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/parameters', (req, res) => {
  try {
    const parameters = platform.getVisibleParameters();
    res.json(parameters);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/metrics', (req, res) => {
  try {
    const metrics = platform.getPerformanceMetrics();
    res.json(metrics);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/workflows', (req, res) => {
  try {
    const workflows = platform.getActiveWorkflows();
    res.json(workflows);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/events', (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const events = platform.getRecentEvents(limit);
    res.json(events);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Platform control endpoints
app.post('/api/start', async (req, res) => {
  try {
    await platform.start();
    res.json({ status: 'started' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/stop', async (req, res) => {
  try {
    await platform.stop();
    res.json({ status: 'stopped' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/emergency-stop', async (req, res) => {
  try {
    await platform.emergencyStop();
    res.json({ status: 'emergency_stopped' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString(),
    platform_status: platform.getPlatformState().status
  });
});

// Error handling middleware
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Server error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : undefined
  });
});

// Start server
app.listen(PORT, async () => {
  console.log(`🚀 LLM Arbitrage Platform server running on port ${PORT}`);
  console.log(`📊 Dashboard available at: http://localhost:${PORT}`);
  console.log(`🔍 API endpoints available at: http://localhost:${PORT}/api/`);
  
  // Auto-start the platform
  try {
    console.log('🔧 Auto-starting arbitrage platform...');
    await platform.start();
    console.log('✅ Platform started successfully');
  } catch (error) {
    console.error('❌ Failed to auto-start platform:', error);
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🛑 Received SIGTERM, shutting down gracefully...');
  
  try {
    await platform.stop();
    process.exit(0);
  } catch (error) {
    console.error('❌ Error during shutdown:', error);
    process.exit(1);
  }
});

process.on('SIGINT', async () => {
  console.log('🛑 Received SIGINT, shutting down gracefully...');
  
  try {
    await platform.stop();
    process.exit(0);
  } catch (error) {
    console.error('❌ Error during shutdown:', error);
    process.exit(1);
  }
});

export default app;