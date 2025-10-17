#!/usr/bin/env node
import { Command } from 'commander';
import { EconomicAgent } from './agents/EconomicAgent';
import { PriceAgent } from './agents/PriceAgent';
import FusionBrain from './fusion/FusionBrain';
import Logger from './utils/logger';
import config from './utils/ConfigLoader';

const logger = Logger.getInstance('Main');

const program = new Command();

program
  .name('arbitrage-platform')
  .description('Agent-Based LLM Arbitrage Platform')
  .version('1.0.0');

program
  .command('start')
  .description('Start all agents and services')
  .option('--agents <agents>', 'Comma-separated list of agents to start', 'all')
  .action(async (options) => {
    logger.info('Starting Arbitrage Platform...', { agents: options.agents });
    
    const agents: any[] = [];
    
    try {
      if (options.agents === 'all' || options.agents.includes('economic')) {
        const economicAgent = new EconomicAgent(3001);
        await economicAgent.start();
        agents.push(economicAgent);
      }
      
      if (options.agents === 'all' || options.agents.includes('price')) {
        const priceAgent = new PriceAgent(3002);
        await priceAgent.start();
        agents.push(priceAgent);
      }
      
      // Add other agents as they're implemented
      
      if (options.agents === 'all' || options.agents.includes('fusion')) {
        const fusion = new FusionBrain();
        await fusion.initialize();
        await fusion.start();
        agents.push(fusion);
      }
      
      logger.info('All requested agents started successfully');
      
      // Handle shutdown
      process.on('SIGINT', async () => {
        logger.info('Shutting down agents...');
        for (const agent of agents) {
          await agent.stop();
        }
        process.exit(0);
      });
      
    } catch (error) {
      logger.error('Failed to start agents', error);
      process.exit(1);
    }
  });

program
  .command('test')
  .description('Run system tests')
  .action(async () => {
    logger.info('Running system tests...');
    
    // Test configuration loading
    const testConfig = config.getAll();
    logger.info('Configuration loaded', { 
      agents: Object.keys(testConfig.agents || {}),
      constraints: testConfig.constraints,
    });
    
    // Test agent initialization
    try {
      const economicAgent = new EconomicAgent(3011);
      await economicAgent.start();
      logger.info('EconomicAgent test passed');
      await economicAgent.stop();
      
      const priceAgent = new PriceAgent(3012);
      await priceAgent.start();
      logger.info('PriceAgent test passed');
      await priceAgent.stop();
      
      logger.info('All tests passed');
      process.exit(0);
    } catch (error) {
      logger.error('Tests failed', error);
      process.exit(1);
    }
  });

program
  .command('backtest')
  .description('Run backtesting')
  .option('--start <date>', 'Start date (YYYY-MM-DD)', '2024-01-01')
  .option('--end <date>', 'End date (YYYY-MM-DD)', '2024-12-31')
  .action(async (options) => {
    logger.info('Starting backtest...', options);
    // Backtesting implementation will be added
    logger.info('Backtest feature coming soon');
  });

program
  .command('dashboard')
  .description('Start monitoring dashboard')
  .option('--port <port>', 'Dashboard port', '8080')
  .action(async (options) => {
    logger.info('Starting dashboard on port', options.port);
    // Dashboard implementation will be added
    logger.info('Dashboard feature coming soon');
  });

program.parse();