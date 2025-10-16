import { execSync } from 'child_process';
import { createRequire } from 'module';
import { register } from 'ts-node/esm';

// Register TypeScript loader
register({
  esm: true,
  experimentalSpecifierResolution: 'node'
});

// Start the demo server
import('./arbitrage/demo-server.ts').catch(console.error);