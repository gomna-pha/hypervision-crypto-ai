import winston from 'winston';
import path from 'path';

const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
  winston.format.errors({ stack: true }),
  winston.format.splat(),
  winston.format.json(),
  winston.format.printf(({ timestamp, level, message, ...metadata }) => {
    let msg = `${timestamp} [${level.toUpperCase()}]: ${message}`;
    if (Object.keys(metadata).length > 0) {
      msg += ` ${JSON.stringify(metadata)}`;
    }
    return msg;
  })
);

class Logger {
  private logger: winston.Logger;
  private static instances: Map<string, Logger> = new Map();

  private constructor(name: string) {
    this.logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: logFormat,
      defaultMeta: { service: name },
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          ),
        }),
        new winston.transports.File({
          filename: path.join(process.env.LOG_DIR || './logs', `${name}-error.log`),
          level: 'error',
          maxsize: 10485760, // 10MB
          maxFiles: 5,
        }),
        new winston.transports.File({
          filename: path.join(process.env.LOG_DIR || './logs', `${name}.log`),
          maxsize: 10485760, // 10MB
          maxFiles: 10,
        }),
      ],
    });
  }

  static getInstance(name: string): Logger {
    if (!Logger.instances.has(name)) {
      Logger.instances.set(name, new Logger(name));
    }
    return Logger.instances.get(name)!;
  }

  debug(message: string, metadata?: any): void {
    this.logger.debug(message, metadata);
  }

  info(message: string, metadata?: any): void {
    this.logger.info(message, metadata);
  }

  warn(message: string, metadata?: any): void {
    this.logger.warn(message, metadata);
  }

  error(message: string, error?: Error | any, metadata?: any): void {
    if (error instanceof Error) {
      this.logger.error(message, { error: error.message, stack: error.stack, ...metadata });
    } else {
      this.logger.error(message, { error, ...metadata });
    }
  }

  metric(name: string, value: number, tags?: Record<string, string>): void {
    this.logger.info('METRIC', { metric: name, value, tags });
  }
}

export default Logger;