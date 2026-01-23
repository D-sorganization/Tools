/**
 * Application Logger
 *
 * Structured logging with support for different log levels and metadata.
 * Works in both server and client environments.
 * Uses pino for high-performance logging in production.
 *
 * For now, uses console with structured formatting.
 * TODO: Add pino when ready for production.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogMetadata {
  [key: string]: unknown;
}

/**
 * Get log level from environment variables
 * Client-safe: Uses NEXT_PUBLIC_LOG_LEVEL for client, falls back to LOG_LEVEL for server
 */
function getLogLevel(): LogLevel {
  // Check if we're on the client (browser)
  const isClient = typeof window !== 'undefined';

  if (isClient) {
    // Client: Use NEXT_PUBLIC_LOG_LEVEL (safe to expose to browser)
    const level = process.env.NEXT_PUBLIC_LOG_LEVEL || 'info';
    return level as LogLevel;
  } else {
    // Server: Can access all env vars, try both public and private
    const level = process.env.LOG_LEVEL || process.env.NEXT_PUBLIC_LOG_LEVEL || 'info';
    return level as LogLevel;
  }
}

class Logger {
  private minLevel: LogLevel;

  constructor() {
    this.minLevel = getLogLevel();
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
    const currentLevelIndex = levels.indexOf(this.minLevel);
    const messageLevelIndex = levels.indexOf(level);
    return messageLevelIndex >= currentLevelIndex;
  }

  private formatMessage(level: LogLevel, message: string, metadata?: LogMetadata): string {
    const timestamp = new Date().toISOString();
    const metadataStr = metadata ? `\n${JSON.stringify(metadata, null, 2)}` : '';
    return `[${timestamp}] ${level.toUpperCase()}: ${message}${metadataStr}`;
  }

  private log(level: LogLevel, message: string, metadata?: LogMetadata): void {
    if (!this.shouldLog(level)) {
      return;
    }

    const formattedMessage = this.formatMessage(level, message, metadata);

    switch (level) {
      case 'debug':
        console.debug(formattedMessage);
        break;
      case 'info':
        console.info(formattedMessage);
        break;
      case 'warn':
        console.warn(formattedMessage);
        break;
      case 'error':
        console.error(formattedMessage);
        break;
    }
  }

  /**
   * Log debug message (lowest priority)
   * Use for detailed debugging information
   */
  debug(message: string, metadata?: LogMetadata): void;
  debug(metadata: LogMetadata, message: string): void;
  debug(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    this.log('debug', message, metadata);
  }

  /**
   * Log info message
   * Use for general informational messages
   */
  info(message: string, metadata?: LogMetadata): void;
  info(metadata: LogMetadata, message: string): void;
  info(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    this.log('info', message, metadata);
  }

  /**
   * Log warning message
   * Use for warning messages that don't prevent operation
   */
  warn(message: string, metadata?: LogMetadata): void;
  warn(metadata: LogMetadata, message: string): void;
  warn(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    this.log('warn', message, metadata);
  }

  /**
   * Log error message (highest priority)
   * Use for errors that prevent operation
   */
  error(message: string, metadata?: LogMetadata): void;
  error(metadata: LogMetadata, message: string): void;
  error(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);

    // If metadata contains an Error object, extract stack trace
    if (metadata && 'error' in metadata && metadata.error instanceof Error) {
      metadata.error = {
        name: metadata.error.name,
        message: metadata.error.message,
        stack: metadata.error.stack,
      };
    }

    this.log('error', message, metadata);

    // Send to error tracking service if configured
    const sentryDsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
    if (sentryDsn && typeof window !== 'undefined') {
      try {
        if ((window as any).Sentry) {
          (window as any).Sentry.captureMessage(message, {
            level: 'error',
            extra: metadata,
          });
        }
      } catch (error) {
        // Silently fail if Sentry is not available
        console.error('Failed to send error to Sentry:', error);
      }
    }
  }

  /**
   * Parse arguments to support both signatures:
   * - logger.info('message', { metadata })
   * - logger.info({ metadata }, 'message')
   */
  private parseArgs(
    messageOrMetadata: string | LogMetadata,
    metadataOrMessage?: LogMetadata | string
  ): [string, LogMetadata | undefined] {
    if (typeof messageOrMetadata === 'string') {
      return [messageOrMetadata, metadataOrMessage as LogMetadata | undefined];
    } else {
      return [metadataOrMessage as string, messageOrMetadata];
    }
  }

  /**
   * Create a child logger with additional context
   * Useful for adding module/component-specific metadata
   */
  child(defaultMetadata: LogMetadata): Logger {
    const childLogger = new Logger();
    const originalLog = childLogger.log.bind(childLogger);

    childLogger.log = (level: LogLevel, message: string, metadata?: LogMetadata) => {
      originalLog(level, message, { ...defaultMetadata, ...metadata });
    };

    return childLogger;
  }
}

// Export singleton instance
export const logger = new Logger();

// Export types
export type { LogLevel, LogMetadata };
