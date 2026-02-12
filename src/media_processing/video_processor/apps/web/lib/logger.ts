/**
 * Application Logger
 *
 * Structured logging with support for different log levels and metadata.
 * Works in both server and client environments.
 * Uses pino for high-performance structured logging.
 */

import pino from 'pino';

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

/**
 * Create a pino logger instance configured for the current environment
 */
function createPinoLogger(): pino.Logger {
  const level = getLogLevel();
  const isClient = typeof window !== 'undefined';

  if (isClient) {
    // Browser: use pino browser mode
    return pino({
      level,
      browser: { asObject: true },
    });
  }

  // Server: use pino with ISO timestamps
  return pino({
    level,
    timestamp: pino.stdTimeFunctions.isoTime,
  });
}

const pinoInstance = createPinoLogger();

class Logger {
  private pino: pino.Logger;

  constructor(pinoLogger?: pino.Logger) {
    this.pino = pinoLogger ?? pinoInstance;
  }

  /**
   * Log debug message (lowest priority)
   * Use for detailed debugging information
   */
  debug(message: string, metadata?: LogMetadata): void;
  debug(metadata: LogMetadata, message: string): void;
  debug(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    if (metadata) {
      this.pino.debug(metadata, message);
    } else {
      this.pino.debug(message);
    }
  }

  /**
   * Log info message
   * Use for general informational messages
   */
  info(message: string, metadata?: LogMetadata): void;
  info(metadata: LogMetadata, message: string): void;
  info(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    if (metadata) {
      this.pino.info(metadata, message);
    } else {
      this.pino.info(message);
    }
  }

  /**
   * Log warning message
   * Use for warning messages that don't prevent operation
   */
  warn(message: string, metadata?: LogMetadata): void;
  warn(metadata: LogMetadata, message: string): void;
  warn(messageOrMetadata: string | LogMetadata, metadataOrMessage?: LogMetadata | string): void {
    const [message, metadata] = this.parseArgs(messageOrMetadata, metadataOrMessage);
    if (metadata) {
      this.pino.warn(metadata, message);
    } else {
      this.pino.warn(message);
    }
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

    if (metadata) {
      this.pino.error(metadata, message);
    } else {
      this.pino.error(message);
    }

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
        this.pino.error('Failed to send error to Sentry');
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
    const childPino = this.pino.child(defaultMetadata as pino.Bindings);
    return new Logger(childPino);
  }
}

// Export singleton instance
export const logger = new Logger();

// Export types
export type { LogLevel, LogMetadata };
