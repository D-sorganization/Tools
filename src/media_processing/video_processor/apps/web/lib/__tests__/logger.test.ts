/**
 * Tests for Logger utility (pino-based)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('pino', () => {
  const mockDebug = vi.fn();
  const mockInfo = vi.fn();
  const mockWarn = vi.fn();
  const mockError = vi.fn();
  const mockChild = vi.fn();

  const pinoFn = () => ({
    debug: mockDebug,
    info: mockInfo,
    warn: mockWarn,
    error: mockError,
    child: mockChild.mockReturnValue({
      debug: mockDebug,
      info: mockInfo,
      warn: mockWarn,
      error: mockError,
      child: mockChild,
    }),
  });

  pinoFn.stdTimeFunctions = {
    isoTime: () => `,"time":"${new Date().toISOString()}"`,
  };

  return { default: pinoFn };
});

const mockDebug = vi.mocked((await import('pino')).default().debug);
const mockInfo = vi.mocked((await import('pino')).default().info);
const mockWarn = vi.mocked((await import('pino')).default().warn);
const mockError = vi.mocked((await import('pino')).default().error);
const mockChild = vi.mocked((await import('pino')).default().child);

// Import after mock setup
import { logger } from '../logger';

describe('Logger', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Logging', () => {
    it('should log debug messages via pino', () => {
      logger.debug('Test debug message');
      expect(mockDebug).toHaveBeenCalledWith('Test debug message');
    });

    it('should log info messages via pino', () => {
      logger.info('Test info message');
      expect(mockInfo).toHaveBeenCalledWith('Test info message');
    });

    it('should log warn messages via pino', () => {
      logger.warn('Test warn message');
      expect(mockWarn).toHaveBeenCalledWith('Test warn message');
    });

    it('should log error messages via pino', () => {
      logger.error('Test error message');
      expect(mockError).toHaveBeenCalledWith('Test error message');
    });
  });

  describe('Metadata Handling', () => {
    it('should log metadata as first arg and message as second (pino convention)', () => {
      const metadata = { userId: 123, action: 'login' };
      logger.info('User logged in', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'User logged in');
    });

    it('should handle nested metadata objects', () => {
      const metadata = {
        user: { id: 123, name: 'John' },
        request: { method: 'GET', path: '/api/users' },
      };
      logger.info('API request', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'API request');
    });

    it('should handle metadata with arrays', () => {
      const metadata = { tags: ['video', 'golf', 'analysis'] };
      logger.info('Tagged content', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'Tagged content');
    });

    it('should handle empty metadata object', () => {
      logger.info('Test', {});
      expect(mockInfo).toHaveBeenCalledWith({}, 'Test');
    });

    it('should not include metadata when not provided', () => {
      logger.info('Test without metadata');
      expect(mockInfo).toHaveBeenCalledWith('Test without metadata');
    });

    it('should handle metadata with null values', () => {
      const metadata = { value: null };
      logger.info('Test', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'Test');
    });

    it('should handle metadata with numbers', () => {
      const metadata = { count: 42, price: 19.99, negative: -5 };
      logger.info('Numbers', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'Numbers');
    });

    it('should handle metadata with booleans', () => {
      const metadata = { isActive: true, isDeleted: false };
      logger.info('Booleans', metadata);
      expect(mockInfo).toHaveBeenCalledWith(metadata, 'Booleans');
    });
  });

  describe('Error Object Handling', () => {
    it('should extract error details from Error object in metadata', () => {
      const error = new Error('Test error');
      logger.error('An error occurred', { error });

      expect(mockError).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            name: 'Error',
            message: 'Test error',
            stack: expect.any(String),
          }),
        }),
        'An error occurred'
      );
    });

    it('should handle TypeError', () => {
      const error = new TypeError('Type mismatch');
      logger.error('Type error occurred', { error });

      expect(mockError).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            name: 'TypeError',
            message: 'Type mismatch',
          }),
        }),
        'Type error occurred'
      );
    });

    it('should handle RangeError', () => {
      const error = new RangeError('Out of range');
      logger.error('Range error occurred', { error });

      expect(mockError).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            name: 'RangeError',
            message: 'Out of range',
          }),
        }),
        'Range error occurred'
      );
    });

    it('should handle error with additional metadata', () => {
      const error = new Error('Test error');
      logger.error('Error with context', { error, userId: 123, action: 'upload' });

      expect(mockError).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: 123,
          action: 'upload',
        }),
        'Error with context'
      );
    });
  });

  describe('Argument Order Flexibility', () => {
    it('should support message first, metadata second', () => {
      logger.info('Message', { key: 'value' });
      expect(mockInfo).toHaveBeenCalledWith({ key: 'value' }, 'Message');
    });

    it('should support metadata first, message second', () => {
      logger.info({ key: 'value' }, 'Message');
      expect(mockInfo).toHaveBeenCalledWith({ key: 'value' }, 'Message');
    });

    it('should work with debug level in both orders', () => {
      logger.debug('Debug message', { debug: true });
      logger.debug({ debug: false }, 'Another debug message');
      expect(mockDebug).toHaveBeenCalledTimes(2);
    });

    it('should work with warn level in both orders', () => {
      logger.warn('Warning', { severity: 'high' });
      logger.warn({ severity: 'low' }, 'Another warning');
      expect(mockWarn).toHaveBeenCalledTimes(2);
    });

    it('should work with error level in both orders', () => {
      logger.error('Error occurred', { code: 500 });
      logger.error({ code: 404 }, 'Not found');
      expect(mockError).toHaveBeenCalledTimes(2);
    });
  });

  describe('Child Logger', () => {
    it('should create child logger via pino.child()', () => {
      const childLogger = logger.child({ module: 'video', component: 'uploader' });
      expect(mockChild).toHaveBeenCalledWith({ module: 'video', component: 'uploader' });

      childLogger.info('Test message');
      expect(mockInfo).toHaveBeenCalledWith('Test message');
    });

    it('should pass metadata to child logger calls', () => {
      const childLogger = logger.child({ module: 'video' });
      childLogger.info('Upload started', { fileName: 'test.mp4' });
      expect(mockInfo).toHaveBeenCalledWith({ fileName: 'test.mp4' }, 'Upload started');
    });

    it('should support multiple child loggers independently', () => {
      logger.child({ module: 'video' });
      logger.child({ module: 'auth' });
      expect(mockChild).toHaveBeenCalledTimes(2);
      expect(mockChild).toHaveBeenCalledWith({ module: 'video' });
      expect(mockChild).toHaveBeenCalledWith({ module: 'auth' });
    });

    it('should allow child logger to use all log levels', () => {
      const childLogger = logger.child({ component: 'test' });
      childLogger.debug('Debug');
      childLogger.info('Info');
      childLogger.warn('Warn');
      childLogger.error('Error');

      expect(mockDebug).toHaveBeenCalled();
      expect(mockInfo).toHaveBeenCalled();
      expect(mockWarn).toHaveBeenCalled();
      expect(mockError).toHaveBeenCalled();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty string message', () => {
      logger.info('');
      expect(mockInfo).toHaveBeenCalledWith('');
    });

    it('should handle very long messages', () => {
      const longMessage = 'A'.repeat(10000);
      logger.info(longMessage);
      expect(mockInfo).toHaveBeenCalledWith(longMessage);
    });

    it('should handle special characters in message', () => {
      logger.info('Message with "quotes" and \'apostrophes\' and \n newlines');
      expect(mockInfo).toHaveBeenCalled();
    });

    it('should handle unicode characters in message', () => {
      logger.info('测试消息');
      expect(mockInfo).toHaveBeenCalledWith('测试消息');
    });

    it('should handle metadata with Date objects', () => {
      const date = new Date('2024-01-01T12:00:00Z');
      logger.info('Date test', { timestamp: date });
      expect(mockInfo).toHaveBeenCalledWith({ timestamp: date }, 'Date test');
    });

    it('should handle metadata with RegExp objects', () => {
      const regex = /test/gi;
      logger.info('Regex test', { pattern: regex });
      expect(mockInfo).toHaveBeenCalledWith({ pattern: regex }, 'Regex test');
    });
  });

  describe('Different Log Levels', () => {
    it('should use pino.debug for debug level', () => {
      logger.debug('Debug message');
      expect(mockDebug).toHaveBeenCalled();
      expect(mockInfo).not.toHaveBeenCalled();
      expect(mockWarn).not.toHaveBeenCalled();
      expect(mockError).not.toHaveBeenCalled();
    });

    it('should use pino.info for info level', () => {
      logger.info('Info message');
      expect(mockInfo).toHaveBeenCalled();
      expect(mockDebug).not.toHaveBeenCalled();
      expect(mockWarn).not.toHaveBeenCalled();
      expect(mockError).not.toHaveBeenCalled();
    });

    it('should use pino.warn for warn level', () => {
      logger.warn('Warning message');
      expect(mockWarn).toHaveBeenCalled();
      expect(mockDebug).not.toHaveBeenCalled();
      expect(mockInfo).not.toHaveBeenCalled();
      expect(mockError).not.toHaveBeenCalled();
    });

    it('should use pino.error for error level', () => {
      logger.error('Error message');
      expect(mockError).toHaveBeenCalled();
      expect(mockDebug).not.toHaveBeenCalled();
      expect(mockInfo).not.toHaveBeenCalled();
      expect(mockWarn).not.toHaveBeenCalled();
    });
  });

  describe('Multiple Calls', () => {
    it('should handle multiple sequential log calls', () => {
      logger.info('First');
      logger.info('Second');
      logger.info('Third');
      expect(mockInfo).toHaveBeenCalledTimes(3);
    });

    it('should handle rapid log calls', () => {
      for (let i = 0; i < 100; i++) {
        logger.info(`Message ${i}`);
      }
      expect(mockInfo).toHaveBeenCalledTimes(100);
    });

    it('should handle mixed level calls', () => {
      logger.debug('Debug');
      logger.info('Info');
      logger.warn('Warn');
      logger.error('Error');
      logger.info('Info again');

      expect(mockDebug).toHaveBeenCalledTimes(1);
      expect(mockInfo).toHaveBeenCalledTimes(2);
      expect(mockWarn).toHaveBeenCalledTimes(1);
      expect(mockError).toHaveBeenCalledTimes(1);
    });
  });

  describe('Real-world Scenarios', () => {
    it('should log video upload started', () => {
      logger.info('Video upload started', {
        fileName: 'golf-swing.mp4',
        fileSize: 1024 * 1024 * 50,
        userId: 'user123',
      });

      expect(mockInfo).toHaveBeenCalledWith(
        { fileName: 'golf-swing.mp4', fileSize: 52428800, userId: 'user123' },
        'Video upload started'
      );
    });

    it('should log API error with details', () => {
      const error = new Error('Network timeout');
      logger.error('API request failed', {
        error,
        endpoint: '/api/videos',
        method: 'POST',
        statusCode: 500,
      });

      expect(mockError).toHaveBeenCalledWith(
        expect.objectContaining({
          endpoint: '/api/videos',
          method: 'POST',
          statusCode: 500,
        }),
        'API request failed'
      );
    });

    it('should log user authentication event', () => {
      const timestamp = new Date().toISOString();
      logger.info('User authenticated', {
        userId: 'user123',
        method: 'oauth',
        provider: 'google',
        timestamp,
      });

      expect(mockInfo).toHaveBeenCalledWith(
        { userId: 'user123', method: 'oauth', provider: 'google', timestamp },
        'User authenticated'
      );
    });

    it('should log video processing completion', () => {
      logger.info('Video processing completed', {
        videoId: 'vid123',
        duration: 45.2,
        fps: 30,
        resolution: '1920x1080',
        processingTime: 12.5,
      });

      expect(mockInfo).toHaveBeenCalledWith(
        expect.objectContaining({
          videoId: 'vid123',
          resolution: '1920x1080',
        }),
        'Video processing completed'
      );
    });

    it('should log rate limit warning', () => {
      logger.warn('Rate limit approaching', {
        ip: '192.168.1.1',
        endpoint: '/api/upload',
        remainingRequests: 2,
        windowReset: new Date(Date.now() + 60000).toISOString(),
      });

      expect(mockWarn).toHaveBeenCalledWith(
        expect.objectContaining({
          ip: '192.168.1.1',
          remainingRequests: 2,
        }),
        'Rate limit approaching'
      );
    });
  });
});
