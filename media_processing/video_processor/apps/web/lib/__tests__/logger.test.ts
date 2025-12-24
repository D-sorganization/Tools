/**
 * Tests for Logger utility
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { logger } from '../logger';

describe('Logger', () => {
  let consoleDebugSpy: ReturnType<typeof vi.spyOn>;
  let consoleInfoSpy: ReturnType<typeof vi.spyOn>;
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    consoleInfoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Basic Logging', () => {
    it('should log debug messages', () => {
      logger.debug('Test debug message');

      expect(consoleDebugSpy).toHaveBeenCalled();
      const loggedMessage = consoleDebugSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('DEBUG');
      expect(loggedMessage).toContain('Test debug message');
    });

    it('should log info messages', () => {
      logger.info('Test info message');

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('INFO');
      expect(loggedMessage).toContain('Test info message');
    });

    it('should log warn messages', () => {
      logger.warn('Test warn message');

      expect(consoleWarnSpy).toHaveBeenCalled();
      const loggedMessage = consoleWarnSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('WARN');
      expect(loggedMessage).toContain('Test warn message');
    });

    it('should log error messages', () => {
      logger.error('Test error message');

      expect(consoleErrorSpy).toHaveBeenCalled();
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('ERROR');
      expect(loggedMessage).toContain('Test error message');
    });
  });

  describe('Message Formatting', () => {
    it('should include ISO timestamp in log message', () => {
      logger.info('Test message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      // Check for ISO timestamp pattern
      expect(loggedMessage).toMatch(/\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\]/);
    });

    it('should format message with level in uppercase', () => {
      logger.info('Test message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('INFO:');
    });

    it('should include message content', () => {
      logger.info('This is my test message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('This is my test message');
    });

    it('should format timestamp, level, and message in correct order', () => {
      logger.info('Test');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      // Should match pattern: [timestamp] LEVEL: message
      expect(loggedMessage).toMatch(/\[.+\] INFO: Test/);
    });
  });

  describe('Metadata Handling', () => {
    it('should log metadata as JSON', () => {
      const metadata = { userId: 123, action: 'login' };
      logger.info('User logged in', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"userId": 123');
      expect(loggedMessage).toContain('"action": "login"');
    });

    it('should format metadata with indentation', () => {
      const metadata = { key: 'value' };
      logger.info('Test', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      // JSON.stringify with 2 spaces should create indented output
      expect(loggedMessage).toContain('{\n');
    });

    it('should handle nested metadata objects', () => {
      const metadata = {
        user: {
          id: 123,
          name: 'John',
        },
        request: {
          method: 'GET',
          path: '/api/users',
        },
      };

      logger.info('API request', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"user"');
      expect(loggedMessage).toContain('"id": 123');
      expect(loggedMessage).toContain('"name": "John"');
      expect(loggedMessage).toContain('"method": "GET"');
    });

    it('should handle metadata with arrays', () => {
      const metadata = {
        tags: ['video', 'golf', 'analysis'],
      };

      logger.info('Tagged content', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"tags"');
      expect(loggedMessage).toContain('video');
      expect(loggedMessage).toContain('golf');
      expect(loggedMessage).toContain('analysis');
    });

    it('should handle empty metadata object', () => {
      logger.info('Test', {});

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Test');
      expect(loggedMessage).toContain('{}');
    });

    it('should not include metadata when not provided', () => {
      logger.info('Test without metadata');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Test without metadata');
      expect(loggedMessage).not.toContain('{');
    });

    it('should handle metadata with null values', () => {
      const metadata = { value: null };
      logger.info('Test', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"value": null');
    });

    it('should handle metadata with undefined values', () => {
      const metadata = { value: undefined };
      logger.info('Test', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      // undefined is not serialized in JSON
      expect(loggedMessage).toContain('Test');
    });

    it('should handle metadata with numbers', () => {
      const metadata = {
        count: 42,
        price: 19.99,
        negative: -5,
      };

      logger.info('Numbers', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"count": 42');
      expect(loggedMessage).toContain('"price": 19.99');
      expect(loggedMessage).toContain('"negative": -5');
    });

    it('should handle metadata with booleans', () => {
      const metadata = {
        isActive: true,
        isDeleted: false,
      };

      logger.info('Booleans', metadata);

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"isActive": true');
      expect(loggedMessage).toContain('"isDeleted": false');
    });
  });

  describe('Error Object Handling', () => {
    it('should extract error details from Error object in metadata', () => {
      const error = new Error('Test error');
      logger.error('An error occurred', { error });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Test error');
      expect(loggedMessage).toContain('"name": "Error"');
      expect(loggedMessage).toContain('"message": "Test error"');
    });

    it('should include stack trace from Error object', () => {
      const error = new Error('Test error');
      logger.error('An error occurred', { error });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"stack"');
    });

    it('should handle TypeError', () => {
      const error = new TypeError('Type mismatch');
      logger.error('Type error occurred', { error });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"name": "TypeError"');
      expect(loggedMessage).toContain('Type mismatch');
    });

    it('should handle RangeError', () => {
      const error = new RangeError('Out of range');
      logger.error('Range error occurred', { error });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"name": "RangeError"');
      expect(loggedMessage).toContain('Out of range');
    });

    it('should handle error with additional metadata', () => {
      const error = new Error('Test error');
      logger.error('Error with context', { error, userId: 123, action: 'upload' });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Test error');
      expect(loggedMessage).toContain('"userId": 123');
      expect(loggedMessage).toContain('"action": "upload"');
    });
  });

  describe('Argument Order Flexibility', () => {
    it('should support message first, metadata second', () => {
      logger.info('Message', { key: 'value' });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Message');
      expect(loggedMessage).toContain('"key": "value"');
    });

    it('should support metadata first, message second', () => {
      logger.info({ key: 'value' }, 'Message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Message');
      expect(loggedMessage).toContain('"key": "value"');
    });

    it('should work with debug level in both orders', () => {
      logger.debug('Debug message', { debug: true });
      logger.debug({ debug: false }, 'Another debug message');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(2);
    });

    it('should work with warn level in both orders', () => {
      logger.warn('Warning', { severity: 'high' });
      logger.warn({ severity: 'low' }, 'Another warning');

      expect(consoleWarnSpy).toHaveBeenCalledTimes(2);
    });

    it('should work with error level in both orders', () => {
      logger.error('Error occurred', { code: 500 });
      logger.error({ code: 404 }, 'Not found');

      expect(consoleErrorSpy).toHaveBeenCalledTimes(2);
    });
  });

  describe('Child Logger', () => {
    it('should create child logger with default metadata', () => {
      const childLogger = logger.child({ module: 'video', component: 'uploader' });
      childLogger.info('Test message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"module": "video"');
      expect(loggedMessage).toContain('"component": "uploader"');
      expect(loggedMessage).toContain('Test message');
    });

    it('should merge child metadata with call metadata', () => {
      const childLogger = logger.child({ module: 'video' });
      childLogger.info('Upload started', { fileName: 'test.mp4' });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"module": "video"');
      expect(loggedMessage).toContain('"fileName": "test.mp4"');
    });

    it('should override child metadata with call metadata', () => {
      const childLogger = logger.child({ priority: 'low' });
      childLogger.info('Important message', { priority: 'high' });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('"priority": "high"');
    });

    it('should support multiple child loggers independently', () => {
      const videoLogger = logger.child({ module: 'video' });
      const authLogger = logger.child({ module: 'auth' });

      videoLogger.info('Video event');
      authLogger.info('Auth event');

      const videoMessage = consoleInfoSpy.mock.calls[0][0];
      const authMessage = consoleInfoSpy.mock.calls[1][0];

      expect(videoMessage).toContain('"module": "video"');
      expect(authMessage).toContain('"module": "auth"');
    });

    it('should allow child logger to use all log levels', () => {
      const childLogger = logger.child({ component: 'test' });

      childLogger.debug('Debug');
      childLogger.info('Info');
      childLogger.warn('Warn');
      childLogger.error('Error');

      expect(consoleDebugSpy).toHaveBeenCalled();
      expect(consoleInfoSpy).toHaveBeenCalled();
      expect(consoleWarnSpy).toHaveBeenCalled();
      expect(consoleErrorSpy).toHaveBeenCalled();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty string message', () => {
      logger.info('');

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('INFO:');
    });

    it('should handle very long messages', () => {
      const longMessage = 'A'.repeat(10000);
      logger.info(longMessage);

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain(longMessage);
    });

    it('should handle special characters in message', () => {
      logger.info('Message with "quotes" and \'apostrophes\' and \n newlines');

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('quotes');
    });

    it('should handle unicode characters in message', () => {
      logger.info('测试消息 🎥 ⛳️');

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('测试消息');
    });

    it('should handle metadata with circular references gracefully', () => {
      const circular: any = { name: 'test' };
      circular.self = circular;

      // This might throw or handle gracefully
      expect(() => {
        logger.info('Circular reference', circular);
      }).toThrow(); // JSON.stringify throws on circular references
    });

    it('should handle very large metadata objects', () => {
      const largeMetadata: any = {};
      for (let i = 0; i < 1000; i++) {
        largeMetadata[`key${i}`] = `value${i}`;
      }

      logger.info('Large metadata', largeMetadata);

      expect(consoleInfoSpy).toHaveBeenCalled();
    });

    it('should handle metadata with Date objects', () => {
      const date = new Date('2024-01-01T12:00:00Z');
      logger.info('Date test', { timestamp: date });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('2024-01-01');
    });

    it('should handle metadata with RegExp objects', () => {
      const regex = /test/gi;
      logger.info('Regex test', { pattern: regex });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('pattern');
    });
  });

  describe('Different Log Levels', () => {
    it('should use console.debug for debug level', () => {
      logger.debug('Debug message');

      expect(consoleDebugSpy).toHaveBeenCalled();
      expect(consoleInfoSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should use console.info for info level', () => {
      logger.info('Info message');

      expect(consoleInfoSpy).toHaveBeenCalled();
      expect(consoleDebugSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should use console.warn for warn level', () => {
      logger.warn('Warning message');

      expect(consoleWarnSpy).toHaveBeenCalled();
      expect(consoleDebugSpy).not.toHaveBeenCalled();
      expect(consoleInfoSpy).not.toHaveBeenCalled();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should use console.error for error level', () => {
      logger.error('Error message');

      expect(consoleErrorSpy).toHaveBeenCalled();
      expect(consoleDebugSpy).not.toHaveBeenCalled();
      expect(consoleInfoSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });
  });

  describe('Multiple Calls', () => {
    it('should handle multiple sequential log calls', () => {
      logger.info('First');
      logger.info('Second');
      logger.info('Third');

      expect(consoleInfoSpy).toHaveBeenCalledTimes(3);
    });

    it('should handle rapid log calls', () => {
      for (let i = 0; i < 100; i++) {
        logger.info(`Message ${i}`);
      }

      expect(consoleInfoSpy).toHaveBeenCalledTimes(100);
    });

    it('should handle mixed level calls', () => {
      logger.debug('Debug');
      logger.info('Info');
      logger.warn('Warn');
      logger.error('Error');
      logger.info('Info again');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(1);
      expect(consoleInfoSpy).toHaveBeenCalledTimes(2);
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe('Real-world Scenarios', () => {
    it('should log video upload started', () => {
      logger.info('Video upload started', {
        fileName: 'golf-swing.mp4',
        fileSize: 1024 * 1024 * 50, // 50MB
        userId: 'user123',
      });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Video upload started');
      expect(loggedMessage).toContain('golf-swing.mp4');
      expect(loggedMessage).toContain('user123');
    });

    it('should log API error with details', () => {
      const error = new Error('Network timeout');
      logger.error('API request failed', {
        error,
        endpoint: '/api/videos',
        method: 'POST',
        statusCode: 500,
      });

      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('API request failed');
      expect(loggedMessage).toContain('Network timeout');
      expect(loggedMessage).toContain('/api/videos');
      expect(loggedMessage).toContain('POST');
    });

    it('should log user authentication event', () => {
      logger.info('User authenticated', {
        userId: 'user123',
        method: 'oauth',
        provider: 'google',
        timestamp: new Date().toISOString(),
      });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('User authenticated');
      expect(loggedMessage).toContain('oauth');
      expect(loggedMessage).toContain('google');
    });

    it('should log video processing completion', () => {
      logger.info('Video processing completed', {
        videoId: 'vid123',
        duration: 45.2,
        fps: 30,
        resolution: '1920x1080',
        processingTime: 12.5,
      });

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Video processing completed');
      expect(loggedMessage).toContain('vid123');
      expect(loggedMessage).toContain('1920x1080');
    });

    it('should log rate limit warning', () => {
      logger.warn('Rate limit approaching', {
        ip: '192.168.1.1',
        endpoint: '/api/upload',
        remainingRequests: 2,
        windowReset: new Date(Date.now() + 60000).toISOString(),
      });

      const loggedMessage = consoleWarnSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Rate limit approaching');
      expect(loggedMessage).toContain('192.168.1.1');
      expect(loggedMessage).toContain('"remainingRequests": 2');
    });
  });
});
