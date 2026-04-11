/**
 * Tests for error classes and utilities
 */

import { describe, it, expect } from 'vitest';
import {
  AppError,
  ValidationError,
  VideoProcessingError,
  AudioRecordingError,
  isAppError,
  getUserMessage,
  assert,
  assertDefined,
  assertNumber,
  assertString,
} from '../errors';

describe('Error Classes', () => {
  describe('AppError', () => {
    it('should create error with message and code', () => {
      const error = new AppError('Test error', 'TEST_ERROR');

      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_ERROR');
      expect(error.statusCode).toBe(500);
      expect(error.name).toBe('AppError');
    });

    it('should accept custom status code', () => {
      const error = new AppError('Test error', 'TEST_ERROR', 400);

      expect(error.statusCode).toBe(400);
    });

    it('should accept metadata', () => {
      const metadata = { userId: '123', action: 'upload' };
      const error = new AppError('Test error', 'TEST_ERROR', 500, metadata);

      expect(error.metadata).toEqual(metadata);
    });

    it('should serialize to JSON', () => {
      const error = new AppError('Test error', 'TEST_ERROR', 400, { key: 'value' });
      const json = error.toJSON();

      expect(json).toMatchObject({
        name: 'AppError',
        message: 'Test error',
        code: 'TEST_ERROR',
        statusCode: 400,
        metadata: { key: 'value' },
      });
      expect(json.stack).toBeDefined();
    });

    it('should have proper prototype chain', () => {
      const error = new AppError('Test error', 'TEST_ERROR');

      expect(error instanceof Error).toBe(true);
      expect(error instanceof AppError).toBe(true);
    });
  });

  describe('ValidationError', () => {
    it('should create validation error with 400 status', () => {
      const error = new ValidationError('Invalid input');

      expect(error.message).toBe('Invalid input');
      expect(error.code).toBe('VALIDATION_ERROR');
      expect(error.statusCode).toBe(400);
      expect(error instanceof AppError).toBe(true);
    });

    it('should return user-friendly message', () => {
      const error = new ValidationError('Invalid input');

      expect(error.getUserMessage()).toBe('Invalid input');
    });
  });

  describe('VideoProcessingError', () => {
    it('should create video processing error with 500 status', () => {
      const error = new VideoProcessingError('Processing failed');

      expect(error.message).toBe('Processing failed');
      expect(error.code).toBe('VIDEO_PROCESSING_ERROR');
      expect(error.statusCode).toBe(500);
    });

    it('should return generic user message', () => {
      const error = new VideoProcessingError('Internal processing error');
      const userMessage = error.getUserMessage();

      expect(userMessage).not.toContain('Internal');
      expect(userMessage).toContain('video');
    });
  });

  describe('AudioRecordingError', () => {
    it('should create audio recording error', () => {
      const error = new AudioRecordingError('Recording failed');

      expect(error.code).toBe('AUDIO_RECORDING_ERROR');
      expect(error.statusCode).toBe(500);
    });

    it('should mention microphone in user message', () => {
      const error = new AudioRecordingError('Recording failed');
      const userMessage = error.getUserMessage();

      expect(userMessage.toLowerCase()).toContain('microphone');
    });
  });
});

describe('Error Utilities', () => {
  describe('isAppError', () => {
    it('should return true for AppError instances', () => {
      const error = new AppError('Test', 'TEST');
      expect(isAppError(error)).toBe(true);
    });

    it('should return true for ValidationError instances', () => {
      const error = new ValidationError('Test');
      expect(isAppError(error)).toBe(true);
    });

    it('should return false for regular Error', () => {
      const error = new Error('Test');
      expect(isAppError(error)).toBe(false);
    });

    it('should return false for non-error objects', () => {
      expect(isAppError('string')).toBe(false);
      expect(isAppError(null)).toBe(false);
      expect(isAppError(undefined)).toBe(false);
      expect(isAppError({})).toBe(false);
    });
  });

  describe('getUserMessage', () => {
    it('should return user message from AppError', () => {
      const error = new ValidationError('Invalid email address');
      expect(getUserMessage(error)).toBe('Invalid email address');
    });

    it('should return generic message for regular Error', () => {
      const error = new Error('Internal server error');
      const message = getUserMessage(error);

      expect(message).not.toContain('Internal');
      expect(message).toContain('unexpected');
    });

    it('should return generic message for unknown errors', () => {
      const message = getUserMessage('string error');

      expect(message).toContain('unknown');
    });

    it('should not expose internal error details', () => {
      const error = new VideoProcessingError('FFmpeg crash: segfault at 0x123456');
      const message = getUserMessage(error);

      expect(message).not.toContain('segfault');
      expect(message).not.toContain('0x123456');
    });
  });
});

describe('Assert Functions', () => {
  describe('assert', () => {
    it('should not throw when condition is true', () => {
      expect(() => {
        assert(true, 'Should not throw');
      }).not.toThrow();

      expect(() => {
        assert(1 === 1, 'Should not throw');
      }).not.toThrow();
    });

    it('should throw ValidationError when condition is false', () => {
      expect(() => {
        assert(false, 'Condition failed');
      }).toThrow(ValidationError);

      expect(() => {
        assert(false, 'Condition failed');
      }).toThrow('Condition failed');
    });

    it('should include metadata in thrown error', () => {
      try {
        assert(false, 'Failed', { key: 'value' });
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError);
        expect((error as ValidationError).metadata).toEqual({ key: 'value' });
      }
    });

    it('should handle falsy values correctly', () => {
      expect(() => assert(0, 'Zero is falsy')).toThrow();
      expect(() => assert('', 'Empty string is falsy')).toThrow();
      expect(() => assert(null, 'Null is falsy')).toThrow();
      expect(() => assert(undefined, 'Undefined is falsy')).toThrow();
    });
  });

  describe('assertDefined', () => {
    it('should not throw for defined values', () => {
      expect(() => assertDefined(0, 'number')).not.toThrow();
      expect(() => assertDefined('', 'string')).not.toThrow();
      expect(() => assertDefined(false, 'boolean')).not.toThrow();
      expect(() => assertDefined({}, 'object')).not.toThrow();
    });

    it('should throw for null', () => {
      expect(() => assertDefined(null, 'value')).toThrow(ValidationError);
      expect(() => assertDefined(null, 'value')).toThrow('value is required');
    });

    it('should throw for undefined', () => {
      expect(() => assertDefined(undefined, 'value')).toThrow(ValidationError);
      expect(() => assertDefined(undefined, 'value')).toThrow('value is required');
    });
  });

  describe('assertNumber', () => {
    it('should not throw for valid numbers', () => {
      expect(() => assertNumber(0, 'number')).not.toThrow();
      expect(() => assertNumber(42, 'number')).not.toThrow();
      expect(() => assertNumber(-10, 'number')).not.toThrow();
      expect(() => assertNumber(3.14, 'number')).not.toThrow();
    });

    it('should throw for non-numbers', () => {
      expect(() => assertNumber('42', 'number')).toThrow(ValidationError);
      expect(() => assertNumber(null as any, 'number')).toThrow();
      expect(() => assertNumber(undefined as any, 'number')).toThrow();
      expect(() => assertNumber({} as any, 'number')).toThrow();
    });

    it('should throw for NaN', () => {
      expect(() => assertNumber(NaN, 'number')).toThrow(ValidationError);
      expect(() => assertNumber(NaN, 'number')).toThrow('must be a valid number');
    });

    it('should enforce minimum value', () => {
      expect(() => assertNumber(5, 'number', { min: 0 })).not.toThrow();
      expect(() => assertNumber(5, 'number', { min: 10 })).toThrow(ValidationError);
      expect(() => assertNumber(5, 'number', { min: 10 })).toThrow('must be at least 10');
    });

    it('should enforce maximum value', () => {
      expect(() => assertNumber(5, 'number', { max: 10 })).not.toThrow();
      expect(() => assertNumber(5, 'number', { max: 3 })).toThrow(ValidationError);
      expect(() => assertNumber(5, 'number', { max: 3 })).toThrow('must be at most 3');
    });

    it('should enforce range', () => {
      expect(() => assertNumber(5, 'number', { min: 0, max: 10 })).not.toThrow();
      expect(() => assertNumber(-1, 'number', { min: 0, max: 10 })).toThrow();
      expect(() => assertNumber(11, 'number', { min: 0, max: 10 })).toThrow();
    });
  });

  describe('assertString', () => {
    it('should not throw for valid strings', () => {
      expect(() => assertString('hello', 'text')).not.toThrow();
      expect(() => assertString('', 'text')).not.toThrow();
    });

    it('should throw for non-strings', () => {
      expect(() => assertString(42 as any, 'text')).toThrow(ValidationError);
      expect(() => assertString(null as any, 'text')).toThrow();
      expect(() => assertString(undefined as any, 'text')).toThrow();
      expect(() => assertString({} as any, 'text')).toThrow();
    });

    it('should enforce minimum length', () => {
      expect(() => assertString('hello', 'text', { minLength: 3 })).not.toThrow();
      expect(() => assertString('hi', 'text', { minLength: 3 })).toThrow(ValidationError);
      expect(() => assertString('hi', 'text', { minLength: 3 })).toThrow(
        'must be at least 3 characters'
      );
    });

    it('should enforce maximum length', () => {
      expect(() => assertString('hi', 'text', { maxLength: 5 })).not.toThrow();
      expect(() => assertString('hello world', 'text', { maxLength: 5 })).toThrow(
        ValidationError
      );
      expect(() => assertString('hello world', 'text', { maxLength: 5 })).toThrow(
        'must be at most 5 characters'
      );
    });

    it('should enforce length range', () => {
      expect(() => assertString('hello', 'text', { minLength: 3, maxLength: 10 })).not.toThrow();
      expect(() => assertString('hi', 'text', { minLength: 3, maxLength: 10 })).toThrow();
      expect(() => assertString('hello world!!', 'text', { minLength: 3, maxLength: 10 })).toThrow();
    });
  });
});
