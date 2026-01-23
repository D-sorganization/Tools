/**
 * Tests for video validation
 */

import { describe, it, expect } from 'vitest';
import {
  quickValidateVideoFile,
  isFileSizeValid,
  isFileTypeAllowed,
  formatFileSize,
  getMaxFileSize,
  MAX_FILE_SIZE,
  ALLOWED_MIME_TYPES,
} from '../video';
import { createMockFile } from '@/test/utils';
import { ValidationError } from '@/lib/errors';

describe('Video Validation', () => {
  describe('quickValidateVideoFile', () => {
    it('should validate valid video file', () => {
      const file = createMockFile('test.mp4', 1024 * 1024, 'video/mp4');

      expect(() => {
        quickValidateVideoFile(file);
      }).not.toThrow();
    });

    it('should reject file with no name', () => {
      const file = createMockFile('', 1024, 'video/mp4');

      expect(() => {
        quickValidateVideoFile(file);
      }).toThrow(ValidationError);
    });

    it('should reject file that is too large', () => {
      const file = createMockFile('test.mp4', MAX_FILE_SIZE + 1, 'video/mp4');

      expect(() => {
        quickValidateVideoFile(file);
      }).toThrow(ValidationError);
    });

    it('should reject file with invalid type', () => {
      const file = createMockFile('test.txt', 1024, 'text/plain');

      expect(() => {
        quickValidateVideoFile(file);
      }).toThrow(ValidationError);
    });

    it('should reject file with zero size', () => {
      const file = createMockFile('test.mp4', 0, 'video/mp4');

      expect(() => {
        quickValidateVideoFile(file);
      }).toThrow(ValidationError);
    });

    it('should accept all allowed MIME types', () => {
      ALLOWED_MIME_TYPES.forEach((mimeType) => {
        const file = createMockFile('test.video', 1024, mimeType);

        expect(() => {
          quickValidateVideoFile(file);
        }).not.toThrow();
      });
    });

    it('should validate file name length', () => {
      const longName = 'a'.repeat(256) + '.mp4';
      const file = createMockFile(longName, 1024, 'video/mp4');

      expect(() => {
        quickValidateVideoFile(file);
      }).toThrow(ValidationError);
    });
  });

  describe('isFileSizeValid', () => {
    it('should return true for valid file sizes', () => {
      expect(isFileSizeValid(createMockFile('test.mp4', 1024))).toBe(true);
      expect(isFileSizeValid(createMockFile('test.mp4', 1024 * 1024))).toBe(true);
      expect(isFileSizeValid(createMockFile('test.mp4', MAX_FILE_SIZE))).toBe(true);
    });

    it('should return false for zero size', () => {
      expect(isFileSizeValid(createMockFile('test.mp4', 0))).toBe(false);
    });

    it('should return false for negative size', () => {
      const file = createMockFile('test.mp4', -1);
      expect(isFileSizeValid(file)).toBe(false);
    });

    it('should return false for oversized files', () => {
      expect(isFileSizeValid(createMockFile('test.mp4', MAX_FILE_SIZE + 1))).toBe(
        false
      );
    });
  });

  describe('isFileTypeAllowed', () => {
    it('should return true for allowed MIME types', () => {
      expect(isFileTypeAllowed(createMockFile('test.mp4', 1024, 'video/mp4'))).toBe(
        true
      );
      expect(isFileTypeAllowed(createMockFile('test.webm', 1024, 'video/webm'))).toBe(
        true
      );
      expect(isFileTypeAllowed(createMockFile('test.ogg', 1024, 'video/ogg'))).toBe(
        true
      );
    });

    it('should return false for disallowed MIME types', () => {
      expect(isFileTypeAllowed(createMockFile('test.txt', 1024, 'text/plain'))).toBe(
        false
      );
      expect(
        isFileTypeAllowed(createMockFile('test.jpg', 1024, 'image/jpeg'))
      ).toBe(false);
      expect(
        isFileTypeAllowed(createMockFile('test.pdf', 1024, 'application/pdf'))
      ).toBe(false);
    });

    it('should be case-sensitive', () => {
      // MIME types should be lowercase
      const file = createMockFile('test.mp4', 1024, 'VIDEO/MP4' as any);
      // Force type to be uppercase (File constructor normalizes to lowercase)
      Object.defineProperty(file, 'type', { value: 'VIDEO/MP4' });
      expect(isFileTypeAllowed(file)).toBe(false);
    });
  });

  describe('formatFileSize', () => {
    it('should format bytes correctly', () => {
      expect(formatFileSize(0)).toBe('0.00 B');
      expect(formatFileSize(100)).toBe('100.00 B');
      expect(formatFileSize(1023)).toBe('1023.00 B');
    });

    it('should format kilobytes correctly', () => {
      expect(formatFileSize(1024)).toBe('1.00 KB');
      expect(formatFileSize(1536)).toBe('1.50 KB');
      expect(formatFileSize(10240)).toBe('10.00 KB');
    });

    it('should format megabytes correctly', () => {
      expect(formatFileSize(1024 * 1024)).toBe('1.00 MB');
      expect(formatFileSize(1.5 * 1024 * 1024)).toBe('1.50 MB');
      expect(formatFileSize(100 * 1024 * 1024)).toBe('100.00 MB');
    });

    it('should format gigabytes correctly', () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe('1.00 GB');
      expect(formatFileSize(2.5 * 1024 * 1024 * 1024)).toBe('2.50 GB');
    });

    it('should round to 2 decimal places', () => {
      expect(formatFileSize(1234)).toBe('1.21 KB');
      expect(formatFileSize(1234567)).toBe('1.18 MB');
    });
  });

  describe('getMaxFileSize', () => {
    it('should return formatted max file size', () => {
      const maxSize = getMaxFileSize();

      expect(maxSize).toContain('MB');
      expect(maxSize).toContain('500');
    });

    it('should be human-readable', () => {
      const maxSize = getMaxFileSize();

      // Should not contain raw bytes
      expect(maxSize).not.toContain('524288000');
    });
  });

  describe('MAX_FILE_SIZE constant', () => {
    it('should be 500MB in bytes', () => {
      expect(MAX_FILE_SIZE).toBe(500 * 1024 * 1024);
    });
  });

  describe('ALLOWED_MIME_TYPES constant', () => {
    it('should include common video formats', () => {
      expect(ALLOWED_MIME_TYPES).toContain('video/mp4');
      expect(ALLOWED_MIME_TYPES).toContain('video/webm');
      expect(ALLOWED_MIME_TYPES).toContain('video/ogg');
    });

    it('should include QuickTime format', () => {
      expect(ALLOWED_MIME_TYPES).toContain('video/quicktime');
    });

    it('should include AVI format', () => {
      expect(ALLOWED_MIME_TYPES).toContain('video/x-msvideo');
    });

    it('should include MKV format', () => {
      expect(ALLOWED_MIME_TYPES).toContain('video/x-matroska');
    });

    it('should have at least 6 formats', () => {
      expect(ALLOWED_MIME_TYPES.length).toBeGreaterThanOrEqual(6);
    });
  });
});

// Note: Full validation tests for magic byte detection and playability
// would require more complex mocking and are better suited for E2E tests
describe('Video Validation Integration', () => {
  it('should provide consistent validation between quick and full validation', () => {
    const validFile = createMockFile('test.mp4', 1024 * 1024, 'video/mp4');

    // Quick validation should not throw
    expect(() => quickValidateVideoFile(validFile)).not.toThrow();

    // File should pass individual checks
    expect(isFileSizeValid(validFile)).toBe(true);
    expect(isFileTypeAllowed(validFile)).toBe(true);
  });

  it('should reject consistently across all validation methods', () => {
    const invalidFile = createMockFile('test.txt', 1024, 'text/plain');

    // Quick validation should throw
    expect(() => quickValidateVideoFile(invalidFile)).toThrow();

    // Individual checks should fail
    expect(isFileTypeAllowed(invalidFile)).toBe(false);
  });
});
