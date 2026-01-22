/**
 * Tests for sanitization functions
 */

import { describe, it, expect } from 'vitest';
import {
  sanitizeText,
  sanitizeFilename,
  sanitizeUrl,
  sanitizeHTML,
  escapeHTML,
  sanitizeJSON,
  sanitizeEmail,
  sanitizeNumber,
  sanitizeColor,
  sanitizeAnnotation,
} from '../sanitize';

describe('Sanitization Functions', () => {
  describe('sanitizeText', () => {
    it('should remove HTML tags', () => {
      expect(sanitizeText('<script>alert("xss")</script>')).toBe('');
      expect(sanitizeText('Hello <b>World</b>')).toBe('Hello World');
      expect(sanitizeText('<div>Test</div>')).toBe('Test');
    });

    it('should remove null bytes', () => {
      expect(sanitizeText('Hello\x00World')).toBe('HelloWorld');
    });

    it('should remove control characters', () => {
      expect(sanitizeText('Hello\x01\x02World')).toBe('HelloWorld');
    });

    it('should trim whitespace', () => {
      expect(sanitizeText('  Hello World  ')).toBe('Hello World');
      expect(sanitizeText('\nTest\n')).toBe('Test');
    });

    it('should return empty string for null/undefined', () => {
      expect(sanitizeText('')).toBe('');
      expect(sanitizeText(null as any)).toBe('');
      expect(sanitizeText(undefined as any)).toBe('');
    });

    it('should preserve safe text', () => {
      expect(sanitizeText('Hello World!')).toBe('Hello World!');
      expect(sanitizeText('Test 123')).toBe('Test 123');
    });
  });

  describe('sanitizeFilename', () => {
    it('should remove path separators', () => {
      expect(sanitizeFilename('../../../etc/passwd')).toBe('etcpasswd');
      expect(sanitizeFilename('..\\..\\windows\\system32')).toBe('windowssystem32');
    });

    it('should remove null bytes', () => {
      expect(sanitizeFilename('test\x00.txt')).toBe('test.txt');
    });

    it('should remove control characters', () => {
      expect(sanitizeFilename('test\x01\x02.txt')).toBe('test.txt');
    });

    it('should remove leading/trailing dots', () => {
      expect(sanitizeFilename('..test.txt')).toBe('test.txt');
      expect(sanitizeFilename('test.txt..')).toBe('test.txt');
    });

    it('should limit length to 255 characters', () => {
      const longName = 'a'.repeat(300) + '.txt';
      const result = sanitizeFilename(longName);
      expect(result.length).toBeLessThanOrEqual(255);
    });

    it('should preserve safe filenames', () => {
      expect(sanitizeFilename('test.txt')).toBe('test.txt');
      expect(sanitizeFilename('my-file_123.pdf')).toBe('my-file_123.pdf');
    });
  });

  describe('sanitizeUrl', () => {
    it('should allow safe protocols', () => {
      expect(sanitizeUrl('http://example.com')).toBe('http://example.com');
      expect(sanitizeUrl('https://example.com')).toBe('https://example.com');
      expect(sanitizeUrl('mailto:test@example.com')).toBe('mailto:test@example.com');
      expect(sanitizeUrl('tel:+1234567890')).toBe('tel:+1234567890');
    });

    it('should allow relative URLs', () => {
      expect(sanitizeUrl('/path/to/page')).toBe('/path/to/page');
      expect(sanitizeUrl('#anchor')).toBe('#anchor');
    });

    it('should reject dangerous protocols', () => {
      expect(sanitizeUrl('javascript:alert("xss")')).toBe('');
      expect(sanitizeUrl('data:text/html,<script>alert("xss")</script>')).toBe('');
      expect(sanitizeUrl('vbscript:msgbox("xss")')).toBe('');
    });

    it('should trim whitespace', () => {
      expect(sanitizeUrl('  http://example.com  ')).toBe('http://example.com');
    });

    it('should return empty string for empty input', () => {
      expect(sanitizeUrl('')).toBe('');
      expect(sanitizeUrl(null as any)).toBe('');
    });
  });

  describe('sanitizeHTML', () => {
    it('should strip all HTML by default', () => {
      expect(sanitizeHTML('<div>Test</div>')).toBe('Test');
      expect(sanitizeHTML('<script>alert("xss")</script>')).toBe('');
    });

    it('should remove dangerous tags', () => {
      expect(sanitizeHTML('<iframe src="evil"></iframe>')).toBe('');
      expect(sanitizeHTML('<object data="evil"></object>')).toBe('');
    });

    // Note: When DOMPurify is added, these tests should be updated
    it('should handle nested tags', () => {
      expect(sanitizeHTML('<div><span>Test</span></div>')).toBe('Test');
    });
  });

  describe('escapeHTML', () => {
    it('should escape special characters', () => {
      const result = escapeHTML('<script>alert("xss")</script>');
      expect(result).toContain('&lt;');
      expect(result).toContain('&gt;');
      expect(result).not.toContain('<script>');
    });

    it('should escape quotes', () => {
      const result = escapeHTML('"test"');
      expect(result).toContain('&quot;');
    });

    it('should escape ampersands', () => {
      const result = escapeHTML('A & B');
      expect(result).toContain('&amp;');
    });

    it('should preserve safe text', () => {
      expect(escapeHTML('Hello World')).toBe('Hello World');
    });
  });

  describe('sanitizeJSON', () => {
    it('should parse and re-stringify valid JSON', () => {
      const input = '{"name":"test","value":123}';
      const result = sanitizeJSON(input);
      expect(JSON.parse(result)).toEqual({ name: 'test', value: 123 });
    });

    it('should throw for invalid JSON', () => {
      expect(() => sanitizeJSON('not json')).toThrow('Invalid JSON string');
      expect(() => sanitizeJSON('{invalid}')).toThrow('Invalid JSON string');
    });

    it('should handle arrays', () => {
      const input = '[1,2,3]';
      const result = sanitizeJSON(input);
      expect(JSON.parse(result)).toEqual([1, 2, 3]);
    });

    it('should return empty string for empty input', () => {
      expect(sanitizeJSON('')).toBe('');
    });
  });

  describe('sanitizeEmail', () => {
    it('should accept valid email addresses', () => {
      expect(sanitizeEmail('test@example.com')).toBe('test@example.com');
      expect(sanitizeEmail('user.name@example.co.uk')).toBe('user.name@example.co.uk');
      expect(sanitizeEmail('user+tag@example.com')).toBe('user+tag@example.com');
    });

    it('should convert to lowercase', () => {
      expect(sanitizeEmail('TEST@EXAMPLE.COM')).toBe('test@example.com');
    });

    it('should trim whitespace', () => {
      expect(sanitizeEmail('  test@example.com  ')).toBe('test@example.com');
    });

    it('should throw for invalid emails', () => {
      expect(() => sanitizeEmail('invalid')).toThrow('Invalid email address');
      expect(() => sanitizeEmail('test@')).toThrow('Invalid email address');
      expect(() => sanitizeEmail('@example.com')).toThrow('Invalid email address');
      expect(() => sanitizeEmail('test @example.com')).toThrow('Invalid email address');
    });
  });

  describe('sanitizeNumber', () => {
    it('should parse string numbers', () => {
      expect(sanitizeNumber('42')).toBe(42);
      expect(sanitizeNumber('3.14')).toBe(3.14);
      expect(sanitizeNumber('-10')).toBe(-10);
    });

    it('should accept number values', () => {
      expect(sanitizeNumber(42)).toBe(42);
      expect(sanitizeNumber(3.14)).toBe(3.14);
    });

    it('should throw for invalid numbers', () => {
      expect(() => sanitizeNumber('not a number')).toThrow('Invalid number');
      expect(() => sanitizeNumber(NaN)).toThrow('Invalid number');
    });

    it('should enforce minimum value', () => {
      expect(sanitizeNumber(5, { min: 0 })).toBe(5);
      expect(sanitizeNumber(-5, { min: 0 })).toBe(0);
      expect(sanitizeNumber(-10, { min: -5 })).toBe(-5);
    });

    it('should enforce maximum value', () => {
      expect(sanitizeNumber(5, { max: 10 })).toBe(5);
      expect(sanitizeNumber(15, { max: 10 })).toBe(10);
    });

    it('should enforce min and max together', () => {
      expect(sanitizeNumber(5, { min: 0, max: 10 })).toBe(5);
      expect(sanitizeNumber(-5, { min: 0, max: 10 })).toBe(0);
      expect(sanitizeNumber(15, { min: 0, max: 10 })).toBe(10);
    });

    it('should round to specified decimals', () => {
      expect(sanitizeNumber(3.14159, { decimals: 2 })).toBe(3.14);
      expect(sanitizeNumber(3.999, { decimals: 0 })).toBe(4);
      expect(sanitizeNumber(2.5, { decimals: 0 })).toBe(2);
    });
  });

  describe('sanitizeColor', () => {
    it('should accept valid hex colors', () => {
      expect(sanitizeColor('#FF0000')).toBe('#FF0000');
      expect(sanitizeColor('#00ff00')).toBe('#00FF00'); // Uppercase
      expect(sanitizeColor('#123456')).toBe('#123456');
    });

    it('should expand short hex colors', () => {
      expect(sanitizeColor('#F00')).toBe('#FF0000');
      expect(sanitizeColor('#0F0')).toBe('#00FF00');
      expect(sanitizeColor('#ABC')).toBe('#AABBCC');
    });

    it('should accept RGB colors', () => {
      expect(sanitizeColor('rgb(255, 0, 0)')).toBe('rgb(255, 0, 0)');
      expect(sanitizeColor('rgba(255, 0, 0, 0.5)')).toBe('rgba(255, 0, 0, 0.5)');
    });

    it('should return default for invalid colors', () => {
      expect(sanitizeColor('invalid')).toBe('#000000');
      expect(sanitizeColor('#GGGGGG')).toBe('#000000');
      expect(sanitizeColor('#12345')).toBe('#000000'); // Wrong length
    });

    it('should handle empty input', () => {
      expect(sanitizeColor('')).toBe('#000000');
      expect(sanitizeColor(null as any)).toBe('#000000');
    });
  });

  describe('sanitizeAnnotation', () => {
    it('should preserve allowed properties', () => {
      const input = {
        type: 'line',
        left: 10,
        top: 20,
        stroke: '#FF0000',
        strokeWidth: 2,
      };

      const result = sanitizeAnnotation(input);

      expect(result).toHaveProperty('type', 'line');
      expect(result).toHaveProperty('left', 10);
      expect(result).toHaveProperty('top', 20);
      expect(result).toHaveProperty('stroke', '#FF0000');
      expect(result).toHaveProperty('strokeWidth', 2);
    });

    it('should remove disallowed properties', () => {
      const input = {
        type: 'line',
        maliciousProperty: 'evil',
        __proto__: { polluted: true },
      };

      const result = sanitizeAnnotation(input);

      expect(result).toHaveProperty('type', 'line');
      expect(result).not.toHaveProperty('maliciousProperty');
      expect(result).not.toHaveProperty('__proto__');
    });

    it('should sanitize text content', () => {
      const input = {
        type: 'text',
        text: '<script>alert("xss")</script>',
      };

      const result = sanitizeAnnotation(input);

      expect(result.text).not.toContain('<script>');
      expect(result.text).not.toContain('alert');
    });

    it('should sanitize color values', () => {
      const input = {
        fill: 'javascript:alert("xss")',
        stroke: '#FF0000',
      };

      const result = sanitizeAnnotation(input);

      expect(result.fill).toBe('#000000'); // Default color for invalid
      expect(result.stroke).toBe('#FF0000');
    });

    it('should handle NaN in numeric properties', () => {
      const input = {
        left: NaN,
        top: 10,
        width: NaN,
      };

      const result = sanitizeAnnotation(input);

      expect(result.left).toBe(0); // NaN replaced with 0
      expect(result.top).toBe(10);
      expect(result.width).toBe(0);
    });

    it('should return input for non-object values', () => {
      expect(sanitizeAnnotation(null)).toBe(null);
      expect(sanitizeAnnotation('string')).toBe('string');
      expect(sanitizeAnnotation(123)).toBe(123);
    });
  });
});
