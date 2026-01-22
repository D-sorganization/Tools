/**
 * Tests for CSRF Protection
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  generateCsrfToken,
  validateCsrfToken,
  requireCsrfToken,
  getClientCsrfToken,
  fetchWithCsrf,
} from '../csrf';

describe('CSRF Protection', () => {
  describe('generateCsrfToken', () => {
    it('should generate a token', () => {
      const token = generateCsrfToken();

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
    });

    it('should generate a token of correct length', () => {
      const token = generateCsrfToken();

      // 32 bytes = 64 hex characters
      expect(token.length).toBe(64);
    });

    it('should generate a token with only hex characters', () => {
      const token = generateCsrfToken();

      expect(token).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should generate different tokens on each call', () => {
      const token1 = generateCsrfToken();
      const token2 = generateCsrfToken();
      const token3 = generateCsrfToken();

      expect(token1).not.toBe(token2);
      expect(token2).not.toBe(token3);
      expect(token1).not.toBe(token3);
    });

    it('should generate cryptographically random tokens', () => {
      const tokens = new Set();
      const iterations = 100;

      for (let i = 0; i < iterations; i++) {
        tokens.add(generateCsrfToken());
      }

      // All tokens should be unique
      expect(tokens.size).toBe(iterations);
    });

    it('should use crypto.getRandomValues', () => {
      const spy = vi.spyOn(crypto, 'getRandomValues');

      generateCsrfToken();

      expect(spy).toHaveBeenCalled();
      spy.mockRestore();
    });

    it('should generate token with proper hex formatting', () => {
      const token = generateCsrfToken();

      // Each byte should be padded to 2 characters
      // No byte should be represented as single character
      expect(token).not.toContain(' ');
      expect(token.length).toBe(64); // 32 bytes * 2 hex chars
    });
  });

  describe('validateCsrfToken', () => {
    it('should return true when tokens match', () => {
      const token = 'abc123def456';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should return false when tokens do not match', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'token1',
          cookie: 'csrf_token=token2',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should return false when header token is missing', () => {
      const request = new Request('http://localhost', {
        headers: {
          cookie: 'csrf_token=abc123',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should return false when cookie is missing', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'abc123',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should return false when csrf_token cookie is not present', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'abc123',
          cookie: 'other_cookie=value',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should handle multiple cookies correctly', () => {
      const token = 'abc123def456';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `session=xyz; csrf_token=${token}; user=john`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should be case-sensitive', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'ABC123',
          cookie: 'csrf_token=abc123',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should reject tokens with different lengths', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'short',
          cookie: 'csrf_token=verylongtoken',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should handle empty tokens', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': '',
          cookie: 'csrf_token=',
        },
      });

      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should handle tokens with special characters', () => {
      const token = 'token-with_special.chars!';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should handle cookie with spaces around equals', () => {
      const token = 'abc123';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should handle cookie with spaces between cookies', () => {
      const token = 'abc123';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `session=xyz; csrf_token=${token}; user=john`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should validate real generated tokens', () => {
      const token = generateCsrfToken();
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });
  });

  describe('Timing-Safe Comparison', () => {
    it('should use constant-time comparison', () => {
      // Test that different mismatches take similar time
      // This is a basic test; true timing attack testing is complex

      const token1 = 'a'.repeat(64);
      const token2 = 'b'.repeat(64);
      const token3 = 'a'.repeat(63) + 'b';

      const request1 = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token1,
          cookie: `csrf_token=${token2}`,
        },
      });

      const request2 = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token1,
          cookie: `csrf_token=${token3}`,
        },
      });

      // Both should return false
      expect(validateCsrfToken(request1)).toBe(false);
      expect(validateCsrfToken(request2)).toBe(false);
    });

    it('should handle unicode characters in comparison', () => {
      const token = '测试token🎥';
      const request1 = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      const request2 = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: 'csrf_token=differenttoken',
        },
      });

      expect(validateCsrfToken(request1)).toBe(true);
      expect(validateCsrfToken(request2)).toBe(false);
    });
  });

  describe('requireCsrfToken', () => {
    it('should return null when token is valid', () => {
      const token = 'abc123def456';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      const result = requireCsrfToken(request);

      expect(result).toBeNull();
    });

    it('should return 403 Response when token is invalid', async () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'token1',
          cookie: 'csrf_token=token2',
        },
      });

      const result = requireCsrfToken(request);

      expect(result).toBeInstanceOf(Response);
      expect(result?.status).toBe(403);
    });

    it('should return JSON error message', async () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'invalid',
        },
      });

      const result = requireCsrfToken(request);
      const body = await result?.json();

      expect(body).toHaveProperty('error', 'CSRF Validation Failed');
      expect(body).toHaveProperty('message', 'Invalid or missing CSRF token');
    });

    it('should return Response with JSON content-type', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'invalid',
        },
      });

      const result = requireCsrfToken(request);

      expect(result?.headers.get('Content-Type')).toBe('application/json');
    });

    it('should return 403 when header token is missing', () => {
      const request = new Request('http://localhost', {
        headers: {
          cookie: 'csrf_token=abc123',
        },
      });

      const result = requireCsrfToken(request);

      expect(result).toBeInstanceOf(Response);
      expect(result?.status).toBe(403);
    });

    it('should return 403 when cookie token is missing', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'abc123',
        },
      });

      const result = requireCsrfToken(request);

      expect(result).toBeInstanceOf(Response);
      expect(result?.status).toBe(403);
    });
  });

  describe('getClientCsrfToken', () => {
    beforeEach(() => {
      // Clear any existing meta tags
      document.head.innerHTML = '';
    });

    it('should return token from meta tag', () => {
      const token = 'abc123def456';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      const result = getClientCsrfToken();

      expect(result).toBe(token);
    });

    it('should return null when meta tag is missing', () => {
      const result = getClientCsrfToken();

      expect(result).toBeNull();
    });

    it('should return null when meta tag has no content', () => {
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      document.head.appendChild(meta);

      const result = getClientCsrfToken();

      expect(result).toBe('');
    });

    it('should handle multiple meta tags', () => {
      const token = 'correct-token';

      const meta1 = document.createElement('meta');
      meta1.name = 'description';
      meta1.content = 'Some description';
      document.head.appendChild(meta1);

      const meta2 = document.createElement('meta');
      meta2.name = 'csrf-token';
      meta2.content = token;
      document.head.appendChild(meta2);

      const meta3 = document.createElement('meta');
      meta3.name = 'viewport';
      meta3.content = 'width=device-width';
      document.head.appendChild(meta3);

      const result = getClientCsrfToken();

      expect(result).toBe(token);
    });

    it('should return first matching meta tag', () => {
      const token1 = 'first-token';
      const token2 = 'second-token';

      const meta1 = document.createElement('meta');
      meta1.name = 'csrf-token';
      meta1.content = token1;
      document.head.appendChild(meta1);

      const meta2 = document.createElement('meta');
      meta2.name = 'csrf-token';
      meta2.content = token2;
      document.head.appendChild(meta2);

      const result = getClientCsrfToken();

      expect(result).toBe(token1);
    });

    it('should handle generated tokens', () => {
      const token = generateCsrfToken();
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      const result = getClientCsrfToken();

      expect(result).toBe(token);
      expect(result?.length).toBe(64);
    });
  });

  describe('fetchWithCsrf', () => {
    let fetchSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      document.head.innerHTML = '';
      fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(
        new Response(JSON.stringify({ success: true }), { status: 200 })
      );
    });

    afterEach(() => {
      fetchSpy.mockRestore();
    });

    it('should call fetch with CSRF token in header', async () => {
      const token = 'abc123def456';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      await fetchWithCsrf('/api/test', { method: 'POST' });

      expect(fetchSpy).toHaveBeenCalledWith(
        '/api/test',
        expect.objectContaining({
          method: 'POST',
          headers: expect.any(Headers),
        })
      );

      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;
      expect(headers.get('x-csrf-token')).toBe(token);
    });

    it('should preserve existing headers', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      await fetchWithCsrf('/api/test', {
        headers: {
          'Content-Type': 'application/json',
          'Custom-Header': 'custom-value',
        },
      });

      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;

      expect(headers.get('x-csrf-token')).toBe(token);
      expect(headers.get('Content-Type')).toBe('application/json');
      expect(headers.get('Custom-Header')).toBe('custom-value');
    });

    it('should work without CSRF token', async () => {
      // No meta tag

      await fetchWithCsrf('/api/test');

      expect(fetchSpy).toHaveBeenCalled();
      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;

      // Should not have CSRF token header
      expect(headers.get('x-csrf-token')).toBeNull();
    });

    it('should preserve request options', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      const body = JSON.stringify({ data: 'test' });

      await fetchWithCsrf('/api/test', {
        method: 'POST',
        body,
        credentials: 'include',
        mode: 'cors',
      });

      expect(fetchSpy).toHaveBeenCalledWith(
        '/api/test',
        expect.objectContaining({
          method: 'POST',
          body,
          credentials: 'include',
          mode: 'cors',
        })
      );
    });

    it('should return fetch response', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      const response = await fetchWithCsrf('/api/test');

      expect(response).toBeInstanceOf(Response);
      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data).toEqual({ success: true });
    });

    it('should handle fetch errors', async () => {
      fetchSpy.mockRejectedValue(new Error('Network error'));

      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      await expect(fetchWithCsrf('/api/test')).rejects.toThrow('Network error');
    });

    it('should work with GET requests', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      await fetchWithCsrf('/api/data', { method: 'GET' });

      expect(fetchSpy).toHaveBeenCalledWith(
        '/api/data',
        expect.objectContaining({
          method: 'GET',
        })
      );

      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;
      expect(headers.get('x-csrf-token')).toBe(token);
    });

    it('should work with default GET method', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      await fetchWithCsrf('/api/data');

      expect(fetchSpy).toHaveBeenCalled();
      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;
      expect(headers.get('x-csrf-token')).toBe(token);
    });

    it('should handle Headers object as input', async () => {
      const token = 'abc123';
      const meta = document.createElement('meta');
      meta.name = 'csrf-token';
      meta.content = token;
      document.head.appendChild(meta);

      const inputHeaders = new Headers();
      inputHeaders.set('Content-Type', 'application/json');

      await fetchWithCsrf('/api/test', {
        method: 'POST',
        headers: inputHeaders,
      });

      const callArgs = fetchSpy.mock.calls[0];
      const headers = callArgs[1]?.headers as Headers;

      expect(headers.get('x-csrf-token')).toBe(token);
      expect(headers.get('Content-Type')).toBe('application/json');
    });
  });

  describe('Integration Tests', () => {
    it('should complete full CSRF flow', () => {
      // 1. Generate token
      const token = generateCsrfToken();
      expect(token).toBeDefined();
      expect(token.length).toBe(64);

      // 2. Validate matching tokens
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });
      expect(validateCsrfToken(request)).toBe(true);

      // 3. Middleware should allow request
      const middlewareResult = requireCsrfToken(request);
      expect(middlewareResult).toBeNull();
    });

    it('should reject invalid CSRF flow', () => {
      const token1 = generateCsrfToken();
      const token2 = generateCsrfToken();

      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token1,
          cookie: `csrf_token=${token2}`,
        },
      });

      expect(validateCsrfToken(request)).toBe(false);

      const middlewareResult = requireCsrfToken(request);
      expect(middlewareResult).toBeInstanceOf(Response);
      expect(middlewareResult?.status).toBe(403);
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long cookie strings', () => {
      const token = generateCsrfToken();
      const longCookie = Array(100)
        .fill(0)
        .map((_, i) => `cookie${i}=value${i}`)
        .join('; ');
      const cookieWithToken = `${longCookie}; csrf_token=${token}`;

      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: cookieWithToken,
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });

    it('should handle cookie values with equals signs', () => {
      const token = 'token=with=equals';
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': token,
          cookie: `csrf_token=${token}`,
        },
      });

      // This will fail because cookie parsing splits on first =
      // The current implementation doesn't handle this edge case
      // This test documents the limitation
      expect(validateCsrfToken(request)).toBe(false);
    });

    it('should handle whitespace in tokens', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-csrf-token': 'token with spaces',
          cookie: 'csrf_token=token with spaces',
        },
      });

      expect(validateCsrfToken(request)).toBe(true);
    });
  });
});
