/**
 * Tests for rate limiting
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  rateLimit,
  getClientIp,
  addRateLimitHeaders,
  globalRateLimiter,
  uploadRateLimiter,
  authRateLimiter,
} from '../rate-limit';

describe('Rate Limiting', () => {
  beforeEach(() => {
    // Clear rate limiters before each test
    globalRateLimiter.clear();
    uploadRateLimiter.clear();
    authRateLimiter.clear();
  });

  afterEach(() => {
    globalRateLimiter.clear();
    uploadRateLimiter.clear();
    authRateLimiter.clear();
  });

  describe('getClientIp', () => {
    it('should get IP from x-forwarded-for header', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-forwarded-for': '192.168.1.1, 10.0.0.1',
        },
      });

      expect(getClientIp(request)).toBe('192.168.1.1');
    });

    it('should get IP from x-real-ip header', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-real-ip': '192.168.1.1',
        },
      });

      expect(getClientIp(request)).toBe('192.168.1.1');
    });

    it('should prefer x-forwarded-for over x-real-ip', () => {
      const request = new Request('http://localhost', {
        headers: {
          'x-forwarded-for': '192.168.1.1',
          'x-real-ip': '10.0.0.1',
        },
      });

      expect(getClientIp(request)).toBe('192.168.1.1');
    });

    it('should return "unknown" when no IP headers present', () => {
      const request = new Request('http://localhost');

      expect(getClientIp(request)).toBe('unknown');
    });
  });

  describe('RateLimiter', () => {
    it('should allow requests under the limit', async () => {
      // Create custom limiter with 3 requests per minute
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(3, 60 * 1000);

      // First 3 requests should be allowed
      const result1 = limiter.check('test-user');
      expect(result1.allowed).toBe(true);
      expect(result1.remaining).toBe(2);

      const result2 = limiter.check('test-user');
      expect(result2.allowed).toBe(true);
      expect(result2.remaining).toBe(1);

      const result3 = limiter.check('test-user');
      expect(result3.allowed).toBe(true);
      expect(result3.remaining).toBe(0);

      limiter.destroy();
    });

    it('should block requests over the limit', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(2, 60 * 1000);

      limiter.check('test-user');
      limiter.check('test-user');

      // Third request should be blocked
      const result = limiter.check('test-user');
      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(0);

      limiter.destroy();
    });

    it('should reset after time window expires', async () => {
      vi.useFakeTimers();

      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(2, 1000); // 2 requests per second

      // Use up the limit
      limiter.check('test-user');
      limiter.check('test-user');

      const blockedResult = limiter.check('test-user');
      expect(blockedResult.allowed).toBe(false);

      // Advance time by 1 second
      vi.advanceTimersByTime(1000);

      // Should be allowed again
      const allowedResult = limiter.check('test-user');
      expect(allowedResult.allowed).toBe(true);

      limiter.destroy();
      vi.useRealTimers();
    });

    it('should track different users separately', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(1, 60 * 1000);

      const result1 = limiter.check('user1');
      expect(result1.allowed).toBe(true);

      const result2 = limiter.check('user2');
      expect(result2.allowed).toBe(true);

      // Both should be blocked on second request
      const result3 = limiter.check('user1');
      expect(result3.allowed).toBe(false);

      const result4 = limiter.check('user2');
      expect(result4.allowed).toBe(false);

      limiter.destroy();
    });

    it('should cleanup expired entries', async () => {
      vi.useFakeTimers();

      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(1, 1000);

      limiter.check('user1');
      limiter.check('user2');

      // Advance past cleanup interval (60 seconds)
      vi.advanceTimersByTime(61 * 1000);

      // Entries should be cleaned up
      const result = limiter.check('user1');
      expect(result.allowed).toBe(true);

      limiter.destroy();
      vi.useRealTimers();
    });
  });

  describe('rateLimit middleware', () => {
    it('should return allowed for request under limit', async () => {
      const request = new Request('http://localhost', {
        headers: { 'x-forwarded-for': '192.168.1.1' },
      });

      const result = await rateLimit(request, globalRateLimiter);

      expect(result.allowed).toBe(true);
      expect(result.response).toBeUndefined();
      expect(result.remaining).toBeGreaterThan(0);
    });

    it('should return 429 response for request over limit', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(1, 60 * 1000);

      const request = new Request('http://localhost', {
        headers: { 'x-forwarded-for': '192.168.1.1' },
      });

      // Use up the limit
      await rateLimit(request, limiter);

      // Second request should be blocked
      const result = await rateLimit(request, limiter);

      expect(result.allowed).toBe(false);
      expect(result.response).toBeDefined();
      expect(result.response?.status).toBe(429);

      limiter.destroy();
    });

    it('should include Retry-After header in 429 response', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(1, 60 * 1000);

      const request = new Request('http://localhost', {
        headers: { 'x-forwarded-for': '192.168.1.1' },
      });

      await rateLimit(request, limiter);
      const result = await rateLimit(request, limiter);

      const retryAfter = result.response?.headers.get('Retry-After');
      expect(retryAfter).toBeDefined();
      expect(parseInt(retryAfter!)).toBeGreaterThan(0);

      limiter.destroy();
    });

    it('should include rate limit headers in 429 response', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(5, 60 * 1000);

      const request = new Request('http://localhost', {
        headers: { 'x-forwarded-for': '192.168.1.1' },
      });

      // Use up all requests
      for (let i = 0; i < 5; i++) {
        await rateLimit(request, limiter);
      }

      const result = await rateLimit(request, limiter);

      expect(result.response?.headers.get('X-RateLimit-Limit')).toBe('5');
      expect(result.response?.headers.get('X-RateLimit-Remaining')).toBe('0');
      expect(result.response?.headers.get('X-RateLimit-Reset')).toBeDefined();

      limiter.destroy();
    });

    it('should return JSON error message in 429 response', async () => {
      const { RateLimiter } = await import('../rate-limit');
      const limiter = new (RateLimiter as any)(1, 60 * 1000);

      const request = new Request('http://localhost', {
        headers: { 'x-forwarded-for': '192.168.1.1' },
      });

      await rateLimit(request, limiter);
      const result = await rateLimit(request, limiter);

      const body = await result.response?.json();

      expect(body).toHaveProperty('error', 'Too many requests');
      expect(body).toHaveProperty('message');
      expect(body).toHaveProperty('retryAfter');

      limiter.destroy();
    });
  });

  describe('addRateLimitHeaders', () => {
    it('should add rate limit headers to response', () => {
      const response = new Response('OK');
      const result = { remaining: 5, resetAt: Date.now() + 60000 };

      const updatedResponse = addRateLimitHeaders(response, result, 10);

      expect(updatedResponse.headers.get('X-RateLimit-Limit')).toBe('10');
      expect(updatedResponse.headers.get('X-RateLimit-Remaining')).toBe('5');
      expect(updatedResponse.headers.get('X-RateLimit-Reset')).toBeDefined();
    });

    it('should format reset time as ISO string', () => {
      const response = new Response('OK');
      const resetAt = new Date('2024-01-01T12:00:00Z').getTime();
      const result = { remaining: 5, resetAt };

      const updatedResponse = addRateLimitHeaders(response, result, 10);

      const resetHeader = updatedResponse.headers.get('X-RateLimit-Reset');
      expect(resetHeader).toBe('2024-01-01T12:00:00.000Z');
    });
  });

  describe('Singleton Rate Limiters', () => {
    it('should have different limits for different limiters', () => {
      // Global: 100 requests per minute
      const global = globalRateLimiter['maxRequests'];
      expect(global).toBe(100);

      // Upload: 5 requests per minute
      const upload = uploadRateLimiter['maxRequests'];
      expect(upload).toBe(5);

      // Auth: 10 requests per 15 minutes
      const auth = authRateLimiter['maxRequests'];
      expect(auth).toBe(10);
    });

    it('should track requests independently', () => {
      const result1 = globalRateLimiter.check('test');
      const result2 = uploadRateLimiter.check('test');

      // Same user, different limiters, different remaining counts
      expect(result1.remaining).not.toBe(result2.remaining);
    });
  });
});
