/**
 * Rate Limiting Middleware
 *
 * Simple in-memory rate limiter for API routes.
 * For production with multiple instances, use Redis (Upstash).
 *
 * This implementation uses a sliding window algorithm.
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

export class RateLimiter {
  private store: Map<string, RateLimitEntry> = new Map();
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(
    private readonly maxRequests: number = 10,
    private readonly windowMs: number = 60 * 1000 // 1 minute
  ) {
    // Clean up old entries every minute
    this.cleanupInterval = setInterval(() => this.cleanup(), 60 * 1000);
  }

  /**
   * Check if request should be rate limited
   * @param identifier - Unique identifier (IP address, user ID, etc.)
   * @returns true if request should be allowed, false if rate limited
   */
  check(identifier: string): { allowed: boolean; remaining: number; resetAt: number } {
    const now = Date.now();
    const entry = this.store.get(identifier);

    // No entry or expired entry
    if (!entry || now >= entry.resetAt) {
      const newEntry: RateLimitEntry = {
        count: 1,
        resetAt: now + this.windowMs,
      };
      this.store.set(identifier, newEntry);

      return {
        allowed: true,
        remaining: this.maxRequests - 1,
        resetAt: newEntry.resetAt,
      };
    }

    // Entry exists and not expired
    if (entry.count >= this.maxRequests) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: entry.resetAt,
      };
    }

    // Increment count
    entry.count++;
    this.store.set(identifier, entry);

    return {
      allowed: true,
      remaining: this.maxRequests - entry.count,
      resetAt: entry.resetAt,
    };
  }

  /**
   * Clean up expired entries
   */
  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.store.entries()) {
      if (now >= entry.resetAt) {
        this.store.delete(key);
      }
    }
  }

  /**
   * Clear all entries (for testing)
   */
  clear(): void {
    this.store.clear();
  }

  /**
   * Destroy the rate limiter
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.clear();
  }
}

// Create singleton instances for different types of rate limiting
export const globalRateLimiter = new RateLimiter(100, 60 * 1000); // 100 requests per minute
export const uploadRateLimiter = new RateLimiter(5, 60 * 1000); // 5 uploads per minute
export const authRateLimiter = new RateLimiter(10, 15 * 60 * 1000); // 10 attempts per 15 minutes

/**
 * Get client IP address from request
 */
export function getClientIp(request: Request): string {
  // Try to get real IP from headers (if behind proxy)
  const forwardedFor = request.headers.get('x-forwarded-for');
  if (forwardedFor) {
    return forwardedFor.split(',')[0].trim();
  }

  const realIp = request.headers.get('x-real-ip');
  if (realIp) {
    return realIp;
  }

  // Fallback to 'unknown' (for development)
  return 'unknown';
}

/**
 * Rate limit middleware for API routes
 *
 * Usage in API route:
 * ```typescript
 * export async function POST(request: Request) {
 *   const rateLimitResult = await rateLimit(request, uploadRateLimiter);
 *   if (!rateLimitResult.allowed) {
 *     return rateLimitResult.response;
 *   }
 *
 *   // Process request...
 * }
 * ```
 */
export async function rateLimit(
  request: Request,
  limiter: RateLimiter = globalRateLimiter
): Promise<{
  allowed: boolean;
  response?: Response;
  remaining: number;
  resetAt: number;
}> {
  const identifier = getClientIp(request);
  const result = limiter.check(identifier);

  if (!result.allowed) {
    const retryAfter = Math.ceil((result.resetAt - Date.now()) / 1000);

    return {
      allowed: false,
      response: new Response(
        JSON.stringify({
          error: 'Too many requests',
          message: 'Rate limit exceeded. Please try again later.',
          retryAfter,
        }),
        {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'Retry-After': retryAfter.toString(),
            'X-RateLimit-Limit': limiter['maxRequests'].toString(),
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': new Date(result.resetAt).toISOString(),
          },
        }
      ),
      remaining: 0,
      resetAt: result.resetAt,
    };
  }

  return {
    allowed: true,
    remaining: result.remaining,
    resetAt: result.resetAt,
  };
}

/**
 * Add rate limit headers to response
 */
export function addRateLimitHeaders(
  response: Response,
  result: { remaining: number; resetAt: number },
  maxRequests: number
): Response {
  response.headers.set('X-RateLimit-Limit', maxRequests.toString());
  response.headers.set('X-RateLimit-Remaining', result.remaining.toString());
  response.headers.set('X-RateLimit-Reset', new Date(result.resetAt).toISOString());

  return response;
}
