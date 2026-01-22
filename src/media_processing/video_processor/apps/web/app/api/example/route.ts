/**
 * Example API Route
 *
 * Demonstrates proper use of:
 * - Rate limiting
 * - CSRF protection
 * - Error handling
 * - Logging
 * - Response formatting
 */

import { NextResponse } from 'next/server';
import { rateLimit, uploadRateLimiter, addRateLimitHeaders } from '@/lib/rate-limit';
import { requireCsrfToken } from '@/lib/csrf';
import { logger } from '@/lib/logger';
import { ValidationError } from '@/lib/errors';
import { z } from 'zod';

// Request validation schema
const RequestSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  message: z.string().min(1).max(1000),
});

/**
 * GET /api/example
 * No rate limiting or CSRF required for GET requests
 */
export async function GET(_request: Request) {
  try {
    logger.info('GET /api/example');

    return NextResponse.json({
      message: 'This is an example API endpoint',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    logger.error('GET /api/example failed', { error });

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/example
 * Requires rate limiting and CSRF token
 */
export async function POST(request: Request) {
  try {
    // 1. Rate limiting
    const rateLimitResult = await rateLimit(request, uploadRateLimiter);
    if (!rateLimitResult.allowed) {
      logger.warn('Rate limit exceeded', {
        ip: request.headers.get('x-forwarded-for'),
        path: '/api/example',
      });
      return rateLimitResult.response!;
    }

    // 2. CSRF protection
    const csrfError = requireCsrfToken(request);
    if (csrfError) {
      logger.warn('CSRF validation failed', {
        ip: request.headers.get('x-forwarded-for'),
        path: '/api/example',
      });
      return csrfError;
    }

    // 3. Parse and validate request body
    const body = await request.json();
    const validatedData = RequestSchema.parse(body);

    logger.info('POST /api/example', {
      name: validatedData.name,
      email: validatedData.email,
    });

    // 4. Process request (your business logic here)
    // For example: Save to database, send email, etc.

    // 5. Return response with rate limit headers
    const response = NextResponse.json({
      success: true,
      message: 'Request processed successfully',
      data: validatedData,
    });

    return addRateLimitHeaders(
      response,
      rateLimitResult,
      uploadRateLimiter['maxRequests']
    );
  } catch (error) {
    // Handle validation errors
    if (error instanceof z.ZodError) {
      logger.warn('Validation error', { error: error.errors });
      return NextResponse.json(
        {
          error: 'Validation failed',
          details: error.errors,
        },
        { status: 400 }
      );
    }

    // Handle application errors
    if (error instanceof ValidationError) {
      logger.warn('Application validation error', { error });
      return NextResponse.json(
        { error: error.message },
        { status: error.statusCode }
      );
    }

    // Handle unexpected errors
    logger.error('POST /api/example failed', { error });
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

/**
 * OPTIONS /api/example
 * Handle CORS preflight requests
 */
export async function OPTIONS(_request: Request) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-CSRF-Token',
    },
  });
}
