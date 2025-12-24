/**
 * CSRF Protection
 *
 * Implements CSRF token generation and validation for forms and API routes.
 * Uses the "Double Submit Cookie" pattern for stateless CSRF protection.
 */

import { cookies } from 'next/headers';

const CSRF_TOKEN_LENGTH = 32;
const CSRF_COOKIE_NAME = 'csrf_token';
const CSRF_HEADER_NAME = 'x-csrf-token';

/**
 * Generate a random CSRF token
 */
export function generateCsrfToken(): string {
  const array = new Uint8Array(CSRF_TOKEN_LENGTH);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Get CSRF token from cookies or generate a new one
 */
export function getCsrfToken(): string {
  const cookieStore = cookies();
  const existingToken = cookieStore.get(CSRF_COOKIE_NAME);

  if (existingToken?.value) {
    return existingToken.value;
  }

  const newToken = generateCsrfToken();
  cookieStore.set(CSRF_COOKIE_NAME, newToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
    maxAge: 60 * 60 * 24, // 24 hours
  });

  return newToken;
}

/**
 * Validate CSRF token from request
 */
export function validateCsrfToken(request: Request): boolean {
  // Get token from header
  const headerToken = request.headers.get(CSRF_HEADER_NAME);
  if (!headerToken) {
    return false;
  }

  // Get token from cookie
  const cookieHeader = request.headers.get('cookie');
  if (!cookieHeader) {
    return false;
  }

  const cookies = Object.fromEntries(
    cookieHeader.split(';').map((cookie) => {
      const [key, value] = cookie.trim().split('=');
      return [key, value];
    })
  );

  const cookieToken = cookies[CSRF_COOKIE_NAME];
  if (!cookieToken) {
    return false;
  }

  // Compare tokens (constant-time comparison to prevent timing attacks)
  return timingSafeEqual(headerToken, cookieToken);
}

/**
 * Timing-safe string comparison
 * Prevents timing attacks by always comparing the full string
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}

/**
 * CSRF protection middleware for API routes
 *
 * Usage:
 * ```typescript
 * export async function POST(request: Request) {
 *   if (!validateCsrfToken(request)) {
 *     return new Response('CSRF token validation failed', { status: 403 });
 *   }
 *
 *   // Process request...
 * }
 * ```
 */
export function requireCsrfToken(request: Request): Response | null {
  if (!validateCsrfToken(request)) {
    return new Response(
      JSON.stringify({
        error: 'CSRF Validation Failed',
        message: 'Invalid or missing CSRF token',
      }),
      {
        status: 403,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  }

  return null;
}

/**
 * CSRF token provider for client-side use
 * Add this to your root layout or pages that need CSRF protection
 *
 * NOTE: To use this, create a React component in a .tsx file:
 *
 * ```tsx
 * 'use server';
 * import { getCsrfToken } from '@/lib/csrf';
 *
 * export function CsrfTokenProvider({ children }: { children: React.ReactNode }) {
 *   const token = getCsrfToken();
 *   return (
 *     <>
 *       <meta name="csrf-token" content={token} />
 *       {children}
 *     </>
 *   );
 * }
 * ```
 *
 * Or manually add the meta tag in your layout.tsx:
 * ```tsx
 * import { getCsrfToken } from '@/lib/csrf';
 *
 * export default function RootLayout({ children }) {
 *   const csrfToken = getCsrfToken();
 *   return (
 *     <html>
 *       <head>
 *         <meta name="csrf-token" content={csrfToken} />
 *       </head>
 *       <body>{children}</body>
 *     </html>
 *   );
 * }
 * ```
 */

/**
 * Client-side helper to get CSRF token
 * Use this when making fetch requests
 */
export function getClientCsrfToken(): string | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta?.getAttribute('content') || null;
}

/**
 * Client-side fetch wrapper with CSRF token
 *
 * Usage:
 * ```typescript
 * const response = await fetchWithCsrf('/api/endpoint', {
 *   method: 'POST',
 *   body: JSON.stringify(data),
 * });
 * ```
 */
export async function fetchWithCsrf(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getClientCsrfToken();

  const headers = new Headers(options.headers);
  if (token) {
    headers.set(CSRF_HEADER_NAME, token);
  }

  return fetch(url, {
    ...options,
    headers,
  });
}
