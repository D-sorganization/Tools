/**
 * Input Sanitization
 *
 * Sanitizes user input to prevent XSS and other injection attacks.
 * All user-provided text should be sanitized before rendering or storage.
 *
 * For now, uses simple string operations.
 * TODO: Add DOMPurify when ready for production.
 *
 * @see PROFESSIONAL_CODE_REVIEW.md section "Security Assessment"
 */

/**
 * Sanitize plain text input
 * Removes all HTML tags and dangerous characters
 */
export function sanitizeText(text: string): string {
  if (!text) {
    return '';
  }

  return (
    text
      // Remove HTML tags
      .replace(/<[^>]*>/g, '')
      // Remove null bytes
      .replace(/\0/g, '')
      // Remove control characters (except newlines and tabs)
      .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '')
      // Trim whitespace
      .trim()
  );
}

/**
 * Sanitize filename
 * Removes dangerous characters that could be used for path traversal
 */
export function sanitizeFilename(filename: string): string {
  if (!filename) {
    return '';
  }

  return (
    filename
      // Remove path separators
      .replace(/[/\\]/g, '')
      // Remove null bytes
      .replace(/\0/g, '')
      // Remove control characters
      .replace(/[\x00-\x1F]/g, '')
      // Remove leading/trailing dots and spaces
      .replace(/^[.\s]+|[.\s]+$/g, '')
      // Limit length
      .substring(0, 255)
      .trim()
  );
}

/**
 * Sanitize URL
 * Ensures URL uses safe protocols
 */
export function sanitizeUrl(url: string): string {
  if (!url) {
    return '';
  }

  const trimmedUrl = url.trim();

  // Check if URL starts with a safe protocol
  const safeProtocols = ['http:', 'https:', 'mailto:', 'tel:'];
  const hasProtocol = safeProtocols.some((protocol) => trimmedUrl.startsWith(protocol));

  if (!hasProtocol && !trimmedUrl.startsWith('/') && !trimmedUrl.startsWith('#')) {
    // If no protocol and not a relative URL, it might be dangerous (javascript:, data:, etc.)
    return '';
  }

  return trimmedUrl;
}

/**
 * Sanitize HTML (basic)
 * For production, use DOMPurify instead
 */
export function sanitizeHTML(html: string): string {
  if (!html) {
    return '';
  }

  // For now, just strip all HTML
  // TODO: Use DOMPurify to allow safe HTML tags
  return sanitizeText(html);
}

/**
 * Escape HTML special characters
 * Useful for displaying user input in HTML context
 * Uses OWASP-recommended character entity encoding
 * Source: https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html
 */
export function escapeHTML(text: string): string {
  if (typeof text !== 'string' || !text) {
    return '';
  }

  // OWASP recommended HTML entity encoding
  const ESCAPE_MAP: { [char: string]: string } = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;', // &apos; is not recommended by OWASP
    '/': '&#x2F;', // Prevents closing script tags
  };

  return text.replace(/[&<>"'/]/g, (char) => ESCAPE_MAP[char]);
}

/**
 * Sanitize JSON string
 * Ensures JSON is valid and safe to parse
 */
export function sanitizeJSON(jsonString: string): string {
  if (!jsonString) {
    return '';
  }

  try {
    // Try to parse and re-stringify to ensure valid JSON
    const parsed = JSON.parse(jsonString);
    return JSON.stringify(parsed);
  } catch {
    throw new Error('Invalid JSON string');
  }
}

/**
 * Validate and sanitize email address
 */
export function sanitizeEmail(email: string): string {
  if (!email) {
    return '';
  }

  const sanitized = email.trim().toLowerCase();

  // Basic email regex
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  if (!emailRegex.test(sanitized)) {
    throw new Error('Invalid email address');
  }

  return sanitized;
}

/**
 * Sanitize number input
 * Ensures value is a valid number within range
 */
export function sanitizeNumber(
  value: string | number,
  options?: {
    min?: number;
    max?: number;
    decimals?: number;
  }
): number {
  const num = typeof value === 'string' ? parseFloat(value) : value;

  if (isNaN(num)) {
    throw new Error('Invalid number');
  }

  let result = num;

  // Apply constraints
  if (options?.min !== undefined && result < options.min) {
    result = options.min;
  }

  if (options?.max !== undefined && result > options.max) {
    result = options.max;
  }

  if (options?.decimals !== undefined) {
    result = parseFloat(result.toFixed(options.decimals));
  }

  return result;
}

/**
 * Sanitize color value (hex or rgb)
 */
export function sanitizeColor(color: string): string {
  if (!color) {
    return '#000000';
  }

  const trimmed = color.trim();

  // Check if hex color
  if (/^#[0-9A-Fa-f]{6}$/.test(trimmed)) {
    return trimmed.toUpperCase();
  }

  // Check if short hex color
  if (/^#[0-9A-Fa-f]{3}$/.test(trimmed)) {
    // Expand short form to full form
    const r = trimmed[1];
    const g = trimmed[2];
    const b = trimmed[3];
    return `#${r}${r}${g}${g}${b}${b}`.toUpperCase();
  }

  // Check if rgb/rgba
  if (trimmed.startsWith('rgb')) {
    // For now, just return as-is if it looks like rgb
    // TODO: Parse and validate RGB values
    return trimmed;
  }

  // Invalid color, return default
  return '#000000';
}

/**
 * Sanitize annotation data from Fabric.js
 * Ensures annotation objects don't contain malicious code
 */
export function sanitizeAnnotation(annotation: any): any {
  if (!annotation || typeof annotation !== 'object') {
    return annotation;
  }

  // Create a clean copy with only safe properties
  const safe: any = {};

  // Allowed properties for annotations
  const allowedProps = [
    'type',
    'left',
    'top',
    'width',
    'height',
    'fill',
    'stroke',
    'strokeWidth',
    'opacity',
    'angle',
    'scaleX',
    'scaleY',
    'x1',
    'y1',
    'x2',
    'y2',
    'text',
    'fontSize',
    'fontFamily',
  ];

  for (const prop of allowedProps) {
    if (prop in annotation) {
      if (prop === 'text') {
        // Sanitize text content
        safe[prop] = sanitizeText(annotation[prop]);
      } else if (prop === 'fill' || prop === 'stroke') {
        // Sanitize color
        safe[prop] = sanitizeColor(annotation[prop]);
      } else if (typeof annotation[prop] === 'number') {
        // Sanitize number (ensure it's valid)
        safe[prop] = isNaN(annotation[prop]) ? 0 : annotation[prop];
      } else {
        // Copy other safe values
        safe[prop] = annotation[prop];
      }
    }
  }

  return safe;
}
