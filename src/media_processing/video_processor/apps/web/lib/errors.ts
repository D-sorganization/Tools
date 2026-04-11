/**
 * Application Error Classes
 *
 * Follows the "Dead Programs Tell No Lies" principle from The Pragmatic Programmer.
 * All errors are typed, logged, and provide context for debugging.
 *
 * @see PROFESSIONAL_CODE_REVIEW.md section "Dead Programs Tell No Lies"
 */

/**
 * Base application error class.
 * All custom errors should extend this class.
 */
export class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly metadata?: Record<string, unknown>
  ) {
    super(message);
    this.name = this.constructor.name;

    // Maintains proper stack trace for where error was thrown (V8 only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }

    // Set prototype explicitly for instanceof checks to work
    Object.setPrototypeOf(this, new.target.prototype);
  }

  /**
   * Convert error to JSON for logging/serialization
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      metadata: this.metadata,
      stack: this.stack,
    };
  }

  /**
   * Get user-friendly error message (safe to display in UI)
   */
  getUserMessage(): string {
    return this.message;
  }
}

/**
 * Validation errors (400 Bad Request)
 * Used when user input is invalid
 */
export class ValidationError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', 400, metadata);
  }

  getUserMessage(): string {
    return this.message; // Validation messages are safe to show users
  }
}

/**
 * Video processing errors (500 Internal Server Error)
 * Used when video upload, processing, or editing fails
 */
export class VideoProcessingError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'VIDEO_PROCESSING_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'Failed to process video. Please try again or use a different file.';
  }
}

/**
 * Audio recording errors (500 Internal Server Error)
 * Used when audio recording fails
 */
export class AudioRecordingError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'AUDIO_RECORDING_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'Failed to record audio. Please check your microphone permissions and try again.';
  }
}

/**
 * Annotation errors (500 Internal Server Error)
 * Used when annotation operations fail
 */
export class AnnotationError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'ANNOTATION_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'Failed to save annotation. Please try again.';
  }
}

/**
 * Pose detection errors (500 Internal Server Error)
 * Used when AI pose detection fails
 */
export class PoseDetectionError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'POSE_DETECTION_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'Failed to detect pose. The video may not contain a visible person.';
  }
}

/**
 * Storage errors (500 Internal Server Error)
 * Used when file upload/download fails
 */
export class StorageError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'STORAGE_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'Failed to upload/download file. Please check your connection and try again.';
  }
}

/**
 * Database errors (500 Internal Server Error)
 * Used when database operations fail
 */
export class DatabaseError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'DATABASE_ERROR', 500, metadata);
  }

  getUserMessage(): string {
    return 'A database error occurred. Please try again later.';
  }
}

/**
 * Authentication errors (401 Unauthorized)
 * Used when authentication fails
 */
export class AuthenticationError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'AUTHENTICATION_ERROR', 401, metadata);
  }

  getUserMessage(): string {
    return 'Authentication failed. Please sign in again.';
  }
}

/**
 * Authorization errors (403 Forbidden)
 * Used when user doesn't have permission
 */
export class AuthorizationError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'AUTHORIZATION_ERROR', 403, metadata);
  }

  getUserMessage(): string {
    return 'You do not have permission to perform this action.';
  }
}

/**
 * Not found errors (404 Not Found)
 * Used when resource doesn't exist
 */
export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    super(
      `${resource}${id ? ` with id ${id}` : ''} not found`,
      'NOT_FOUND_ERROR',
      404,
      { resource, id }
    );
  }

  getUserMessage(): string {
    return 'The requested resource was not found.';
  }
}

/**
 * Rate limit errors (429 Too Many Requests)
 * Used when user exceeds rate limits
 */
export class RateLimitError extends AppError {
  constructor(message: string, metadata?: Record<string, unknown>) {
    super(message, 'RATE_LIMIT_ERROR', 429, metadata);
  }

  getUserMessage(): string {
    return 'Too many requests. Please slow down and try again later.';
  }
}

/**
 * Type guard to check if an error is an AppError
 */
export function isAppError(error: unknown): error is AppError {
  return error instanceof AppError;
}

/**
 * Get user-friendly message from any error
 */
export function getUserMessage(error: unknown): string {
  if (isAppError(error)) {
    return error.getUserMessage();
  }

  if (error instanceof Error) {
    // Don't expose internal error messages to users
    return 'An unexpected error occurred. Please try again.';
  }

  return 'An unknown error occurred. Please try again.';
}

/**
 * Assert that a condition is true, throw ValidationError if not
 * Follows "Design by Contract" principle
 */
export function assert(
  condition: unknown,
  message: string,
  metadata?: Record<string, unknown>
): asserts condition {
  if (!condition) {
    throw new ValidationError(message, metadata);
  }
}

/**
 * Assert that a value is defined (not null or undefined)
 */
export function assertDefined<T>(
  value: T | null | undefined,
  name: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new ValidationError(`${name} is required`, { name, value });
  }
}

/**
 * Assert that a value is a valid number
 */
export function assertNumber(
  value: unknown,
  name: string,
  options?: { min?: number; max?: number }
): asserts value is number {
  if (typeof value !== 'number' || isNaN(value)) {
    throw new ValidationError(`${name} must be a valid number`, { name, value });
  }

  if (options?.min !== undefined && value < options.min) {
    throw new ValidationError(
      `${name} must be at least ${options.min}`,
      { name, value, min: options.min }
    );
  }

  if (options?.max !== undefined && value > options.max) {
    throw new ValidationError(
      `${name} must be at most ${options.max}`,
      { name, value, max: options.max }
    );
  }
}

/**
 * Assert that a value is a valid string
 */
export function assertString(
  value: unknown,
  name: string,
  options?: { minLength?: number; maxLength?: number }
): asserts value is string {
  if (typeof value !== 'string') {
    throw new ValidationError(`${name} must be a string`, { name, value });
  }

  if (options?.minLength !== undefined && value.length < options.minLength) {
    throw new ValidationError(
      `${name} must be at least ${options.minLength} characters`,
      { name, value, minLength: options.minLength }
    );
  }

  if (options?.maxLength !== undefined && value.length > options.maxLength) {
    throw new ValidationError(
      `${name} must be at most ${options.maxLength} characters`,
      { name, value, maxLength: options.maxLength }
    );
  }
}
