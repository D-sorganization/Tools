/**
 * Video File Validation
 *
 * Validates video files to prevent:
 * - File type spoofing (magic byte validation)
 * - Oversized files
 * - Invalid video formats
 * - Corrupted files
 *
 * Follows "Design by Contract" and "Dead Programs Tell No Lies" principles.
 *
 * @see PROFESSIONAL_CODE_REVIEW.md section "Design by Contract"
 */

import { ValidationError } from '@/lib/errors';
import { z } from 'zod';

// ============================================================================
// Constants
// ============================================================================

// Maximum file size in bytes (500MB)
const MAX_FILE_SIZE = 500 * 1024 * 1024;

// Allowed video MIME types
const ALLOWED_MIME_TYPES = [
  'video/mp4',
  'video/webm',
  'video/ogg',
  'video/quicktime',      // .mov
  'video/x-msvideo',      // .avi
  'video/x-matroska',     // .mkv
] as const;

// Magic bytes for video file format detection
// These are the first few bytes of each video format
const MAGIC_BYTES: Record<string, Uint8Array[]> = {
  'video/mp4': [
    // MP4 files start with an 'ftyp' atom/box (0x66 0x74 0x79 0x70 = 'ftyp' in ASCII)
    // The first 4 bytes represent the box size in big-endian format:
    // 0x18 (24 bytes), 0x1c (28 bytes), 0x20 (32 bytes) are common ftyp box sizes
    // Format: [size (4 bytes)][type (4 bytes 'ftyp')]
    new Uint8Array([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70]), // 24-byte ftyp
    new Uint8Array([0x00, 0x00, 0x00, 0x1c, 0x66, 0x74, 0x79, 0x70]), // 28-byte ftyp
    new Uint8Array([0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70]), // 32-byte ftyp
  ],
  'video/webm': [
    // WebM files start with EBML header
    new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]),
  ],
  'video/ogg': [
    // OGG files start with 'OggS'
    new Uint8Array([0x4f, 0x67, 0x67, 0x53]),
  ],
  'video/quicktime': [
    // QuickTime MOV files
    new Uint8Array([0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70, 0x71, 0x74]),
  ],
  'video/x-msvideo': [
    // AVI files start with 'RIFF....AVI'
    new Uint8Array([0x52, 0x49, 0x46, 0x46]),
  ],
  'video/x-matroska': [
    // MKV files start with EBML header
    new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]),
  ],
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get human-readable file size
 */
export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(2)} ${units[unitIndex]}`;
}

/**
 * Get max file size in human-readable format
 */
export function getMaxFileSize(): string {
  return formatFileSize(MAX_FILE_SIZE);
}

// ============================================================================
// Schemas
// ============================================================================

const VideoFileSchema = z.object({
  name: z.string().min(1, "File name is required").max(255, "File name is too long"),
  size: z.number().int()
    .min(1, "File is empty")
    .max(MAX_FILE_SIZE, `File size too large. Maximum size: ${getMaxFileSize()}`),
  type: z.enum(ALLOWED_MIME_TYPES as any, {
    errorMap: () => ({ message: "File type not supported" })
  }),
});

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Check if byte array starts with any of the given patterns
 */
function startsWithAny(bytes: Uint8Array, patterns: Uint8Array[]): boolean {
  return patterns.some((pattern) => {
    if (bytes.length < pattern.length) {
      return false;
    }

    for (let i = 0; i < pattern.length; i++) {
      if (bytes[i] !== pattern[i]) {
        return false;
      }
    }

    return true;
  });
}

/**
 * Detect actual file type from magic bytes
 */
async function detectFileType(file: File): Promise<string | null> {
  try {
    // Read first 512 bytes (enough for most magic byte checks)
    const buffer = await file.slice(0, 512).arrayBuffer();
    const bytes = new Uint8Array(buffer);

    // Check against known magic bytes
    for (const [mimeType, patterns] of Object.entries(MAGIC_BYTES)) {
      if (startsWithAny(bytes, patterns)) {
        return mimeType;
      }
    }

    return null;
  } catch (error) {
    throw new ValidationError('Failed to read file for type detection', { error });
  }
}

/**
 * Validate video file metadata (name, size, declared type)
 */
function validateVideoMetadata(file: File): void {
  try {
    VideoFileSchema.parse({
      name: file.name,
      size: file.size,
      type: file.type,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      const issues = error.issues.map(i => i.message).join(', ');
      throw new ValidationError(`Invalid video file: ${issues}`, {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
        zodError: error,
      });
    }
    throw error;
  }
}

/**
 * Validate that file type matches declared MIME type (prevents file type spoofing)
 */
async function validateFileTypeMatch(file: File): Promise<void> {
  const declaredType = file.type;
  const actualType = await detectFileType(file);

  if (!actualType) {
    throw new ValidationError(
      'Unable to determine file type. The file may be corrupted or unsupported.',
      { fileName: file.name, declaredType }
    );
  }

  // For video formats that share magic bytes (WebM and MKV both use EBML),
  // we accept either
  const isWebmOrMkv =
    (declaredType === 'video/webm' || declaredType === 'video/x-matroska') &&
    (actualType === 'video/webm' || actualType === 'video/x-matroska');

  if (actualType !== declaredType && !isWebmOrMkv) {
    throw new ValidationError(
      `File type mismatch. Expected ${declaredType}, but file appears to be ${actualType}.`,
      { fileName: file.name, declaredType, actualType }
    );
  }
}

/**
 * Validate that video file is playable
 */
async function validateVideoPlayability(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    let timeoutId: NodeJS.Timeout | null = null;

    video.preload = 'metadata';
    video.src = url;

    const cleanup = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      URL.revokeObjectURL(url);
      video.remove();
    };

    video.onloadedmetadata = () => {
      cleanup();
      resolve();
    };

    video.onerror = (error) => {
      cleanup();
      reject(
        new ValidationError('Invalid or corrupted video file. Unable to load video metadata.', {
          fileName: file.name,
          error,
        })
      );
    };

    // Timeout after 10 seconds
    timeoutId = setTimeout(() => {
      cleanup();
      reject(
        new ValidationError('Video validation timeout. The file may be corrupted or too large.', {
          fileName: file.name,
          fileSize: file.size,
        })
      );
    }, 10000);
  });
}

/**
 * Comprehensive video file validation
 *
 * Validates:
 * 1. File metadata (name, size, type)
 * 2. File type match (magic bytes vs declared MIME type)
 * 3. Video playability (can browser parse it?)
 *
 * @throws {ValidationError} if validation fails
 */
export async function validateVideoFile(file: File): Promise<void> {
  // Step 1: Validate metadata
  validateVideoMetadata(file);

  // Step 2: Validate file type (magic bytes)
  await validateFileTypeMatch(file);

  // Step 3: Validate playability
  await validateVideoPlayability(file);
}

/**
 * Quick validation for file upload (before full validation)
 * Checks only metadata, not file content
 */
export function quickValidateVideoFile(file: File): void {
  validateVideoMetadata(file);
}

/**
 * Check if file size is within limits
 */
export function isFileSizeValid(file: File): boolean {
  return file.size > 0 && file.size <= MAX_FILE_SIZE;
}

/**
 * Check if file type is allowed
 */
export function isFileTypeAllowed(file: File): boolean {
  return ALLOWED_MIME_TYPES.includes(file.type as any);
}

// Export constants for use in components
export { MAX_FILE_SIZE, ALLOWED_MIME_TYPES };
