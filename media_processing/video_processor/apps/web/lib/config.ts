/**
 * Application Configuration Module
 *
 * This module validates and exports all environment variables.
 * It fails fast on startup if required variables are missing,
 * following the "Dead Programs Tell No Lies" principle.
 *
 * @see .env.example for all available configuration options
 */

import { z } from 'zod';

// ============================================================================
// Validation Schemas
// ============================================================================

const NodeEnvSchema = z.enum(['development', 'production', 'test']);

const DatabaseConfigSchema = z.object({
  url: z.string().url('DATABASE_URL must be a valid PostgreSQL connection string'),
  directUrl: z.string().url().optional(),
});

const StorageConfigSchema = z.object({
  accountId: z.string().min(1, 'R2_ACCOUNT_ID is required'),
  accessKeyId: z.string().min(1, 'R2_ACCESS_KEY_ID is required'),
  secretAccessKey: z.string().min(1, 'R2_SECRET_ACCESS_KEY is required'),
  bucketName: z.string().min(1, 'R2_BUCKET_NAME is required'),
  publicUrl: z.string().url('R2_PUBLIC_URL must be a valid URL'),
  endpoint: z.string().url('R2_ENDPOINT must be a valid URL'),
});

const VideoConfigSchema = z.object({
  maxFileSizeMB: z.number().positive().max(2000),
  defaultFPS: z.number().positive().max(120),
  supportedTypes: z.array(z.string()).min(1),
  retentionDays: z.number().positive().optional(),
});

const AiConfigSchema = z.object({
  mediaPipeModelPath: z.string().url(),
  enablePoseDetection: z.boolean(),
  modelComplexity: z.number().min(0).max(2),
  minConfidence: z.number().min(0).max(1),
});

const FeatureFlagsSchema = z.object({
  matlabIntegration: z.boolean(),
  visualization3D: z.boolean(),
  advancedEditing: z.boolean(),
  videoSharing: z.boolean(),
});

const AuthConfigSchema = z.object({
  nextAuthUrl: z.string().url(),
  nextAuthSecret: z.string().min(32, 'NEXTAUTH_SECRET must be at least 32 characters'),
  google: z.object({
    clientId: z.string().optional(),
    clientSecret: z.string().optional(),
  }).optional(),
  github: z.object({
    clientId: z.string().optional(),
    clientSecret: z.string().optional(),
  }).optional(),
});

const MonitoringConfigSchema = z.object({
  sentry: z.object({
    dsn: z.string().optional(),
    org: z.string().optional(),
    project: z.string().optional(),
  }),
  posthog: z.object({
    key: z.string().optional(),
    host: z.string().url().optional(),
  }),
  logLevel: z.enum(['debug', 'info', 'warn', 'error']),
});

const RateLimitConfigSchema = z.object({
  redisUrl: z.string().url().optional(),
  redisToken: z.string().optional(),
  requests: z.number().positive().optional(),
  window: z.string().optional(),
});

// ============================================================================
// Configuration Class
// ============================================================================

class Config {
  public readonly env: z.infer<typeof NodeEnvSchema>;
  public readonly app: {
    url: string;
    name: string;
  };
  public readonly database: z.infer<typeof DatabaseConfigSchema>;
  public readonly storage: z.infer<typeof StorageConfigSchema>;
  public readonly video: z.infer<typeof VideoConfigSchema>;
  public readonly ai: z.infer<typeof AiConfigSchema>;
  public readonly features: z.infer<typeof FeatureFlagsSchema>;
  public readonly auth: z.infer<typeof AuthConfigSchema>;
  public readonly monitoring: z.infer<typeof MonitoringConfigSchema>;
  public readonly rateLimit: z.infer<typeof RateLimitConfigSchema>;
  public readonly debug: boolean;

  constructor() {
    this.validateRequiredEnvVars();

    // Environment
    this.env = NodeEnvSchema.parse(process.env.NODE_ENV || 'development');

    // Application
    this.app = {
      url: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
      name: process.env.NEXT_PUBLIC_APP_NAME || 'Golf Swing Video Analyzer',
    };

    // Database
    this.database = DatabaseConfigSchema.parse({
      url: process.env.DATABASE_URL,
      directUrl: process.env.DIRECT_URL,
    });

    // Storage (R2)
    this.storage = StorageConfigSchema.parse({
      accountId: process.env.R2_ACCOUNT_ID,
      accessKeyId: process.env.R2_ACCESS_KEY_ID,
      secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
      bucketName: process.env.R2_BUCKET_NAME,
      publicUrl: process.env.R2_PUBLIC_URL,
      endpoint: process.env.R2_ENDPOINT || `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    });

    // Video Configuration
    this.video = VideoConfigSchema.parse({
      maxFileSizeMB: parseInt(process.env.NEXT_PUBLIC_MAX_VIDEO_FILE_SIZE_MB || '500', 10),
      defaultFPS: parseInt(process.env.NEXT_PUBLIC_DEFAULT_VIDEO_FPS || '30', 10),
      supportedTypes: (process.env.NEXT_PUBLIC_SUPPORTED_VIDEO_TYPES || 'video/mp4,video/webm,video/ogg').split(','),
      retentionDays: process.env.VIDEO_RETENTION_DAYS ? parseInt(process.env.VIDEO_RETENTION_DAYS, 10) : undefined,
    });

    // AI Configuration
    this.ai = AiConfigSchema.parse({
      mediaPipeModelPath: process.env.NEXT_PUBLIC_MEDIAPIPE_MODEL_PATH || 'https://cdn.jsdelivr.net/npm/@mediapipe/pose',
      enablePoseDetection: process.env.NEXT_PUBLIC_ENABLE_POSE_DETECTION === 'true',
      modelComplexity: parseInt(process.env.NEXT_PUBLIC_MEDIAPIPE_MODEL_COMPLEXITY || '1', 10),
      minConfidence: parseFloat(process.env.NEXT_PUBLIC_POSE_DETECTION_MIN_CONFIDENCE || '0.5'),
    });

    // Feature Flags
    this.features = FeatureFlagsSchema.parse({
      matlabIntegration: process.env.NEXT_PUBLIC_FEATURE_MATLAB_INTEGRATION === 'true',
      visualization3D: process.env.NEXT_PUBLIC_FEATURE_3D_VISUALIZATION === 'true',
      advancedEditing: process.env.NEXT_PUBLIC_FEATURE_ADVANCED_EDITING !== 'false',
      videoSharing: process.env.NEXT_PUBLIC_FEATURE_VIDEO_SHARING !== 'false',
    });

    // Authentication
    this.auth = AuthConfigSchema.parse({
      nextAuthUrl: process.env.NEXTAUTH_URL || this.app.url,
      nextAuthSecret: process.env.NEXTAUTH_SECRET || this.generateDevSecret(),
      google: {
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      },
      github: {
        clientId: process.env.GITHUB_CLIENT_ID,
        clientSecret: process.env.GITHUB_CLIENT_SECRET,
      },
    });

    // Monitoring
    this.monitoring = MonitoringConfigSchema.parse({
      sentry: {
        dsn: process.env.SENTRY_DSN || process.env.NEXT_PUBLIC_SENTRY_DSN,
        org: process.env.SENTRY_ORG,
        project: process.env.SENTRY_PROJECT,
      },
      posthog: {
        key: process.env.NEXT_PUBLIC_POSTHOG_KEY,
        host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
      },
      logLevel: (process.env.LOG_LEVEL || 'info') as z.infer<typeof MonitoringConfigSchema>['logLevel'],
    });

    // Rate Limiting
    this.rateLimit = RateLimitConfigSchema.parse({
      redisUrl: process.env.UPSTASH_REDIS_URL,
      redisToken: process.env.UPSTASH_REDIS_TOKEN,
      requests: process.env.RATE_LIMIT_REQUESTS ? parseInt(process.env.RATE_LIMIT_REQUESTS, 10) : 10,
      window: process.env.RATE_LIMIT_WINDOW || '10s',
    });

    // Debug
    this.debug = process.env.DEBUG === 'true';

    this.logConfiguration();
  }

  /**
   * Validate that all required environment variables are present.
   * Fails fast on startup if any are missing.
   */
  private validateRequiredEnvVars(): void {
    const requiredEnvVars = [
      'DATABASE_URL',
    ] as const;

    const requiredInProduction = [
      'R2_ACCOUNT_ID',
      'R2_ACCESS_KEY_ID',
      'R2_SECRET_ACCESS_KEY',
      'R2_BUCKET_NAME',
      'R2_PUBLIC_URL',
      'NEXTAUTH_SECRET',
    ] as const;

    const missing: string[] = [];

    // Check always-required variables
    for (const key of requiredEnvVars) {
      if (!process.env[key]) {
        missing.push(key);
      }
    }

    // Check production-required variables
    if (process.env.NODE_ENV === 'production') {
      for (const key of requiredInProduction) {
        if (!process.env[key]) {
          missing.push(key);
        }
      }
    }

    if (missing.length > 0) {
      const errorMessage = [
        '❌ Missing required environment variables:',
        '',
        ...missing.map(key => `  - ${key}`),
        '',
        '📝 Please copy .env.example to .env and fill in the values:',
        '   cp .env.example .env',
        '',
        '📚 See .env.example for documentation on each variable.',
      ].join('\n');

      throw new Error(errorMessage);
    }
  }

  /**
   * Generate a development-only secret.
   * In production, NEXTAUTH_SECRET must be explicitly set.
   */
  private generateDevSecret(): string {
    if (this.env === 'production') {
      throw new Error(
        'NEXTAUTH_SECRET must be set in production. Generate one with: openssl rand -base64 32'
      );
    }

    console.warn(
      '⚠️  Using auto-generated NEXTAUTH_SECRET for development. ' +
      'Set NEXTAUTH_SECRET in .env for production.'
    );

    return 'dev-secret-' + Math.random().toString(36).substring(2, 15);
  }

  /**
   * Log configuration on startup (excluding secrets).
   */
  private logConfiguration(): void {
    if (this.env === 'development' && this.debug) {
      console.log('📋 Configuration loaded:');
      console.log('  Environment:', this.env);
      console.log('  App URL:', this.app.url);
      console.log('  Database:', this.database.url.split('@')[1]); // Hide credentials
      console.log('  Storage:', this.storage.bucketName);
      console.log('  Video max size:', this.video.maxFileSizeMB, 'MB');
      console.log('  Default FPS:', this.video.defaultFPS);
      console.log('  Pose detection:', this.ai.enablePoseDetection ? 'enabled' : 'disabled');
      console.log('  Feature flags:', this.features);
    }
  }

  /**
   * Check if running in production environment.
   */
  public isProduction(): boolean {
    return this.env === 'production';
  }

  /**
   * Check if running in development environment.
   */
  public isDevelopment(): boolean {
    return this.env === 'development';
  }

  /**
   * Check if running in test environment.
   */
  public isTest(): boolean {
    return this.env === 'test';
  }

  /**
   * Get the maximum file size in bytes.
   */
  public getMaxFileSizeBytes(): number {
    return this.video.maxFileSizeMB * 1024 * 1024;
  }

  /**
   * Get frame duration in seconds for current FPS.
   */
  public getFrameDuration(): number {
    return 1 / this.video.defaultFPS;
  }

  /**
   * Check if a feature is enabled.
   */
  public isFeatureEnabled(feature: keyof z.infer<typeof FeatureFlagsSchema>): boolean {
    return this.features[feature];
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

// Create singleton instance
// This will validate configuration on module load and fail fast if invalid
let configInstance: Config;

try {
  configInstance = new Config();
} catch (error) {
  console.error('Failed to load configuration:', error);
  throw error;
}

export const config = configInstance;

// Export types
export type AppConfig = Config;
