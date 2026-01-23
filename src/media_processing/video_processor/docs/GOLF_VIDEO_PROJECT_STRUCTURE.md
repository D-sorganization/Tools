# Golf Swing Video Analysis Platform - Project Structure

## Repository Organization

```
golf-swing-analyzer/
├── README.md
├── package.json
├── tsconfig.json
├── next.config.js
├── tailwind.config.ts
├── .env.example
├── .cursorrules.md
│
├── apps/
│   └── web/                          # Next.js frontend application
│       ├── app/                      # App router (Next.js 13+)
│       │   ├── (auth)/              # Auth routes
│       │   │   ├── login/
│       │   │   ├── register/
│       │   │   └── layout.tsx
│       │   ├── (dashboard)/         # Protected dashboard routes
│       │   │   ├── projects/
│       │   │   ├── videos/
│       │   │   └── layout.tsx
│       │   ├── (public)/            # Public routes
│       │   │   ├── share/[id]/
│       │   │   └── embed/[id]/
│       │   ├── api/                 # API routes
│       │   │   ├── auth/
│       │   │   ├── videos/
│       │   │   ├── annotations/
│       │   │   └── ai/
│       │   ├── layout.tsx
│       │   └── page.tsx
│       │
│       ├── components/              # React components
│       │   ├── ui/                  # shadcn/ui components
│       │   │   ├── button.tsx
│       │   │   ├── dialog.tsx
│       │   │   ├── slider.tsx
│       │   │   └── ...
│       │   ├── video/               # Video-related components
│       │   │   ├── VideoPlayer.tsx
│       │   │   ├── VideoTimeline.tsx
│       │   │   ├── VideoControls.tsx
│       │   │   └── VideoUploader.tsx
│       │   ├── editor/              # Editor components
│       │   │   ├── EditorCanvas.tsx
│       │   │   ├── ToolsPanel.tsx
│       │   │   ├── PropertiesPanel.tsx
│       │   │   └── LayersPanel.tsx
│       │   ├── tools/               # Drawing/annotation tools
│       │   │   ├── DrawingTool.tsx
│       │   │   ├── LineTool.tsx
│       │   │   ├── PlaneTool.tsx
│       │   │   ├── TextTool.tsx
│       │   │   └── FreehandTool.tsx
│       │   ├── ai/                  # AI feature components
│       │   │   ├── PoseDetector.tsx
│       │   │   ├── PendulumViewer.tsx
│       │   │   ├── FeatureTracker.tsx
│       │   │   └── UpscalerControl.tsx
│       │   └── share/               # Sharing components
│       │       ├── ShareDialog.tsx
│       │       └── EmbedCode.tsx
│       │
│       ├── lib/                     # Utility libraries
│       │   ├── video/
│       │   │   ├── ffmpeg.ts        # FFmpeg.wasm wrapper
│       │   │   ├── processor.ts     # Video processing
│       │   │   └── formats.ts       # Format handling
│       │   ├── canvas/
│       │   │   ├── fabric-setup.ts  # Fabric.js configuration
│       │   │   ├── annotations.ts   # Annotation helpers
│       │   │   └── export.ts        # Canvas export utilities
│       │   ├── ai/
│       │   │   ├── mediapipe.ts     # MediaPipe setup
│       │   │   ├── pose-detection.ts
│       │   │   ├── tracking.ts      # Feature tracking
│       │   │   └── pendulum.ts      # Pendulum model
│       │   ├── three/
│       │   │   ├── scene-setup.ts   # Three.js setup
│       │   │   ├── plane-overlay.ts # 3D plane rendering
│       │   │   └── camera.ts        # Camera controls
│       │   ├── audio/
│       │   │   ├── recorder.ts      # Audio recording
│       │   │   └── mixer.ts         # Audio mixing
│       │   ├── db/
│       │   │   ├── client.ts        # Prisma client
│       │   │   └── queries.ts       # Database queries
│       │   └── utils/
│       │       ├── math.ts          # Math utilities
│       │       ├── validation.ts
│       │       └── formatting.ts
│       │
│       ├── hooks/                   # React hooks
│       │   ├── useVideoPlayer.ts
│       │   ├── useAnnotations.ts
│       │   ├── useAudioRecorder.ts
│       │   ├── usePoseDetection.ts
│       │   ├── useFeatureTracking.ts
│       │   └── useWebSocket.ts
│       │
│       ├── types/                   # TypeScript types
│       │   ├── video.ts
│       │   ├── annotation.ts
│       │   ├── ai.ts
│       │   ├── user.ts
│       │   └── database.ts
│       │
│       ├── styles/
│       │   └── globals.css
│       │
│       └── public/
│           ├── models/              # AI model files
│           ├── workers/             # Web workers
│           └── assets/
│
├── packages/
│   ├── database/                    # Prisma schema & migrations
│   │   ├── prisma/
│   │   │   ├── schema.prisma
│   │   │   └── migrations/
│   │   └── package.json
│   │
│   ├── shared/                      # Shared utilities
│   │   ├── src/
│   │   │   ├── constants.ts
│   │   │   ├── types.ts
│   │   │   └── validators.ts
│   │   └── package.json
│   │
│   └── ai-engine/                   # AI processing engine
│       ├── src/
│       │   ├── pose/
│       │   ├── tracking/
│       │   ├── pendulum/
│       │   └── upscaling/
│       └── package.json
│
├── services/
│   ├── api/                         # Backend API (Node.js/Express)
│   │   ├── src/
│   │   │   ├── routes/
│   │   │   ├── controllers/
│   │   │   ├── middleware/
│   │   │   ├── services/
│   │   │   └── server.ts
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   └── ai-worker/                   # Python AI service (FastAPI)
│       ├── src/
│       │   ├── routers/
│       │   ├── models/
│       │   │   ├── real_esrgan/
│       │   │   └── custom_models/
│       │   ├── processing/
│       │   └── main.py
│       ├── requirements.txt
│       └── Dockerfile
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT.md
│   ├── DEVELOPMENT_GUIDE.md
│   ├── AI_MODELS.md
│   └── USER_MANUAL.md
│
├── scripts/
│   ├── setup.sh                     # Initial setup
│   ├── dev.sh                       # Start development servers
│   ├── build.sh                     # Build all services
│   ├── deploy.sh                    # Deploy to production
│   └── seed-db.ts                   # Database seeding
│
├── tests/
│   ├── e2e/                         # End-to-end tests (Playwright)
│   ├── integration/                 # Integration tests
│   └── unit/                        # Unit tests
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── deploy-frontend.yml
        ├── deploy-backend.yml
        └── test.yml
```

---

## Key Files & Their Purpose

### Root Configuration

#### `package.json`
```json
{
  "name": "golf-swing-analyzer",
  "version": "0.1.0",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*",
    "services/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "type-check": "turbo run type-check"
  },
  "devDependencies": {
    "turbo": "^1.10.0",
    "typescript": "^5.0.0",
    "prettier": "^3.0.0",
    "eslint": "^8.0.0"
  }
}
```

#### `turbo.json` (Turborepo configuration)
```json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {},
    "type-check": {}
  }
}
```

#### `tsconfig.json` (Base TypeScript config)
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "preserve",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "incremental": true,
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./components/*"],
      "@/lib/*": ["./lib/*"],
      "@/types/*": ["./types/*"]
    }
  }
}
```

---

## Core Module Descriptions

### 1. Video Processing Module

**Location**: `apps/web/lib/video/`

**Key Files**:
- `ffmpeg.ts`: FFmpeg.wasm initialization and operations
- `processor.ts`: High-level video processing functions
- `formats.ts`: Format detection and conversion

**Responsibilities**:
- Video upload and validation
- Format conversion
- Trimming, cropping, rotation
- Thumbnail generation
- Video concatenation
- Export to various formats

**Example Interface**:
```typescript
export interface VideoProcessor {
  loadVideo(file: File): Promise<VideoMetadata>;
  trim(start: number, end: number): Promise<Blob>;
  crop(rect: Rectangle): Promise<Blob>;
  rotate(degrees: number): Promise<Blob>;
  generateThumbnail(timestamp: number): Promise<Blob>;
  exportVideo(options: ExportOptions): Promise<Blob>;
}
```

### 2. Drawing & Annotation Module

**Location**: `apps/web/lib/canvas/`

**Key Files**:
- `fabric-setup.ts`: Fabric.js canvas initialization
- `annotations.ts`: Annotation management
- `tools.ts`: Drawing tool implementations
- `export.ts`: Canvas export utilities

**Responsibilities**:
- Canvas overlay management
- Drawing tools (line, arrow, freehand, text)
- Layer management
- Undo/redo system
- Annotation persistence
- Export overlays with video

**Example Interface**:
```typescript
export interface AnnotationManager {
  addAnnotation(type: AnnotationType, data: AnnotationData): void;
  removeAnnotation(id: string): void;
  updateAnnotation(id: string, data: Partial<AnnotationData>): void;
  getAnnotationsForFrame(frame: number): Annotation[];
  exportAnnotations(): AnnotationExport;
  importAnnotations(data: AnnotationExport): void;
}
```

### 3. AI & Computer Vision Module

**Location**: `apps/web/lib/ai/`

**Key Files**:
- `mediapipe.ts`: MediaPipe initialization
- `pose-detection.ts`: Pose estimation
- `tracking.ts`: Feature tracking (OpenCV.js)
- `pendulum.ts`: Pendulum model fitting
- `club-detection.ts`: Custom club detection model

**Responsibilities**:
- Real-time pose detection
- Multi-frame pose tracking
- Feature point tracking
- Pendulum model creation and fitting
- Club detection and tracking
- Swing event detection

**Example Interface**:
```typescript
export interface PoseDetector {
  initialize(): Promise<void>;
  detectPose(frame: ImageData): Promise<MediaPipePose>;
  detectPosesInVideo(video: HTMLVideoElement): AsyncGenerator<MediaPipePose>;
  dispose(): void;
}

export interface PendulumFitter {
  createModel(segments: SegmentDefinition[]): PendulumModel;
  fit(poses: MediaPipePose[], constraints: PhysicsConstraints): FittedModel;
  simulate(model: FittedModel, initialConditions: State): Trajectory;
}
```

### 4. 3D Visualization Module

**Location**: `apps/web/lib/three/`

**Key Files**:
- `scene-setup.ts`: Three.js scene initialization
- `plane-overlay.ts`: 3D plane rendering
- `camera.ts`: Camera controls and perspective
- `sync.ts`: Synchronization with 2D video

**Responsibilities**:
- 3D scene management
- Plane overlay rendering
- Camera perspective adjustment
- Coordinate transformation (2D ↔ 3D)
- Real-time manipulation

**Example Interface**:
```typescript
export interface PlaneOverlay {
  create(options: PlaneOptions): Plane3D;
  update(plane: Plane3D, changes: Partial<PlaneOptions>): void;
  projectToScreen(plane: Plane3D, camera: Camera): Polygon2D;
  adjustTilt(plane: Plane3D, delta: number): void;
  adjustRoll(plane: Plane3D, delta: number): void;
  fitToPoints(points: Vector3[]): Plane3D;
}
```

### 5. Audio Recording Module

**Location**: `apps/web/lib/audio/`

**Key Files**:
- `recorder.ts`: Audio recording using MediaRecorder API
- `mixer.ts`: Audio mixing with video
- `waveform.ts`: Waveform visualization
- `sync.ts`: Audio-video synchronization

**Responsibilities**:
- Microphone access and recording
- Real-time recording during playback
- Waveform visualization
- Audio mixing with video
- Multiple audio track management

**Example Interface**:
```typescript
export interface AudioRecorder {
  startRecording(videoTimestamp: number): Promise<void>;
  stopRecording(): Promise<AudioBlob>;
  pauseRecording(): void;
  resumeRecording(): void;
  getWaveformData(): Float32Array;
  dispose(): void;
}
```

### 6. Real-Time Collaboration Module

**Location**: `apps/web/lib/websocket/`

**Key Files**:
- `client.ts`: Socket.io client
- `collaboration.ts`: Collaborative editing
- `presence.ts`: User presence tracking
- `sync.ts`: State synchronization

**Responsibilities**:
- WebSocket connection management
- Real-time annotation sharing
- Cursor/presence tracking
- Conflict resolution
- Event broadcasting

**Example Interface**:
```typescript
export interface CollaborationManager {
  connect(projectId: string): Promise<void>;
  disconnect(): void;
  broadcastAnnotation(annotation: Annotation): void;
  subscribeToAnnotations(callback: (annotation: Annotation) => void): void;
  updatePresence(cursor: CursorPosition): void;
  subscribeToPresence(callback: (users: UserPresence[]) => void): void;
}
```

---

## Database Schema Details

### Core Models

```prisma
// packages/database/prisma/schema.prisma

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

generator client {
  provider = "prisma-client-js"
}

model User {
  id            String    @id @default(cuid())
  email         String    @unique
  name          String
  avatar        String?
  role          Role      @default(COACH)

  // Relationships
  projects      Project[]
  sharedProjects ProjectShare[]

  // Timestamps
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  lastLoginAt   DateTime?

  @@index([email])
}

model Project {
  id          String   @id @default(cuid())
  name        String
  description String?

  // Ownership
  userId      String
  user        User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Content
  videos      Video[]
  shares      ProjectShare[]

  // Settings
  settings    Json     @default("{}")

  // Timestamps
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  @@index([userId])
}

model Video {
  id              String   @id @default(cuid())

  // Project relationship
  projectId       String
  project         Project  @relation(fields: [projectId], references: [id], onDelete: Cascade)

  // File information
  originalUrl     String   // S3 URL
  processedUrl    String?  // Processed version URL
  thumbnailUrl    String?
  filename        String
  mimeType        String
  size            Int      // Bytes

  // Video metadata
  duration        Float    // Seconds
  width           Int      // Pixels
  height          Int      // Pixels
  fps             Float    // Frames per second
  codec           String?

  // Content
  annotations     Annotation[]
  audioTracks     AudioTrack[]
  segments        VideoSegment[]
  aiAnalysis      AiAnalysis?

  // Processing status
  processingStatus ProcessingStatus @default(PENDING)
  processingError String?

  // Timestamps
  uploadedAt      DateTime @default(now())
  processedAt     DateTime?

  @@index([projectId])
}

model Annotation {
  id          String          @id @default(cuid())

  // Video relationship
  videoId     String
  video       Video           @relation(fields: [videoId], references: [id], onDelete: Cascade)

  // Annotation details
  type        AnnotationType
  data        Json            // Type-specific data

  // Timing
  startFrame  Int
  endFrame    Int?            // Null for single-frame annotations

  // Appearance
  style       Json            @default("{}")

  // Metadata
  createdBy   String?         // User ID
  createdAt   DateTime        @default(now())
  updatedAt   DateTime        @updatedAt

  @@index([videoId, startFrame])
}

model AudioTrack {
  id          String   @id @default(cuid())

  // Video relationship
  videoId     String
  video       Video    @relation(fields: [videoId], references: [id], onDelete: Cascade)

  // Audio file
  url         String   // S3 URL
  filename    String
  mimeType    String
  size        Int
  duration    Float

  // Timing
  startTime   Float    @default(0) // Seconds into video

  // Metadata
  recordedBy  String?  // User ID
  recordedAt  DateTime @default(now())

  @@index([videoId])
}

model VideoSegment {
  id          String       @id @default(cuid())

  // Video relationship
  videoId     String
  video       Video        @relation(fields: [videoId], references: [id], onDelete: Cascade)

  // Segment definition
  type        SegmentType
  name        String?
  startTime   Float        // Seconds
  endTime     Float        // Seconds

  // AI-detected or manual
  isAiGenerated Boolean   @default(false)
  confidence    Float?     // For AI-generated segments

  // Metadata
  createdAt   DateTime    @default(now())

  @@index([videoId])
}

model AiAnalysis {
  id              String   @id @default(cuid())

  // Video relationship (one-to-one)
  videoId         String   @unique
  video           Video    @relation(fields: [videoId], references: [id], onDelete: Cascade)

  // Analysis data
  poseData        Json     // Array of poses per frame
  pendulumModel   Json?    // Fitted pendulum parameters
  trackingData    Json?    // Feature tracking results
  swingEvents     Json?    // Detected swing events

  // Processing metadata
  modelVersions   Json     // Track which models were used
  processingTime  Float    // Seconds
  confidence      Float?   // Overall confidence score

  // Timestamps
  analyzedAt      DateTime @default(now())
}

model ProjectShare {
  id          String       @id @default(cuid())

  // Relationships
  projectId   String
  project     Project      @relation(fields: [projectId], references: [id], onDelete: Cascade)

  userId      String?      // Null for public links
  user        User?        @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Access control
  permission  Permission   @default(VIEW)
  shareToken  String       @unique @default(cuid())

  // Expiration
  expiresAt   DateTime?

  // Usage tracking
  accessCount Int          @default(0)
  lastAccessedAt DateTime?

  // Timestamps
  createdAt   DateTime     @default(now())

  @@index([projectId])
  @@index([shareToken])
}

// Enums
enum Role {
  COACH
  STUDENT
  ADMIN
}

enum ProcessingStatus {
  PENDING
  PROCESSING
  COMPLETED
  FAILED
}

enum AnnotationType {
  LINE
  ARROW
  CIRCLE
  RECTANGLE
  FREEHAND
  TEXT
  PLANE_3D
  POINT
}

enum SegmentType {
  ADDRESS
  BACKSWING
  TOP_OF_BACKSWING
  DOWNSWING
  IMPACT
  FOLLOW_THROUGH
  FINISH
  CUSTOM
}

enum Permission {
  VIEW
  COMMENT
  EDIT
  ADMIN
}
```

---

## API Routes Structure

### REST API Endpoints

```
POST   /api/auth/register          - Create account
POST   /api/auth/login             - Login
POST   /api/auth/logout            - Logout
GET    /api/auth/session           - Get current session

GET    /api/projects               - List user's projects
POST   /api/projects               - Create new project
GET    /api/projects/:id           - Get project details
PATCH  /api/projects/:id           - Update project
DELETE /api/projects/:id           - Delete project

POST   /api/projects/:id/videos    - Upload video to project
GET    /api/videos/:id             - Get video details
PATCH  /api/videos/:id             - Update video metadata
DELETE /api/videos/:id             - Delete video

GET    /api/videos/:id/annotations - List annotations
POST   /api/videos/:id/annotations - Create annotation
PATCH  /api/annotations/:id        - Update annotation
DELETE /api/annotations/:id        - Delete annotation

POST   /api/videos/:id/audio       - Upload audio commentary
DELETE /api/audio/:id               - Delete audio track

POST   /api/ai/pose-detect         - Run pose detection
POST   /api/ai/track-features      - Track features
POST   /api/ai/fit-pendulum        - Fit pendulum model
POST   /api/ai/upscale             - Upscale video (premium)

POST   /api/shares                 - Create share link
GET    /api/shares/:token          - Access shared content
DELETE /api/shares/:id             - Revoke share

POST   /api/export                 - Export video with overlays
GET    /api/export/:id/status      - Check export status
GET    /api/export/:id/download    - Download export
```

### WebSocket Events

```
// Client → Server
connect                             - Establish connection
join-project                        - Join project room
leave-project                       - Leave project room
annotation-add                      - Add annotation
annotation-update                   - Update annotation
annotation-delete                   - Delete annotation
cursor-move                         - Update cursor position
video-seek                          - Sync video position

// Server → Client
user-joined                         - User joined project
user-left                           - User left project
annotation-added                    - Annotation added by other user
annotation-updated                  - Annotation updated
annotation-deleted                  - Annotation deleted
cursor-updated                      - User's cursor moved
video-synced                        - Video position synced
```

---

## Development Workflow

### Initial Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd golf-swing-analyzer

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your values

# 4. Set up database
npm run db:generate
npm run db:migrate
npm run db:seed

# 5. Start development servers
npm run dev
```

### Development Commands

```bash
# Start all services in development mode
npm run dev

# Start specific service
npm run dev --filter=web
npm run dev --filter=api
npm run dev --filter=ai-worker

# Build for production
npm run build

# Run tests
npm run test
npm run test:e2e
npm run test:unit

# Lint and format
npm run lint
npm run format

# Type checking
npm run type-check

# Database operations
npm run db:generate     # Generate Prisma client
npm run db:migrate      # Run migrations
npm run db:studio       # Open Prisma Studio
npm run db:seed         # Seed database
```

---

## Testing Strategy

### Unit Tests
- **Location**: Colocated with source files (`*.test.ts`)
- **Framework**: Vitest
- **Coverage**: > 80% for business logic

### Integration Tests
- **Location**: `tests/integration/`
- **Framework**: Vitest + Supertest
- **Focus**: API endpoints, database operations

### End-to-End Tests
- **Location**: `tests/e2e/`
- **Framework**: Playwright
- **Focus**: Critical user flows

### Example Test Structure

```typescript
// apps/web/lib/video/processor.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { VideoProcessor } from './processor';

describe('VideoProcessor', () => {
  let processor: VideoProcessor;

  beforeEach(() => {
    processor = new VideoProcessor();
  });

  describe('trim', () => {
    it('should trim video to specified range', async () => {
      const mockVideo = await loadMockVideo();
      const trimmed = await processor.trim(mockVideo, 1, 5);

      expect(trimmed.duration).toBe(4); // 5 - 1
    });

    it('should throw error for invalid range', async () => {
      const mockVideo = await loadMockVideo();

      await expect(
        processor.trim(mockVideo, 5, 1)
      ).rejects.toThrow('Start time must be before end time');
    });
  });
});
```

---

## Deployment Architecture

### Production Environment

```
┌─────────────────────────────────────────────────┐
│                 Cloudflare CDN                   │
│              (DNS, WAF, DDoS Protection)        │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────┐             ┌──────────┐
│ Vercel  │             │ Railway  │
│ (Web)   │             │ (API +   │
│         │◄───────────►│  AI)     │
└────┬────┘             └─────┬────┘
     │                        │
     │    ┌───────────────────┘
     │    │
     ▼    ▼
  ┌──────────┐
  │PostgreSQL│
  │(Railway) │
  └──────────┘
     │
     ▼
  ┌──────────┐
  │   S3/R2  │
  │ (Video   │
  │ Storage) │
  └──────────┘
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run test

  deploy-frontend:
    needs: [lint, test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}

  deploy-backend:
    needs: [lint, test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: api
```

---

## Next Steps

1. **Set up repository** with this structure
2. **Initialize core modules** (video, canvas, AI)
3. **Build MVP features** (upload, playback, basic drawing)
4. **Integrate AI capabilities** (pose detection, tracking)
5. **Add advanced features** (3D overlays, pendulum)
6. **Polish UI/UX**
7. **Deploy to production**

---

*This structure provides a scalable, maintainable foundation for your golf swing video analysis platform.*
