# Setup Guide - Golf Swing Video Analysis Platform

## ✅ Current Status

**Branch**: `feat/golf-swing-video-platform`
**Status**: Project structure created, ready for implementation

## 📁 What's Been Created

### Project Structure
✅ Monorepo structure (Turborepo)
✅ Next.js app scaffold
✅ TypeScript configuration
✅ Tailwind CSS setup
✅ Database schema (Prisma)
✅ MATLAB integration folder
✅ Folder structure for all components

### Files Created
- `package.json` - Root package.json with workspaces
- `turbo.json` - Turborepo configuration
- `apps/web/` - Next.js application
- `packages/database/` - Prisma schema
- `matlab/` - MATLAB models directory
- `docs/` - All documentation

## 🚀 Next Steps

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Set Up Database

```bash
# Install Prisma CLI
npm install -D prisma

# Generate Prisma client
cd packages/database
npx prisma generate

# Create database (if not exists)
# Then run migrations
npx prisma migrate dev
```

### 4. Start Development Server

```bash
npm run dev
```

Visit http://localhost:3000

## 📚 Documentation

All documentation is in the `docs/` folder:

1. **Start Here**: `docs/GOLF_VIDEO_QUICK_REFERENCE.md`
2. **Budget**: `docs/GOLF_VIDEO_BUDGET_GUIDE.md` (for home developers)
3. **Tech Stack**: `docs/GOLF_VIDEO_EDITOR_TECH_STACK.md`
4. **Structure**: `docs/GOLF_VIDEO_PROJECT_STRUCTURE.md`
5. **Action Plan**: `docs/GOLF_VIDEO_ACTION_PLAN.md`
6. **MATLAB**: `docs/GOLF_VIDEO_MATLAB_INTEGRATION.md`
7. **Languages**: `docs/GOLF_VIDEO_PROGRAMMING_LANGUAGES.md`

## 🎯 Priority Features to Build First

### Phase 1: MVP (Start Here)

1. **Video Upload** (`apps/web/components/video/VideoUploader.tsx`)
   - Drag-and-drop upload
   - File validation
   - Upload to Cloudflare R2

2. **Video Player** (`apps/web/components/video/VideoPlayer.tsx`)
   - Playback controls
   - Timeline scrubber
   - Frame-by-frame navigation

3. **Basic Drawing** (`apps/web/components/tools/LineTool.tsx`)
   - Line drawing on video
   - Save annotations

4. **User Authentication** (`apps/web/app/(auth)/`)
   - NextAuth.js setup
   - Email/password auth

5. **Projects** (`apps/web/app/(dashboard)/projects/`)
   - Create projects
   - List projects
   - Basic CRUD

## 💡 MATLAB Integration

### For You (Developer)

1. **Develop Models in MATLAB**:
   ```matlab
   cd matlab/models
   % Create Simscape Multibody pendulum model
   % Test with sample data
   ```

2. **Export Results**:
   ```matlab
   results = pendulum_model(pose_data);
   export_results_to_web(results, 'results.json');
   ```

3. **Import in Web Platform**:
   ```typescript
   import { loadMATLABResults } from '@/lib/matlab/import-results';
   const results = await loadMATLABResults('/matlab/results.json');
   ```

See `docs/GOLF_VIDEO_MATLAB_INTEGRATION.md` for complete guide.

## 🛠️ Development Tools

### Recommended VS Code Extensions

- ESLint
- Prettier
- Tailwind CSS IntelliSense
- Prisma
- TypeScript and JavaScript Language Features

### Useful Commands

```bash
# Development
npm run dev              # Start all services
npm run dev --filter=web # Start only web app

# Building
npm run build           # Build all packages

# Testing
npm run test           # Run all tests

# Database
cd packages/database
npx prisma studio      # Open Prisma Studio (database GUI)
npx prisma migrate dev # Run migrations
```

## 🎨 UI Components

We'll use **shadcn/ui** for UI components. To set up:

```bash
cd apps/web
npx shadcn-ui@latest init
```

Then install components as needed:
```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add dialog
# etc.
```

## 📦 Key Dependencies to Install

### Core (Will install when needed)
```bash
# Video processing
npm install @ffmpeg/ffmpeg @ffmpeg/core

# Drawing
npm install fabric

# 3D
npm install three @types/three

# AI
npm install @mediapipe/pose @mediapipe/drawing_utils
npm install @tensorflow/tfjs

# Database
npm install @prisma/client
npm install prisma --save-dev

# Auth
npm install next-auth

# UI
npx shadcn-ui@latest init
```

Install these as you build each feature (don't install everything at once).

## ✅ Checklist

### Setup
- [x] Project structure created
- [x] TypeScript configured
- [x] Next.js scaffold ready
- [x] Database schema defined
- [x] MATLAB folder structure
- [ ] Install dependencies (`npm install`)
- [ ] Set up environment variables
- [ ] Set up database
- [ ] Set up Cloudflare R2 (for video storage)
- [ ] Set up Supabase (for database/auth)

### Phase 1 Features
- [ ] Video upload
- [ ] Video playback
- [ ] Basic drawing tools
- [ ] User authentication
- [ ] Project management
- [ ] Basic sharing

### Phase 2 Features
- [ ] AI pose detection
- [ ] Feature tracking
- [ ] Swing segmentation

### Phase 3 Features
- [ ] 3D plane overlays
- [ ] Pendulum model (MATLAB integration)
- [ ] Camera perspective

### Phase 4 Features
- [ ] Video upscaling
- [ ] Advanced collaboration
- [ ] Mobile optimization

## 🚀 Start Building!

1. **Install dependencies**: `npm install`
2. **Read Quick Reference**: `docs/GOLF_VIDEO_QUICK_REFERENCE.md`
3. **Start with MVP features** (video upload → playback → drawing)
4. **Add MATLAB models** as you develop physics features
5. **Iterate based on feedback**

Good luck! 🚀⛳
