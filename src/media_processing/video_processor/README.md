# Golf Swing Video Analysis Platform

Advanced AI-powered golf swing video analysis platform for coaches to analyze and share swing videos with students.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm 9+
- (Optional) MATLAB with Simscape Multibody for physics modeling
- Git

### Installation

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run development server
npm run dev
```

### Development

```bash
# Start all services
npm run dev

# Start specific workspace
npm run dev --filter=web

# Build for production
npm run build

# Run tests
npm run test

# Type check
npm run type-check
```

## 📁 Project Structure

```
golf-swing-analyzer/
├── apps/
│   └── web/              # Next.js frontend (TypeScript)
├── packages/
│   ├── database/         # Prisma schema
│   ├── shared/           # Shared utilities
│   └── ai-engine/        # AI processing engine
├── services/
│   ├── api/              # Node.js backend API
│   └── matlab-worker/    # MATLAB integration (optional)
├── matlab/                # MATLAB models (Simscape Multibody)
└── docs/                  # Documentation
```

## 🎯 Features

- ✅ Video upload and playback
- ✅ AI pose detection (MediaPipe)
- ✅ Drawing and annotations
- ✅ Audio commentary recording
- ✅ 3D plane overlays
- ✅ MATLAB integration for physics modeling
- ✅ Video editing (trim, crop, rotate)
- ✅ Sharing and collaboration

## 📚 Documentation

- **Quick Start**: `docs/GOLF_VIDEO_QUICK_REFERENCE.md`
- **Budget Guide**: `docs/GOLF_VIDEO_BUDGET_GUIDE.md` (for home developers)
- **Technology Stack**: `docs/GOLF_VIDEO_EDITOR_TECH_STACK.md`
- **Project Structure**: `docs/GOLF_VIDEO_PROJECT_STRUCTURE.md`
- **Action Plan**: `docs/GOLF_VIDEO_ACTION_PLAN.md`
- **MATLAB Integration**: `docs/GOLF_VIDEO_MATLAB_INTEGRATION.md`
- **Programming Languages**: `docs/GOLF_VIDEO_PROGRAMMING_LANGUAGES.md`

## 💰 Costs

### Development

- **Home Developer**: $0 (your time)
- **If Outsourcing**: $140-210k

### Monthly Operating Costs

- **MVP (0-50 users)**: $1-2/month
- **Growth (50-200 users)**: $2-5/month
- **Scale (200-1,000 users)**: $5-10/month

See `docs/GOLF_VIDEO_BUDGET_GUIDE.md` for detailed breakdown.

## 🛠️ Technology Stack

- **Frontend**: Next.js + React + TypeScript
- **Backend**: Node.js + Express + TypeScript
- **Database**: PostgreSQL + Prisma
- **Storage**: Cloudflare R2 (free 10GB, then $0.015/GB)
- **AI**: MediaPipe + TensorFlow.js (runs in browser)
- **Video**: FFmpeg.wasm (runs in browser)
- **3D**: Three.js
- **Drawing**: Fabric.js
- **Deployment**: Vercel (free) + Supabase (free)

## 🎨 MATLAB Integration

The platform supports MATLAB integration for physics modeling:

- Develop Simscape Multibody models locally
- Export results to JSON
- Import into web platform
- Optional: Python bridge to MATLAB Runtime (no license needed)

See `docs/GOLF_VIDEO_MATLAB_INTEGRATION.md` for details.

## 📝 License

MIT

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome!

---

**Status**: In development
**Branch**: `feat/golf-swing-video-platform`
