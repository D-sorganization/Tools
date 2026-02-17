# Golf Swing Video Analysis Platform - Quick Reference

## 🎯 One-Sentence Summary

**Build a web-based golf swing video analysis platform using Next.js + MediaPipe + Three.js to enable coaches to analyze swings with AI-powered pose detection, 3D overlays, and audio commentary, then share with students.**

---

## 💡 Core Recommendation

### Technology Stack

```
Frontend:  Next.js + React + TypeScript + Tailwind CSS
Video:     FFmpeg.wasm + Video.js
Drawing:   Fabric.js
3D:        Three.js
AI:        MediaPipe + TensorFlow.js + OpenCV.js
Backend:   Node.js + Express + PostgreSQL + Prisma
Storage:   AWS S3 or Cloudflare R2
Deploy:    Vercel (frontend) + Railway (backend)
```

### Why Web-Based?

✅ Universal access (any device)
✅ No installation required
✅ Easy sharing via links
✅ Automatic updates
✅ Single codebase

---

## 📋 Feature Checklist with Solutions

| Your Requirement   | Technology Solution           | Complexity  |
| ------------------ | ----------------------------- | ----------- |
| Edit & share video | Next.js + FFmpeg.wasm         | ⭐ Easy     |
| Audio commentary   | Web Audio API + MediaRecorder | ⭐ Easy     |
| Draw lines/planes  | Fabric.js                     | ⭐ Easy     |
| 3D plane overlays  | Three.js                      | ⭐⭐ Medium |
| Camera perspective | Three.js + OpenCV.js          | ⭐⭐ Medium |
| Motion capture     | MediaPipe (Google)            | ⭐⭐ Medium |
| Pendulum model     | Custom + Matter.js            | ⭐⭐⭐ Hard |
| Feature tracking   | OpenCV.js                     | ⭐⭐ Medium |
| Video upscaling    | Real-ESRGAN (backend)         | ⭐⭐⭐ Hard |
| Crop/rotate/trim   | FFmpeg.wasm                   | ⭐ Easy     |
| All video formats  | FFmpeg.wasm                   | ⭐ Easy     |

---

## 🚀 4-Phase Development Plan

### Phase 1: MVP (10 weeks)

**Goal**: Upload, playback, drawing, audio, sharing

**Features**:

- Video upload (all formats)
- Playback with timeline
- Drawing tools (line, arrow, text, freehand)
- Audio commentary recording
- Trim/crop/rotate
- Export annotated videos
- User accounts
- Share via links

**Cost**: $40-60k (outsourced) | **$0 + 3 months** (home developer)
**Monthly Cost**: **$1-2/month**

### Phase 2: AI (8 weeks)

**Goal**: Intelligent motion tracking and analysis

**Features**:

- Pose detection (33 landmarks)
- Feature tracking
- Club detection & tracking
- Auto swing segmentation
- AI insights dashboard

**Cost**: $35-50k (outsourced) | **$0 + 2 months** (home developer)
**Monthly Cost**: **$2-5/month**

### Phase 3: 3D & Physics (8 weeks)

**Goal**: 3D visualization and physics modeling

**Features**:

- 3D plane overlays (adjustable)
- Camera perspective estimation
- Triple pendulum model
- 3D swing path visualization
- Physics-based analysis

**Cost**: $40-60k (outsourced) | **$0 + 2 months** (home developer)
**Monthly Cost**: **$5-10/month**

### Phase 4: Premium & Polish (6 weeks)

**Goal**: Advanced features and production readiness

**Features**:

- AI video upscaling
- Mobile optimization
- Real-time collaboration
- Performance optimization
- Final polish

**Cost**: $25-40k (outsourced) | **$0 + 1.5 months** (home developer)
**Monthly Cost**: **$10-30/month**

**Total**: 32 weeks | $140-210k (outsourced) | **$0 + 8-9 months** (home developer)

**💰 See `docs/GOLF_VIDEO_BUDGET_GUIDE.md` for detailed cost breakdown**

---

## 💰 Cost Summary

### Development Costs

**⚠️ IMPORTANT**: If you're developing yourself, development cost = **$0** (your time)!

| Approach                 | Cost                          | Timeline    |
| ------------------------ | ----------------------------- | ----------- |
| **Home Developer (You)** | **$0** (your time)            | 8-9 months  |
| **Hiring Agency**        | $140-210k fixed               | 6-8 months  |
| **Hiring Developers**    | Salaries (~$200k/year loaded) | 9-12 months |

### Monthly Operating Costs (Infrastructure)

**These are the REAL costs - much cheaper than listed elsewhere!**

| Users                    | Monthly Cost     | Services Used                      |
| ------------------------ | ---------------- | ---------------------------------- |
| **0-50 (MVP)**           | **$1-2/month**   | Domain only, everything else FREE! |
| **50-200 (Beta)**        | **$2-5/month**   | Mostly free tiers                  |
| **200-1,000 (Growth)**   | **$5-10/month**  | Some paid tiers                    |
| **1,000-10,000 (Scale)** | **$30-60/month** | Paid tiers but still cheap         |

**📖 For detailed budget breakdown: `docs/GOLF_VIDEO_BUDGET_GUIDE.md`**

### Why So Cheap?

- ✅ **Vercel**: FREE (enough for 100k+ users!)
- ✅ **Supabase**: FREE (500MB database = thousands of users)
- ✅ **Cloudflare R2**: FREE (10GB storage, then $0.015/GB)
- ✅ **Processing**: Done in browser (no server costs!)

---

## 🎨 Key Technical Components

### 1. Video Processing

```typescript
// Using FFmpeg.wasm for browser-side video processing
import { createFFmpeg, fetchFile } from "@ffmpeg/ffmpeg";

const ffmpeg = createFFmpeg({ log: true });
await ffmpeg.load();

// Trim video
ffmpeg.FS("writeFile", "input.mp4", await fetchFile(videoFile));
await ffmpeg.run(
  "-i",
  "input.mp4",
  "-ss",
  "00:00:01",
  "-to",
  "00:00:05",
  "output.mp4",
);
const data = ffmpeg.FS("readFile", "output.mp4");
```

### 2. Pose Detection

```typescript
// Using MediaPipe for body tracking
import { Pose } from "@mediapipe/pose";

const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
});

pose.onResults((results) => {
  const landmarks = results.poseLandmarks;
  const leftWrist = landmarks[15];
  const leftShoulder = landmarks[11];
  // Analyze swing positions
});
```

### 3. Drawing Annotations

```typescript
// Using Fabric.js for canvas overlays
import { fabric } from "fabric";

const canvas = new fabric.Canvas("canvas");

// Draw line
const line = new fabric.Line([x1, y1, x2, y2], {
  stroke: "red",
  strokeWidth: 2,
});
canvas.add(line);

// Save annotations per frame
const annotations = canvas.toJSON();
```

### 4. 3D Plane Overlay

```typescript
// Using Three.js for 3D visualization
import * as THREE from "three";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);

// Create plane
const geometry = new THREE.PlaneGeometry(10, 10);
const material = new THREE.MeshBasicMaterial({
  color: 0xff0000,
  transparent: true,
  opacity: 0.3,
});
const plane = new THREE.Mesh(geometry, material);
scene.add(plane);
```

### 5. Audio Recording

```typescript
// Using Web Audio API for commentary
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: "audio/webm;codecs=opus",
});

const chunks: Blob[] = [];
mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

mediaRecorder.onstop = () => {
  const audioBlob = new Blob(chunks, { type: "audio/webm" });
  // Save or mix with video
};

mediaRecorder.start();
```

---

## 📊 Success Criteria

### MVP Success (Phase 1)

- [ ] 50 beta users (coaches)
- [ ] 500 videos processed
- [ ] < 2 second video load time
- [ ] 80% share at least 1 video
- [ ] Net Promoter Score > 40

### Full Platform Success (Phase 4)

- [ ] 1,000+ active users
- [ ] 10,000+ videos analyzed
- [ ] 99% uptime
- [ ] NPS > 60
- [ ] Revenue positive

---

## 🛠️ Starter Commands

### Create New Project

```bash
# Create Next.js app with TypeScript
npx create-next-app@latest golf-swing-analyzer --typescript --tailwind --app

# Navigate to project
cd golf-swing-analyzer

# Install core dependencies
npm install @ffmpeg/ffmpeg @ffmpeg/core
npm install fabric
npm install three @types/three
npm install @mediapipe/pose @mediapipe/drawing_utils
npm install video.js

# Install backend dependencies
npm install prisma @prisma/client
npm install next-auth

# Set up Prisma
npx prisma init

# Install UI library
npx shadcn-ui@latest init
```

### Development Workflow

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm run test

# Lint
npm run lint

# Type check
npm run type-check
```

---

## 📁 Project Structure (Simplified)

```
golf-swing-analyzer/
├── app/                      # Next.js app router
│   ├── (dashboard)/         # Dashboard pages
│   ├── api/                 # API routes
│   └── share/               # Public sharing pages
├── components/              # React components
│   ├── video/               # Video player, timeline
│   ├── editor/              # Editor canvas, tools
│   ├── ai/                  # AI features
│   └── ui/                  # UI components (shadcn)
├── lib/                     # Core logic
│   ├── video/               # Video processing (FFmpeg)
│   ├── canvas/              # Drawing (Fabric.js)
│   ├── ai/                  # AI models (MediaPipe)
│   ├── three/               # 3D (Three.js)
│   └── db/                  # Database (Prisma)
├── prisma/                  # Database schema
└── public/                  # Static assets
```

---

## 🎯 Critical Design Decisions

### 1. Web vs. Desktop vs. Mobile?

**Decision**: **Web-first**
**Reason**: Maximum reach, easy sharing, single codebase

### 2. Real-time vs. Batch Processing?

**Decision**: **Hybrid**
**Reason**: Lightweight processing (pose detection) in browser, heavy processing (upscaling) on server

### 3. Subscription vs. One-time Purchase?

**Decision**: **Freemium subscription**
**Reason**: Better LTV, aligns with ongoing value (cloud storage, AI features)

### 4. Build vs. Buy AI Models?

**Decision**: **Use existing (MediaPipe)**
**Reason**: Faster time-to-market, proven accuracy, free

### 5. Monorepo vs. Separate Repos?

**Decision**: **Monorepo (Turborepo)**
**Reason**: Shared code, easier refactoring, better DX

---

## ⚠️ Key Risks & Mitigations

| Risk                            | Mitigation                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| **AI performance in browser**   | Use Web Workers, lazy loading, progressive enhancement       |
| **Pose detection accuracy**     | Multiple models, confidence thresholds, manual correction UI |
| **Video storage costs**         | Compression, CDN, tiered storage, user limits                |
| **Pendulum model complexity**   | Start with double pendulum, allow manual adjustment          |
| **Real-time collaboration lag** | Optimistic UI updates, websocket with fallback               |

---

## 📚 Essential Resources

### Documentation

- [Next.js Docs](https://nextjs.org/docs)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Three.js Manual](https://threejs.org/manual/)
- [FFmpeg.wasm](https://ffmpegwasm.netlify.app/)

### Tools

- [Vercel](https://vercel.com) - Deployment
- [Railway](https://railway.app) - Backend hosting
- [Prisma Studio](https://www.prisma.io/studio) - Database GUI
- [Postman](https://www.postman.com) - API testing

### Communities

- [Next.js Discord](https://nextjs.org/discord)
- [Three.js Discourse](https://discourse.threejs.org/)
- [r/webdev](https://reddit.com/r/webdev)

---

## 🚦 Decision Framework

### Should I Start with MVP or Build Everything?

**Start with MVP.** Get feedback, validate market fit, iterate.

### Should I Hire Agency or Build In-House?

- **Agency**: If you have $150k+ and want fast (6 months)
- **In-House**: If you have technical co-founder or are technical
- **Hybrid**: MVP with agency ($40-60k), then hire developers

### Should I Use MediaPipe or Train Custom Models?

**MediaPipe first.** 90% accuracy out-of-box. Train custom only if needed.

### Should I Build 3D Features in MVP?

**No.** Validate core video analysis first. 3D is Phase 3.

### Should I Support Mobile from Day 1?

**Make it responsive, but optimize for desktop first.** Coaches likely use desktops/laptops for analysis.

---

## 🎬 Next 3 Steps

1. **Validate** (1 week)

   - Interview 5-10 golf coaches
   - Confirm feature priorities
   - Test willingness to pay ($29-99/month)

2. **Build POC** (2 weeks)

   - Set up Next.js project
   - Implement video upload + playback
   - Integrate MediaPipe (pose detection)
   - Demo to 3 coaches, get feedback

3. **Execute MVP** (10 weeks)
   - Full Phase 1 development
   - Weekly progress reviews
   - Beta launch with 20-50 coaches

---

## 💬 Sample Pricing (for reference)

### Freemium Model

- **Free**: 3 videos/month, basic features, watermark
- **Coach ($29/month)**: 25 videos/month, AI features, no watermark, 5GB storage
- **Pro Coach ($79/month)**: Unlimited videos, 3D analysis, 50GB storage, priority support
- **Academy ($249/month)**: Unlimited coaches, unlimited videos, team features, 500GB storage

### Add-Ons

- **Video Upscaling**: $5 per video
- **Extra Storage**: $10/month per 10GB
- **White Label**: $500/month (remove branding)

---

## ✅ Pre-Launch Checklist

### Technical

- [ ] All core features working
- [ ] 99%+ uptime (tested)
- [ ] < 2s page load time
- [ ] < 5% error rate
- [ ] Mobile responsive
- [ ] Security audit passed
- [ ] GDPR compliant
- [ ] Backups configured

### Business

- [ ] Pricing determined
- [ ] Payment processing (Stripe)
- [ ] Terms of Service
- [ ] Privacy Policy
- [ ] Support system (email/chat)
- [ ] Marketing website
- [ ] Demo videos
- [ ] Onboarding flow

### Launch

- [ ] 20+ beta users committed
- [ ] Analytics configured (PostHog/Mixpanel)
- [ ] Error tracking (Sentry)
- [ ] Monitoring (Uptime Robot)
- [ ] Launch plan documented
- [ ] Press release (optional)

---

## 🎯 Remember

1. **Start small, iterate fast** - Don't build everything at once
2. **Talk to users constantly** - Build what they need, not what you think is cool
3. **Focus on UX** - Golf coaches are not tech experts
4. **Make sharing frictionless** - This is your viral loop
5. **Charge early** - Validate willingness to pay

---

## 📞 Questions?

This platform is 100% technically feasible with the recommended stack. All features can be built using open-source technologies and modern web APIs.

**You have everything you need to start building. Good luck! 🚀⛳**

---

_Last Updated: November 1, 2025_
