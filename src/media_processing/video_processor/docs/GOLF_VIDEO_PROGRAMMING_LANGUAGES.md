# Golf Swing Video Analysis Platform - Programming Languages

## 🎯 Quick Answer

**Yes, primarily JavaScript/TypeScript!**

The entire platform is built in **TypeScript** (typed JavaScript) for both frontend and backend, with **optional Python** only for advanced server-side AI processing.

---

## 📊 Language Breakdown

### Primary: TypeScript (95% of code)

**What is TypeScript?**

- TypeScript is JavaScript with type safety
- Compiles to JavaScript (runs everywhere)
- Better for large projects (catches errors early)
- Same syntax as JavaScript (easy to learn)

**Used For**:

- ✅ Frontend (React/Next.js)
- ✅ Backend API (Node.js/Express)
- ✅ Video processing (FFmpeg.wasm)
- ✅ AI/ML in browser (MediaPipe, TensorFlow.js)
- ✅ Drawing/Canvas (Fabric.js)
- ✅ 3D graphics (Three.js)
- ✅ Database queries (Prisma ORM)

**Why TypeScript?**

- Type safety catches bugs early
- Better IDE autocomplete
- Easier refactoring
- Industry standard for modern web apps

---

## 🗂️ Language Usage by Component

### Frontend (100% TypeScript/JavaScript)

```typescript
// Next.js React components (TypeScript)
// Example: Video player component
import React from 'react';
import { VideoPlayer } from '@/components/video/VideoPlayer';

interface VideoPlayerProps {
  videoUrl: string;
  onPlay: () => void;
  onPause: () => void;
}

export const VideoEditor: React.FC<VideoPlayerProps> = ({ videoUrl }) => {
  return <VideoPlayer src={videoUrl} />;
};
```

**Technologies**:

- **Next.js**: React framework (TypeScript)
- **React**: UI library (TypeScript)
- **Tailwind CSS**: Styling (CSS, but configured via TypeScript)
- **Fabric.js**: Canvas drawing (JavaScript/TypeScript)
- **Three.js**: 3D graphics (JavaScript/TypeScript)

---

### Backend API (100% TypeScript/JavaScript)

```typescript
// Node.js/Express API routes (TypeScript)
// Example: Video upload endpoint
import { NextApiRequest, NextApiResponse } from "next";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method === "POST") {
    const video = await prisma.video.create({
      data: {
        url: req.body.url,
        userId: req.body.userId,
      },
    });
    return res.status(200).json(video);
  }
}
```

**Technologies**:

- **Node.js**: JavaScript runtime
- **Express/Next.js API**: API framework
- **Prisma**: Database ORM (TypeScript)

---

### Video Processing (100% JavaScript/TypeScript)

```typescript
// FFmpeg.wasm for browser-side video processing
import { createFFmpeg, fetchFile } from "@ffmpeg/ffmpeg";

const ffmpeg = createFFmpeg({ log: true });

async function trimVideo(
  videoFile: File,
  startTime: number,
  endTime: number,
): Promise<Blob> {
  await ffmpeg.load();
  ffmpeg.FS("writeFile", "input.mp4", await fetchFile(videoFile));

  await ffmpeg.run(
    "-i",
    "input.mp4",
    "-ss",
    startTime.toString(),
    "-to",
    endTime.toString(),
    "output.mp4",
  );

  const data = ffmpeg.FS("readFile", "output.mp4");
  return new Blob([data.buffer], { type: "video/mp4" });
}
```

**Technologies**:

- **FFmpeg.wasm**: WebAssembly version of FFmpeg (runs in browser)
- All processing done in JavaScript/TypeScript

---

### AI & Computer Vision (100% JavaScript/TypeScript in Browser)

```typescript
// MediaPipe pose detection (runs in browser!)
import { Pose } from "@mediapipe/pose";
import { drawConnections, drawLandmarks } from "@mediapipe/drawing_utils";

const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
});

pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

pose.onResults((results) => {
  if (results.poseLandmarks) {
    // Draw pose on canvas
    drawConnections(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 2,
    });
  }
});
```

**Technologies**:

- **MediaPipe**: Google's AI library (JavaScript/TypeScript)
- **TensorFlow.js**: Machine learning (JavaScript/TypeScript)
- **OpenCV.js**: Computer vision (JavaScript/TypeScript)

**All AI runs in the browser - no server needed!**

---

### Database (SQL, but managed via TypeScript)

```typescript
// Prisma ORM (TypeScript) manages PostgreSQL (SQL)
// You write TypeScript, Prisma generates SQL

// TypeScript schema definition
model Video {
  id        String   @id @default(cuid())
  url       String
  userId    String
  user      User     @relation(fields: [userId], references: [id])
  createdAt DateTime @default(now())
}

// TypeScript query (Prisma generates SQL)
const videos = await prisma.video.findMany({
  where: { userId: currentUser.id },
  include: { user: true }
});
```

**Technologies**:

- **PostgreSQL**: Database (SQL)
- **Prisma**: ORM (TypeScript) - you write TypeScript, Prisma handles SQL

**You write TypeScript, database is SQL underneath**

---

### Optional: Python (Only for Advanced Features)

**When is Python used?**

- ❌ **NOT needed** for MVP or most features
- ✅ **Only** for advanced video upscaling (Real-ESRGAN)
- ✅ **Optional** server-side AI processing

**Example**: Video Upscaling Service (Optional)

```python
# services/ai-worker/src/main.py
# Only needed for advanced video upscaling
# NOT required for MVP or core features

from fastapi import FastAPI, UploadFile
from realesrgan import RealESRGAN
import cv2

app = FastAPI()
model = RealESRGAN(device='cuda')
model.load_weights('weights/RealESRGAN_x4plus.pth')

@app.post("/upscale")
async def upscale_video(file: UploadFile):
    # Upscale video using AI
    # This is OPTIONAL premium feature
    pass
```

**Technologies**:

- **FastAPI**: Python web framework (only for upscaling service)
- **Real-ESRGAN**: Python AI model (only for upscaling)

**Can skip Python entirely if you don't need upscaling!**

---

## 📦 Complete Stack Breakdown

### Frontend (100% TypeScript)

```
✅ Next.js        - TypeScript
✅ React          - TypeScript
✅ Tailwind CSS   - CSS (configured via TypeScript)
✅ Fabric.js      - JavaScript/TypeScript
✅ Three.js       - JavaScript/TypeScript
✅ MediaPipe      - JavaScript/TypeScript
✅ TensorFlow.js  - JavaScript/TypeScript
✅ OpenCV.js      - JavaScript/TypeScript
✅ FFmpeg.wasm    - JavaScript/TypeScript (WASM binary)
```

### Backend (100% TypeScript)

```
✅ Node.js        - JavaScript runtime
✅ Next.js API    - TypeScript
✅ Express        - TypeScript (optional)
✅ Prisma         - TypeScript ORM
✅ PostgreSQL     - SQL (managed via Prisma TypeScript)
```

### Optional Services (Python - Can Skip!)

```
❌ FastAPI        - Python (only for upscaling)
❌ Real-ESRGAN    - Python (only for upscaling)
```

---

## 🎯 What You Need to Know

### For MVP (Phase 1)

**Only JavaScript/TypeScript!**

```typescript
// Everything you need:
- TypeScript/JavaScript (95% of code)
- CSS (styling)
- SQL (database, but managed via Prisma TypeScript)

// NOT needed:
- Python (skip entirely for MVP)
- Any other languages
```

### For Full Platform (Phase 4)

**Still mostly TypeScript, Python only for optional features**

```typescript
// Core features: TypeScript/JavaScript only
// Optional premium feature: Python (video upscaling)

// You can skip Python entirely if you don't need upscaling!
```

---

## 💻 Learning Path

### If You Know JavaScript

**You're 90% there!**

TypeScript is just JavaScript with types:

```typescript
// JavaScript
function greet(name) {
  return `Hello, ${name}!`;
}

// TypeScript (just adds types)
function greet(name: string): string {
  return `Hello, ${name}!`;
}
```

### If You Know Another Language

**TypeScript is easy to learn**

**If you know:**

- Python → TypeScript syntax similar
- Java/C# → TypeScript has types like you're used to
- C++ → TypeScript is simpler
- Ruby → TypeScript is similar but typed

### If You're New to Programming

**Start with JavaScript, then add TypeScript**

**Learning path**:

1. Learn JavaScript basics (1-2 weeks)
2. Learn React (2-3 weeks)
3. Learn TypeScript (1 week - just add types!)
4. Build MVP (3 months)

---

## 📚 Required Skills

### Must Know

- ✅ **JavaScript** (or TypeScript) - Core language
- ✅ **React** - UI framework
- ✅ **HTML/CSS** - Styling

### Should Know

- ✅ **TypeScript** - Adds type safety
- ✅ **Node.js** - Backend runtime
- ✅ **SQL** - Database (basic queries)
- ✅ **Git** - Version control

### Nice to Have (Not Required)

- ⚠️ **Python** - Only if you want server-side upscaling
- ⚠️ **WebAssembly** - FFmpeg.wasm uses it, but you don't need to learn it
- ⚠️ **Docker** - Only if self-hosting

---

## 🔧 Development Setup

### Install (All JavaScript/TypeScript)

```bash
# Install Node.js (JavaScript runtime)
# Download from: https://nodejs.org/

# Create Next.js project with TypeScript
npx create-next-app@latest golf-swing-analyzer --typescript --tailwind --app

# Install dependencies (all JavaScript packages)
npm install

# That's it! Everything is JavaScript/TypeScript
```

### Code Editor

**VS Code** (free) with TypeScript support built-in

- Automatic type checking
- Autocomplete
- Error detection

---

## 🎨 Example: Complete Feature in TypeScript

### Video Upload with Pose Detection (100% TypeScript)

```typescript
// 1. Upload video (TypeScript)
async function uploadVideo(file: File): Promise<string> {
  const formData = new FormData();
  formData.append("video", file);

  const response = await fetch("/api/videos/upload", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  return data.videoUrl;
}

// 2. Process video in browser (TypeScript)
import { createFFmpeg } from "@ffmpeg/ffmpeg";

const ffmpeg = createFFmpeg();
await ffmpeg.load();

// 3. Detect poses in browser (TypeScript)
import { Pose } from "@mediapipe/pose";

const pose = new Pose();
pose.onResults((results) => {
  // Process poses (all TypeScript)
  const landmarks = results.poseLandmarks;
  // Save to database via API (TypeScript)
});

// 4. Draw on canvas (TypeScript)
import { fabric } from "fabric";

const canvas = new fabric.Canvas("canvas");
canvas.add(
  new fabric.Line([x1, y1, x2, y2], {
    stroke: "red",
    strokeWidth: 2,
  }),
);

// 5. Save to database (TypeScript)
const response = await fetch("/api/annotations", {
  method: "POST",
  body: JSON.stringify({ videoId, annotations: canvas.toJSON() }),
});
```

**Everything in TypeScript! No other languages needed.**

---

## ❓ Frequently Asked Questions

### Q: Do I need to know Python?

**A: No!** Python is only needed for optional video upscaling. You can skip it entirely.

### Q: Can I use plain JavaScript instead of TypeScript?

**A: Yes!** TypeScript is just JavaScript with types. You can use plain JavaScript, but TypeScript is recommended for:

- Better error detection
- Better IDE autocomplete
- Easier refactoring

### Q: Do I need to know SQL?

**A: Basic knowledge helps**, but Prisma ORM handles most SQL for you. You write TypeScript, Prisma generates SQL.

### Q: Is everything really JavaScript/TypeScript?

**A: Yes!** 95% of the code is TypeScript/JavaScript. Only optional Python service for upscaling, and SQL database (but managed via TypeScript).

### Q: How hard is it to learn TypeScript if I know JavaScript?

**A: Very easy!** TypeScript is JavaScript with types. If you know JavaScript, learning TypeScript takes about 1 week.

### Q: What about WebAssembly?

**A: You don't need to learn it!** FFmpeg.wasm is already compiled to WebAssembly. You use it from JavaScript like any other library.

---

## 📖 Learning Resources

### JavaScript Basics

- **MDN JavaScript Guide**: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide
- **JavaScript.info**: https://javascript.info/

### TypeScript

- **TypeScript Handbook**: https://www.typescriptlang.org/docs/handbook/intro.html
- **TypeScript in 5 minutes**: https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes.html

### React/Next.js

- **Next.js Learn**: https://nextjs.org/learn
- **React Docs**: https://react.dev/

### MediaPipe (AI)

- **MediaPipe Web**: https://google.github.io/mediapipe/solutions/pose.html
- **TensorFlow.js**: https://www.tensorflow.org/js

---

## ✅ Bottom Line

### Languages You'll Use:

1. **TypeScript/JavaScript** - 95% of code
2. **SQL** - Basic knowledge (database, managed via Prisma)
3. **CSS** - Styling
4. **Python** - Only if you want optional video upscaling (can skip!)

### Recommended Path:

1. Learn JavaScript (1-2 weeks)
2. Learn TypeScript basics (1 week - just add types!)
3. Learn React/Next.js (2-3 weeks)
4. Start building (everything else learned along the way)

**You can build the entire MVP with just JavaScript/TypeScript knowledge!**

---

_Last Updated: Focused on JavaScript/TypeScript as primary language_
