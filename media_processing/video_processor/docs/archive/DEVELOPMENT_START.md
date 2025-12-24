# Development Start Guide - Golf Swing Video Platform

## ⚠️ Prerequisites Check

Before we can start building, you need to install:

### Required Software

1. **Node.js** (v18 or higher)
   - Download: https://nodejs.org/
   - Check version: `node --version` (should be 18+)

2. **npm** (comes with Node.js)
   - Check version: `npm --version`

3. **Git** (already installed ✓)

4. **VS Code** (recommended)
   - With extensions I've configured (see VS_CODE_SETUP.md)

### Optional but Recommended

5. **PostgreSQL** (for local database) OR **Supabase** (free cloud database)
6. **Cloudflare R2** account (for video storage)

---

## 🚀 Step-by-Step Setup

### Step 1: Install Node.js

1. Go to https://nodejs.org/
2. Download **LTS version** (Long Term Support)
3. Install with default options
4. Restart your terminal
5. Verify:
   ```bash
   node --version  # Should show v18.x or higher
   npm --version   # Should show v9.x or higher
   ```

### Step 2: Install VS Code (If Not Already Installed)

1. Download: https://code.visualstudio.com/
2. Install with default options
3. Install recommended extensions:
   - Open VS Code in this project folder
   - VS Code will prompt: "This workspace has extension recommendations"
   - Click **"Install All"**

   Or install manually (see `VS_CODE_SETUP.md`)

### Step 3: Install Project Dependencies

Once Node.js is installed:

```bash
# Navigate to project directory
cd /c/Users/diete/Repositories/Project_Template

# Install all dependencies
npm install
```

This will:
- Install Next.js, React, TypeScript, and all dependencies
- Set up the monorepo workspace
- Take 2-5 minutes the first time

### Step 4: Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
# For now, you can leave defaults for local development
```

### Step 5: Install Prisma CLI

```bash
# Install Prisma CLI globally (optional but helpful)
npm install -g prisma

# Or use npx (no global install needed)
npx prisma --version
```

### Step 6: Set Up Database

**Option A: Supabase (Recommended - Free & Easy)**

1. Go to https://supabase.com/
2. Create free account
3. Create new project
4. Go to Settings → Database
5. Copy connection string
6. Add to `.env`:
   ```
   DATABASE_URL="postgresql://postgres:your-password@your-project.supabase.co:5432/postgres"
   ```

**Option B: Local PostgreSQL**

1. Install PostgreSQL: https://www.postgresql.org/download/windows/
2. Create database named `golf_swing_analyzer`
3. Add to `.env`:
   ```
   DATABASE_URL="postgresql://postgres:password@localhost:5432/golf_swing_analyzer"
   ```

Then run:
```bash
cd packages/database
npx prisma generate
npx prisma migrate dev --name init
```

### Step 7: Start Development Server

```bash
# From project root
npm run dev

# Should see:
# ▲ Next.js 14.x.x
# - Local:        http://localhost:3000
# - Ready in Xs
```

Open http://localhost:3000 in your browser!

---

## 🐛 Troubleshooting

### "npm: command not found"

**Solution**: Node.js is not installed or not in PATH
1. Install Node.js from https://nodejs.org/
2. Restart terminal
3. Try `node --version` again

### "Cannot find module"

**Solution**: Dependencies not installed
```bash
npm install
```

### "Port 3000 already in use"

**Solution**: Another app is using port 3000
```bash
# Kill the process using port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or change port in package.json scripts
"dev": "next dev -p 3001"
```

### "Prisma schema file not found"

**Solution**: Navigate to correct directory
```bash
cd packages/database
npx prisma generate
```

### Database Connection Issues

**Solution**: Check DATABASE_URL in .env
1. Make sure Supabase/PostgreSQL is running
2. Verify connection string is correct
3. Check firewall isn't blocking connection

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `node --version` shows v18+
- [ ] `npm --version` shows v9+
- [ ] `npm install` completes without errors
- [ ] VS Code extensions installed
- [ ] `.env` file created
- [ ] Database connection works
- [ ] `npm run dev` starts without errors
- [ ] http://localhost:3000 shows the app

---

## 📚 Next Steps After Setup

Once everything is installed and working:

1. **Read**: `VS_CODE_SETUP.md` for development tips
2. **Read**: `docs/GOLF_VIDEO_QUICK_REFERENCE.md` for project overview
3. **Read**: `docs/GOLF_VIDEO_BUDGET_GUIDE.md` for cost optimization
4. **Read**: `.cursor/rules/webdevrules.md` for coding standards

5. **Start Building**:
   - Check `docs/GOLF_VIDEO_ACTION_PLAN.md` for Phase 1 tasks
   - Start with video upload component
   - Build incrementally

---

## 🎯 Phase 1 MVP Features to Build

Once setup is complete, build these in order:

1. **Video Upload** (`apps/web/components/video/VideoUploader.tsx`)
2. **Video Player** (`apps/web/components/video/VideoPlayer.tsx`)
3. **Basic Drawing** (`apps/web/components/tools/LineTool.tsx`)
4. **User Auth** (`apps/web/app/(auth)/`)
5. **Projects** (`apps/web/app/(dashboard)/projects/`)

---

## 💡 Getting Help

### Documentation
- All docs in `docs/` folder
- Start with `docs/GOLF_VIDEO_QUICK_REFERENCE.md`

### VS Code Help
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Help" to see options
- Or check `VS_CODE_SETUP.md`

### Next.js Help
- Official docs: https://nextjs.org/docs
- Learn course: https://nextjs.org/learn

### TypeScript Help
- Official docs: https://www.typescriptlang.org/docs/
- Quick start: https://www.typescriptlang.org/docs/handbook/intro.html

---

## 🚀 Ready to Build?

Once all prerequisites are installed:

1. Run `npm install`
2. Run `npm run dev`
3. Start coding!

**I'll help you every step of the way! 🎉**

---

*If you run into issues, check the troubleshooting section above or let me know!*
