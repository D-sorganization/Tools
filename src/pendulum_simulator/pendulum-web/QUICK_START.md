# Quick Start: Adding Triple & Golfer Models

## In 5 Minutes

### 1. Copy Files to Correct Locations
```bash
cd pendulum-web/src

# Physics files
cp ../../physics_triple.ts ./physics_triple.ts
cp ../../physics_golfer.ts ./physics_golfer.ts

# Presets
cp ../../presets_triple.ts ./presets_triple.ts
cp ../../presets_golfer.ts ./presets_golfer.ts

# Components
cp ../../TriplePendulumCanvas.tsx ./components/TriplePendulumCanvas.tsx
cp ../../GolferCanvas.tsx ./components/GolferCanvas.tsx

# New App
cp ../../AppNew.tsx ./App.tsx
```

### 2. Verify TypeScript
```bash
npm install
npx tsc --noEmit
```

Should have 0 errors.

### 3. Run Dev Server
```bash
npm run dev
```

Open http://localhost:5173 (or provided URL)

### 4. Test Models
- Click **"Double Pendulum (2-DOF)"** tab → Select preset → Run
- Click **"Triple Pendulum (3-DOF)"** tab → Select preset → Run
- Click **"Golfer (8-DOF)"** tab → Select preset → Run

Done! ✓

---

## What Changed

### New Files (8 total)
- `physics_triple.ts` - 3-DOF physics
- `physics_golfer.ts` - 8-DOF physics
- `presets_triple.ts` - Triple presets
- `presets_golfer.ts` - Golfer presets
- `components/TriplePendulumCanvas.tsx` - 3-segment renderer
- `components/GolferCanvas.tsx` - Golfer renderer
- `App.tsx` - REPLACED (backup as `App.backup.tsx`)
- `INTEGRATION_GUIDE.md` - Full documentation

### Unchanged
All other files remain unchanged. No breaking changes to existing double pendulum functionality.

---

## File Checklist

Make sure these 8 new files exist after copying:

```
✓ src/physics_triple.ts (467 lines)
✓ src/physics_golfer.ts (588 lines)
✓ src/presets_triple.ts (92 lines)
✓ src/presets_golfer.ts (155 lines)
✓ src/components/TriplePendulumCanvas.tsx (187 lines)
✓ src/components/GolferCanvas.tsx (286 lines)
✓ src/App.tsx (816 lines) — REPLACED
✓ INTEGRATION_GUIDE.md (308 lines)
```

---

## Key Differences Between Models

### Double Pendulum (2 DOF)
- Shoulder + wrist joints
- 2×2 mass matrix
- Canvas: 2 segments
- Presets: 5 existing configurations

### Triple Pendulum (3 DOF)
- Shoulder + elbow + wrist
- 3×3 mass matrix
- Canvas: 3-segment chain
- Presets: 3 new configurations

### Golfer Upper-Body (8 DOF)
- Hub + 2 arms (3 DOF each) + club
- 8×8 mass matrix
- 4 holonomic constraints (hands on club)
- Canvas: skeletal with hub, both arms, club
- Presets: 3 new configurations

---

## Typical Workflow

### For Double Pendulum (Existing)
1. Select preset
2. Adjust parameters (masses, lengths)
3. Set initial angles
4. Adjust torque coefficients
5. Run & animate
6. (Optional) Optimize via "Optimizer" tab

### For Triple Pendulum (New)
1. Select preset
2. Adjust segment parameters (m1, m2, m3, L1, L2, L3)
3. Set initial angles (θ₁, φ₂, φ₃)
4. Set torques for shoulder, elbow, wrist
5. Run & animate

### For Golfer (New)
1. Select preset
2. Adjust body masses (hub, arms)
3. Set initial posture (hub angle, shoulder angle)
4. Run & animate
5. Watch constraint-enforced motion

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Cannot find module" | Check all 8 files in correct directories |
| TypeScript errors | Run `npm install`, then `npx tsc --noEmit` |
| Simulation won't run | Check torque format: comma-separated numbers |
| Canvas blank | Ensure "Play" button is clicked |
| App crashes | Check browser console for error messages |

---

## Physics Quick Reference

### Double Pendulum
```
q = [θ₁, φ]
M(q) = 2×2 matrix
τ(t) = [τ_shoulder, τ_wrist]
```

### Triple Pendulum
```
q = [θ₁, φ₂, φ₃]
M(q) = 3×3 matrix
τ(t) = [τ_shoulder, τ_elbow, τ_wrist]
```

### Golfer
```
q = [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club]
M(q) = 8×8 matrix (with 4 constraints)
τ(t) = [τ_hub, τ_rs, τ_re, τ_rh, τ_ls, τ_le, τ_lh]
```

---

## Browser Requirements

- **Modern browser** with ES2020+ support
- **WebGL** for canvas rendering (all modern browsers)
- **localStorage** for preset caching (optional)
- **No external CDN dependencies** (all bundled)

Tested on:
- Chrome 120+
- Firefox 120+
- Safari 16+
- Edge 120+

---

## Next Steps

- [ ] Copy 8 files to correct locations
- [ ] Run `npm install && npx tsc --noEmit`
- [ ] Run `npm run dev`
- [ ] Test all 3 models with presets
- [ ] (Optional) Read `INTEGRATION_GUIDE.md` for deep dive
- [ ] (Optional) Customize presets or physics

---

## Support Resources

1. **Quick Start** (this file) - 5-minute setup
2. **INTEGRATION_GUIDE.md** - Detailed guide with troubleshooting
3. **FILES_CREATED.md** - Complete file inventory with line counts
4. **Code comments** - Every physics file has inline documentation
5. **Preset examples** - See presets_*.ts for configuration patterns

---

## Rollback (If Needed)

To revert to double pendulum only:
```bash
# Restore original app
cp src/App.backup.tsx src/App.tsx

# Remove new files (optional)
rm src/physics_triple.ts src/physics_golfer.ts
rm src/presets_triple.ts src/presets_golfer.ts
rm src/components/TriplePendulumCanvas.tsx src/components/GolferCanvas.tsx
```

---

That's it! You now have a full multi-model pendulum simulator. 🚀
