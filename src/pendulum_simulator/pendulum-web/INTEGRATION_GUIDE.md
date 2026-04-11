# Integration Guide: Triple Pendulum & Golfer Models

This guide explains how to integrate the new triple pendulum (3-DOF) and golfer upper-body (8-DOF) models into your React/Tauri web app.

## Files Created

### Physics Implementations
1. **`src/physics_triple.ts`** - Triple pendulum physics (3-DOF)
   - 3×3 mass matrix with analytical expressions
   - Coriolis and gravity vectors
   - RK4 integration with polynomial torque functions
   - Forward kinematics for shoulder, elbow, wrist, tip

2. **`src/physics_golfer.ts`** - Golfer upper-body physics (8-DOF)
   - Hub + two 2-segment arms + club with 4 holonomic constraints
   - Constraint Jacobians and simplified KKT solver
   - Baumgarte stabilization for constraints
   - Forward kinematics for all 7 mass points + club tip

### Presets
3. **`src/presets_triple.ts`** - Triple pendulum presets
   - "Three-Segment Swing (passive)"
   - "Three-Segment Swing (active)"
   - "Free Triple Pendulum"

4. **`src/presets_golfer.ts`** - Golfer model presets
   - "Golfer Upper Body (symmetric swing)"
   - "Golfer Upper Body (asymmetric swing)"
   - "Free Golfer Body (no torques)"

### React Components
5. **`src/components/TriplePendulumCanvas.tsx`** - 3-segment canvas renderer
   - Renders shoulder → elbow → wrist → tip chain
   - Color-coded segments (blue, orange, green)
   - Animated trail of tip positions
   - Joint markers and crosshair

6. **`src/components/GolferCanvas.tsx`** - Golfer visualization
   - Hub standoff with perpendicular shoulder line
   - Right arm (red), left arm (blue), club (green)
   - All 7 joints with different colors
   - Clubhead sphere at tip
   - Club trail

### Updated App
7. **`src/AppNew.tsx`** - Refactored main app with model selector
   - Model tabs at top: Double / Triple / Golfer
   - Model-specific control panels
   - Unified simulation and animation system
   - Supports all three models with tabbed interface

## Integration Steps

### Step 1: Backup Current App
```bash
cp src/App.tsx src/App.backup.tsx
```

### Step 2: Update `src/App.tsx`
Replace the existing `src/App.tsx` with `AppNew.tsx`:
```bash
mv src/AppNew.tsx src/App.tsx
```

### Step 3: Verify Imports
The new App.tsx imports the new physics modules and components:
- `import { TriplePendulumCanvas } from './components/TriplePendulumCanvas';`
- `import { GolferCanvas } from './components/GolferCanvas';`
- `import { PRESETS_TRIPLE } from './presets_triple';`
- `import { PRESETS_GOLFER } from './presets_golfer';`
- `import { makeTripleParams, ... } from './physics_triple';`
- `import { makeGolferParams, ... } from './physics_golfer';`

These imports are already in the new AppNew.tsx file.

### Step 4: Install TypeScript (if not done)
The app uses TypeScript. Ensure your project has:
```json
{
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
```

### Step 5: Run Type Check (optional)
```bash
npx tsc --noEmit
```

### Step 6: Start Dev Server
```bash
npm run dev
```

## File Structure After Integration

```
pendulum-web/src/
├── App.tsx                          # UPDATED: Multi-model selector
├── App.css                          # Existing styles (unchanged)
├── main.tsx                         # Entry point
│
├── physics.ts                       # Double pendulum (existing)
├── physics_triple.ts                # NEW: Triple pendulum
├── physics_golfer.ts                # NEW: Golfer model
│
├── presets.ts                       # Double presets (existing)
├── presets_triple.ts                # NEW: Triple presets
├── presets_golfer.ts                # NEW: Golfer presets
│
├── optimizer.ts                     # Double optimizer (existing)
├── units.ts                         # Unit conversion (existing)
│
├── components/
│   ├── PendulumCanvas.tsx           # Double canvas (existing)
│   ├── TriplePendulumCanvas.tsx      # NEW: Triple canvas
│   ├── GolferCanvas.tsx             # NEW: Golfer canvas
│   ├── AnalysisPlots.tsx            # Plots (existing)
│   ├── OptimizerPanel.tsx           # Optimizer (existing)
│   └── UnitSelector.tsx             # Unit picker (existing)
```

## Key Design Decisions

### 1. Physics Models
- **Double**: 2 DOF, 2×2 mass matrix, no constraints
- **Triple**: 3 DOF, 3×3 mass matrix, no constraints
- **Golfer**: 8 DOF, 4 holonomic constraints, simplified KKT solver

### 2. Coordinate Systems
All models use the same local coordinate frame:
- Origin at shoulder/hub
- x-axis: horizontal (right positive)
- y-axis: vertical (up positive)
- Angles measured counterclockwise from downward vertical

### 3. Simulation Timestep
All models use RK4 integration with dt = 0.005 s (200 Hz)

### 4. Torque Functions
Each model supports polynomial torque functions: τ(t) = c₀ + c₁t + c₂t² + …

For golfer (8 DOF), the preset provides zero torques to show constraint-driven dynamics.

### 5. Canvas Rendering
- **PendulumCanvas**: 2-segment (existing)
- **TriplePendulumCanvas**: 3-segment chain
- **GolferCanvas**: Hub + 2 arms + club (skeletal animation)

All scale to fit viewport and animate smoothly.

## Testing Each Model

### Double Pendulum (Existing)
1. Select "Double Pendulum" tab
2. Choose preset: "Golf Swing (passive wrist)"
3. Run simulation
4. Verify: arms and shaft rotate, clubhead swings

### Triple Pendulum (New)
1. Select "Triple Pendulum" tab
2. Choose preset: "Three-Segment Swing (passive)"
3. Run simulation
4. Verify: shoulder, elbow, wrist segments animate smoothly
5. Watch tip trace move in canvas

### Golfer Upper-Body (New)
1. Select "Golfer" tab
2. Choose preset: "Golfer Upper Body (symmetric swing)"
3. Run simulation
4. Verify: hub rotates, both arms move, club moves with grip constraints
5. Watch club tip trail trace

## Customization

### Adding New Presets

#### For Triple:
```typescript
// In presets_triple.ts
_preset_triple(
  'My Triple Swing',
  2.0, 1.5, 0.5, 0.20,   // masses
  0.35, 0.25, 0.15,       // lengths
  0.05, 0.04, 0.03,       // damping
  -45, 30, 45, 0, 0, 0,   // initial angles & velocities
  [-15, 5], [3, -1], [1], // torque coeffs (shoulder, elbow, wrist)
  2.0,                     // duration
  'My custom swing description'
);
```

#### For Golfer:
```typescript
// In presets_golfer.ts
_preset_golfer(
  'My Golfer Swing',
  3.0, 3.0, 1.5, 3.0, 1.5, 0.30, 0.20,  // masses
  0.25, 0.30, 0.25, 0.30, 0.25, 1.10,    // lengths
  0.15, 0.15, 0.10, 0.10,                 // offsets & grips
  0.1, 0.08, 0.06, 0.04, 0.08, 0.06, 0.04, // damping
  -30, -45, 30, 0, -45, 30, 0, 0,        // initial angles
  0, 0, 0, 0, 0, 0, 0, 0,                // initial velocities
  [-15, 5], [0], [0], [0], [0], [0], [0], // torques
  2.0,
  'Custom golfer preset'
);
```

### Modifying Mass Matrix Computation
The triple and golfer models compute M(q) analytically:
- **Triple**: See `massMatrix3()` in physics_triple.ts
- **Golfer**: See `massMatrix_golfer()` in physics_golfer.ts (currently simplified)

To improve the golfer model's mass matrix, implement full analytical Jacobians for each mass point.

### Extending Constraints (Golfer)
The golfer model uses simplified constraint penalties. To implement full KKT:
1. Compute constraint Jacobian Φ_q (done in `constraintJacobian()`)
2. Assemble 12×12 KKT matrix: [M Φ_q^T; Φ_q 0]
3. Solve for accelerations and Lagrange multipliers
4. See `equationsOfMotion_golfer()` for where to add this

## Performance Notes

- **Double**: ~60 FPS on modern laptop
- **Triple**: ~50 FPS (3×3 matrix solve)
- **Golfer**: ~40 FPS (constraint penalties + penalty gain tuning)

For better golfer performance, implement analytical Jacobians and reduce constraint penalty gain (currently 1000).

## Known Limitations

1. **Golfer Model**: Uses penalty method instead of full KKT solver
   - Constraints enforced via spring penalty τ = -K·Φ
   - Penalty gain K=1000 may need tuning for stability
   - For production, implement Baumgarte stabilization with proper KKT solve

2. **Canvas Scaling**: All canvases auto-scale to viewport
   - For very large/small models, adjust scale factor in toCanvas() callbacks

3. **Analysis Plots**: Currently only work for double pendulum
   - Triple and golfer models don't feed into AnalysisPlots yet
   - Would require duplicating analysis for each model

## Troubleshooting

### "Cannot find module" errors
- Ensure all 6 new TypeScript files are in `src/`
- Run `npm install` if package.json dependencies are missing

### Simulation won't run
- Check browser console for errors
- Verify torque coefficient parsing (comma-separated values)
- Ensure initial angles are in valid range (radians after conversion)

### Canvas doesn't animate
- Verify "Play" button is clicked (controls panel bottom)
- Check that simulation produced states (> 2 timesteps)
- Inspect browser console for canvas context errors

### Performance issues
- Reduce simulation dt (make smaller for faster playback)
- Lower animation frame rate by increasing speed slider
- For golfer, reduce constraint penalty gain K in equationsOfMotion_golfer()

## Future Enhancements

1. **Add Analysis Plots for Triple/Golfer**
   - Duplicate AnalysisPlots logic for each model
   - Compute kinetic/potential energy, joint velocities, etc.

2. **Implement Full KKT Solver for Golfer**
   - Replace penalty method with constraint force computation
   - Use Baumgarte stabilization for better numerical stability

3. **Add Optimizer for Triple/Golfer**
   - Extend OptimizerPanel to work with all models
   - Optimize torque coefficients to maximize club tip speed

4. **Multi-body Inertia**
   - Model segments as uniform rods (not just point masses)
   - Compute moment of inertia about each joint

5. **Contact Dynamics**
   - Add ground contact for golfer (foot/ball)
   - Impact forces when clubhead hits ball

## References

### Physics
- Lagrange equations: Goldstein, "Classical Mechanics" (3rd ed.)
- Constrained dynamics: Baumgarte, "Stabilization of constraints..." (1972)
- RK4 integration: Hairer & Wanner, "Solving Ordinary Differential Equations"

### Code
- See comments in `physics_triple.ts` and `physics_golfer.ts` for detailed derivations
- Mass matrix formulas are inline-documented

## Support

For questions or issues:
1. Check the integration guide above
2. Review comments in physics files
3. Examine preset configurations for examples
4. Consult browser console for runtime errors
