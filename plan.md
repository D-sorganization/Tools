1. **Analyze `src/p1am_control_system/frontend/src/hooks/useTrendBackfill.ts`**:
   - In `useTrendBackfill`, there is a `.map().filter()` chain that maps timestamps to `BackfillPoint` objects and filters out invalid ones.
     ```typescript
          setPoints(
            ts
              .map((iso, i) => ({ t: parseHistorianTs(iso), v: (vs[i] ?? 0) * scale }))
              .filter((p) => Number.isFinite(p.t)),
          );
     ```
   - This creates an intermediate array containing objects that are then potentially discarded during the `.filter()` step.

2. **Refactor `useTrendBackfill.ts`**:
   - Replace the chained `.map().filter()` passes with a single-pass `for` loop that avoids creating and discarding intermediate objects. This aligns with Bolt's `.jules/bolt.md` learning: "Replace chained .filter() and .map() with a single-pass for loop".
   ```typescript
        const pts: BackfillPoint[] = [];
        for (let i = 0; i < ts.length; i++) {
          const t = parseHistorianTs(ts[i]);
          if (Number.isFinite(t)) {
            pts.push({ t, v: (vs[i] ?? 0) * scale });
          }
        }
        setPoints(pts);
   ```

3. **Check memory constraints**:
   "DO NOT journal routine work like generic React performance tips"
   "Avoid micro-optimizing operations that run on tiny static arrays executed during initialization or renders."
   - The array `ts` from the historian backfill can be up to `maxPoints` (which defaults to `4000` via `TREND_BACKFILL_MAX_POINTS`), meaning this is NOT a tiny static array. It's an array of up to 4000 items that gets processed on every backfill. Eliminating the intermediate allocation of 4000 objects is a worthwhile optimization for large data streams.

4. **Prepare diff**:
   ```typescript
<<<<<<< SEARCH
        setPoints(
          ts
            .map((iso, i) => ({ t: parseHistorianTs(iso), v: (vs[i] ?? 0) * scale }))
            .filter((p) => Number.isFinite(p.t)),
        );
=======
        // ⚡ Bolt Optimization: Replace chained .map().filter() passes with a single-pass for loop
        // to avoid intermediate array allocations and closure overhead on large historian backfill arrays.
        const pts: BackfillPoint[] = [];
        for (let i = 0; i < ts.length; i++) {
          const t = parseHistorianTs(ts[i]);
          if (Number.isFinite(t)) {
            pts.push({ t, v: (vs[i] ?? 0) * scale });
          }
        }
        setPoints(pts);
>>>>>>> REPLACE
   ```

5. **Verify the change**:
   - `cd /app/src/p1am_control_system/frontend && pnpm run type-check` (or `tsc`)
   - `npm run test` for the workspace.
   - Run the full project test suite from the root.
