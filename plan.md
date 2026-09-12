1. **Optimize array spreading in `NeuralModelLabPanel.tsx`:**
   - In `CapabilityPlot`, `vendors` is mapped into an array of `strictRowCount` and spread into `Math.max`.
   - Update it to use a single-pass `for` loop to compute the `maximum` instead of mapping and spreading, which improves performance and avoids GC/call stack pressure on larger arrays.
2. **Review and commit changes:**
   - Verify changes with `pnpm run lint` and `pnpm test`.
   - Update `CHANGELOG.md` or `.jules/bolt.md` if necessary to record this optimization.
   - Run `pre_commit_instructions` and submit PR using the "Bolt" persona guidelines.
