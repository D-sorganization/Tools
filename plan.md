1. **Optimize `applyFormula` inside `useDataProcessor`**: Replace `filteredData.map()` and `safeUsedSignals.map()` inside the loop with a single-pass `for` loop to avoid excessive garbage collection and object allocations during formula execution on large datasets.
2. **Pre-commit checks**: Run `npm run lint`, `npm run type-check`, and testing to verify everything still passes, and follow any instructions provided by the pre-commit checker.
3. **Submit the PR**: Ensure the PR is formatted properly and measures performance impact according to the "Bolt" specific instructions.
