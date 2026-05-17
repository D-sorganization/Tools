1. **Optimize Array Pre-allocation in `odeSolver.ts`:**
   - In `src/ode_solver/web/src/lib/odeSolver.ts`, `results` is instantiated as an empty array: `const results: ODEResultPoint[] = []`.
   - In the RK4 integration loop, `point` objects are pushed to it `numPoints` times using `results.push(point)`.
   - To align with "Bolt's Array Pre-allocation" principle for large arrays in high-frequency/numerical loops, change `results` to be pre-allocated to the known size: `const results = new Array<ODEResultPoint>(numPoints);`.
   - Instead of using `results.push(point)`, assign it directly by index: `results[i] = point;`.

2. **Complete Pre-commit Steps:**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

3. **Submit the PR:**
   - Verify all tests pass, and submit a PR with the title format "⚡ Bolt: [performance improvement]".
