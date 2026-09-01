1. **Apply Optimization in `src/p1am_control_system/frontend/src/lib/curveFit.ts`**
   - In `rSquared()`, replace `points.reduce()` for computing `meanY` with a standard `for` loop to eliminate array method callback overhead and GC pressure.
   - In `linearFit.fit()`, replace the two separate `points.reduce()` calls for computing `meanX` and `meanY` with a single standard `for` loop that computes both sums simultaneously.

2. **Verify changes**
   - Read `src/p1am_control_system/frontend/src/lib/curveFit.ts` to confirm changes apply correctly.

3. **Run CI and tests locally**
   - `cd /app/src/p1am_control_system/frontend && pnpm install && pnpm exec tsc && npm run test`
   - Run the repository-wide tests: `cd /app && npm install && npm run test`

4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**

5. **Submit the change.**
1. **Update `SPEC.md`**
   - Use `replace_with_git_merge_diff` to add a new bullet point under the `### Goals & Non-Goals` or a recent spec update section documenting the performance improvement in `src/p1am_control_system/frontend/src/lib/curveFit.ts`.
   - Update `Last Spec Update` and `Spec Version` in the `1. Identity` table if necessary.
   - Run `cat SPEC.md` to verify the modifications.
2. **Commit and Submit**
   - Call `submit` with the appropriate branch name and commit message.
