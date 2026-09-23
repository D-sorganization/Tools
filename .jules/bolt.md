
## 2026-09-23 - Eliminate Array Spreads for Min/Max Calculations
**Learning:** Using the spread operator with `Math.min(...tx)` and `Math.max(...tx)` on arrays whose length dynamically changes based on target configurations can result in 'Maximum call stack size exceeded' exceptions. Additionally, calling spread operators inside hot rendering functions creates significant garbage collection pressure due to `O(N)` allocations.
**Action:** Always compute numerical boundaries across target/geometry arrays using a single-pass `for` loop to safely operate in `O(N)` time without mapping the arrays directly to the call stack.
