1. **Analyze `AlarmsHeader.tsx`**
   - The file currently uses `.filter()` to count unacknowledged alarms and `.reduce()` to find the highest severity.
   - For a system that might handle high-frequency events or many alarms, replacing these with a single-pass `for` loop will eliminate intermediate array allocations and reduce GC overhead.

2. **Refactor `AlarmsHeader.tsx`**
   - Implement a single `for` loop to track the number of unacknowledged alarms and find the maximum severity.

3. **Verify the Optimization**
   - Run linter/compiler to ensure type safety.
   - Run tests if applicable.
   - Verify visually/functionally that the refactoring maintains exact original behavior.

4. **Update Bolt's Journal**
   - Log the learning about replacing `.filter()` and `.reduce()` with standard `for` loops in React components for aggregating list data.

5. **Pre-commit Steps**
   - Run `pre_commit_instructions` tool to verify CI passes before submitting.

6. **Submit PR**
   - Submit the changes with an appropriate commit message outlining the performance benefits.
