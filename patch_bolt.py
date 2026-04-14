with open('.jules/bolt.md', 'r') as f:
    content = f.read()

content = content.replace("**Action:** When calculating statistics, use a single pass over the object array to accumulate sums and extract numerical values into a pre-allocated typed array (e.g., `Float64Array`). Then, perform secondary calculations (like variance) in a tight loop over the contiguous, pre-populated typed array. This drastically reduces object property access and speeds up execution by ~15-20%.", "**Action:** When calculating statistics over an array of objects (`RowData[]`), delay typed-array allocation until the valid count is known via a first pass. Over-allocating to the maximum possible size for every signal before knowing the count creates an O(row-count) allocation even for sparse datasets, which can cause UI stalls or OOM errors. Use a two-pass approach where the first pass counts and sums, and the second pass extracts values into an exactly-sized `Float64Array` and calculates variance.")

content = content.replace("## 2026-04-14 - [Optimize Two-Pass Statistics Calculation]", "## 2026-04-14 - [Optimize Typed Array Allocation in Statistics]")

content = content.replace("**Learning:** In JavaScript/TypeScript, when calculating statistics (like variance or median) over an array of objects (`RowData[]`), iterating over the large object array multiple times causes significant overhead due to property access (`data[i][signal]`) and type checks.", "**Learning:** Over-allocating `Float64Array` to the total row count before filtering out invalid/non-numeric entries introduces a data-dependent performance regression for sparse datasets, creating huge unnecessary allocations when the valid count is tiny or zero.")

with open('.jules/bolt.md', 'w') as f:
    f.write(content)
