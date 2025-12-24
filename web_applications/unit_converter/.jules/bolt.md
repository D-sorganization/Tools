## 2024-05-22 - [Optimizing Search Units]
**Learning:** The `searchUnits` function was performing an O(N*M) operation where N is the number of units and M is the total number of aliases. By inverting the alias map to O(N) lookup, we reduced the complexity significantly. `Object.entries()` inside a loop is a performance killer as it allocates new arrays on every iteration.
**Action:** Always look for nested loops that iterate over static or semi-static data structures. Pre-compute or cache reverse mappings to avoid O(N) lookups in hot paths like search-as-you-type.
