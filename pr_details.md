Title: ⚡ Bolt: Replace map and reduce array methods with single-pass loops in default tab initializers

Description:
💡 What: Replaced `.reduce()` in `defaultTabVisibility()` and `.map()` in `defaultTabOrder()` with single-pass `for` loops in `tabs.ts`.
🎯 Why: `defaultTabVisibility` and `defaultTabOrder` are called frequently (e.g., during localStorage hydration or component state initialization). Using `.reduce()` and `.map()` on the static `TABS` array allocates unnecessary intermediate array elements, function calls, and closures on every pass, generating garbage collection pressure.
📊 Impact: Eliminates callback allocation and intermediate array creation for these tab initialization routines, slightly reducing GC overhead.
🔬 Measurement: Verified that unit tests for components using these tabs (like TabBar and PanelStack) continue to pass.
