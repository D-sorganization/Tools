# Defense Paragraphs (Orthogonality Review)
**Date:** 2026-03-09

### Injection Point: Section 3.1 Import Path Fragility
**Preemptive Defense:** The widespread use of `sys.path` manipulation represents a critical boundary condition of the monorepo's evolution. Prior to the centralized `PluginManager`, tools required multi-level fallbacks to operate as standalone applications. The active mandate is migrating to `ensure_utils_in_path()`.

### Injection Point: Section 3.2 Duplicate File I/O Implementations
**Preemptive Defense:** Duplicate JSON utilities were an intentional, temporary methodological choice to preserve module autonomy during the shared library transition. Forcing an immediate dependency would have broken backwards compatibility.

### Injection Point: Section 3.3 Logging Infrastructure Split
**Preemptive Defense:** The logging split is an artifact of a phased deprecation strategy. Immediate consolidation would have disrupted active workflows, hence the active deprecation warnings (Issue #208).

### Injection Point: Section 3.4 Constants Duplication
**Preemptive Defense:** While global primitives (e.g., $g$, $\pi$) should be centralized, tightly coupling disparate simulations to a single constants file can introduce varying precision bugs. A nuanced decoupling is required.

### Injection Point: Section 3.5 Configuration Management Inconsistency
**Preemptive Defense:** Decentralized configuration supported the "standalone tool" philosophy. However, hardcoded Windows paths breach portability constraints and necessitate a unified `config_manager.py` pattern.