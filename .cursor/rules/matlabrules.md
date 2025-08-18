# MATLAB Best Practices Rules for Cursor

This document consolidates style, correctness, and performance optimization guidelines for MATLAB programming. 
It acts like Ruff (lint checks) and Black (formatter) combined—Cursor should apply these rules automatically where possible.

---

## 🎯 Core Principles
- **MUST** write clear, maintainable code before optimizing.
- **MUST** profile before optimizing performance.
- **SHOULD** use vectorization, preallocation, and built-ins when possible.
- **AVOID** unsafe constructs (`inv`, `eval`, `global`).
- **Clarity First**: Write simple, readable, maintainable code.
- **Efficiency**: Use vectorization, preallocation, and built-ins to maximize speed.
- **Column-Major Awareness**: MATLAB stores arrays in column-major order; loop accordingly.
- **Correctness**: Avoid unsafe constructs (`inv`, `eval`, `global`). Validate all inputs and outputs.
- **Profiling**: Optimize only after profiling with `profile`, `timeit`, or `gputimeit`.

---

## 💅 Code Formatting & Style
- **MUST** use 4 spaces (no tabs) and keep lines ≤100 chars.
- **MUST** use camelCase for vars/functions, PascalCase for classes/scripts, UPPER_CASE for constants.
- **MUST** one function per file, filename == function name.
- **SHOULD** use descriptive names (`velocity` not `v`).
- **SHOULD** comment *why*, not *what*.
- **AVOID** `clear all`, `clc`, `close all`, or `addpath` in library code.
- **Indentation**: 4 spaces, no tabs.
- **Line Length**: Max 80–100 characters.
- **Naming**:
  - Variables & functions: `camelCase`
  - Classes & scripts: `PascalCase`
  - Constants: `UPPER_CASE`
  - Booleans: prefix with `is`, `has` (`isValid`, `hasConverged`)
- **Comments**:
  - Use `%` for single line, `%%` for cells, `%{ ... %}` for blocks.
  - Comment *why*, not *what*.
- **Functions**:
  - One function per `.m` file. Filename == function name.
  - Start with an H1 help line, purpose, inputs, outputs, and examples.
  - Use `arguments` blocks (R2019b+) or `validateattributes`.
  - Always end with `end`.
- **Hygiene**:
  - No `clear all`, `clc`, `close all`, or `addpath` in library code.
  - Avoid shadowing built-ins (`sum`, `table`, `length`).

---

## ✅ Correctness & Reliability
- **MUST** use `A\b`, not `inv(A)`.
- **MUST** compare floats with tolerance, not `==`.
- **MUST** validate all inputs and outputs.
- **MUST** seed RNG in tests (`rng(0,"twister")`).
- **SHOULD** use `numel`, `size`, `isempty` instead of `length`.
- **AVOID** `eval`, `assignin`, `global`, or shadowing built-ins.
- **NEVER** use `inv(A)`; use `A\b`, `chol`, `qr`, or `svd`.
- Use tolerances for floating-point comparisons: `abs(a-b) <= tol`.
- Seed RNG in tests: `rng(0, "twister")`.
- Avoid `eval`, `assignin`, `feval("...")`, `load` without outputs, and `global`.
- Prefer `numel`, `size`, `isempty` over `length`.
- Use `string` instead of `char` when interfacing with APIs.
- Use `assert`, `error`, and descriptive IDs for error handling.

---

## 🚀 Performance Optimization
- **MUST** profile (`profile`, `timeit`, `gputimeit`) before optimizing.
- **MUST** preallocate arrays (zeros/ones/nan/cell/spalloc).
- **MUST** respect column-major memory order in loops.
- **SHOULD** vectorize loops with built-ins where possible.
- **AVOID** growing arrays, row-major loops, or redundant temporaries.

### 1. Vectorization & Built-ins
- Replace loops with vectorized ops:
  ```matlab
  y = sin(x) + cos(x); % instead of looping
  ```
- Use logical indexing instead of `find` unless indices are required.
- Favor built-ins (`sum`, `mean`, `accumarray`, `conv`, `fft`, `unique`) over manual loops.
- Avoid `arrayfun`/`cellfun` for speed (fine for clarity).

### 2. Preallocation
- **Critical Rule**: Preallocate arrays with `zeros`, `ones`, `nan`, `cell`, or `spalloc`. 
- Never grow arrays in a loop.
- Preallocate cell arrays and structs if filled in loops.

### 3. Memory & Loop Layout
- MATLAB is column-major:
  ```matlab
  for j = 1:n
      for i = 1:m
          A(i,j) = A(i,j) + c;
      end
  end
  ```
- Access by columns for contiguous memory.
- Minimize transposes and temporary copies.
- Use sparse matrices for large, mostly-zero data.

### 4. Linear Algebra
- Use `A\b` instead of `inv(A)*b`.
- Reuse factorizations (`chol`, `qr`, `lu`) instead of solving repeatedly.
- Avoid forming `A'*A` explicitly unless well-conditioned.

### 5. ODEs & Simulation Kernels
- Match solver to problem: `ode45` (non-stiff), `ode15s`/`ode23t` (stiff), `ode113` (high-accuracy).
- Provide Jacobians/mass matrices via `odeset` for speed.
- Keep RHS functions pure: no plotting, no I/O, no array growth.
- Vectorize RHS for multi-trajectory integration.

### 6. Parallel & GPU
- Use `parfor` when iterations are independent; preallocate sliced outputs.
- Use `parfeval` for asynchronous tasks.
- For GPU: transfer once (`gpuArray`), compute, then `gather` once.
- Benchmark GPU code with `gputimeit`.

### 7. File I/O & Tables
- Use `readmatrix`/`readtable`, not `xlsread`.
- Convert tables to arrays in inner loops.
- Always close files (`fclose(fid)`).

### 8. Graphics
- Precreate graphics objects; update properties (`XData`, `YData`) instead of recreating.
- Throttle redraws with `drawnow limitrate`.

---

## 🔧 MATLAB Idioms & Patterns
- Preallocate lists via cells then `cat` once.
- Hoist invariants out of loops.
- Use `accumarray`, `blkdiag`, `kron` for aggregation and block ops.
- Prefer linear indexing (`A(idx)`) with `sub2ind` when possible.

---

## 🧪 Testing & Tooling
- **MUST** write deterministic unit tests (`matlab.unittest`).
- **MUST** assert correctness with tolerances.
- **SHOULD** use fixtures for CI-friendly tests.
- **AVOID** including figures or UI in tests.
- Write unit tests with `matlab.unittest`.
- Use fixtures for temp files, CI-friendly tests.
- Time with `timeit`, `tic/toc`, or `profile`.
- Monitor memory with `whos`, `memory`.

---

## 🛑 Common Anti-Patterns
- **AVOID** growing arrays dynamically.
- **AVOID** using `eval` for variable names.
- **AVOID** using `global` unnecessarily.
- **AVOID** row-major loop order (slow).
- **AVOID** repeated plotting in compute loops.
- **AVOID** using `find` when logical indexing suffices.
- Growing arrays dynamically.
- Using `eval` for variable names.
- Using `global` unnecessarily.
- Row-major loop order (slow).
- Repeated plotting inside loops.
- Using `find` where logical indexing suffices.

---

## 📋 Code Review Checklist
- [ ] Preallocation everywhere arrays grow
- [ ] Loops vectorized where possible
- [ ] Column-major loop order respected
- [ ] No `inv`, `eval`, or globals
- [ ] Built-ins used over custom loops
- [ ] Input validation present
- [ ] Descriptive names and comments
- [ ] Unit tests written
- [ ] Plots efficient (no redraw in compute loops)
- [ ] Profiling evidence for performance claims

---

## 📚 Resources
- [MATLAB Performance Tips](https://www.mathworks.com/help/matlab/matlab_prog/techniques-for-improving-performance.html)
- [Vectorization](https://www.mathworks.com/help/matlab/matlab_prog/vectorization.html)
- [Memory Management](https://www.mathworks.com/help/matlab/matlab_prog/resolving-out-of-memory-errors.html)
- [Profiling](https://www.mathworks.com/help/matlab/ref/profile.html)

---

## 🔑 Attitude Check
- If it's slow, **prove it** with the profiler before changing code.
- If it's clever, make it clearer—or make it simpler.
- And remember: **never use `inv`**.
