# Aurora CAS Functionality Coverage

This calculator carries forward the TI-89’s computer-algebra strengths while layering in modern
touch controls, clearer UX cues, and a richer linear algebra surface.

## Implemented core math and CAS features

- **Constants and numeric tools:** `pi`, `e`, `I`, rounding, floor/ceiling, integer factorials,
  permutations `nPr`, and combinations `nCr`.
- **Trigonometry and hyperbolic sets:** full circular, inverse, and hyperbolic functions
  (`sin`, `cos`, `tan`, `csc`, `sec`, `cot`, `asin`, `acos`, `atan`, `csch`, `sech`, `coth`, and
  their inverses), plus power/root helpers (`sqrt`, `cbrt`).
- **Complex arithmetic:** component access (`re`/`im`/`real`/`imag`), magnitude/argument (`abs`,
  `norm`, `arg`), conjugation, and polar/rectangular helpers (`cis`, `polar`, `rect`).
- **Algebra and simplification:** factor/expand/cancel, rational and trigonometric simplifiers,
  partial fraction decomposition (`apart`), `collect`, `together`, `factor_terms`, `gcd`, `lcm`,
  and generic `simplify` plus direct equation balancing for symbolic unknowns.
- **Summations and products:** discrete `sum` and `product` over indexed ranges.
- **Matrix and vector math:** constructors (`Matrix`, `eye`, `ones`, `zeros`), structure helpers
  (`diag`, `rank`, `trace`), reductions (`rref`/`row_reduce`), inverses/transposes/determinants,
  vector ops (`dot`, `cross`, `norm`), decompositions (`qr`, `lu`, `svd`), eigentools
  (`eigenvals`, `eigenvects`, `charpoly`), null/row/column spaces, pseudoinverse, and linear
  system helpers (`linsolve`, `solve_linear`).
- **Matrix exponentials and powers:** `matrix_exp`/`expm`, `matrix_log`/`logm`, `matrix_power`,
  and `block_diag` for quickly assembling dynamic models.
- **Robotics and screw theory:** skew/vee utilities, SE(3) hat/vee, screw axis builder, twist
  exponentials for rigid transforms, and adjoint mappings for frame changes.
- **Calculus and series:** symbolic derivatives, definite/indefinite integrals, directional limits,
  Taylor series, and solutions for ordinary differential equations.
- **Equation solving:** single equations, simultaneous linear systems, symbolic substitution, and
  ANS recall for chained calculations.

## UX highlights

- Mode-specific soft keys cover CAS, algebra, solving, calculus, systems, limits, series, and
  differential equations with clear, modern labels.
- Keypad rows expose matrix, algebraic, and complex tokens alongside EE/ANS/n! variables, plus
  one-tap access to eigentools, decompositions, and equation simplifiers.
- A touch-focused editing layer allows tap-to-place cursor, history recall, and ANS insertion
  directly from the on-screen display for tablet-friendly use.
- Copy buttons sit beside the result display for fast clipboard transfers of inputs or outputs.

## Not yet covered

- Graphing (2D/3D), geometry, and statistics applications.
- File/app management, program editor, and data/matrix editors.
- Numeric solvers for regressions, custom units conversions, or probability distributions beyond
  factorial/nCr/nPr.
- Exact hardware key chords and physical IO features.
