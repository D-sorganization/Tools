# Status Quo & Competitor Analysis Log

**Last Updated:** 2026-01-14
**Maintainer:** Jules
**Purpose:** A running log of competitor analysis and market positioning for the Tools Monorepo projects.

---

## 1. Web Application: Aurora CAS Calculator

**Status Quo:**

- **Tech Stack:** Python (Flask, SymPy), Vanilla JS Frontend.
- **Core Value:** High-precision symbolic mathematics with specialized support for Robotics (Screw Theory, SE(3)) and Linear Algebra.
- **Form Factor:** Mobile-friendly Web Application.

### Competitor Landscape (By Estimated Market Share)

#### 1. Wolfram Alpha (Market Leader)

- **Type:** Computational Knowledge Engine.
- **Tech:** Proprietary (Wolfram Language).
- **Features:** Natural language processing, step-by-step solutions (Pro), massive database of physical constants/data.
- **Gap:** Aurora is less "black box" and allows for Python-style syntax favored by engineers; Wolfram is closed-source.

#### 2. Desmos

- **Type:** Graphing Calculator.
- **Tech:** TypeScript/JavaScript (Client-side).
- **Features:** Best-in-class interactive graphing, highly intuitive UI for education.
- **Gap:** Aurora offers symbolic algebra (CAS) which Desmos lacks; Desmos is purely numerical/graphical.

#### 3. GeoGebra

- **Type:** Interactive Geometry/Algebra System.
- **Tech:** Java/HTML5.
- **Features:** Strong integration of geometry and algebra, widely used in schools.
- **Gap:** Aurora focuses more on engineering/robotics workflows rather than K-12 education.

#### 4. Symbolab

- **Type:** Math Solver.
- **Tech:** Proprietary AI/Rule-based.
- **Features:** Excellent at showing steps for calculus/algebra problems.
- **Gap:** Aurora targets power users needing complex matrix/robotics operations, not just homework help.

---

## 2. Web Application: Unit Converter

**Status Quo:**

- **Tech Stack:** Vanilla JavaScript (PWA), LocalStorage.
- **Core Value:** Offline-first, NIST-compliant precision, specialized Engineering conversions (Gas Flow, Heating Value).
- **Form Factor:** Progressive Web App (Installable).

### Competitor Landscape (By Estimated Market Share)

#### 1. Google Search Converter

- **Type:** Search Feature.
- **Features:** Instant, zero-click access.
- **Gap:** Online only, limited to basic units, no "Gas Flow" or specialized engineering contexts.

#### 2. UnitConverters.net / ConvertUnits.com

- **Type:** Ad-supported Websites.
- **Features:** SEO-heavy, covers thousands of units.
- **Gap:** Cluttered interfaces, requires internet, ads. Aurora is ad-free and cleaner.

#### 3. Mobile Apps (e.g., "Metric Conversion Tool", "Unit Converter Ultimate")

- **Type:** Native Apps (iOS/Android).
- **Features:** Offline support, good UI.
- **Gap:** Aurora offers the same "App" experience via PWA without needing an App Store download, plus specialized engineering logic (NIST specs).

---

## 3. Scientific Modeling: Solar System

**Status Quo:**

- **Tech Stack:** Python (PyGame, PyOpenGL).
- **Core Value:** Educational visualization, Orbital Mechanics (Hohmann Transfers), Open Source.
- **Form Factor:** Desktop Application.

### Competitor Landscape (By Estimated Market Share)

#### 1. Universe Sandbox

- **Type:** Physics Simulator / Game.
- **Tech:** Unity (C#).
- **Features:** Stunning visuals, n-body physics, terraforming, catastrophic collisions.
- **Gap:** Aurora is more "textbook accurate" for orbital transfers and planning, whereas Universe Sandbox focuses on visual spectacle and "what if" scenarios.

#### 2. NASA Eyes

- **Type:** Visualization Tool.
- **Tech:** Web/Unity.
- **Features:** Real mission data, authentic spacecraft models.
- **Gap:** Aurora allows for user-driven trajectory planning/manipulation, NASA Eyes is mostly for viewing.

#### 3. Celestia

- **Type:** 3D Space Simulator.
- **Tech:** C++ (OpenGL).
- **Features:** Massive catalog of stars/galaxies, open-source.
- **Gap:** Aurora is a lighter-weight Python implementation, easier for students to modify/script.

#### 4. SpaceEngine

- **Type:** Procedural Universe Generator.
- **Tech:** Proprietary Engine.
- **Features:** Explorable procedural universe (billions of galaxies).
- **Gap:** Different scale; Aurora focuses on our specific Solar System mechanics.

---

## 4. Research: Golf Biomechanics Simulator

**Status Quo:**

- **Tech Stack:** MATLAB.
- **Core Value:** Research-grade reproducibility, golf-specific physics engine.
- **Form Factor:** Research Codebase.

### Competitor Landscape (By Estimated Market Share)

#### 1. OpenSim (Stanford)

- **Type:** Biomechanics Research Platform.
- **Tech:** C++/Python/Java.
- **Features:** The academic standard for musculoskeletal modeling.
- **Gap:** Generic (requires custom models for golf); Aurora is purpose-built for Golf.

#### 2. Commercial Simulators (Golfzon, Trackman, Full Swing)

- **Type:** Hardware + Software Ent.
- **Features:** Radar/Camera tracking, gamification, luxury rendering.
- **Gap:** These are "Black Boxes" for entertainment/training. Aurora is "Glass Box" for physics research and algorithm development.

#### 3. Visual3D (C-Motion)

- **Type:** Analysis Software.
- **Features:** Processing Motion Capture data.
- **Gap:** Focuses on data processing, whereas Aurora focuses on _simulation_ and _modeling_.

---

## 5. Research & Education: Star Wars RRT Path Planner

**Status Quo:**

- **Tech Stack:** Python (PyGame, PyOpenGL) & MATLAB (App Designer).
- **Core Value:** "Edutainment" focused visualization of RRT algorithms, high-performance Python implementation vs Scientific MATLAB version.
- **Form Factor:** Desktop Application / Simulation.

### Competitor Landscape (By Estimated Market Share)

#### 1. OMPL (Open Motion Planning Library)

- **Type:** Standard Library.
- **Tech:** C++.
- **Features:** The academic and industrial standard for sampling-based algorithms (RRT, PRM, EST).
- **Gap:** OMPL is a backend library without inherent visualization; Aurora provides a "gamified" visual experience out of the box.

#### 2. MoveIt! (ROS)

- **Type:** Robotics Framework.
- **Tech:** C++/Python (ROS).
- **Features:** Full robotic manipulation pipeline (Inverse Kinematics + Path Planning + Collision Checking).
- **Gap:** MoveIt! has a steep learning curve and requires ROS; Aurora is a standalone, lightweight tool for algorithm demonstration.

#### 3. MATLAB Robotics System Toolbox

- **Type:** Commercial Toolbox.
- **Tech:** MATLAB.
- **Features:** Robust, validated algorithms for industry.
- **Gap:** Expensive commercial license; Aurora's Python version is free and open source.

#### 4. Game Engines (Unity/Unreal Navigation)

- **Type:** Game Middleware.
- **Tech:** C#/C++.
- **Features:** NavMesh based pathfinding (A\* variants).
- **Gap:** Game engines typically use A\* on NavMeshes; Aurora specifically visualizes RRT/Sampling-based methods which are better for high-dimensional configuration spaces (like robotic arms).

---

## 6. Industrial Tool: Data Processor

**Status Quo:**

- **Tech Stack:** Python (CustomTkinter, SciPy, Pandas).
- **Core Value:** Standalone GUI for Chemical/Industrial signal processing (filtering, smoothing, FFT) without needing coding skills.
- **Form Factor:** Desktop GUI & CLI.

### Competitor Landscape (By Estimated Market Share)

#### 1. Microsoft Excel

- **Type:** Spreadsheet.
- **Features:** Ubiquitous, basic charts.
- **Gap:** Excel struggles with massive time-series datasets and lacks advanced signal processing (FFT, Savitzky-Golay) out of the box.

#### 2. Seeq

- **Type:** Advanced Analytics for Process Manufacturing.
- **Features:** Enterprise-grade time-series search, cleansing, and context adding.
- **Gap:** Seeq is an expensive enterprise SaaS; Aurora Data Processor is a lightweight, local tool for quick ad-hoc analysis.

#### 3. TrendMiner

- **Type:** Self-Service Industrial Analytics.
- **Features:** Pattern recognition, predictive maintenance.
- **Gap:** Similar to Seeq, high cost and complexity.

#### 4. Python Notebooks (Jupyter/Pandas)

- **Type:** Development Environment.
- **Features:** Infinite flexibility.
- **Gap:** Requires programming knowledge. Aurora Data Processor wraps this power in a GUI accessible to non-coding process engineers.
