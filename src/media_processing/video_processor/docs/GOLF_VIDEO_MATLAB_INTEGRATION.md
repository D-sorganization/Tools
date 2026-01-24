# Golf Swing Video Analysis Platform - MATLAB Integration Guide

## 🎯 MATLAB Integration Overview

Since you're familiar with Python and have MATLAB with Simscape Multibody, we can create a hybrid approach that leverages MATLAB for physics modeling while using web technologies for the platform.

---

## ❓ Can You Build This Entirely in MATLAB?

### Short Answer: **Not Recommended for a Web-Based Sharing Platform**

### Why MATLAB Alone Won't Work Well:

#### ❌ Limitations:
1. **Home License Restriction**: Can't publish apps (you mentioned this)
2. **Web Deployment**: MATLAB Compiler Web Apps require server license
3. **Sharing**: Users would need MATLAB installed
4. **Cost**: Each user needs MATLAB license ($2,150/year)
5. **Performance**: Web apps are slower than native browser apps
6. **Mobile**: MATLAB apps don't work on mobile devices
7. **Cloud Costs**: MATLAB Web App Server is expensive ($$$)

#### ✅ What MATLAB IS Great For:
1. **Physics Modeling**: Simscape Multibody for pendulum models
2. **Data Analysis**: Advanced signal processing
3. **Simulation**: Swing dynamics modeling
4. **Prototyping**: Rapid development of physics models

---

## ✅ Recommended Hybrid Approach

### Best Strategy: **MATLAB for Modeling, Web for Platform**

```
┌─────────────────────────────────────────┐
│  Web Platform (TypeScript/JavaScript)    │
│  - Video upload/playback                  │
│  - User interface                         │
│  - Sharing & collaboration                │
│  - Real-time analysis                     │
└───────────────┬─────────────────────────┘
                 │
                 │ API calls to MATLAB
                 ▼
┌─────────────────────────────────────────┐
│  MATLAB Services (Python-like API)      │
│  - Physics models (Simscape Multibody)   │
│  - Advanced signal processing           │
│  - Swing analysis algorithms            │
│  - Export to JSON/API                   │
└─────────────────────────────────────────┘
```

---

## 🏗️ Architecture: MATLAB Integration

### Option 1: MATLAB as Backend Service (Recommended)

**MATLAB Component**:
- Runs physics models (Simscape Multibody)
- Processes swing data
- Returns results via API

**Web Component**:
- Handles user interface
- Manages videos
- Calls MATLAB for physics analysis
- Handles sharing/collaboration

### Option 2: MATLAB as Standalone Tool

**For You (Developer)**:
- MATLAB app for creating/switching models
- Export model parameters
- Test physics simulations

**For Users (Platform)**:
- Web platform uses pre-computed models
- Import MATLAB results
- Real-time browser-based analysis

---

## 📁 Project Structure with MATLAB Integration

```
golf-swing-analyzer/
├── apps/
│   └── web/                    # Next.js frontend (TypeScript)
│       ├── app/
│       ├── components/
│       └── lib/
│
├── matlab/                     # MATLAB code for physics modeling
│   ├── models/
│   │   ├── pendulum_model.m    # Pendulum simulation
│   │   ├── swing_analysis.m    # Swing analysis functions
│   │   └── simscape_models/    # Simscape Multibody models
│   ├── api/
│   │   ├── matlab_api.m        # MATLAB REST API server
│   │   └── export_json.m       # Export results to JSON
│   ├── utils/
│   │   ├── pose_to_angles.m    # Convert MediaPipe poses to angles
│   │   └── model_fitting.m     # Fit pendulum to data
│   └── tests/
│       └── test_models.m       # Test MATLAB models
│
├── services/
│   ├── api/                    # Node.js backend (TypeScript)
│   │   └── routes/
│   │       └── matlab.ts       # Call MATLAB services
│   └── matlab-worker/          # Optional: MATLAB Runtime service
│       └── matlab_api.py       # Python bridge to MATLAB
│
└── docs/
    └── matlab_integration.md   # This file
```

---

## 🔧 Implementation Strategy

### Phase 1: Build Web Platform
- ✅ Web interface (Next.js/TypeScript)
- ✅ Video upload/playback
- ✅ Pose detection (MediaPipe - browser)
- ✅ Basic analysis

### Phase 2: Add MATLAB Integration
- ✅ Develop Simscape Multibody models
- ✅ Create MATLAB API endpoints
- ✅ Connect web platform to MATLAB
- ✅ Import MATLAB model results

### Phase 3: Hybrid Analysis
- ✅ Browser: Real-time pose detection
- ✅ MATLAB: Advanced physics simulation
- ✅ Web: Combine results

---

## 💻 MATLAB Implementation

### MATLAB Pendulum Model (Simscape Multibody)

```matlab
% matlab/models/pendulum_model.m
% Triple pendulum model for golf swing

function results = pendulum_model(pose_data, options)
    % PENDULUM_MODEL Simulate golf swing as triple pendulum
    %
    % Inputs:
    %   pose_data - MediaPipe pose landmarks (JSON or struct)
    %   options   - Model parameters (mass, length, etc.)
    %
    % Outputs:
    %   results   - Struct with:
    %               .trajectory - Joint angles over time
    %               .forces    - Forces at joints
    %               .energy    - Kinetic/potential energy
    %               .fit_error - Error between model and data

    arguments
        pose_data (1,1) struct
        options.mass_upper_arm (1,1) double = 3.0      % kg
        options.mass_forearm (1,1) double = 1.5        % kg
        options.mass_club (1,1) double = 0.4          % kg
        options.length_upper_arm (1,1) double = 0.35  % m
        options.length_forearm (1,1) double = 0.30    % m
        options.length_club (1,1) double = 1.15       % m
    end

    % Extract key points from pose data
    left_shoulder = pose_data.landmarks(11, :);
    left_elbow = pose_data.landmarks(13, :);
    left_wrist = pose_data.landmarks(15, :);

    % Calculate joint angles
    upper_arm_angle = calculate_angle(left_shoulder, left_elbow);
    forearm_angle = calculate_angle(left_elbow, left_wrist);

    % Simulate with Simscape Multibody
    model_name = 'golf_swing_pendulum';
    simOut = sim(model_name, ...
        'StopTime', num2str(length(pose_data.frames)), ...
        'InitialState', [upper_arm_angle; forearm_angle; 0; 0], ...
        'MassUpperArm', options.mass_upper_arm, ...
        'MassForearm', options.mass_forearm, ...
        'MassClub', options.mass_club);

    % Extract results
    results.trajectory = simOut.logsout.get('angles').Values.Data;
    results.forces = simOut.logsout.get('forces').Values.Data;
    results.energy = simOut.logsout.get('energy').Values.Data;

    % Calculate fit error
    measured_angles = [upper_arm_angle; forearm_angle];
    model_angles = results.trajectory(1:2, :);
    results.fit_error = mean((measured_angles - model_angles).^2);
end
```

### MATLAB REST API (Export Results)

```matlab
% matlab/api/matlab_api.m
% Create REST API endpoint to serve MATLAB results

function start_matlab_api(port)
    % START_MATLAB_API Start MATLAB REST API server
    %
    % Usage:
    %   start_matlab_api(8080)

    arguments
        port (1,1) double = 8080
    end

    % Create web server
    server = matlab.webapps.Server('Port', port);

    % Define endpoints
    server.addRoute('POST', '/api/matlab/pendulum', @handle_pendulum);
    server.addRoute('POST', '/api/matlab/analyze', @handle_analyze);
    server.addRoute('GET', '/api/matlab/health', @handle_health);

    % Start server
    server.start();
    fprintf('MATLAB API server running on port %d\n', port);
end

function response = handle_pendulum(request)
    % Handle pendulum model request
    pose_data = jsondecode(request.Body);

    % Run pendulum model
    results = pendulum_model(pose_data);

    % Return JSON response
    response = matlab.webapps.HttpResponse();
    response.StatusCode = 200;
    response.Body = jsonencode(results);
    response.ContentType = 'application/json';
end
```

---

## 🔗 Connecting Web Platform to MATLAB

### Option 1: MATLAB Compiler (Server License Needed)

```typescript
// services/api/routes/matlab.ts
// Call MATLAB REST API

import axios from 'axios';

const MATLAB_API_URL = process.env.MATLAB_API_URL || 'http://localhost:8080';

export async function analyzeSwingWithMATLAB(poseData: any) {
  try {
    const response = await axios.post(
      `${MATLAB_API_URL}/api/matlab/pendulum`,
      poseData,
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000, // 30 seconds
      }
    );

    return response.data;
  } catch (error) {
    console.error('MATLAB API error:', error);
    throw new Error('MATLAB analysis failed');
  }
}
```

### Option 2: MATLAB Runtime + Python Bridge

```python
# services/matlab-worker/matlab_api.py
# Python bridge to MATLAB Runtime (no license needed!)

import matlab.engine
import json

class MATLABAnalyzer:
    def __init__(self):
        # Start MATLAB engine (Runtime - no license needed!)
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('./matlab/models')

    def analyze_pendulum(self, pose_data):
        """Run pendulum model in MATLAB"""
        # Convert Python dict to MATLAB struct
        pose_struct = self._dict_to_matlab_struct(pose_data)

        # Call MATLAB function
        results = self.eng.pendulum_model(pose_struct, nargout=1)

        # Convert MATLAB struct to Python dict
        return self._matlab_struct_to_dict(results)

    def _dict_to_matlab_struct(self, data):
        """Convert Python dict to MATLAB struct"""
        # Implementation here
        pass
```

```typescript
// services/api/routes/matlab.ts
// Call Python bridge (which calls MATLAB Runtime)

export async function analyzeSwingWithMATLAB(poseData: any) {
  // Call Python service (which uses MATLAB Runtime)
  const response = await fetch('http://localhost:5000/api/matlab/pendulum', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(poseData),
  });

  return await response.json();
}
```

---

## 🎯 Recommended Workflow for You

### For Development (Using Your MATLAB License)

1. **Build Models in MATLAB** (Simscape Multibody)
   ```matlab
   % Develop pendulum model
   % Test with sample data
   % Export model parameters
   ```

2. **Export Results to Web Platform**
   ```matlab
   % Save results to JSON
   results = pendulum_model(test_data);
   writestruct(results, 'model_results.json');
   ```

3. **Web Platform Uses Results**
   ```typescript
   // Import MATLAB results
   import modelResults from '@/matlab/model_results.json';

   // Use in web app
   displayPendulumResults(modelResults);
   ```

### For Production (No MATLAB License Needed for Users)

1. **Convert MATLAB Models to JavaScript**
   - Rewrite pendulum physics in TypeScript
   - Use same algorithms but in JavaScript
   - No MATLAB dependency

2. **Use MATLAB Runtime** (Free, No License)
   - Deploy MATLAB Runtime on server
   - Python bridge calls MATLAB Runtime
   - Users don't need MATLAB

3. **Pre-Compute Models**
   - Generate model library in MATLAB
   - Export to JSON/TypeScript
   - Web platform uses pre-computed models

---

## 📊 MATLAB vs Web Platform Comparison

| Feature | MATLAB Only | Web Platform | Hybrid (Recommended) |
|---------|-------------|--------------|----------------------|
| **User Access** | ❌ Needs MATLAB | ✅ Any browser | ✅ Any browser |
| **Sharing** | ❌ Hard (need MATLAB) | ✅ Easy (just links) | ✅ Easy (just links) |
| **Cost** | ❌ $2,150/user/year | ✅ Free for users | ✅ Free for users |
| **Physics Models** | ✅ Excellent (Simscape) | ⚠️ Must rewrite | ✅ Use MATLAB for models |
| **Real-Time** | ❌ Slow (web apps) | ✅ Fast (browser) | ✅ Fast (browser) |
| **Mobile** | ❌ No | ✅ Yes | ✅ Yes |
| **Development** | ✅ Easy for you | ⚠️ New language | ✅ Use both strengths |

---

## 🚀 Recommended Approach for You

### Phase 1: Build Web Platform (TypeScript)
- ✅ User interface
- ✅ Video upload/playback
- ✅ Pose detection (MediaPipe)
- ✅ Basic analysis

### Phase 2: Add MATLAB Integration
- ✅ Develop Simscape Multibody models locally
- ✅ Test models with MATLAB
- ✅ Export model parameters/results
- ✅ Import into web platform

### Phase 3: Convert to JavaScript (Optional)
- ✅ Rewrite pendulum physics in TypeScript
- ✅ Remove MATLAB dependency
- ✅ Faster (no API calls)

### Alternative: Keep MATLAB for You
- ✅ Develop models in MATLAB
- ✅ Export results to JSON
- ✅ Web platform imports JSON
- ✅ No MATLAB needed in production

---

## 💡 MATLAB Integration Examples

### Example 1: Export MATLAB Results to JSON

```matlab
% matlab/utils/export_results.m
function export_results_to_web(results, output_file)
    % EXPORT_RESULTS_TO_WEB Export MATLAB results for web platform

    arguments
        results (1,1) struct
        output_file (1,1) string = 'results.json'
    end

    % Convert MATLAB struct to JSON-compatible format
    json_data = struct();
    json_data.trajectory = results.trajectory';
    json_data.forces = results.forces';
    json_data.energy = results.energy';
    json_data.fit_error = results.fit_error;
    json_data.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');

    % Write JSON file
    fid = fopen(output_file, 'w');
    fprintf(fid, '%s', jsonencode(json_data));
    fclose(fid);

    fprintf('Results exported to %s\n', output_file);
end
```

### Example 2: Import MATLAB Results in Web

```typescript
// apps/web/lib/matlab/import-results.ts
// Import MATLAB-exported results

export interface MATLABResults {
  trajectory: number[][];
  forces: number[][];
  energy: {
    kinetic: number[];
    potential: number[];
  };
  fit_error: number;
  timestamp: string;
}

export async function loadMATLABResults(
  jsonPath: string
): Promise<MATLABResults> {
  const response = await fetch(jsonPath);
  return await response.json();
}

export function visualizePendulumResults(results: MATLABResults) {
  // Visualize MATLAB results in web interface
  // Use Three.js to display pendulum motion
  // Show forces, energy, etc.
}
```

---

## ✅ Bottom Line

### Can You Build Entirely in MATLAB?
**Technically possible, but NOT recommended for web-based sharing platform**

### Recommended Approach:
**Hybrid: MATLAB for Modeling, Web for Platform**

- ✅ **Develop models in MATLAB** (Simscape Multibody)
- ✅ **Test locally with MATLAB**
- ✅ **Export results to web platform**
- ✅ **Web platform handles sharing/user interface**
- ✅ **Users access via browser (no MATLAB needed)**

### For You:
- Use MATLAB for rapid physics model development
- Use web platform for user interface and sharing
- Best of both worlds!

---

## 📚 Next Steps

1. **Set up project structure** (TypeScript/JavaScript)
2. **Create MATLAB models folder** (Simscape Multibody)
3. **Build web platform** (Next.js/React)
4. **Create MATLAB export functions** (JSON/API)
5. **Integrate MATLAB results** into web platform

**You get to use MATLAB strengths (Simscape Multibody) while building a modern web platform!**

---

*Last Updated: MATLAB Integration Guide*
