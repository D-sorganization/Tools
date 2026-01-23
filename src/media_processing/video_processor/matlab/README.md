# MATLAB Integration for Golf Swing Video Analysis

This directory contains MATLAB code for physics modeling and analysis of golf swings.

## Structure

```
matlab/
├── models/           # Simscape Multibody models and physics simulations
├── api/              # MATLAB REST API endpoints (optional)
├── utils/            # Helper functions (pose conversion, model fitting)
└── tests/            # Test scripts for MATLAB models
```

## Key Features

- **Triple Pendulum Model**: Simscape Multibody model of golf swing
- **Swing Analysis**: Convert MediaPipe poses to joint angles
- **Model Fitting**: Fit pendulum parameters to detected poses
- **Physics Simulation**: Calculate forces, energy, dynamics

## Usage

### Local Development

1. Open MATLAB and navigate to this directory
2. Run models locally:
   ```matlab
   cd matlab/models
   results = pendulum_model(pose_data);
   ```

3. Export results for web platform:
   ```matlab
   export_results_to_web(results, 'results.json');
   ```

### Integration with Web Platform

The web platform can:
- Import MATLAB results via JSON
- Call MATLAB API endpoints (if configured)
- Use MATLAB-computed model parameters

## MATLAB Runtime (No License Needed)

For production, you can use MATLAB Runtime (free, no license):
- Python bridge calls MATLAB Runtime
- No MATLAB license needed for deployment
- Users don't need MATLAB installed

See `docs/GOLF_VIDEO_MATLAB_INTEGRATION.md` for details.
