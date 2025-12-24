# Star Wars RRT Path Planner - MATLAB Version 🔬

## 📁 Directory Structure

```
matlab/
├── src/                    # Source code
│   ├── core/              # Core RRT algorithms
│   │   ├── RRT.m          # Main RRT path planning algorithm
│   │   ├── RRTPlanner.m   # Unified wrapper for RRT variants
│   │   ├── collisionCheck.m # Collision detection system
│   │   ├── collisionCheckEdge.m # Path-wide collision validation
│   │   ├── stlread.m      # STL file loader
│   │   └── shipOrientation.m # Ship orientation calculator
│   ├── environment/       # Environment management utilities
│   │   ├── AsteroidFieldPresets.m # Curated obstacle presets
│   │   └── ObstacleManager.m     # Programmatic obstacle generation
│   ├── visualization/     # Plotting and animation
│   │   ├── plotRRT_Static.m # Static path visualization
│   │   ├── plotRRT_Moving.m # Original moving visualization
│   │   ├── plotRRT_Moving_Improved.m # Enhanced visualization
│   │   └── plotPursuit_Improved.m # Pursuit visualization
│   ├── ai/                # Pursuit AI system
│   │   └── pursuitSystem.m # Intelligent pursuit AI
│   ├── ships/             # Ship asset management
│   │   └── ShipLibrary.m  # STL loader and cache
│   ├── gui/               # User interface
│   │   └── starWarsPathPlannerGUI.m # Comprehensive GUI
│   ├── main_improved.m    # Enhanced main entry point
│   ├── main_cinematic.m   # Cinematic mode
│   └── main_static.m      # Static mode
├── models/                # 3D ship models
│   └── falcon_clean_fixed.stl # Millennium Falcon model
├── data/                  # Obstacle configurations
│   ├── obstacles3D.csv    # Default obstacles
│   ├── velocities.mat     # Velocity data
│   └── angular_velocities.mat # Angular velocity data
├── examples/              # Example scripts
├── tests/                 # Unit tests
│   └── test_system.m      # System verification
└── README.md              # This file
```

## 🚀 Quick Start

### **Option 1: GUI Mode (Recommended)**
```matlab
% Navigate to MATLAB source directory
cd matlab/src

% Launch the comprehensive GUI
starWarsPathPlannerGUI
```

### **Option 2: Command Line Mode**
```matlab
% Navigate to MATLAB source directory
cd matlab/src

% Run enhanced main script
main_improved

% Or run specific modes
main_cinematic    % Cinematic animation
main_static       % Static visualization
```

### **Option 3: Direct Function Calls**
```matlab
% Load obstacles
obstacles = readmatrix('data/obstacles3D.csv');

% Define environment
bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
start = [-0.8, 0, 0];
goal = [0.8, 0, 0];

% Plan path
[nodes, path] = RRT(start, goal, obstacles, bounds);

% Visualize
plotRRT_Moving_Improved(obstacles, path, nodes, start, goal, bounds);
```

## 🎮 Features

### **Core Functionality**
- **RRT Path Planning**: Rapidly-exploring Random Trees algorithm
- **3D Collision Detection**: Sphere and cube obstacle avoidance
- **STL Model Loading**: High-quality 3D ship models
- **Ship Orientation**: Proper nose-forward movement

### **Visualization**
- **Cinematic Animation**: Star Wars-style camera work
- **Dynamic Starfield**: Thousands of animated stars
- **Path Traces**: Real-time path visualization
- **Multiple Views**: Cinematic, chase cam, top-down, side view

### **AI System**
- **Intelligent Pursuit**: Target ships evade when pursued
- **Dynamic Replanning**: Real-time path updates
- **Behavioral States**: Different AI personalities
- **Capture Detection**: Victory conditions and effects

### **GUI Interface**
- **Comprehensive Controls**: Full parameter adjustment
- **Real-time Updates**: Live visualization changes
- **Mode Switching**: Single ship vs pursuit modes
- **Environment Customization**: Load/generate obstacles

## 🔧 Technical Details

### **Core Algorithms**
- **RRT.m**: Main path planning algorithm with goal biasing
- **collisionCheck.m**: Efficient collision detection for spheres and cubes
- **stlread.m**: Robust STL file loader (ASCII and binary)
- **shipOrientation.m**: Proper ship orientation calculation

### **Visualization Engine**
- **plotRRT_Moving_Improved.m**: Enhanced cinematic visualization
- **plotPursuit_Improved.m**: Pursuit scenario visualization
- **Dynamic Camera**: Multiple preset and free camera modes
- **Real-time Effects**: Starfield, laser shots, explosions

### **AI System**
- **pursuitSystem.m**: Intelligent pursuit and evasion AI
- **Behavioral Logic**: Evasion when pursued, goal-seeking when safe
- **Dynamic Goals**: Random goal generation and switching
- **Capture Mechanics**: Distance-based capture detection

## 🎯 Usage Modes

### **Single Ship Navigation**
1. Set start and goal positions
2. Choose ship type and camera view
3. Plan path using RRT algorithm
4. Watch cinematic animation

### **Pursuit Mode**
1. Configure pursuer and target ships
2. Set AI behavior parameters
3. Start pursuit scenario
4. Watch intelligent chase sequence

### **Camera Views**
- **Cinematic**: Classic Star Wars angles
- **Chase Cam**: Follow ships from behind
- **Top Down**: Strategic overview
- **Side View**: Profile perspective
- **Free Camera**: User-controlled movement

## 🛠️ Configuration

### **Path Planning Parameters**
```matlab
% RRT parameters
max_iterations = 5000;    % Maximum RRT iterations
step_size = 0.05;         % Step size for path expansion
goal_radius = 0.1;        % Goal reaching threshold
goal_bias = 0.2;          % Probability of sampling goal
```

### **Animation Settings**
```matlab
% Animation parameters
animation_speed = 0.02;   % Animation speed
show_path = true;         % Display path traces
show_obstacles = true;    % Display obstacles
show_starfield = true;    % Display starfield
```

### **AI Behavior**
```matlab
% Pursuit AI parameters
pursuer_speed = 0.02;     % Pursuer movement speed
target_speed = 0.015;     % Target movement speed
evasion_radius = 0.15;    % Distance to start evading
capture_radius = 0.05;    % Distance for capture
```

## 📊 Performance

### **Path Planning**
- **Simple Environment**: ~1-2 seconds
- **Complex Environment**: ~3-5 seconds
- **Real-time Replanning**: ~0.5-1 seconds

### **Animation**
- **Static Scenes**: 30+ FPS
- **Dynamic Animation**: 10-30 FPS
- **Complex Scenes**: 5-15 FPS

### **Memory Usage**
- **Base Application**: ~100MB
- **With Models**: ~200MB
- **Large Scenes**: ~500MB

## 🧪 Testing

### **System Verification**
```matlab
% Run comprehensive system test
cd matlab/tests
test_system
```

### **Component Testing**
```matlab
% Test individual components
test_rrt_algorithm
test_collision_detection
test_visualization
test_ai_system
```

## 🔧 Troubleshooting

### **Common Issues**

**STL Loading Error:**
```matlab
% Ensure STL file is in models directory
% Check file path in stlread.m
% Verify STL file format (ASCII or binary)
```

**GUI Not Launching:**
```matlab
% Check MATLAB version compatibility
% Ensure all dependencies are available
% Try command-line mode as fallback
```

**Performance Issues:**
```matlab
% Reduce animation speed
% Decrease number of obstacles
% Close other MATLAB windows
% Use static visualization mode
```

## 🚀 Advanced Usage

### **Custom Obstacles**
```matlab
% Generate custom obstacle field
obstacles = generateRandomObstacles(bounds, num_obstacles);

% Load custom obstacle file
obstacles = readmatrix('my_obstacles.csv');
```

### **Custom Ship Models**
```matlab
% Load custom STL model
[V, F] = stlread('my_ship.stl');

% Apply custom orientation
R = shipOrientation(current_pos, next_pos, forward, up);
```

### **Custom AI Behavior**
```matlab
% Modify pursuit parameters
ai.evasion_radius = 0.2;
ai.capture_radius = 0.03;
ai.pursuer_speed = 0.03;
```

## 📚 Related Documentation

- **[Main Project README](../README.md)**: Overall project overview
- **[Python Version](../python/README.md)**: Python implementation
- **[Performance Comparison](../docs/comparison/README.md)**: MATLAB vs Python
- **[API Reference](../docs/api/README.md)**: Function documentation

---

**May the Force be with your MATLAB path planning!** 🌟🔬 