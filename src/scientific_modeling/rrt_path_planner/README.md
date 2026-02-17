# Star Wars RRT Path Planner 🌟

A comprehensive, cinematic Star Wars-style path planning simulator featuring intelligent pursuit scenarios, real-time 3D visualization, and advanced AI behavior.

## 🚀 Project Overview

This project provides **two parallel implementations** of an enhanced RRT (Rapidly-exploring Random Tree) path planner:

- **MATLAB Version**: Research-focused with comprehensive GUI and scientific visualization
- **Python Version**: Performance-optimized with real-time 3D rendering and GPU acceleration

Both versions feature Star Wars-themed pursuit scenarios with intelligent AI, cinematic camera views, and professional-grade visual effects.

## 📁 Project Structure

```
Star_Wars_RRT_Planner/
├── 📁 matlab/                    # MATLAB Implementation
│   ├── 📁 src/                   # Source code
│   │   ├── core/                 # Core RRT algorithms
│   │   ├── visualization/        # Plotting and animation
│   │   ├── ai/                   # Pursuit AI system
│   │   └── gui/                  # User interface
│   ├── 📁 models/                # 3D ship models (STL files)
│   ├── 📁 data/                  # Obstacle configurations
│   ├── 📁 examples/              # Example scripts
│   └── 📁 tests/                 # Unit tests
│
├── 📁 python/                    # Python Implementation
│   ├── 📁 src/                   # Source code
│   │   ├── core/                 # Core RRT algorithms
│   │   ├── rendering/            # OpenGL rendering engine
│   │   ├── ai/                   # Pursuit AI system
│   │   └── gui/                  # User interface
│   ├── 📁 models/                # 3D ship models
│   ├── 📁 data/                  # Obstacle configurations
│   ├── 📁 examples/              # Example scripts
│   └── 📁 tests/                 # Unit tests
│
├── 📁 docs/                      # Documentation
│   ├── matlab/                   # MATLAB-specific docs
│   ├── python/                   # Python-specific docs
│   └── comparison/               # Performance comparisons
│
├── 📁 assets/                    # Shared assets
│   ├── images/                   # Screenshots and diagrams
│   ├── videos/                   # Demo videos
│   └── models/                   # Shared 3D models
│
├── 📁 examples/                  # Cross-platform examples
└── 📁 tests/                     # Integration tests
```

## 🎯 Features

### **Core Functionality**

- **RRT Path Planning**: Rapidly-exploring Random Trees algorithm
- **3D Environment**: Dynamic obstacle avoidance in 3D space
- **Ship Navigation**: Realistic ship movement with proper orientation
- **Pursuit Scenarios**: Intelligent AI-driven chase sequences

### **Visual Effects**

- **Cinematic Starfield**: Dynamic background with thousands of stars
- **Ship Models**: High-quality 3D models (Millennium Falcon, etc.)
- **Path Visualization**: Real-time path traces and RRT trees
- **Camera System**: Multiple cinematic camera views

### **AI Behavior**

- **Intelligent Pursuit**: Target ships evade when pursued
- **Dynamic Replanning**: Real-time path updates
- **Collision Avoidance**: Sophisticated obstacle navigation
- **Behavioral States**: Different AI personalities

## 🛠️ Quick Start

### MATLAB Version

```matlab
% Navigate to MATLAB directory
cd matlab/src

% Run the main application
main_improved

% Or launch GUI directly
starWarsPathPlannerGUI
```

### Python Version

```bash
# Navigate to Python directory
cd python/src

# Install dependencies
pip install -r requirements.txt

# Run the application
python star_wars_rrt.py
```

## 📊 Performance Comparison

| Feature            | MATLAB      | Python           |
| ------------------ | ----------- | ---------------- |
| **Animation FPS**  | 10-30       | 60+              |
| **Path Planning**  | ~1-5s       | ~0.1-1s          |
| **Memory Usage**   | 100-500MB   | 50-200MB         |
| **GPU Support**    | Limited     | Full CUDA/OpenCL |
| **Cross-platform** | Windows/Mac | All platforms    |
| **Cost**           | Expensive   | Free             |

## 🎮 Usage Modes

### **Single Ship Navigation**

- Plan optimal path through obstacle field
- Cinematic camera following
- Real-time path visualization

### **Pursuit Mode**

- Two ships engage in intelligent chase
- Dynamic evasion and pursuit behavior
- Capture detection and victory effects

### **Camera Views**

- **Cinematic**: Classic Star Wars-style angles
- **Chase Cam**: Follow ships from behind
- **Top Down**: Strategic overview
- **Side View**: Profile perspective
- **Free Camera**: User-controlled movement

## 🔧 Technical Details

### **MATLAB Implementation**

- **GUI Framework**: MATLAB App Designer
- **3D Graphics**: MATLAB's built-in 3D plotting
- **Path Planning**: Custom RRT implementation
- **File Formats**: STL, CSV, MAT

### **Python Implementation**

- **Rendering**: OpenGL with PyGame
- **3D Graphics**: Hardware-accelerated rendering
- **Path Planning**: NumPy-optimized RRT
- **Performance**: GPU acceleration with CUDA

## 🚀 Advanced Features

### **MATLAB Version**

- **Comprehensive GUI**: Full control panel with real-time updates
- **Scientific Visualization**: Publication-quality plots
- **Simulink Integration**: System modeling capabilities
- **Toolbox Support**: Robotics, Computer Vision toolboxes

### **Python Version**

- **Real-time Rendering**: 60+ FPS animations
- **GPU Acceleration**: CUDA/OpenCL support
- **Advanced AI**: Machine learning integration
- **Cross-platform**: Windows, Mac, Linux

## 📚 Documentation

- **[MATLAB Documentation](docs/matlab/README.md)**: Complete MATLAB guide
- **[Python Documentation](docs/python/README.md)**: Complete Python guide
- **[Performance Comparison](docs/comparison/README.md)**: Detailed benchmarks
- **[API Reference](docs/api/README.md)**: Function documentation

## 🧪 Testing

### **Unit Tests**

```bash
# MATLAB Tests
cd matlab/tests
run_tests

# Python Tests
cd python/tests
pytest
```

### **Integration Tests**

```bash
# Cross-platform tests
cd tests
python run_integration_tests.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Areas**

1. **Performance**: Optimize algorithms and rendering
2. **AI**: Enhance ship behavior and tactics
3. **Graphics**: Add visual effects and shaders
4. **Physics**: Implement realistic space physics
5. **Testing**: Add comprehensive test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original RRT implementation team
- Star Wars community for inspiration
- MATLAB and Python communities
- 3D model creators and contributors

## 🌟 Star Wars References

This project pays homage to the iconic Star Wars universe with:

- **Millennium Falcon**: Primary ship model
- **Asteroid Field**: Dynamic obstacle environments
- **Pursuit Scenarios**: Inspired by classic chase sequences
- **Cinematic Style**: Star Wars-inspired camera work

---

**May the Force be with your path planning!** 🌟

_"Sir, the possibility of successfully navigating an asteroid field is approximately 3,720 to 1!"_ - C-3PO
