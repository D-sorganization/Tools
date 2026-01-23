# Star Wars RRT Path Planner - Project Structure 📁

## 🎯 Overview

This document outlines the complete, organized structure of the Star Wars RRT Path Planner project, featuring parallel MATLAB and Python implementations with clear separation of concerns and professional organization.

## 📂 Complete Directory Structure

```
Star_Wars_RRT_Planner/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 .gitignore                   # Git ignore rules
├── 📄 PROJECT_STRUCTURE.md         # This file
│
├── 📁 matlab/                      # MATLAB Implementation
│   ├── 📄 README.md               # MATLAB-specific documentation
│   ├── 📁 src/                    # Source code
│   │   ├── 📁 core/               # Core RRT algorithms
│   │   │   ├── RRT.m              # Main RRT path planning
│   │   │   ├── collisionCheck.m   # Collision detection
│   │   │   ├── stlread.m          # STL file loader
│   │   │   └── shipOrientation.m  # Ship orientation calculator
│   │   ├── 📁 visualization/      # Plotting and animation
│   │   │   ├── plotRRT_Static.m   # Static visualization
│   │   │   ├── plotRRT_Moving.m   # Original moving visualization
│   │   │   ├── plotRRT_Moving_Improved.m # Enhanced visualization
│   │   │   └── plotPursuit_Improved.m # Pursuit visualization
│   │   ├── 📁 ai/                 # Pursuit AI system
│   │   │   └── pursuitSystem.m    # Intelligent pursuit AI
│   │   ├── 📁 gui/                # User interface
│   │   │   └── starWarsPathPlannerGUI.m # Comprehensive GUI
│   │   ├── main_improved.m        # Enhanced main entry point
│   │   ├── main_cinematic.m       # Cinematic mode
│   │   └── main_static.m          # Static mode
│   ├── 📁 models/                 # 3D ship models
│   │   └── falcon_clean_fixed.stl # Millennium Falcon model
│   ├── 📁 data/                   # Obstacle configurations
│   │   ├── obstacles3D.csv        # Default obstacles
│   │   ├── velocities.mat         # Velocity data
│   │   └── angular_velocities.mat # Angular velocity data
│   ├── 📁 examples/               # Example scripts
│   └── 📁 tests/                  # Unit tests
│       └── test_system.m          # System verification
│
├── 📁 python/                     # Python Implementation
│   ├── 📄 requirements.txt        # Python dependencies
│   ├── 📄 setup.py               # Installation script
│   ├── 📁 src/                   # Source code
│   │   ├── 📁 core/              # Core RRT algorithms
│   │   ├── 📁 rendering/         # OpenGL rendering engine
│   │   ├── 📁 ai/                # Pursuit AI system
│   │   ├── 📁 gui/               # User interface
│   │   └── star_wars_rrt.py      # Main application
│   ├── 📁 models/                # 3D ship models
│   ├── 📁 data/                  # Obstacle configurations
│   ├── 📁 examples/              # Example scripts
│   └── 📁 tests/                 # Unit tests
│
├── 📁 docs/                       # Documentation
│   ├── 📁 matlab/                # MATLAB-specific docs
│   │   └── README.md             # MATLAB documentation
│   ├── 📁 python/                # Python-specific docs
│   │   └── README.md             # Python documentation
│   ├── 📁 comparison/            # Performance comparisons
│   └── 📁 api/                   # API reference
│
├── 📁 assets/                     # Shared assets
│   ├── 📁 images/                # Screenshots and diagrams
│   │   └── rrt_static_view.png   # Static visualization example
│   ├── 📁 videos/                # Demo videos
│   │   └── falcon_flight.mp4     # Sample animation
│   └── 📁 models/                # Shared 3D models
│
├── 📁 examples/                   # Cross-platform examples
└── 📁 tests/                      # Integration tests
```

## 🎯 Organization Principles

### **1. Parallel Implementations**
- **MATLAB Version**: Research-focused with comprehensive GUI
- **Python Version**: Performance-optimized with real-time rendering
- **Shared Assets**: Common models, data, and documentation

### **2. Clear Separation of Concerns**
- **Core Algorithms**: RRT, collision detection, path planning
- **Visualization**: Plotting, animation, rendering
- **AI System**: Pursuit behavior, evasion strategies
- **User Interface**: GUIs and control systems

### **3. Professional Structure**
- **Documentation**: Comprehensive guides for each version
- **Testing**: Unit tests and integration tests
- **Examples**: Ready-to-run demonstration scripts
- **Configuration**: Proper dependency management

## 🚀 Key Benefits of This Structure

### **For Developers**
- **Easy Navigation**: Clear folder structure
- **Modular Design**: Independent components
- **Version Control**: Separate tracking of MATLAB and Python
- **Testing**: Dedicated test directories

### **For Users**
- **Quick Start**: Clear entry points for each version
- **Documentation**: Comprehensive guides
- **Examples**: Ready-to-run demonstrations
- **Flexibility**: Choose MATLAB or Python based on needs

### **For Contributors**
- **Clear Guidelines**: Contributing documentation
- **Code Standards**: Style guides for both languages
- **Testing Framework**: Proper test organization
- **Documentation**: API references and tutorials

## 📊 File Organization Details

### **MATLAB Structure**
```
matlab/src/
├── core/           # Core algorithms (RRT, collision detection)
├── visualization/  # Plotting and animation functions
├── ai/            # Pursuit AI system
├── gui/           # User interface components
└── *.m            # Main entry points
```

### **Python Structure**
```
python/src/
├── core/          # Core algorithms (RRT, collision detection)
├── rendering/     # OpenGL rendering engine
├── ai/           # Pursuit AI system
├── gui/          # User interface components
└── *.py          # Main application files
```

### **Shared Resources**
```
assets/
├── images/        # Screenshots, diagrams, examples
├── videos/        # Demo videos, animations
└── models/        # Shared 3D models

docs/
├── matlab/        # MATLAB-specific documentation
├── python/        # Python-specific documentation
├── comparison/    # Performance comparisons
└── api/          # API reference
```

## 🎮 Usage Patterns

### **MATLAB Users**
1. Navigate to `matlab/src/`
2. Run `starWarsPathPlannerGUI` for GUI mode
3. Run `main_improved` for command-line mode
4. Check `matlab/README.md` for detailed instructions

### **Python Users**
1. Navigate to `python/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python src/star_wars_rrt.py`
4. Check `docs/python/README.md` for detailed instructions

### **Researchers**
1. Use MATLAB version for prototyping and analysis
2. Use Python version for performance-critical applications
3. Compare results using `docs/comparison/` benchmarks
4. Extend functionality in respective `core/` directories

## 🔧 Maintenance Benefits

### **Version Control**
- **Separate Tracking**: MATLAB and Python changes tracked independently
- **Clear History**: Easy to see which version introduced features
- **Branch Strategy**: Can develop features in parallel

### **Testing**
- **Unit Tests**: Each component has dedicated tests
- **Integration Tests**: Cross-platform functionality testing
- **Performance Tests**: Benchmarking between versions

### **Documentation**
- **Version-Specific**: Each implementation has its own docs
- **Cross-Reference**: Easy to compare features between versions
- **API Reference**: Complete function documentation

## 🌟 Future Extensibility

### **New Features**
- **MATLAB**: Add to appropriate subdirectory in `matlab/src/`
- **Python**: Add to appropriate subdirectory in `python/src/`
- **Shared**: Add to `assets/` or `docs/` as appropriate

### **New Ship Models**
- **3D Models**: Add to respective `models/` directories
- **Behaviors**: Add to respective `ai/` directories
- **Visualization**: Add to respective `visualization/` or `rendering/` directories

### **New Environments**
- **Obstacle Data**: Add to respective `data/` directories
- **Visual Effects**: Add to respective visualization systems
- **AI Behaviors**: Add to respective AI systems

---

**This structure provides a professional, maintainable, and extensible foundation for the Star Wars RRT Path Planner project!** 🌟 