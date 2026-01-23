# Contributing to Star Wars RRT Path Planner 🌟

Thank you for your interest in contributing to the Star Wars RRT Path Planner! This document provides guidelines for contributing to both the MATLAB and Python versions of the project.

## 🚀 Getting Started

### **Prerequisites**
- **MATLAB Version**: MATLAB R2019b or later
- **Python Version**: Python 3.8 or later
- **Git**: For version control
- **Basic knowledge**: RRT algorithms, 3D graphics, or Star Wars enthusiasm! 😄

### **Setup Development Environment**

#### MATLAB Setup
```matlab
% Clone the repository
git clone <repository-url>
cd Star_Wars_RRT_Planner/matlab/src

% Add paths
addpath(genpath('.'));

% Test installation
test_system
```

#### Python Setup
```bash
# Clone the repository
git clone <repository-url>
cd Star_Wars_RRT_Planner/python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/star_wars_rrt.py
```

## 🎯 Development Areas

### **High Priority**
1. **Performance Optimization**
   - Faster path planning algorithms
   - Improved rendering performance
   - Memory usage optimization

2. **AI Enhancement**
   - More sophisticated pursuit behavior
   - Machine learning integration
   - Advanced evasion strategies

3. **Visual Effects**
   - Enhanced 3D graphics
   - Particle effects and explosions
   - Improved lighting and shadows

### **Medium Priority**
1. **New Features**
   - Additional ship types
   - Multiplayer support
   - Mission editor

2. **Documentation**
   - API documentation
   - Tutorial videos
   - Code examples

3. **Testing**
   - Unit test coverage
   - Integration tests
   - Performance benchmarks

### **Low Priority**
1. **Platform Support**
   - Mobile versions
   - VR integration
   - Web version

2. **Community**
   - User guides
   - Community examples
   - Plugin system

## 📝 Code Style Guidelines

### **MATLAB Guidelines**
```matlab
% Use descriptive function names
function [path, nodes] = planOptimalPath(start, goal, obstacles)
    % Function description
    % Inputs: start - starting position [x,y,z]
    %         goal - goal position [x,y,z]
    %         obstacles - obstacle matrix
    % Outputs: path - planned path
    %          nodes - RRT tree nodes
    
    % Use meaningful variable names
    max_iterations = 5000;
    step_size = 0.05;
    
    % Add comments for complex logic
    % Goal-biased sampling for faster convergence
    if rand() < goal_bias
        sample = goal;
    else
        sample = generateRandomPoint(bounds);
    end
end
```

### **Python Guidelines**
```python
# Use descriptive function names and type hints
def plan_optimal_path(
    start: np.ndarray, 
    goal: np.ndarray, 
    obstacles: List[Obstacle]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Plan optimal path using RRT algorithm.
    
    Args:
        start: Starting position [x, y, z]
        goal: Goal position [x, y, z]
        obstacles: List of obstacles
        
    Returns:
        path: Planned path as numpy array
        nodes: RRT tree nodes
    """
    # Use meaningful variable names
    max_iterations = 5000
    step_size = 0.05
    
    # Add comments for complex logic
    # Goal-biased sampling for faster convergence
    if random.random() < goal_bias:
        sample = goal
    else:
        sample = generate_random_point(bounds)
```

## 🧪 Testing Guidelines

### **MATLAB Testing**
```matlab
% Create test functions
function test_rrt_algorithm()
    % Test RRT path planning
    start = [-0.8, 0, 0];
    goal = [0.8, 0, 0];
    obstacles = generateTestObstacles();
    
    [nodes, path] = RRT(start, goal, obstacles, bounds);
    
    % Assertions
    assert(~isempty(path), 'Path should not be empty');
    assert(norm(path(1,:) - start) < 0.1, 'Path should start near start point');
    assert(norm(path(end,:) - goal) < 0.1, 'Path should end near goal');
end
```

### **Python Testing**
```python
# Use pytest for testing
import pytest
import numpy as np

def test_rrt_algorithm():
    """Test RRT path planning algorithm."""
    start = np.array([-0.8, 0, 0])
    goal = np.array([0.8, 0, 0])
    obstacles = generate_test_obstacles()
    
    path, nodes = planner.plan_path(start, goal, obstacles)
    
    # Assertions
    assert path is not None, "Path should not be None"
    assert len(path) > 0, "Path should not be empty"
    assert np.linalg.norm(path[0] - start) < 0.1, "Path should start near start point"
    assert np.linalg.norm(path[-1] - goal) < 0.1, "Path should end near goal"
```

## 🔄 Pull Request Process

### **Before Submitting**
1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Test thoroughly** on both MATLAB and Python versions

### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test addition

## Testing
- [ ] MATLAB tests pass
- [ ] Python tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

### **Bug Report Template**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- MATLAB Version: R2021a
- Python Version: 3.9.7
- Operating System: Windows 10
- Graphics Card: NVIDIA GTX 1660

## Additional Information
Screenshots, error messages, etc.
```

## 💡 Feature Requests

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature would be useful

## Proposed Implementation
How you think it should be implemented

## Alternatives Considered
Other approaches you've considered

## Additional Information
Mockups, examples, etc.
```

## 🌟 Star Wars Themed Contributions

We love Star Wars-themed contributions! Here are some ideas:

### **Ship Models**
- Add new Star Wars ships (X-Wing, TIE Fighter, etc.)
- Improve existing ship models
- Add ship-specific behaviors

### **Environments**
- Create new Star Wars environments (Death Star, Hoth, etc.)
- Add environmental effects (asteroid fields, space debris)
- Implement Star Wars physics

### **AI Behaviors**
- Implement Star Wars character personalities
- Add iconic Star Wars maneuvers
- Create faction-based AI (Rebels vs Empire)

### **Visual Effects**
- Add lightsaber effects
- Implement Star Wars explosions
- Create atmospheric effects

## 📞 Getting Help

### **Communication Channels**
- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Code Review**: Provide constructive feedback on pull requests

### **Code of Conduct**
- Be respectful and inclusive
- Help newcomers
- Share knowledge and expertise
- Have fun with Star Wars references! 😄

## 🎉 Recognition

Contributors will be recognized in:
- **README.md**: List of contributors
- **Release Notes**: Credit for significant contributions
- **Documentation**: Attribution for major features
- **Star Wars Hall of Fame**: Special recognition for outstanding contributions

---

**May the Force be with your contributions!** 🌟

*"Do or do not. There is no try."* - Yoda 