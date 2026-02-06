# Getting Started with the Unified Tools Launcher

Welcome to the Unified Tools Launcher! This guide will help you get started with the comprehensive tool collection in the Tools repository.

## Prerequisites

Before using the launcher, ensure you have:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Core runtime |
| PyQt6 | 6.6.0+ | GUI framework |
| Git | Latest | Version control |
| MATLAB | R2020a+ | MATLAB-based tools (optional) |
| Node.js | 18+ | Web applications (optional) |

## Quick Start

### 1. Launch the Application

```bash
# From the repository root
python UnifiedToolsLauncher.py
```

### 2. Navigate the Interface

The launcher presents a tabbed interface organized by tool category:

- **Process Engineering** - 24+ industrial calculators
- **Scientific Modeling** - Simulation and modeling tools
- **Signal Processing** - Function generators and filters
- **Data Processing** - Data analysis platforms
- **Robotics** - URDF builders, inertia calculators
- **Media Processing** - Audio/video tools
- **Web Applications** - Browser-based tools
- **Development** - Project utilities

### 3. Launch a Tool

1. Click on a category tab
2. Browse the tool cards
3. Click **Launch Tool** on the desired tool

## Interface Elements

### Header

- **Title**: Shows the launcher name
- **Debug Mode**: Checkbox to enable verbose logging

### Tool Cards

Each tool is displayed as a card showing:
- Tool name
- Type badge (Python, MATLAB, Web, Browser)
- Description
- Path to the tool
- Launch button

### Activity Log

The bottom panel shows:
- Tool launch status
- Error messages
- Debug output (when enabled)

### Status Bar

Shows the repository root path for reference.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F1 | Open User Manual |
| Ctrl+L | Launch selected tool |
| Ctrl+Tab | Next category tab |
| Ctrl+Shift+Tab | Previous category tab |
| Ctrl+Q | Quit launcher |

## Theme Support

The launcher supports multiple color themes. Access the Theme menu to switch between:
- Light and Dark themes
- Editor themes (Monokai, Dracula, One Dark)
- Office themes (MS Word, MS Excel)
- Custom themes

## Getting Help

### User Manual

Press **F1** or use Help > User Manual to open the comprehensive documentation.

### Context Help

Click the **?** button next to any tool for specific help about that tool.

### Tool Descriptions

Each tool card includes a brief description. Hover over elements for additional tooltips.

## Debug Mode

Enable Debug Mode by checking the box in the header. This provides:
- Detailed launch logging
- Command-line arguments shown
- Error stack traces
- Process output capture

## Troubleshooting

### Tool Won't Launch

1. Check the activity log for error messages
2. Verify the tool path exists
3. Ensure required dependencies are installed
4. Enable Debug Mode for more details

### Missing Tools

If tools are missing from the launcher:
1. Verify `tools.json` exists in the repository root
2. Check that tool paths are correct
3. Ensure the plugin system found the tool manifest

### Performance Issues

- Close unused tool windows
- Disable Debug Mode for better performance
- Check system resources (CPU, memory)

## Next Steps

- Explore the **Process Engineering** tab for industrial calculators
- Try the **Signal Processing** tools for waveform generation
- Check the **Scientific Modeling** tools for simulation capabilities
- Review the full User Manual for detailed documentation

---

For more detailed information, see the full [User Manual](../USER_MANUAL.md).
