# Media Processing

Tools for audio and video analysis, processing, and editing.

## Overview

This directory contains comprehensive media processing applications implemented in MATLAB and web technologies, providing capabilities for audio signal processing, multi-track mixing, and video analysis.

## Components

### [Audio Processor](audio_processor/README.md)

Professional audio signal processing and multi-track mixing tools featuring:

- **Multi-format Support**: WAV, MP3, FLAC, OGG, M4A
- **Advanced Filtering**: FFT-based filters, Butterworth, custom FIR/IIR
- **Audio Effects**: Reverb, delay, EQ, compression, chorus, pitch shifting
- **Multi-Track Mixing**: 8+ tracks with per-track effects chains
- **Analysis Tools**: Spectrogram, FFT analyzer, loudness metering
- **MATLAB Implementation**: Built with MATLAB Signal Processing Toolbox

### [Video Processor](video_processor/README.md)

AI-powered golf swing video analysis platform featuring:

- **Video Analysis**: Upload, playback, and annotation tools
- **AI Pose Detection**: MediaPipe integration for pose tracking
- **Drawing Tools**: Annotations and overlays
- **Audio Commentary**: Record and attach audio notes
- **3D Visualization**: Three.js plane overlays
- **MATLAB Integration**: Physics modeling with Simscape Multibody

## Quick Start

### Audio Processor

```matlab
% Navigate to the audio processor directory
cd media_processing/audio_processor/matlab/audio_signal_processor

% Launch standard version
launch_audio_processor

% Or launch Pro version
launch_audio_processor_pro
```

### Video Processor

```bash
cd media_processing/video_processor

# Install dependencies
npm install

# Run development server
npm run dev
```

## Requirements

### Audio Processor

- MATLAB R2020b or later
- Signal Processing Toolbox
- Audio Toolbox (recommended)
- DSP System Toolbox (for advanced features)

### Video Processor

- Node.js 18+ and npm 9+
- (Optional) MATLAB with Simscape Multibody

## Integration

This suite integrates with:

- **Scientific Modeling** (`scientific_modeling/`) - Analysis integration
- **MATLAB Core** (`matlab/`) - Shared MATLAB infrastructure
- **Data Processing** (`data_processing/`) - Signal analysis pipelines

## License

Part of the Tools repository. See main repository license for details.
