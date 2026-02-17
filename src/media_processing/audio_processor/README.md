# Audio Processor

Professional audio signal processing and multi-track mixing tools for the Golf Biomechanics Simulator & Game Engine repository.

## Overview

This directory contains advanced audio processing tools implemented in MATLAB, providing comprehensive capabilities for audio filtering, effects processing, multi-track mixing, and analysis.

## Directory Structure

```
audio_processor/
├── README.md                              # This file
└── matlab/
    └── audio_signal_processor/            # Main audio processing suite
        ├── README.md                      # Detailed documentation
        ├── launch_audio_processor.m       # Standard launcher
        ├── launch_audio_processor_pro.m   # Pro version launcher
        ├── core/                          # Core processing modules
        │   ├── AudioFilterEngine.m        # Filter engine
        │   ├── FFTFilters.m               # FFT-based filters
        │   ├── AudioLoader.m              # Audio file loading
        │   ├── AudioEffects.m             # Effects library
        │   ├── MixerCore.m                # Multi-track mixer
        │   ├── SoundLibraryManager.m      # Sample library
        │   ├── ConvolutionReverb.m        # Convolution reverb
        │   └── ...
        ├── gui/                           # GUI components
        │   └── MainWindow.m               # Main application window
        ├── utils/                         # Utility functions
        │   ├── SpectrogramGenerator.m
        │   ├── FrequencyAnalyzer.m
        │   ├── AudioExporter.m
        │   └── MetadataExtractor.m
        ├── examples/                      # Demo scripts
        └── tests/                         # Unit tests
```

## Features

### Audio Processing

- Multi-format audio loading (WAV, MP3, FLAC, OGG, M4A)
- Sample rate conversion
- Stereo/mono processing
- Large file streaming

### Advanced Filtering

- FFT-based filters (low-pass, high-pass, band-pass, band-stop)
- Multiple window functions (Gaussian, Hamming, Hann, Blackman, Kaiser)
- Time-domain filters (Butterworth, Moving Average, Median)
- Custom FIR/IIR design

### Audio Effects

- Reverb (algorithmic and convolution-based)
- Delay/Echo with feedback
- Parametric EQ
- Compression/Limiting
- Distortion/Overdrive
- Chorus/Flanger
- Pitch shifting
- Time stretching

### Multi-Track Mixing

- 8+ track support
- Per-track volume, pan, solo, mute
- Effect chains per track
- Master bus processing
- Timeline editing
- Real-time monitoring

### Analysis Tools

- Real-time spectrogram
- FFT spectrum analyzer
- Waveform viewer
- Phase correlation meter
- Loudness metering (LUFS)

## Quick Start

### Requirements

- MATLAB R2020b or later
- Signal Processing Toolbox
- Audio Toolbox (recommended)
- DSP System Toolbox (for advanced features)

### Launch the Application

```matlab
% Navigate to the audio processor directory
cd media_processing/audio_processor/matlab/audio_signal_processor

% Launch standard version
launch_audio_processor

% Or launch Pro version
launch_audio_processor_pro
```

### Basic Workflow

1. **Load Audio**: Import files or select from sample library
2. **Apply Filters**: Use FFT or time-domain filters
3. **Add Effects**: Apply reverb, delay, EQ, compression
4. **Mix Tracks**: Adjust levels, panning, and effects
5. **Export**: Save mixed audio or individual stems

## Documentation

Comprehensive documentation is available in the `matlab/audio_signal_processor/` directory:

| Document                                                                                 | Description              |
| ---------------------------------------------------------------------------------------- | ------------------------ |
| [README.md](matlab/audio_signal_processor/README.md)                                     | Main documentation       |
| [QUICK_START.md](matlab/audio_signal_processor/QUICK_START.md)                           | Getting started guide    |
| [API_DOCUMENTATION.md](matlab/audio_signal_processor/API_DOCUMENTATION.md)               | API reference            |
| [GUI_QUICK_START_GUIDE.md](matlab/audio_signal_processor/GUI_QUICK_START_GUIDE.md)       | GUI usage guide          |
| [CONVOLUTION_REVERB_GUIDE.md](matlab/audio_signal_processor/CONVOLUTION_REVERB_GUIDE.md) | Reverb processing        |
| [ANTI_ALIASING_GUIDE.md](matlab/audio_signal_processor/ANTI_ALIASING_GUIDE.md)           | Anti-aliasing techniques |

## Integration

This tool integrates with:

- **Video Processor** (`media_processing/video_processor/`) - Audio for video
- **Scientific Modeling** (`scientific_modeling/`) - Analysis integration
- **MATLAB Core** (`matlab/`) - Shared MATLAB infrastructure

## License

Part of the Tools repository. See main repository license for details.
