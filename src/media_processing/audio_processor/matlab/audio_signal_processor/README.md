# Audio Signal Processor & Mixer

A comprehensive MATLAB-based audio signal processing and multi-track mixing application with advanced filtering, effects, and analysis capabilities.

## Features

### Core Audio Processing

- **Multi-format Audio Loading**: WAV, MP3, FLAC, OGG, M4A support
- **Sample Rate Conversion**: Automatic and manual sample rate handling
- **Stereo/Mono Processing**: Full channel support and conversion
- **Large File Streaming**: Chunk-based processing for memory efficiency

### Advanced Filtering Suite

- **FFT-based Filters**: Low-pass, High-pass, Band-pass, Band-stop
- **Window Functions**: Gaussian, Rectangular, Hamming, Hann, Blackman, Kaiser, Tukey, Bartlett
- **Time-domain Filters**: Butterworth, Moving Average, Median
- **Custom FIR/IIR Design**: Advanced filter design capabilities

### Audio Effects Library

- **Reverb**: Algorithmic and convolution-based
- **Delay/Echo**: With feedback control and tempo sync
- **Parametric EQ**: Multi-band equalization
- **Compression/Limiting**: Dynamic range control
- **Distortion/Overdrive**: Harmonic enhancement
- **Chorus/Flanger**: Modulation effects
- **Pitch Shifting**: Real-time pitch manipulation
- **Time Stretching**: Independent tempo/pitch control

### Multi-Track Mixer

- **8+ Track Support**: Independent track processing
- **Per-track Controls**: Volume, pan, solo, mute
- **Effect Chains**: Multiple effects per track
- **Master Bus**: Global effects and processing
- **Timeline Editing**: Visual waveform editing
- **Real-time Monitoring**: Live audio preview

### Analysis Tools

- **Real-time Spectrogram**: Frequency domain visualization
- **FFT Spectrum Analyzer**: Detailed frequency analysis
- **Waveform Viewer**: Zoom and navigation
- **Phase Correlation Meter**: Stereo field analysis
- **Loudness Metering**: LUFS compliance monitoring

### Sound Library System

- **Sample Library**: Organized drum, bass, synth, guitar, vocal samples
- **Instrument Effects**: Preset effect chains for different instruments
- **Impulse Responses**: Reverb and cabinet modeling
- **User Library**: Custom sample upload and management
- **Metadata System**: Comprehensive tagging and search

### Export Capabilities

- **Mixed Audio Output**: High-quality audio export
- **Individual Track Export**: Stem generation
- **Format Conversion**: Multiple output formats
- **Batch Processing**: Automated workflow support

## Installation

1. **MATLAB Requirements**:

   - MATLAB R2020b or later
   - Signal Processing Toolbox
   - Audio Toolbox (recommended)
   - DSP System Toolbox (for advanced features)

2. **Launch the Application**:

   ```matlab
   cd audio_signal_processor
   launch_audio_processor
   ```

## File Structure

```
audio_signal_processor/
├── README.md
├── launch_audio_processor.m
├── core/
│   ├── AudioFilterEngine.m
│   ├── FFTFilters.m
│   ├── AudioLoader.m
│   ├── AudioEffects.m
│   ├── MixerCore.m
│   ├── SoundLibraryManager.m
│   └── InstrumentEffectsLibrary.m
├── gui/
│   ├── AudioProcessorApp.mlapp
│   ├── FilterPanel.m
│   ├── MixerPanel.m
│   ├── AnalysisPanel.m
│   └── LibraryBrowserPanel.m
├── utils/
│   ├── SpectrogramGenerator.m
│   ├── FrequencyAnalyzer.m
│   ├── AudioExporter.m
│   └── MetadataExtractor.m
├── library/
│   ├── samples/ (organized by instrument type)
│   ├── instrument_effects/ (preset effect chains)
│   ├── impulse_responses/ (reverb/cabinet models)
│   └── user_library/ (user uploads)
├── examples/
│   └── demo_projects/
└── tests/
    ├── test_fft_filters.m
    ├── test_audio_effects.m
    └── test_library_manager.m
```

## Usage

### Basic Workflow

1. **Load Audio**: Import audio files or select from sample library
2. **Apply Filters**: Use FFT or time-domain filters for signal processing
3. **Add Effects**: Apply reverb, delay, EQ, compression, etc.
4. **Mix Tracks**: Adjust levels, panning, and effects per track
5. **Export**: Save mixed audio or individual stems

### Advanced Features

- **Batch Processing**: Process multiple files with same settings
- **Preset Management**: Save and load effect chains
- **Real-time Analysis**: Monitor frequency content and levels
- **Custom Effects**: Create and save custom effect combinations

## Technical Details

### Filter Implementation

- **FFT Filters**: Frequency domain processing with customizable window functions
- **Zero-phase Filtering**: Forward-backward filtering for phase preservation
- **Transition Bandwidth**: Adjustable roll-off characteristics
- **Real-time Processing**: Optimized for live audio applications

### Effect Processing

- **Algorithmic Reverb**: Schroeder and Moorer algorithms
- **Convolution Reverb**: High-quality impulse response processing
- **Dynamic Processing**: Look-ahead compression and limiting
- **Modulation Effects**: LFO-based chorus and flanger

### Performance Optimization

- **Vectorized Operations**: MATLAB-optimized signal processing
- **Memory Management**: Efficient handling of large audio files
- **Parallel Processing**: Multi-core utilization for batch operations
- **Caching System**: Intelligent caching for repeated operations

## Contributing

This project follows MATLAB best practices and coding standards:

- Function documentation with help text
- Input validation using `arguments` blocks
- Unit testing for core functionality
- Modular design for easy extension

## License

Part of the Tools repository. See main repository license for details.

## Support

For issues and feature requests, please refer to the main Tools repository documentation.
