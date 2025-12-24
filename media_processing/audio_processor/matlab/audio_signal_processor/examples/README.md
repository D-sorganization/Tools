# Audio Signal Processor Examples

This directory contains comprehensive examples demonstrating the capabilities of the MATLAB Audio Signal Processor.

## Available Examples

### `demo_audio_processor.m`

**Comprehensive demonstration of all features**

This is the main demo script that showcases the complete workflow:

- **MATLAB Built-in Sounds**: Loading and using MATLAB's extensive sound library
- **FFT Filtering**: All 4 filter types with 8 window functions
- **Audio Effects**: Reverb, delay, compression, distortion, EQ
- **Multi-track Mixing**: 4-track mixer with effects chains
- **Frequency Analysis**: Spectrum analysis and peak detection
- **Audio Export**: Multiple formats with metadata

**Usage:**

```matlab
demo_audio_processor()
```

### `run_all.m`

**Run all examples and tests**

Executes the complete test suite and demonstrations.

**Usage:**

```matlab
run_all()
```

## Example Workflows

### Basic Audio Processing

```matlab
% Load audio
[data, fs] = audioread('song.wav');

% Apply FFT low-pass filter
filtered = FFTFilters(data, 'Low-pass', 'FreqLow', 0.2, 'SampleRate', fs);

% Add reverb effect
processed = AudioEffects(filtered, 'Reverb', 'RoomSize', 0.5, 'SampleRate', fs);

% Export result
AudioExporter(processed, 'output.wav', 'SampleRate', fs);
```

### Multi-track Mixing

```matlab
% Create mixer
mixer = MixerCore(4, 44100);

% Load tracks
mixer.loadTrack(1, kickData, 44100);
mixer.loadTrack(2, bassData, 44100);
mixer.loadTrack(3, leadData, 44100);

% Set levels and pan
mixer.setTrackVolume(1, 0.8);
mixer.setTrackPan(2, -0.3);

% Add effects
mixer.addEffect(3, 'Reverb', struct('RoomSize', 0.4));

% Process mix
mixedAudio = mixer.processMix();
```

### Frequency Analysis

```matlab
% Create analyzer
analyzer = FrequencyAnalyzer(audioData, sampleRate);
analyzer.analyze();

% Get spectrum
[spectrum, frequencies] = analyzer.getSpectrum();

% Detect peaks
peaks = analyzer.detectPeaks();

% Generate spectrogram
[S, F, T] = SpectrogramGenerator(audioData, 'SampleRate', sampleRate);
```

### Library Management

```matlab
% Create library manager
libMgr = SoundLibraryManager();

% Load MATLAB built-in sound
[data, fs] = libMgr.loadMATLABSound('handel');

% Search samples
results = libMgr.searchSamples('classical');

% Get available sounds
matlabSounds = libMgr.getMATLABSounds();
```

## MATLAB Built-in Sound Integration

The Audio Signal Processor integrates seamlessly with MATLAB's built-in sound library:

### Available Sound Categories

- **Classical**: Handel's Hallelujah Chorus
- **Percussion**: Gong sounds
- **Voice**: Human laughter
- **Effects**: Splat sound effects
- **Environmental**: Train whistle
- **Synthetic**: Chirp, sawtooth, square waves
- **Speech**: Audio Toolbox speech examples
- **DSP**: Signal processing examples

### Loading MATLAB Sounds

```matlab
% Load specific sound
[data, fs] = libMgr.loadMATLABSound('handel');

% Browse available sounds
sounds = libMgr.getMATLABSounds();
soundNames = fieldnames(sounds);

% Load with metadata
[data, fs, info] = libMgr.loadMATLABSound('gong');
fprintf('Description: %s\n', info.Description);
fprintf('Tags: %s\n', info.Tags);
```

## Effect Presets

The system includes pre-configured effect chains for different instruments:

### Guitar Presets

- **Rock Overdrive**: Distortion + EQ + Reverb
- **Clean Guitar**: EQ + Compression + Reverb

### Vocal Presets

- **Pop Vocal**: Compression + EQ + Reverb
- **Radio Vocal**: Heavy compression + EQ + Limiting

### Synth Presets

- **Pad Synth**: Chorus + Delay + Reverb
- **Lead Synth**: Distortion + Chorus + Delay

### Master Presets

- **Mastering Chain**: EQ + Compression + Limiting
- **Live Performance**: EQ + Limiting

### Using Presets

```matlab
% Create effects library
effectsLib = InstrumentEffectsLibrary();

% Get preset
preset = effectsLib.getPreset('guitar', 'rock_overdrive');

% Apply preset effects
for i = 1:length(preset.Effects)
    effect = preset.Effects{i};
    audioData = AudioEffects(audioData, effect.Type, effect.Parameters);
end
```

## Performance Tips

### Large File Processing

```matlab
% Use chunked loading for large files
[data, fs] = AudioLoader('large_file.wav', 'ChunkSize', 1e6);
```

### Batch Processing

```matlab
% Process multiple files
files = {'file1.wav', 'file2.wav', 'file3.wav'};
for i = 1:length(files)
    [data, fs] = audioread(files{i});
    processed = AudioEffects(data, 'Compression', 'SampleRate', fs);
    audiowrite(['processed_' files{i}], processed, fs);
end
```

### Memory Management

```matlab
% Clear large variables when done
clear largeAudioData;
```

## Troubleshooting

### Common Issues

1. **"Audio Toolbox required"**: Some features require the Audio Toolbox
2. **"File not found"**: Check file paths and permissions
3. **"Out of memory"**: Use chunked processing for large files
4. **"Invalid parameters"**: Check parameter ranges and types

### Getting Help

```matlab
% Get help for any function
help FFTFilters
help AudioEffects
help MixerCore

% Run tests
test_fft_filters()
```

## Next Steps

1. **Run the demo**: `demo_audio_processor()`
2. **Launch the GUI**: `launch_audio_processor()`
3. **Explore the library**: Browse MATLAB built-in sounds
4. **Create custom presets**: Save your own effect chains
5. **Process your audio**: Load and process your own files

## Contributing

To add new examples:

1. Create a new `.m` file in this directory
2. Follow the existing naming convention
3. Include comprehensive help text
4. Test with different audio files
5. Update this README

For more information, see the main README.md file.
