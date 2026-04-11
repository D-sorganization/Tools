# Audio Signal Processor - Programmatic API Documentation

Complete guide to using the Audio Signal Processor programmatically (without GUI).

---

## Table of Contents

1. [Audio Loading](#1-audio-loading)
2. [FFT-Based Filtering](#2-fft-based-filtering)
3. [Time-Domain Filtering](#3-time-domain-filtering)
4. [Audio Effects](#4-audio-effects)
5. [Multi-Track Mixing](#5-multi-track-mixing)
6. [Frequency Analysis](#6-frequency-analysis)
7. [Spectrogram Generation](#7-spectrogram-generation)
8. [Sound Library Management](#8-sound-library-management)
9. [Metadata Extraction](#9-metadata-extraction)
10. [Audio Export](#10-audio-export)

---

## 1. Audio Loading

### AudioLoader

Load audio files with advanced options and metadata extraction.

**Syntax:**

```matlab
[audioData, sampleRate, info] = AudioLoader(filepath)
[audioData, sampleRate, info] = AudioLoader(filepath, Name, Value)
```

**Parameters:**

- `filepath` - Path to audio file or MATLAB sound name
- `'Metadata'` - Extract metadata (default: false)
- `'Normalize'` - Normalize audio (default: false)
- `'SampleRate'` - Resample to target sample rate (default: original)
- `'Channels'` - Convert to mono/stereo (default: original)

**Examples:**

```matlab
% Basic loading
[audioData, fs] = AudioLoader('song.wav');

% Load with metadata
[audioData, fs, info] = AudioLoader('song.wav', 'Metadata', true);
fprintf('Duration: %.2f seconds\n', info.Duration);
fprintf('Bit depth: %d bits\n', info.BitDepth);

% Load and normalize
[audioData, fs] = AudioLoader('song.wav', 'Normalize', true);

% Load and resample to 44.1 kHz
[audioData, fs] = AudioLoader('song.mp3', 'SampleRate', 44100);

% Convert to mono
[audioData, fs] = AudioLoader('stereo.wav', 'Channels', 1);
```

---

## 2. FFT-Based Filtering

### FFTFilters

Frequency-domain filtering with various window functions.

**Syntax:**

```matlab
filtered = FFTFilters(audioData, filterType, Name, Value)
```

**Filter Types:**

- `'Low Pass'` - Low-pass filter
- `'High Pass'` - High-pass filter
- `'Band Pass'` - Band-pass filter
- `'Band Stop'` - Band-stop (notch) filter

**Parameters:**

- `'CutoffFrequency'` - Cutoff frequency in Hz (default: 1000)
- `'LowCutoff'` - Low cutoff for band-pass/stop (default: 300)
- `'HighCutoff'` - High cutoff for band-pass/stop (default: 3000)
- `'TransitionBandwidth'` - Transition bandwidth in Hz (default: 100)
- `'WindowType'` - Window function (default: 'Gaussian')
  - Options: 'Gaussian', 'Rectangular', 'Hamming', 'Hann', 'Blackman', 'Kaiser', 'Tukey', 'Bartlett'
- `'ZeroPhase'` - Zero-phase filtering (default: true)
- `'SampleRate'` - Sample rate in Hz (default: 44100)

**Examples:**

```matlab
% Low-pass filter
filtered = FFTFilters(audioData, 'Low Pass', ...
    'CutoffFrequency', 2000, ...
    'TransitionBandwidth', 500, ...
    'WindowType', 'Gaussian', ...
    'SampleRate', fs);

% High-pass filter with Hamming window
filtered = FFTFilters(audioData, 'High Pass', ...
    'CutoffFrequency', 500, ...
    'WindowType', 'Hamming', ...
    'ZeroPhase', true, ...
    'SampleRate', fs);

% Band-pass filter (300-3000 Hz)
filtered = FFTFilters(audioData, 'Band Pass', ...
    'LowCutoff', 300, ...
    'HighCutoff', 3000, ...
    'TransitionBandwidth', 100, ...
    'SampleRate', fs);

% Band-stop (notch) filter to remove 60 Hz hum
filtered = FFTFilters(audioData, 'Band Stop', ...
    'LowCutoff', 55, ...
    'HighCutoff', 65, ...
    'TransitionBandwidth', 5, ...
    'SampleRate', fs);
```

---

## 3. Time-Domain Filtering

### AudioFilterEngine

Time-domain filtering with various filter types.

**Syntax:**

```matlab
filtered = AudioFilterEngine(audioData, filterType, Name, Value)
```

**Filter Types:**

- `'Butterworth'` - Butterworth IIR filter
- `'MovingAverage'` - Moving average filter
- `'Median'` - Median filter

**Parameters:**

**Butterworth:**

- `'CutoffFrequency'` - Cutoff frequency in Hz (default: 1000)
- `'FilterOrder'` - Filter order (default: 4)
- `'FilterMode'` - 'lowpass', 'highpass', 'bandpass' (default: 'lowpass')
- `'SampleRate'` - Sample rate in Hz (default: 44100)

**Moving Average / Median:**

- `'WindowSize'` - Window size in samples (default: 5, must be odd)

**Examples:**

```matlab
% Butterworth low-pass filter
filtered = AudioFilterEngine(audioData, 'Butterworth', ...
    'CutoffFrequency', 1500, ...
    'FilterOrder', 6, ...
    'FilterMode', 'lowpass', ...
    'SampleRate', fs);

% Moving average filter for smoothing
filtered = AudioFilterEngine(audioData, 'MovingAverage', ...
    'WindowSize', 5);

% Median filter for impulse noise removal
filtered = AudioFilterEngine(audioData, 'Median', ...
    'WindowSize', 7);
```

---

## 4. Audio Effects

### AudioEffects

Comprehensive audio effects processing.

**Syntax:**

```matlab
processed = AudioEffects(audioData, effectType, Name, Value)
```

**Effect Types:**

- `'Reverb'` - Algorithmic reverb
- `'Delay'` - Delay/echo effect
- `'EQ'` - Parametric equalizer
- `'Compression'` - Dynamic range compression
- `'Limiting'` - Peak limiting
- `'Distortion'` - Harmonic distortion
- `'Chorus'` - Chorus effect
- `'Flanger'` - Flanger effect
- `'PitchShift'` - Pitch shifting
- `'TimeStretch'` - Time stretching

### Common Parameters (All Effects)

- `'SampleRate'` - Sample rate in Hz (default: 44100)
- `'Mix'` - Dry/wet mix ratio 0-1 (default: 0.5)
- `'Bypass'` - Bypass effect (default: false)

### Reverb Parameters

- `'RoomSize'` - Room size 0-1 (default: 0.5)
- `'DecayTime'` - Decay time in seconds (default: 2.0)
- `'Damping'` - High-frequency damping 0-1 (default: 0.5)
- `'PreDelay'` - Pre-delay in seconds (default: 0.02)

### Delay Parameters

- `'DelayTime'` - Delay time in seconds (default: 0.25)
- `'Feedback'` - Feedback amount 0-0.95 (default: 0.3)
- `'TempoSync'` - Sync to tempo (default: false)
- `'Tempo'` - Tempo in BPM (default: 120)

### EQ Parameters

- `'LowGain'` - Low frequency gain in dB (default: 0)
- `'MidGain'` - Mid frequency gain in dB (default: 0)
- `'HighGain'` - High frequency gain in dB (default: 0)
- `'LowFreq'` - Low frequency crossover in Hz (default: 250)
- `'HighFreq'` - High frequency crossover in Hz (default: 4000)

### Compression Parameters

- `'Threshold'` - Compression threshold in dB (default: -12)
- `'Ratio'` - Compression ratio (default: 4)
- `'Attack'` - Attack time in ms (default: 10)
- `'Release'` - Release time in ms (default: 100)
- `'Knee'` - Soft knee width in dB (default: 2)

### Distortion Parameters

- `'Drive'` - Distortion amount 0-1 (default: 0.5)
- `'Tone'` - Tone control 0-1 (default: 0.5)
- `'Level'` - Output level 0-1 (default: 0.7)

### Modulation Parameters (Chorus/Flanger)

- `'Rate'` - LFO rate in Hz (default: 0.5)
- `'Depth'` - Modulation depth 0-1 (default: 0.3)

**Examples:**

```matlab
% Reverb
reverb = AudioEffects(audioData, 'Reverb', ...
    'RoomSize', 0.7, ...
    'DecayTime', 3.0, ...
    'Damping', 0.5, ...
    'Mix', 0.3, ...
    'SampleRate', fs);

% Delay with tempo sync
delay = AudioEffects(audioData, 'Delay', ...
    'DelayTime', 0.25, ...
    'Feedback', 0.5, ...
    'TempoSync', true, ...
    'Tempo', 120, ...
    'Mix', 0.4, ...
    'SampleRate', fs);

% Parametric EQ
eq = AudioEffects(audioData, 'EQ', ...
    'LowGain', 3, ...
    'MidGain', -2, ...
    'HighGain', 4, ...
    'LowFreq', 200, ...
    'HighFreq', 5000, ...
    'SampleRate', fs);

% Compression
compressed = AudioEffects(audioData, 'Compression', ...
    'Threshold', -12, ...
    'Ratio', 4, ...
    'Attack', 5, ...
    'Release', 50, ...
    'SampleRate', fs);

% Distortion
distorted = AudioEffects(audioData, 'Distortion', ...
    'Drive', 0.7, ...
    'Tone', 0.6, ...
    'Mix', 0.5, ...
    'SampleRate', fs);

% Chorus
chorus = AudioEffects(audioData, 'Chorus', ...
    'Rate', 0.8, ...
    'Depth', 0.4, ...
    'Mix', 0.5, ...
    'SampleRate', fs);

% Chain multiple effects
processed = audioData;
processed = AudioEffects(processed, 'EQ', 'LowGain', 2, 'SampleRate', fs);
processed = AudioEffects(processed, 'Compression', 'Threshold', -10, 'SampleRate', fs);
processed = AudioEffects(processed, 'Reverb', 'RoomSize', 0.6, 'Mix', 0.2, 'SampleRate', fs);
```

---

## 5. Multi-Track Mixing

### MixerCore

Professional multi-track mixer with per-track controls.

**Syntax:**

```matlab
mixer = MixerCore(numTracks, sampleRate)
```

**Key Methods:**

- `loadTrack(trackIndex, audioData, trackSampleRate)` - Load audio to track
- `setTrackVolume(trackIndex, volume)` - Set volume (0-1)
- `setTrackPan(trackIndex, pan)` - Set pan (-1 to 1)
- `setTrackSolo(trackIndex, solo)` - Solo track
- `setTrackMute(trackIndex, mute)` - Mute track
- `addEffect(trackIndex, effectType, params)` - Add effect to track
- `removeEffect(trackIndex, effectIndex)` - Remove effect
- `processMix()` - Mix all tracks and return result

**Examples:**

```matlab
% Create 8-track mixer at 44.1 kHz
mixer = MixerCore(8, 44100);

% Load tracks
mixer.loadTrack(1, drums, 44100);
mixer.loadTrack(2, bass, 44100);
mixer.loadTrack(3, guitar, 48000); % Auto-resampled to 44100
mixer.loadTrack(4, vocals, 44100);

% Set levels (0-1)
mixer.setTrackVolume(1, 0.8);  % Drums
mixer.setTrackVolume(2, 0.7);  % Bass
mixer.setTrackVolume(3, 0.6);  % Guitar
mixer.setTrackVolume(4, 0.9);  % Vocals

% Pan tracks (-1=left, 0=center, 1=right)
mixer.setTrackPan(1, 0);       % Drums center
mixer.setTrackPan(2, 0);       % Bass center
mixer.setTrackPan(3, -0.5);    % Guitar left
mixer.setTrackPan(4, 0.5);     % Vocals right

% Solo a track
mixer.setTrackSolo(4, true);  % Solo vocals

% Mute a track
mixer.setTrackMute(2, true);  % Mute bass

% Add effects to track
effectParams = struct('RoomSize', 0.5, 'DecayTime', 1.5);
mixer.addEffect(4, 'Reverb', effectParams);

% Process mix
mixedAudio = mixer.processMix();

% Access master bus
mixer.MasterBus.Volume = 0.9;
mixer.MasterBus.Pan = 0;
```

---

## 6. Frequency Analysis

### FrequencyAnalyzer

FFT-based frequency analysis.

**Syntax:**

```matlab
[freqs, magnitudes] = FrequencyAnalyzer(audioData, Name, Value)
[freqs, magnitudes, phases] = FrequencyAnalyzer(audioData, Name, Value)
```

**Parameters:**

- `'SampleRate'` - Sample rate in Hz (default: 44100)
- `'FFTSize'` - FFT size (default: 2048)
- `'Window'` - Window function (default: 'Hann')
- `'AveragingMode'` - 'none', 'time', 'frequency' (default: 'none')

**Examples:**

```matlab
% Basic spectrum analysis
[freqs, mags] = FrequencyAnalyzer(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 2048);

% Plot spectrum
figure;
plot(freqs, 20*log10(mags));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum');
grid on;

% Find dominant frequencies
[pks, locs] = findpeaks(mags, 'SortStr', 'descend', 'NPeaks', 5);
dominantFreqs = freqs(locs);
fprintf('Top 5 frequencies: ');
fprintf('%.1f Hz ', dominantFreqs);
fprintf('\n');

% Large FFT for better resolution
[freqs, mags] = FrequencyAnalyzer(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 8192);
```

---

## 7. Spectrogram Generation

### SpectrogramGenerator

Generate time-frequency spectrograms.

**Syntax:**

```matlab
[S, F, T] = SpectrogramGenerator(audioData, Name, Value)
```

**Parameters:**

- `'SampleRate'` - Sample rate in Hz (default: 44100)
- `'FFTSize'` - FFT size (default: 2048)
- `'Overlap'` - Overlap ratio 0-1 (default: 0.75)
- `'Window'` - Window function (default: 'Hann')
- `'FrequencyLimits'` - [fMin, fMax] in Hz (default: [0, fs/2])

**Examples:**

```matlab
% Generate spectrogram
[S, F, T] = SpectrogramGenerator(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 2048, ...
    'Overlap', 0.75);

% Plot spectrogram
figure;
imagesc(T, F, 10*log10(abs(S)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');
colormap('jet');
colorbar;
caxis([-80, 0]); % dB range

% High-resolution spectrogram
[S, F, T] = SpectrogramGenerator(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 4096, ...
    'Overlap', 0.90, ...
    'FrequencyLimits', [20, 20000]);

% Focus on specific frequency range (bass frequencies)
[S, F, T] = SpectrogramGenerator(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 4096, ...
    'FrequencyLimits', [20, 500]);
```

---

## 8. Sound Library Management

### SoundLibraryManager

Manage audio sample libraries with metadata and search.

**Syntax:**

```matlab
libraryManager = SoundLibraryManager()
libraryManager = SoundLibraryManager(libraryPath)
```

**Key Methods:**

- `loadSample(category, filename)` - Load sample from library
- `loadMATLABSound(soundName)` - Load MATLAB built-in sound
- `searchSamples(query)` - Search samples by metadata
- `getCategories()` - Get available categories
- `getMATLABSounds()` - Get MATLAB built-in sounds
- `addSample(filepath, metadata)` - Add sample to library
- `updateCatalog()` - Refresh library catalog

**Examples:**

```matlab
% Create library manager
libMgr = SoundLibraryManager();

% Load MATLAB built-in sound
[data, fs, info] = libMgr.loadMATLABSound('handel');
fprintf('Loaded: %s\n', info.Description);

% List available MATLAB sounds
matlabSounds = libMgr.getMATLABSounds();
soundNames = fieldnames(matlabSounds);
for i = 1:length(soundNames)
    name = soundNames{i};
    desc = matlabSounds.(name).Description;
    fprintf('%s: %s\n', name, desc);
end

% Load sample from library (if samples exist)
categories = libMgr.getCategories();
if ~isempty(categories)
    [data, fs, info] = libMgr.loadSample('drums', 'kick_01.wav');
end

% Search for samples
results = libMgr.searchSamples('bass');
fprintf('Found %d matches\n', results.Count);
for i = 1:results.Count
    match = results.Matches{i};
    fprintf('  %s / %s\n', match.Category, match.Filename);
end

% Add custom sample to library
metadata = struct('Category', 'drums', 'Tags', 'snare acoustic');
libMgr.addSample('/path/to/snare.wav', metadata);

% Refresh catalog
libMgr.updateCatalog();
```

---

## 9. Metadata Extraction

### MetadataExtractor

Extract comprehensive metadata from audio files.

**Syntax:**

```matlab
metadata = MetadataExtractor(audioData, Name, Value)
```

**Parameters:**

- `'SampleRate'` - Sample rate in Hz (default: 44100)
- `'Format'` - File format (default: 'WAV')
- `'IncludeSpectral'` - Include spectral features (default: false)

**Extracted Metadata:**

- Duration, SampleRate, Channels, BitDepth
- PeakLevel, RMSLevel, DynamicRange
- ZeroCrossings, SpectralCentroid, SpectralRolloff
- And more...

**Examples:**

```matlab
% Basic metadata extraction
metadata = MetadataExtractor(audioData, ...
    'SampleRate', fs, ...
    'Format', 'WAV');

fprintf('Duration: %.2f s\n', metadata.Duration);
fprintf('Peak level: %.2f dB\n', metadata.PeakLevel);
fprintf('RMS level: %.2f dB\n', metadata.RMSLevel);
fprintf('Dynamic range: %.2f dB\n', metadata.DynamicRange);
fprintf('Zero crossings: %d\n', metadata.ZeroCrossings);

% Include spectral features
metadata = MetadataExtractor(audioData, ...
    'SampleRate', fs, ...
    'IncludeSpectral', true);

fprintf('Spectral centroid: %.2f Hz\n', metadata.SpectralCentroid);
fprintf('Spectral rolloff: %.2f Hz\n', metadata.SpectralRolloff);
```

---

## 10. Audio Export

### AudioExporter

Export audio with various formats and options.

**Syntax:**

```matlab
AudioExporter(audioData, outputPath, Name, Value)
```

**Parameters:**

- `'SampleRate'` - Sample rate in Hz (default: 44100)
- `'BitDepth'` - Bit depth: 16, 24, 32 (default: 16)
- `'Format'` - Output format (default: auto-detect from extension)
- `'Normalize'` - Normalize audio (default: false)
- `'DitherType'` - Dither type: 'none', 'TPDF' (default: 'none')
- `'Metadata'` - Metadata struct (default: empty)

**Examples:**

```matlab
% Basic export
AudioExporter(audioData, 'output.wav', 'SampleRate', fs);

% High-quality export (24-bit)
AudioExporter(audioData, 'output.wav', ...
    'SampleRate', 48000, ...
    'BitDepth', 24, ...
    'Normalize', true);

% Export with metadata
metadata = struct('Title', 'My Song', 'Artist', 'Me', 'Year', '2025');
AudioExporter(audioData, 'output.wav', ...
    'SampleRate', fs, ...
    'BitDepth', 24, ...
    'Metadata', metadata);

% Export to different formats
AudioExporter(audioData, 'output.mp3', 'SampleRate', fs);
AudioExporter(audioData, 'output.flac', 'SampleRate', fs, 'BitDepth', 24);

% Export with dithering (for bit depth reduction)
AudioExporter(audioData, 'output.wav', ...
    'SampleRate', fs, ...
    'BitDepth', 16, ...
    'DitherType', 'TPDF');
```

---

## Complete Workflow Example

Here's a complete workflow combining multiple features:

```matlab
% 1. Load audio
[audio, fs] = AudioLoader('input.wav', 'Normalize', true);

% 2. Apply pre-processing
audio = FFTFilters(audio, 'High Pass', ...
    'CutoffFrequency', 80, ...
    'SampleRate', fs);

% 3. Apply EQ
audio = AudioEffects(audio, 'EQ', ...
    'LowGain', 2, ...
    'HighGain', 3, ...
    'SampleRate', fs);

% 4. Apply compression
audio = AudioEffects(audio, 'Compression', ...
    'Threshold', -12, ...
    'Ratio', 3, ...
    'SampleRate', fs);

% 5. Add reverb
audio = AudioEffects(audio, 'Reverb', ...
    'RoomSize', 0.6, ...
    'Mix', 0.2, ...
    'SampleRate', fs);

% 6. Analyze result
[freqs, mags] = FrequencyAnalyzer(audio, 'SampleRate', fs);
metadata = MetadataExtractor(audio, 'SampleRate', fs);

% 7. Export
AudioExporter(audio, 'output_mastered.wav', ...
    'SampleRate', fs, ...
    'BitDepth', 24, ...
    'Normalize', true);

fprintf('Processing complete!\n');
fprintf('Peak level: %.2f dB\n', metadata.PeakLevel);
fprintf('RMS level: %.2f dB\n', metadata.RMSLevel);
```

---

## Additional Resources

- **Demo Script**: See `examples/demo_all_features.m` for a comprehensive demonstration
- **Unit Tests**: Check `tests/` directory for usage examples
- **GUI Application**: Run `launch_audio_processor` for the graphical interface

---

## Tips and Best Practices

1. **Always normalize audio** before processing to avoid clipping:

   ```matlab
   audio = audio / max(abs(audio(:)));
   ```

2. **Chain effects carefully** - order matters:
   - EQ → Compression → Reverb is typically best
   - Distortion → EQ → Delay for creative effects

3. **Use appropriate FFT sizes**:
   - 1024-2048 for real-time applications
   - 4096-8192 for offline analysis
   - Larger = better frequency resolution, slower

4. **Monitor levels** throughout the signal chain:

   ```matlab
   peakLevel = 20*log10(max(abs(audio(:))));
   rmsLevel = 20*log10(rms(audio(:)));
   ```

5. **Save intermediate results** for complex workflows:
   ```matlab
   AudioExporter(audio, 'step1_filtered.wav', 'SampleRate', fs);
   ```

---

## Support

For issues, questions, or feature requests, please refer to the main repository documentation.

**Version:** 1.0
**Last Updated:** 2025
