# Audio Signal Processor - Complete Feature Guide

## 📋 Quick Navigation

**For Users:**

- [Getting Started](#getting-started)
- [Current GUI (5 Tabs)](#current-gui)
- [Available Features](#all-available-features)
- [Quick Examples](#quick-examples)

**For Developers:**

- [GUI Reorganization Plan](GUI_ARCHITECTURE_REVIEW.md)
- [Implementation Guide](GUI_QUICK_START_GUIDE.md)
- [Integration Summary](INTEGRATION_SUMMARY.md)

---

## 🎯 What is This?

A **professional-grade audio signal processor** with capabilities rivaling commercial DAWs, plus **unique research features** unavailable in consumer software.

### Unique Selling Points

1. **Professional Audio Processing** - Filtering, effects, mixing
2. **Music Production Tools** - Autotune, key/tempo detection, audio-to-MIDI
3. **Research-Grade Analysis** - Wavelet transforms, feature extraction, Nyquist analysis
4. **Convolution Reverb** - Hollywood-quality IR-based reverb
5. **MATLAB Integration** - Leverage Audio Toolbox and Wavelet Toolbox
6. **Open Architecture** - Full access to backend for custom processing

---

## 🚀 Getting Started

### Launch the Application

```matlab
cd matlab/audio_signal_processor
launch_audio_processor  % or
MainWindow()  % Direct GUI launch
```

### Quick Tutorial (5 minutes)

1. **Load Audio**: File → Load Audio (or use Library tab)
2. **View Waveform**: Waveform tab shows your audio
3. **Apply Filter**: Filters tab → Select Low Pass → Adjust frequency → Apply
4. **Mix Tracks**: Mixer tab → Load multiple files → Adjust levels → Process Mix
5. **Analyze**: Analysis tab → Generate Spectrogram

---

## 📊 Current GUI

### Tab 1: Waveform

**Purpose:** View and navigate audio

**Features:**

- Waveform visualization
- Zoom in/out
- Time navigation
- File loading

### Tab 2: Filters

**Purpose:** Frequency filtering

**Features:**

- **FFT Filters:** Low-pass, High-pass, Band-pass, Band-stop
- **Window Functions:** Gaussian, Hamming, Hann, Blackman, etc.
- **Time-Domain:** Butterworth, Moving Average, Median
- Filter response preview
- Zero-phase filtering

### Tab 3: Mixer

**Purpose:** Multi-track mixing

**Features:**

- 8 independent tracks
- Volume and pan per track
- Solo and mute
- Effect chains per track
- Master bus processing
- Mix export

⚠️ **Note:** Current mixer uses basic `MixerCore`. Enhanced version with time offsets available but not integrated.

### Tab 4: Analysis

**Purpose:** Audio analysis

**Features:**

- Real-time spectrogram
- FFT spectrum analyzer
- Phase correlation meter (stereo)
- Loudness metering (Peak, RMS, LUFS)
- Configurable FFT size and overlap

### Tab 5: Library

**Purpose:** Sample management

**Features:**

- Sample browser by category
- MATLAB built-in sounds
- Search functionality
- Sample preview
- User library management
- Sample collections

---

## 🎨 All Available Features

### ✅ Accessible via GUI (Current)

| Feature                 | Tab      | Status     |
| ----------------------- | -------- | ---------- |
| Waveform display        | Waveform | ✅ Working |
| Audio loading           | Waveform | ✅ Working |
| FFT filters             | Filters  | ✅ Working |
| Time-domain filters     | Filters  | ✅ Working |
| Basic mixing (8 tracks) | Mixer    | ✅ Working |
| Volume/pan controls     | Mixer    | ✅ Working |
| Spectrogram             | Analysis | ✅ Working |
| FFT spectrum            | Analysis | ✅ Working |
| Loudness metering       | Analysis | ✅ Working |
| Sample browser          | Library  | ✅ Working |

### ⚠️ Accessible via Code Only (Not in GUI)

| Feature                           | Backend Class              | Documentation                  |
| --------------------------------- | -------------------------- | ------------------------------ |
| **Audio Editing**                 | `AudioEditor`              | `ENHANCEMENTS_README.md`       |
| - Trim, cut, copy, paste          |                            |                                |
| - Fade in/out                     |                            |                                |
| - Normalize (Peak, RMS, LUFS)     |                            |                                |
| - Remove silence                  |                            |                                |
| - Undo/redo (50 levels)           |                            |                                |
| **Audio Effects**                 | `AudioEffects`             | Built-in help                  |
| - Reverb (algorithmic)            |                            |                                |
| - Convolution Reverb              | `ConvolutionReverb`        | `CONVOLUTION_REVERB_GUIDE.md`  |
| - Delay/Echo                      |                            |                                |
| - Parametric EQ                   |                            |                                |
| - Compression                     |                            |                                |
| - Limiting                        |                            |                                |
| - Distortion                      |                            |                                |
| - Chorus                          |                            |                                |
| - Flanger                         |                            |                                |
| - Pitch shift                     |                            |                                |
| - Time stretch                    |                            |                                |
| **Enhanced Mixing**               | `MixerCoreEnhanced`        | `ENHANCEMENTS_README.md`       |
| - Track time offsets              |                            |                                |
| - Per-track fades                 |                            |                                |
| - Automation curves               |                            |                                |
| - Timeline markers                |                            |                                |
| - Auto-alignment                  |                            |                                |
| **Music Production**              | `MusicProductionTools`     | `MUSIC_PRODUCTION_FEATURES.md` |
| - Autotune (pitch correction)     |                            |                                |
| - Key detection                   |                            |                                |
| - Tempo detection                 |                            |                                |
| - Chord detection                 |                            |                                |
| - Audio-to-MIDI                   |                            |                                |
| - Harmonizer                      |                            |                                |
| - Vocoder                         |                            |                                |
| - Click track generation          |                            |                                |
| - Audio quantization              |                            |                                |
| **Wavelet Analysis**              | `WaveletProcessor`         | `ENHANCEMENTS_README.md`       |
| - Wavelet denoising               |                            |                                |
| - Time-frequency analysis         |                            |                                |
| - Transient/tonal separation      |                            |                                |
| **Advanced Analysis**             | `AdvancedAudioProcessor`   | `ENHANCEMENTS_README.md`       |
| - Neural pitch detection          |                            |                                |
| - Onset detection                 |                            |                                |
| - Feature extraction (MFCC, etc.) |                            |                                |
| - Loudness (LUFS)                 |                            |                                |
| - Time stretching (advanced)      |                            |                                |
| **Anti-Aliasing**                 | `AntiAliasingTools`        | `ANTI_ALIASING_GUIDE.md`       |
| - Nyquist frequency analysis      |                            |                                |
| - Aliasing detection              |                            |                                |
| - Anti-aliasing filters           |                            |                                |
| - Oversampling/downsampling       |                            |                                |
| **Instrument Presets**            | `InstrumentEffectsLibrary` | Built-in help                  |
| - Vintage Keys                    |                            |                                |
| - Electric Guitar                 |                            |                                |
| - Acoustic Guitar                 |                            |                                |
| - Bass Guitar                     |                            |                                |
| - Lead Synth                      |                            |                                |
| - Pad Synth                       |                            |                                |
| - Vocals                          |                            |                                |
| - Drums                           |                            |                                |

---

## 💻 Quick Examples

### Example 1: Load and Filter Audio

```matlab
% Load audio
[audio, fs] = audioread('mysong.wav');

% Apply low-pass filter
filtered = FFTFilters(audio, 'Low Pass', ...
    'CutoffFrequency', 5000, ...
    'SampleRate', fs);

% Save result
audiowrite('mysong_filtered.wav', filtered, fs);
```

### Example 2: Apply Autotune

```matlab
% Load vocal
[vocal, fs] = audioread('vocal.wav');

% Create production tools
tools = MusicProductionTools();

% Apply autotune in C Major
autotuned = tools.autotune(vocal, fs, ...
    'Key', 'C', ...
    'Scale', 'major', ...
    'Strength', 0.8);  % 0-1, 1 = robotic

% Save result
audiowrite('vocal_autotuned.wav', autotuned, fs);
```

### Example 3: Convolution Reverb

```matlab
% Load audio
[drums, fs] = audioread('drums.wav');

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.4, 0.6);  % 40% reverb, 60% dry

% Process
result = reverb.process(drums, fs);

% Save
audiowrite('drums_hall.wav', result, fs);
```

### Example 4: Multi-track Mix with Offsets

```matlab
% Create enhanced mixer
mixer = MixerCoreEnhanced(4, 44100);

% Load tracks
[drums, fs] = audioread('drums.wav');
[bass, ~] = audioread('bass.wav');
[guitar, ~] = audioread('guitar.wav');
[vocal, ~] = audioread('vocal.wav');

% Load into mixer
mixer.loadTrack(1, drums, fs);
mixer.loadTrack(2, bass, fs);
mixer.loadTrack(3, guitar, fs);
mixer.loadTrack(4, vocal, fs);

% Set time offsets (in seconds)
mixer.setTrackOffset(2, 0.5);  % Bass starts 0.5s later
mixer.setTrackOffset(3, 1.0);  % Guitar starts 1s later
mixer.setTrackOffset(4, 2.0);  % Vocal starts 2s later

% Add fades
mixer.setTrackFadeIn(4, 0.5, 'scurve');  % Vocal fade in

% Set volumes
mixer.setTrackVolume(1, 0.8);
mixer.setTrackVolume(2, 0.9);
mixer.setTrackVolume(3, 0.7);
mixer.setTrackVolume(4, 1.0);

% Process mix
mixed = mixer.processMix();

% Export
audiowrite('full_mix.wav', mixed, fs);
```

### Example 5: Audio Editing

```matlab
% Load audio
[audio, fs] = audioread('podcast.wav');

% Create editor
editor = AudioEditor(audio, fs);

% Trim to keep only 10s-30s
editor.setSelection(10, 30);
editor.trim();

% Add fade in
editor.fadeIn(0.5, 'scurve');

% Add fade out
editor.fadeOut(1.0, 'exponential');

% Normalize to -16 LUFS (broadcast standard)
editor.normalize('lufs', -16);

% Get result
result = editor.getAudio();

% Save
audiowrite('podcast_edited.wav', result, fs);
```

### Example 6: Detect Key and Tempo

```matlab
% Load song
[audio, fs] = audioread('song.wav');

% Create production tools
tools = MusicProductionTools();

% Detect key
[key, scale, confidence] = tools.detectKey(audio, fs);
fprintf('Key: %s %s (confidence: %.2f)\n', key, scale, confidence);

% Detect tempo
[bpm, beats] = tools.detectTempo(audio, fs);
fprintf('Tempo: %.1f BPM\n', bpm);

% Detect chords
[chords, times] = tools.detectChords(audio, fs);
fprintf('Found %d chords\n', length(chords));
for i = 1:length(chords)
    fprintf('%.2fs: %s\n', times(i), chords{i});
end
```

### Example 7: Wavelet Denoising

```matlab
% Load noisy audio
[noisy, fs] = audioread('noisy_recording.wav');

% Create wavelet processor
wp = WaveletProcessor();

% Denoise using wavelet transform
denoised = wp.denoise(noisy, 'Method', 'Bayes', 'Wavelet', 'db4');

% Save result
audiowrite('denoised.wav', denoised, fs);
```

### Example 8: Anti-Aliasing Check

```matlab
% Load audio
[audio, fs] = audioread('myfile.wav');

% Create anti-aliasing tools
aa = AntiAliasingTools();

% Check Nyquist compliance
compliance = aa.checkNyquistCompliance(audio, fs);
fprintf('Nyquist Frequency: %.0f Hz\n', compliance.nyquistFreq);
fprintf('Content above Nyquist: %.2f%%\n', compliance.percentAbove * 100);

% Detect aliasing
aliasing = aa.detectAliasing(audio, fs);
if aliasing.hasAliasing
    fprintf('WARNING: Aliasing detected! Level: %.2f dB\n', aliasing.level);
else
    fprintf('No aliasing detected\n');
end

% Plot spectrum with Nyquist line
aa.plotSpectrum(audio, fs);
```

---

## 📚 Documentation Index

### User Guides

- **This File** - Overview and quick examples
- `README.md` - Original project README
- `CONVOLUTION_REVERB_GUIDE.md` - Complete reverb guide
- `MUSIC_PRODUCTION_FEATURES.md` - Music production tutorial
- `ANTI_ALIASING_GUIDE.md` - Anti-aliasing explained

### Developer Guides

- `GUI_ARCHITECTURE_REVIEW.md` - **START HERE** for GUI reorganization
- `GUI_QUICK_START_GUIDE.md` - Step-by-step implementation
- `IMPLEMENTATION_ROADMAP.md` - Original enhancement roadmap
- `INTEGRATION_SUMMARY.md` - What's been completed

### Reference

- `ENHANCEMENTS_README.md` - Executive summary of all enhancements
- `AUDIO_PROCESSOR_CRITICAL_REVIEW.md` - Original critical review
- `ENHANCEMENT_EXAMPLES.m` - 15 working examples
- `CONVOLUTION_REVERB_EXAMPLES.m` - 13 reverb examples

---

## 🔄 Recommended Workflow

### For Music Production

1. **Import** tracks via Mixer tab
2. **Align** tracks using code (time offsets)
3. **Apply Effects** via code (reverb, compression, EQ)
4. **Mix** using Mixer tab (volume, pan)
5. **Master** using code (limiting, final EQ)
6. **Export** via Mixer tab

### For Audio Editing

1. **Load** audio via Waveform tab
2. **Edit** via code (trim, fade, normalize)
3. **Process** via Filters tab
4. **Analyze** via Analysis tab
5. **Export** via File menu

### For Research/Analysis

1. **Load** audio via Waveform tab
2. **Initial Analysis** via Analysis tab
3. **Deep Analysis** via code (wavelets, features)
4. **Custom Processing** via code
5. **Export** results via code

---

## 🚨 Known Limitations (Current GUI)

1. **No audio editing** - Must use code to trim, cut, fade
2. **No effects interface** - Must use code to apply effects
3. **No time offsets in mixer** - Basic mixer only (MixerCore)
4. **No music production tools** - Autotune etc. only via code
5. **No research features** - Wavelets, anti-aliasing only via code

**Solution:** See `GUI_ARCHITECTURE_REVIEW.md` for complete reorganization plan.

---

## 🎓 Learning Path

### Beginner (GUI Only)

1. Load audio files
2. View waveforms
3. Apply basic filters
4. Create simple mixes
5. Generate spectrograms

### Intermediate (GUI + Basic Code)

1. Use AudioEffects for reverb/compression
2. Use MixerCoreEnhanced for offset mixing
3. Use AudioEditor for editing
4. Export high-quality audio

### Advanced (Full Code Access)

1. Autotune vocals
2. Detect key/tempo/chords
3. Wavelet denoising
4. Feature extraction for ML
5. Custom effect chains
6. Batch processing workflows

### Expert (Extend the System)

1. Create custom effects
2. Implement new analysis algorithms
3. Contribute to GUI reorganization
4. Integrate with other MATLAB toolboxes
5. Develop specialized workflows

---

## 💡 Pro Tips

1. **Always normalize after processing** - Use LUFS (-16 for broadcast, -14 for streaming)
2. **Use convolution reverb for realism** - Download IRs from OpenAIR
3. **Check for aliasing** - Especially after pitch shifting or distortion
4. **Use time offsets for natural mix** - Not everything should start at 0s
5. **Apply fades liberally** - Prevents clicks and pops
6. **Monitor in stereo** - Use phase correlation to check stereo field
7. **Batch process** - Process multiple files with same settings
8. **Save presets** - Save your effect chains for reuse
9. **Use wavelets for noise** - Better than simple filters for broadband noise
10. **Trust your ears** - Meters are guides, not rules

---

## 🆘 Getting Help

### Documentation

- Read the relevant guide for your feature
- Check example scripts
- Review class documentation (`help ClassName`)

### Common Issues

- **"Function not found"** - Add to MATLAB path: `addpath('core')`
- **"Invalid audio format"** - Use WAV for best compatibility
- **"Out of memory"** - Process in chunks for large files
- **"Distortion/clipping"** - Normalize before and after processing

### Support

- Check existing documentation first
- Review example scripts
- Consult MATLAB documentation for toolbox features

---

## 🎯 Summary

### What You Have

**Backend (100% Complete):**

- ✅ Professional filtering and effects
- ✅ Advanced mixing with time offsets
- ✅ Complete audio editing suite
- ✅ Music production tools (autotune!)
- ✅ Research-grade analysis (wavelets!)
- ✅ Convolution reverb
- ✅ Anti-aliasing tools

**GUI (40% Complete):**

- ✅ Basic waveform viewing
- ✅ Filtering interface
- ✅ Basic mixer (no offsets)
- ✅ Analysis tools
- ✅ Sample library
- ❌ No editing interface
- ❌ No effects interface
- ❌ No production tools interface
- ❌ No research tools interface

### What's Next

**GUI Reorganization** - See `GUI_ARCHITECTURE_REVIEW.md`

- 9-tab structure
- 100% feature exposure
- Professional workflow
- 7-week implementation

---

**Your audio processor rivals commercial DAWs in capability. Complete the GUI reorganization to make it accessible to everyone! 🎵**
