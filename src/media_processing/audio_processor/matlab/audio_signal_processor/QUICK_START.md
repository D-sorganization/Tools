# Audio Signal Processor - Quick Start Guide

Get up and running with the Audio Signal Processor in minutes!

---

## 🚀 Launch the Application

```matlab
cd matlab/audio_signal_processor
launch_audio_processor
```

That's it! The GUI will launch with all features ready to use.

---

## 📖 5-Minute Tutorial

### 1. Load Audio (Waveform Tab)

1. Click **Waveform** tab
2. Click **Load Audio** button
3. Browse to select an audio file OR use MATLAB built-in sounds
4. Waveform displays automatically

**Try this:**
```matlab
% Or load programmatically:
[audio, fs] = load('handel');
```

### 2. Apply Filters (Filters Tab)

1. Click **Filters** tab
2. Select filter type (Low Pass recommended for first try)
3. Set cutoff frequency (try 2000 Hz)
4. Click **Preview Response** to see filter curve
5. Click **Apply Filter**
6. Return to Waveform tab to see filtered audio

### 3. Add Effects (Use Programmatically)

```matlab
% Load audio
[audio, fs] = AudioLoader('your_file.wav');

% Add reverb
audio = AudioEffects(audio, 'Reverb', 'RoomSize', 0.7, 'Mix', 0.3, 'SampleRate', fs);

% Play result
sound(audio, fs);
```

### 4. Analyze Audio (Analysis Tab)

1. Click **Analysis** tab
2. Click **Analyze Spectrum** to see frequency content
3. Click **Generate Spectrogram** for time-frequency view
4. Click **Measure Loudness** for level metrics

### 5. Mix Multiple Tracks (Mixer Tab)

1. Click **Mixer** tab
2. Click **Load** on Track 1, select first audio file
3. Click **Load** on Track 2, select second audio file
4. Adjust volume faders and pan knobs
5. Click **Process Mix**
6. Click **Export Mix** to save

---

## 🎯 Common Workflows

### Workflow 1: Clean Up Audio

```matlab
% Load
[audio, fs] = AudioLoader('noisy_recording.wav');

% Remove low-frequency rumble
audio = FFTFilters(audio, 'High Pass', 'CutoffFrequency', 80, 'SampleRate', fs);

% Apply light compression
audio = AudioEffects(audio, 'Compression', 'Threshold', -12, 'SampleRate', fs);

% Export
AudioExporter(audio, 'clean_audio.wav', 'SampleRate', fs, 'BitDepth', 24);
```

### Workflow 2: Add Professional Effects

```matlab
% Load
[audio, fs] = AudioLoader('vocal.wav');

% EQ boost
audio = AudioEffects(audio, 'EQ', 'LowGain', 2, 'HighGain', 3, 'SampleRate', fs);

% Compression
audio = AudioEffects(audio, 'Compression', 'Threshold', -10, 'Ratio', 3, 'SampleRate', fs);

% Reverb
audio = AudioEffects(audio, 'Reverb', 'RoomSize', 0.6, 'Mix', 0.25, 'SampleRate', fs);

% Export
AudioExporter(audio, 'vocal_processed.wav', 'SampleRate', fs, 'BitDepth', 24);
```

### Workflow 3: Frequency Analysis

```matlab
% Load
[audio, fs] = AudioLoader('music.wav');

% Get spectrum
[freqs, mags] = FrequencyAnalyzer(audio, 'SampleRate', fs, 'FFTSize', 2048);

% Plot
figure;
plot(freqs, 20*log10(mags));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum');
grid on;

% Find dominant frequencies
[pks, locs] = findpeaks(mags, 'SortStr', 'descend', 'NPeaks', 5);
fprintf('Top 5 frequencies: ');
fprintf('%.1f Hz ', freqs(locs));
fprintf('\n');
```

---

## 📚 Where to Find More

- **Complete API Reference**: See `API_DOCUMENTATION.md`
- **Full Demo Script**: Run `examples/demo_all_features.m`
- **Feature List**: Check `README.md`
- **Implementation Details**: Read `IMPLEMENTATION_SUMMARY.md`

---

## 🎓 Key Functions Cheat Sheet

### Audio I/O
```matlab
[audio, fs] = AudioLoader('file.wav');              % Load
AudioExporter(audio, 'out.wav', 'SampleRate', fs);  % Export
```

### Filtering
```matlab
% FFT filter
filtered = FFTFilters(audio, 'Low Pass', 'CutoffFrequency', 2000, 'SampleRate', fs);

% Time-domain filter
filtered = AudioFilterEngine(audio, 'Butterworth', 'CutoffFrequency', 1500, 'SampleRate', fs);
```

### Effects
```matlab
% Reverb
audio = AudioEffects(audio, 'Reverb', 'RoomSize', 0.7, 'Mix', 0.3, 'SampleRate', fs);

% Compression
audio = AudioEffects(audio, 'Compression', 'Threshold', -12, 'Ratio', 4, 'SampleRate', fs);

% EQ
audio = AudioEffects(audio, 'EQ', 'LowGain', 3, 'HighGain', -2, 'SampleRate', fs);
```

### Analysis
```matlab
% Spectrum
[freqs, mags] = FrequencyAnalyzer(audio, 'SampleRate', fs);

% Spectrogram
[S, F, T] = SpectrogramGenerator(audio, 'SampleRate', fs);

% Metadata
metadata = MetadataExtractor(audio, 'SampleRate', fs);
```

### Mixing
```matlab
mixer = MixerCore(8, fs);                  % Create mixer
mixer.loadTrack(1, audio1, fs);            % Load track
mixer.setTrackVolume(1, 0.8);              % Set volume
mixer.setTrackPan(1, -0.5);                % Set pan
mixedAudio = mixer.processMix();           % Mix
```

---

## 💡 Pro Tips

1. **Always normalize audio** before processing:
   ```matlab
   audio = audio / max(abs(audio(:)));
   ```

2. **Chain effects in order**: EQ → Compression → Reverb

3. **Use high bit depth** for exports:
   ```matlab
   AudioExporter(audio, 'out.wav', 'BitDepth', 24, 'Normalize', true);
   ```

4. **Monitor levels**:
   ```matlab
   peakDB = 20*log10(max(abs(audio(:))));
   fprintf('Peak level: %.2f dB\n', peakDB);
   ```

5. **Use MATLAB built-in sounds** for testing:
   ```matlab
   [audio, fs] = load('handel');  % Free test audio
   ```

---

## ❓ Troubleshooting

### Application won't launch
```matlab
% Make sure you're in the right directory
cd matlab/audio_signal_processor
launch_audio_processor
```

### Missing toolbox warning
- Signal Processing Toolbox is **required**
- Audio Toolbox is recommended but not required

### Audio sounds distorted
- Check peak levels: `20*log10(max(abs(audio(:))))`
- Normalize if needed: `audio = audio / max(abs(audio(:)))`

### GUI panel is empty
- Make sure audio is loaded in Waveform tab first
- Check for error messages in console

---

## 🎯 Your First Project

Here's a complete beginner project to try:

```matlab
% 1. Load MATLAB's built-in sound
[audio, fs] = load('handel');

% 2. Normalize it
audio = audio / max(abs(audio(:)));

% 3. Apply a low-pass filter to make it sound mellow
filtered = FFTFilters(audio, 'Low Pass', ...
    'CutoffFrequency', 2000, ...
    'SampleRate', fs);

% 4. Add some reverb for atmosphere
reverb = AudioEffects(filtered, 'Reverb', ...
    'RoomSize', 0.7, ...
    'DecayTime', 2.5, ...
    'Mix', 0.3, ...
    'SampleRate', fs);

% 5. Analyze the result
[freqs, mags] = FrequencyAnalyzer(reverb, ...
    'SampleRate', fs);

% 6. Plot it
figure;
subplot(2,1,1);
plot((0:length(audio)-1)/fs, audio);
title('Original');
xlabel('Time (s)');

subplot(2,1,2);
plot((0:length(reverb)-1)/fs, reverb);
title('Processed (Filtered + Reverb)');
xlabel('Time (s)');

% 7. Play it
sound(reverb, fs);

% 8. Export it
AudioExporter(reverb, 'my_first_processed_audio.wav', ...
    'SampleRate', fs, ...
    'BitDepth', 24, ...
    'Normalize', true);

fprintf('Success! Check my_first_processed_audio.wav\n');
```

---

## 🎉 You're Ready!

You now know enough to:
- ✅ Load and process audio
- ✅ Apply filters and effects
- ✅ Mix multiple tracks
- ✅ Analyze frequency content
- ✅ Export professional-quality audio

**Explore the full documentation for advanced features!**

---

**Happy Processing! 🎵🎧**
