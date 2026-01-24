%% Audio Signal Processor - Complete Feature Demonstration
% This script demonstrates all features of the Audio Signal Processor
% in a programmatic way (without GUI).
%
% This demo showcases:
% - Audio loading and basic operations
% - FFT-based and time-domain filtering
% - Audio effects (reverb, delay, EQ, compression, etc.)
% - Multi-track mixing
% - Frequency analysis and spectrograms
% - Sound library management
% - Audio export
%
% Requirements:
% - Signal Processing Toolbox
% - Audio Toolbox (recommended)
%
% Author: Audio Signal Processor Team
% Date: 2025

clear; close all; clc;

fprintf('========================================\n');
fprintf('Audio Signal Processor - Feature Demo\n');
fprintf('========================================\n\n');

% Add paths
addpath('../core');
addpath('../utils');

%% 1. AUDIO LOADING
fprintf('1. AUDIO LOADING\n');
fprintf('   Loading MATLAB built-in sounds...\n');

% Load MATLAB built-in sound
[audioData, fs] = load('handel');
fprintf('   ✓ Loaded Handel sound: %d samples at %d Hz\n', length(audioData), fs);
fprintf('   ✓ Duration: %.2f seconds\n', length(audioData)/fs);
fprintf('   ✓ Channels: %d\n\n', size(audioData, 2));

% Use AudioLoader for more advanced loading
fprintf('   Using AudioLoader for detailed info...\n');
try
    [data, sampleRate, info] = AudioLoader('handel', 'Metadata', true);
    fprintf('   ✓ Loaded with metadata\n');
    fprintf('   ✓ Sample format: %s\n', info.Format);
catch
    fprintf('   Using basic load function\n');
    data = audioData;
    sampleRate = fs;
end

% Normalize audio
audioData = audioData / max(abs(audioData(:)));

%% 2. FFT-BASED FILTERING
fprintf('\n2. FFT-BASED FILTERING\n');

% Low-pass filter
fprintf('   Applying low-pass filter (cutoff: 2000 Hz)...\n');
filteredLowpass = FFTFilters(audioData, 'Low Pass', ...
    'CutoffFrequency', 2000, ...
    'TransitionBandwidth', 500, ...
    'WindowType', 'Gaussian', ...
    'ZeroPhase', true, ...
    'SampleRate', fs);
fprintf('   ✓ Low-pass filter applied\n');

% High-pass filter
fprintf('   Applying high-pass filter (cutoff: 500 Hz)...\n');
filteredHighpass = FFTFilters(audioData, 'High Pass', ...
    'CutoffFrequency', 500, ...
    'TransitionBandwidth', 200, ...
    'WindowType', 'Hamming', ...
    'SampleRate', fs);
fprintf('   ✓ High-pass filter applied\n');

% Band-pass filter
fprintf('   Applying band-pass filter (300-3000 Hz)...\n');
filteredBandpass = FFTFilters(audioData, 'Band Pass', ...
    'LowCutoff', 300, ...
    'HighCutoff', 3000, ...
    'TransitionBandwidth', 100, ...
    'SampleRate', fs);
fprintf('   ✓ Band-pass filter applied\n\n');

%% 3. TIME-DOMAIN FILTERING
fprintf('3. TIME-DOMAIN FILTERING\n');

% Butterworth filter
fprintf('   Applying Butterworth filter...\n');
butterworthFiltered = AudioFilterEngine(audioData, 'Butterworth', ...
    'CutoffFrequency', 1500, ...
    'FilterOrder', 6, ...
    'SampleRate', fs);
fprintf('   ✓ Butterworth filter applied\n');

% Moving average filter
fprintf('   Applying moving average filter...\n');
movAvgFiltered = AudioFilterEngine(audioData, 'MovingAverage', ...
    'WindowSize', 5);
fprintf('   ✓ Moving average filter applied\n\n');

%% 4. AUDIO EFFECTS
fprintf('4. AUDIO EFFECTS\n');

% Reverb
fprintf('   Applying reverb effect...\n');
reverbAudio = AudioEffects(audioData, 'Reverb', ...
    'RoomSize', 0.7, ...
    'DecayTime', 2.5, ...
    'Damping', 0.5, ...
    'PreDelay', 0.02, ...
    'Mix', 0.3, ...
    'SampleRate', fs);
fprintf('   ✓ Reverb applied (room size: 0.7, decay: 2.5s)\n');

% Delay/Echo
fprintf('   Applying delay effect...\n');
delayAudio = AudioEffects(audioData, 'Delay', ...
    'DelayTime', 0.25, ...
    'Feedback', 0.4, ...
    'Mix', 0.4, ...
    'SampleRate', fs);
fprintf('   ✓ Delay applied (250ms with 40%% feedback)\n');

% Parametric EQ
fprintf('   Applying parametric EQ...\n');
eqAudio = AudioEffects(audioData, 'EQ', ...
    'LowGain', 3, ...
    'MidGain', 0, ...
    'HighGain', -2, ...
    'LowFreq', 250, ...
    'HighFreq', 4000, ...
    'SampleRate', fs);
fprintf('   ✓ EQ applied (Low: +3dB, High: -2dB)\n');

% Compression
fprintf('   Applying compression...\n');
compressedAudio = AudioEffects(audioData, 'Compression', ...
    'Threshold', -12, ...
    'Ratio', 4, ...
    'Attack', 10, ...
    'Release', 100, ...
    'Mix', 1.0, ...
    'SampleRate', fs);
fprintf('   ✓ Compression applied (threshold: -12dB, ratio: 4:1)\n');

% Distortion
fprintf('   Applying distortion...\n');
distortedAudio = AudioEffects(audioData, 'Distortion', ...
    'Drive', 0.6, ...
    'Tone', 0.5, ...
    'Level', 0.7, ...
    'Mix', 0.5, ...
    'SampleRate', fs);
fprintf('   ✓ Distortion applied (drive: 60%%)\n');

% Chorus
fprintf('   Applying chorus effect...\n');
chorusAudio = AudioEffects(audioData, 'Chorus', ...
    'Rate', 0.5, ...
    'Depth', 0.3, ...
    'Mix', 0.5, ...
    'SampleRate', fs);
fprintf('   ✓ Chorus applied (rate: 0.5 Hz, depth: 30%%)\n\n');

%% 5. MULTI-TRACK MIXING
fprintf('5. MULTI-TRACK MIXING\n');

% Create mixer with 4 tracks
fprintf('   Creating mixer with 4 tracks...\n');
mixer = MixerCore(4, fs);
fprintf('   ✓ Mixer created\n');

% Load different processed versions into tracks
fprintf('   Loading tracks...\n');
mixer.loadTrack(1, audioData, fs);
fprintf('   ✓ Track 1: Original audio\n');

mixer.loadTrack(2, reverbAudio, fs);
fprintf('   ✓ Track 2: Reverb version\n');

mixer.loadTrack(3, filteredLowpass, fs);
fprintf('   ✓ Track 3: Low-pass filtered\n');

mixer.loadTrack(4, delayAudio, fs);
fprintf('   ✓ Track 4: Delay version\n');

% Set track levels
fprintf('   Setting track levels and panning...\n');
mixer.setTrackVolume(1, 0.5);
mixer.setTrackVolume(2, 0.3);
mixer.setTrackVolume(3, 0.4);
mixer.setTrackVolume(4, 0.2);

% Pan tracks
mixer.setTrackPan(1, 0);    % Center
mixer.setTrackPan(2, -0.5); % Left
mixer.setTrackPan(3, 0.5);  % Right
mixer.setTrackPan(4, 0);    % Center

fprintf('   ✓ Track levels and panning configured\n');

% Process mix
fprintf('   Processing mix...\n');
mixedAudio = mixer.processMix();
fprintf('   ✓ Mix processed: %d samples\n\n', size(mixedAudio, 1));

%% 6. FREQUENCY ANALYSIS
fprintf('6. FREQUENCY ANALYSIS\n');

% Spectrum analysis
fprintf('   Performing FFT spectrum analysis...\n');
[freqs, magnitudes] = FrequencyAnalyzer(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 2048);
fprintf('   ✓ Spectrum analyzed (%d frequency bins)\n', length(freqs));

% Find peak frequencies
[pks, locs] = findpeaks(magnitudes, 'SortStr', 'descend', 'NPeaks', 5);
fprintf('   ✓ Top 5 frequency components:\n');
for i = 1:5
    fprintf('      %d. %.1f Hz (%.2f dB)\n', i, freqs(locs(i)), 20*log10(pks(i)));
end

% Generate spectrogram
fprintf('\n   Generating spectrogram...\n');
[S, F, T] = SpectrogramGenerator(audioData, ...
    'SampleRate', fs, ...
    'FFTSize', 2048, ...
    'Overlap', 0.75);
fprintf('   ✓ Spectrogram generated\n');
fprintf('   ✓ Time bins: %d, Frequency bins: %d\n\n', length(T), length(F));

%% 7. SOUND LIBRARY MANAGEMENT
fprintf('7. SOUND LIBRARY MANAGEMENT\n');

% Create library manager
fprintf('   Initializing sound library manager...\n');
libraryManager = SoundLibraryManager();
fprintf('   ✓ Library manager created\n');

% List MATLAB sounds
fprintf('   Available MATLAB sounds:\n');
matlabSounds = libraryManager.getMATLABSounds();
soundNames = fieldnames(matlabSounds);
for i = 1:min(5, length(soundNames))
    soundName = soundNames{i};
    soundInfo = matlabSounds.(soundName);
    fprintf('      - %s: %s\n', soundName, soundInfo.Description);
end

% Load a MATLAB sound
fprintf('\n   Loading gong sound from library...\n');
try
    [gongData, gongFs, gongInfo] = libraryManager.loadMATLABSound('gong');
    fprintf('   ✓ Gong sound loaded: %d samples at %d Hz\n', length(gongData), gongFs);
catch
    fprintf('   (Gong sound not available)\n');
end

fprintf('\n');

%% 8. METADATA EXTRACTION
fprintf('8. METADATA EXTRACTION\n');

fprintf('   Extracting metadata from processed audio...\n');
metadata = MetadataExtractor(audioData, ...
    'SampleRate', fs, ...
    'Format', 'MAT');
fprintf('   ✓ Duration: %.2f seconds\n', metadata.Duration);
fprintf('   ✓ Peak level: %.2f dB\n', metadata.PeakLevel);
fprintf('   ✓ RMS level: %.2f dB\n', metadata.RMSLevel);
fprintf('   ✓ Dynamic range: %.2f dB\n', metadata.DynamicRange);
fprintf('   ✓ Zero crossings: %d\n\n', metadata.ZeroCrossings);

%% 9. AUDIO EXPORT
fprintf('9. AUDIO EXPORT\n');

% Export processed audio
outputDir = '../output';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('   Exporting processed audio files...\n');

% Export reverb version
outputFile1 = fullfile(outputDir, 'demo_reverb.wav');
AudioExporter(reverbAudio, outputFile1, ...
    'SampleRate', fs, ...
    'BitDepth', 24, ...
    'Normalize', true);
fprintf('   ✓ Exported: demo_reverb.wav\n');

% Export mixed audio
outputFile2 = fullfile(outputDir, 'demo_mixed.wav');
AudioExporter(mixedAudio, outputFile2, ...
    'SampleRate', fs, ...
    'BitDepth', 24, ...
    'Normalize', true);
fprintf('   ✓ Exported: demo_mixed.wav\n');

% Export filtered audio
outputFile3 = fullfile(outputDir, 'demo_lowpass.wav');
AudioExporter(filteredLowpass, outputFile3, ...
    'SampleRate', fs, ...
    'BitDepth', 16, ...
    'Normalize', true);
fprintf('   ✓ Exported: demo_lowpass.wav\n\n');

%% 10. VISUALIZATION
fprintf('10. VISUALIZATION\n');
fprintf('   Creating comprehensive analysis plots...\n\n');

figure('Name', 'Audio Signal Processor - Demo Results', 'Position', [100, 100, 1400, 900]);

% Original waveform
subplot(3, 3, 1);
t = (0:length(audioData)-1) / fs;
plot(t, audioData);
title('Original Audio');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Filtered waveforms
subplot(3, 3, 2);
plot(t, filteredLowpass);
title('Low-Pass Filtered');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 3, 3);
plot(t, filteredHighpass);
title('High-Pass Filtered');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Effect-processed waveforms
subplot(3, 3, 4);
plot(t, reverbAudio);
title('With Reverb');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 3, 5);
plot(t, compressedAudio);
title('Compressed');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 3, 6);
plot(t, distortedAudio);
title('With Distortion');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Spectrum analysis
subplot(3, 3, 7);
plot(freqs, 20*log10(magnitudes));
title('Frequency Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
xlim([0, fs/2]);

% Spectrogram
subplot(3, 3, 8);
imagesc(T, F, 10*log10(abs(S)));
axis xy;
title('Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colormap('jet');
colorbar;

% Mixed audio
subplot(3, 3, 9);
tMixed = (0:size(mixedAudio, 1)-1) / fs;
plot(tMixed, mixedAudio);
title('Mixed Audio (4 tracks)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

fprintf('   ✓ Plots created\n\n');

%% SUMMARY
fprintf('========================================\n');
fprintf('DEMONSTRATION COMPLETE\n');
fprintf('========================================\n\n');

fprintf('Summary of operations:\n');
fprintf('  ✓ Loaded and processed audio\n');
fprintf('  ✓ Applied 5+ different filters\n');
fprintf('  ✓ Applied 6+ different effects\n');
fprintf('  ✓ Mixed 4 tracks with individual controls\n');
fprintf('  ✓ Performed frequency analysis\n');
fprintf('  ✓ Generated spectrogram\n');
fprintf('  ✓ Managed sound library\n');
fprintf('  ✓ Extracted metadata\n');
fprintf('  ✓ Exported 3 audio files\n');
fprintf('  ✓ Created visualization plots\n\n');

fprintf('All features demonstrated successfully!\n');
fprintf('Check the output folder for exported audio files.\n\n');

%% INTERACTIVE PLAYBACK (OPTIONAL)
fprintf('To play the processed audio:\n');
fprintf('  sound(audioData, fs)         - Original\n');
fprintf('  sound(reverbAudio, fs)       - With reverb\n');
fprintf('  sound(compressedAudio, fs)   - Compressed\n');
fprintf('  sound(mixedAudio, fs)        - Mixed tracks\n\n');
