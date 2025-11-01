%% Audio Processor Enhancement Examples
% Demonstrates new professional-level features
%
% This script showcases the enhanced capabilities of the Audio Signal
% Processor, including:
% - Time offsets and alignment
% - Audio trimming and editing
% - Wavelet-based processing
% - Advanced Audio Toolbox features
%
% Author: Audio Signal Processor Team
% Date: November 2025

%% Setup
% Add paths
addpath('core');
addpath('utils');

%% Example 1: Multi-Track Mixing with Time Offsets
fprintf('=== Example 1: Time Offset Mixing ===\n');

% Create enhanced mixer
mixer = MixerCoreEnhanced(8, 44100);

% Load sample audio files
[drums, fs1] = audioread('drums.wav');  % Or use: load handel; drums = y; fs1 = Fs;
[bass, fs2] = audioread('bass.wav');
[vocal, fs3] = audioread('vocal.wav');

% Load tracks
mixer.loadTrack(1, drums, fs1);
mixer.loadTrack(2, bass, fs2);
mixer.loadTrack(3, vocal, fs3);

% Set track names
mixer.setTrackName(1, 'Drums');
mixer.setTrackName(2, 'Bass');
mixer.setTrackName(3, 'Vocals');

% Set time offsets (vocals start 1 second later)
mixer.setTrackOffset(1, 0.0);    % Drums at start
mixer.setTrackOffset(2, 0.0);    % Bass at start
mixer.setTrackOffset(3, 1.0);    % Vocals delayed by 1 second

% Add fades to vocals
mixer.setTrackFadeIn(3, 0.5, 'linear');      % 0.5s fade in
mixer.setTrackFadeOut(3, 1.0, 'exponential'); % 1.0s fade out

% Add markers
mixer.addMarker(0.0, 'Intro');
mixer.addMarker(8.0, 'Verse');
mixer.addMarker(24.0, 'Chorus');
mixer.addMarker(40.0, 'Bridge');

% Set levels
mixer.setTrackVolume(1, 0.8);
mixer.setTrackVolume(2, 0.7);
mixer.setTrackVolume(3, 0.9);

% Pan tracks
mixer.setTrackPan(1, 0.0);    % Center
mixer.setTrackPan(2, -0.3);   % Slightly left
mixer.setTrackPan(3, 0.2);    % Slightly right

% Process mix with offsets
fprintf('Processing mix with time offsets...\n');
mixedAudio = mixer.processMix();

% Export
fprintf('Total duration: %.2f seconds\n', mixer.getTotalDuration());
AudioExporter(mixedAudio, 'mixed_with_offsets.wav', 'SampleRate', 44100);

%% Example 2: Audio Editing and Trimming
fprintf('\n=== Example 2: Audio Editing ===\n');

% Load audio
[audio, fs] = audioread('speech.wav');

% Create editor
editor = AudioEditor(audio, fs);

% Get info
info = editor.getInfo();
fprintf('Original duration: %.2f seconds\n', info.Duration);
fprintf('Peak: %.4f, RMS: %.4f\n', info.Peak, info.RMS);

% Select and trim first 0.5 seconds
editor.setSelection(0, 0.5);
editor.delete();  % Remove first 0.5s

% Apply fade in
editor.fadeIn(0.3, 'scurve');

% Normalize to -3 dB
editor.normalize('peak', -3);

% Remove DC offset
editor.removeOffset();

% Remove silence (threshold 0.01, min duration 0.5s)
editor.removeSilence(0.01, 0.5);

% Get processed audio
processedAudio = editor.getAudio();
fprintf('Processed duration: %.2f seconds\n', length(processedAudio)/fs);

% Export
editor.export('edited_speech.wav', 'BitDepth', 24);

%% Example 3: Auto-Alignment of Tracks
fprintf('\n=== Example 3: Auto-Alignment ===\n');

% Create mixer
mixer2 = MixerCoreEnhanced(4, 44100);

% Load multiple takes of same performance
[take1, fs] = audioread('take1.wav');
[take2, ~] = audioread('take2.wav');
[take3, ~] = audioread('take3.wav');

mixer2.loadTrack(1, take1, fs);
mixer2.loadTrack(2, take2, fs);
mixer2.loadTrack(3, take3, fs);

% Auto-align tracks by peak
fprintf('Aligning tracks by peak detection...\n');
mixer2.alignTracks('peak');

% Show offsets
for i = 1:3
    fprintf('Track %d offset: %.4f seconds\n', i, mixer2.Tracks(i).StartOffset);
end

% Process aligned mix
alignedMix = mixer2.processMix();
AudioExporter(alignedMix, 'aligned_mix.wav', 'SampleRate', fs);

%% Example 4: Wavelet Denoising
fprintf('\n=== Example 4: Wavelet Denoising ===\n');

% Load noisy audio
[noisyAudio, fs] = audioread('noisy_recording.wav');

% Create wavelet processor
wp = WaveletProcessor();

% Check if Wavelet Toolbox is available
if wp.HasWaveletToolbox
    fprintf('Wavelet Toolbox detected\n');

    % Denoise using Bayesian method
    fprintf('Applying wavelet denoising...\n');
    cleanAudio = wp.denoise(noisyAudio, 'Wavelet', 'db4', ...
        'Method', 'Bayes', 'Threshold', 'Soft');

    % Calculate SNR improvement
    noise = noisyAudio - cleanAudio;
    snr_before = 10 * log10(sum(noisyAudio.^2) / sum(noise.^2));
    fprintf('SNR improvement: %.2f dB\n', snr_before);

    % Export
    AudioExporter(cleanAudio, 'denoised_audio.wav', 'SampleRate', fs);
else
    fprintf('Wavelet Toolbox not available - using fallback\n');
    cleanAudio = wp.denoise(noisyAudio);
end

%% Example 5: Wavelet Time-Frequency Analysis
fprintf('\n=== Example 5: Wavelet Time-Frequency Analysis ===\n');

% Load audio
[audio, fs] = audioread('music.wav');

% Convert to mono
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Create wavelet processor
wp = WaveletProcessor();

if wp.HasWaveletToolbox
    % Perform CWT
    fprintf('Computing continuous wavelet transform...\n');
    [cfs, frequencies, time] = wp.timeFrequencyAnalysis(audio, fs, ...
        'Wavelet', 'amor', 'FrequencyLimits', [50, 2000]);

    % Plot scalogram
    wp.plotScalogram(cfs, frequencies, time, 'Title', 'Music Scalogram');

    % Synchrosqueeze for better resolution
    fprintf('Computing synchrosqueezed transform...\n');
    [sst, freq] = wp.synchrosqueeze(audio, fs, 'FrequencyLimits', [50, 2000]);

    figure;
    imagesc(time, freq, abs(sst));
    axis xy;
    set(gca, 'YScale', 'log');
    colormap('parula');
    colorbar;
    title('Synchrosqueezed Wavelet Transform');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
end

%% Example 6: Transient/Tonal Separation
fprintf('\n=== Example 6: Component Separation ===\n');

% Load drum mix
[drumMix, fs] = audioread('drums.wav');

% Create wavelet processor
wp = WaveletProcessor();

% Separate transient and tonal components
fprintf('Separating transient and tonal components...\n');
[transients, tonal] = wp.separateTransientTonal(drumMix, fs, ...
    'Wavelet', 'db4', 'Level', 5);

% Export separated components
AudioExporter(transients, 'drums_transients.wav', 'SampleRate', fs);
AudioExporter(tonal, 'drums_tonal.wav', 'SampleRate', fs);

fprintf('Transient energy: %.4f\n', sum(transients(:).^2));
fprintf('Tonal energy: %.4f\n', sum(tonal(:).^2));

%% Example 7: Pitch Detection and Tracking
fprintf('\n=== Example 7: Pitch Detection ===\n');

% Load vocal recording
[vocal, fs] = audioread('vocal.wav');

% Create advanced audio processor
ap = AdvancedAudioProcessor();

if ap.HasAudioToolbox
    fprintf('Audio Toolbox detected - using neural network pitch detection\n');

    % Detect pitch
    [pitch, confidence] = ap.detectPitch(vocal, fs, 'Range', [80, 400]);

    % Plot pitch track
    figure;
    subplot(2,1,1);
    plot((0:length(vocal)-1)/fs, vocal);
    title('Audio Waveform');
    xlabel('Time (s)');
    ylabel('Amplitude');

    subplot(2,1,2);
    plot((0:length(pitch)-1) * 0.052, pitch, 'b-', 'LineWidth', 1.5);
    hold on;
    plot((0:length(confidence)-1) * 0.052, confidence * 200, 'r--');
    title('Pitch Tracking');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    legend('Pitch', 'Confidence (scaled)');
    grid on;

    % Calculate statistics
    validPitch = pitch(confidence > 0.7);
    fprintf('Mean pitch: %.2f Hz\n', mean(validPitch, 'omitnan'));
    fprintf('Pitch range: %.2f - %.2f Hz\n', min(validPitch), max(validPitch));
else
    fprintf('Audio Toolbox not available - using fallback\n');
end

%% Example 8: Onset Detection
fprintf('\n=== Example 8: Onset Detection ===\n');

% Load percussive audio
[audio, fs] = audioread('drums.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

% Detect onsets
fprintf('Detecting onsets...\n');
onsetTimes = ap.detectOnsets(audio, fs, 'Threshold', 0.5);

fprintf('Detected %d onsets\n', length(onsetTimes));

% Plot audio with onset markers
figure;
t = (0:length(audio)-1) / fs;
plot(t, audio);
hold on;
for i = 1:length(onsetTimes)
    xline(onsetTimes(i), 'r--', 'LineWidth', 1.5);
end
title(sprintf('Onset Detection (%d onsets)', length(onsetTimes)));
xlabel('Time (s)');
ylabel('Amplitude');

% Estimate tempo
tempo = ap.estimateTempo(audio, fs);
fprintf('Estimated tempo: %.1f BPM\n', tempo);

%% Example 9: Psychoacoustic Analysis
fprintf('\n=== Example 9: Psychoacoustic Analysis ===\n');

% Load audio
[audio, fs] = audioread('music.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

if ap.HasAudioToolbox
    % Measure acoustic loudness
    fprintf('Measuring acoustic loudness...\n');
    loudness = ap.measureLoudness(audio, fs, 'Calibration', 94);

    fprintf('Loudness: %.2f phons\n', mean(loudness.phons));
    fprintf('Loudness: %.2f sones\n', mean(loudness.sones));

    % Measure SPL
    spl = ap.measureSPL(audio, fs, 'Weighting', 'A');
    fprintf('SPL (A-weighted): %.2f dB\n', mean(spl.levels));

    % Bark scale analysis
    fprintf('Performing Bark scale analysis...\n');
    barkAnalysis = ap.barkScaleAnalysis(audio, fs);

    figure;
    bar(barkAnalysis.frequencies, 10*log10(barkAnalysis.energies + eps));
    title('Bark Scale Energy Distribution');
    xlabel('Frequency (Hz)');
    ylabel('Energy (dB)');
    grid on;
end

%% Example 10: Feature Extraction for Machine Learning
fprintf('\n=== Example 10: Feature Extraction ===\n');

% Load audio samples
[audio1, fs] = audioread('sample1.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

if ap.HasAudioToolbox
    fprintf('Extracting comprehensive feature set...\n');

    % Extract all features
    features = ap.extractAllFeatures(audio1, fs);

    % Display feature matrix size
    fprintf('Feature matrix size: %d x %d\n', size(features, 1), size(features, 2));

    % Extract MFCC specifically
    mfcc = ap.extractMFCC(audio1, fs, 'NumCoeffs', 13);
    fprintf('MFCC matrix size: %d x %d\n', size(mfcc, 1), size(mfcc, 2));

    % Plot MFCC
    figure;
    imagesc(mfcc');
    title('MFCC Features');
    xlabel('Frame');
    ylabel('Coefficient');
    colorbar;
else
    fprintf('Audio Toolbox not available\n');
end

%% Example 11: Advanced Filtering
fprintf('\n=== Example 11: Advanced Filtering ===\n');

% Load audio
[audio, fs] = audioread('speech.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

% Apply octave band filter at 1 kHz
fprintf('Applying octave band filter...\n');
filtered = ap.octaveFilter(audio, fs, 'CenterFrequency', 1000);

% Apply 31-band graphic EQ
fprintf('Applying 31-band graphic EQ...\n');
gains = zeros(31, 1);  % All bands at 0 dB
gains(15:17) = 6;      % Boost around 1 kHz (+6 dB)
gains(1:5) = -3;       % Cut low end (-3 dB)

eqAudio = ap.graphicEQ(audio, fs, gains);

% Export
AudioExporter(filtered, 'octave_filtered.wav', 'SampleRate', fs);
AudioExporter(eqAudio, 'graphic_eq.wav', 'SampleRate', fs);

%% Example 12: Time Scaling
fprintf('\n=== Example 12: Time Scaling ===\n');

% Load audio
[audio, fs] = audioread('speech.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

if ap.HasAudioToolbox
    fprintf('Time scaling without pitch change...\n');

    % Make it 1.5x faster (same pitch)
    faster = ap.timeScale(audio, fs, 0.67);  % factor < 1 = faster

    % Make it slower
    slower = ap.timeScale(audio, fs, 1.5);   % factor > 1 = slower

    fprintf('Original duration: %.2f s\n', length(audio)/fs);
    fprintf('Faster duration: %.2f s\n', length(faster)/fs);
    fprintf('Slower duration: %.2f s\n', length(slower)/fs);

    % Export
    AudioExporter(faster, 'speech_faster.wav', 'SampleRate', fs);
    AudioExporter(slower, 'speech_slower.wav', 'SampleRate', fs);
else
    fprintf('Audio Toolbox required for time scaling\n');
end

%% Example 13: Stereo Processing
fprintf('\n=== Example 13: Stereo Processing ===\n');

% Load stereo audio
[stereoAudio, fs] = audioread('stereo_music.wav');

% Create advanced processor
ap = AdvancedAudioProcessor();

% Widen stereo image
fprintf('Widening stereo image...\n');
widened = ap.stereoWiden(stereoAudio, 0.5);  % 50% wider

% Mid-side processing
fprintf('M/S processing - boost mid, reduce sides...\n');
msProcessed = ap.midSideProcess(stereoAudio, 1.2, 0.8);  % +20% mid, -20% side

% Export
AudioExporter(widened, 'widened_stereo.wav', 'SampleRate', fs);
AudioExporter(msProcessed, 'ms_processed.wav', 'SampleRate', fs);

%% Example 14: Crossfading
fprintf('\n=== Example 14: Crossfading ===\n');

% Load two audio files
[audio1, fs] = audioread('track1.wav');
[audio2, ~] = audioread('track2.wav');

% Create editors
editor1 = AudioEditor(audio1, fs);

% Crossfade with second audio (1 second crossfade)
fprintf('Applying 1-second crossfade...\n');
editor1.crossfade(audio2, 1.0, 'scurve');

% Export
crossfaded = editor1.getAudio();
AudioExporter(crossfaded, 'crossfaded.wav', 'SampleRate', fs);

%% Example 15: Complete Workflow
fprintf('\n=== Example 15: Complete Production Workflow ===\n');

% 1. Load and trim raw recordings
fprintf('Step 1: Loading and trimming...\n');
[vocal, fs] = audioread('raw_vocal.wav');
editor = AudioEditor(vocal, fs);
editor.setSelection(0.5, length(vocal)/fs - 0.5);  % Trim ends
editor.trim();

% 2. Denoise with wavelets
fprintf('Step 2: Denoising...\n');
wp = WaveletProcessor();
cleanVocal = wp.denoise(editor.getAudio(), 'Wavelet', 'db4');

% 3. Normalize and apply fades
fprintf('Step 3: Normalization and fades...\n');
editor2 = AudioEditor(cleanVocal, fs);
editor2.fadeIn(0.2, 'scurve');
editor2.fadeOut(0.5, 'exponential');
editor2.normalize('lufs', -16);

% 4. Create mix with multiple tracks
fprintf('Step 4: Mixing...\n');
mixer = MixerCoreEnhanced(4, fs);

% Load backing tracks
[drums, ~] = audioread('drums.wav');
[bass, ~] = audioread('bass.wav');
[guitar, ~] = audioread('guitar.wav');

mixer.loadTrack(1, drums, fs);
mixer.loadTrack(2, bass, fs);
mixer.loadTrack(3, guitar, fs);
mixer.loadTrack(4, editor2.getAudio(), fs);

% Set names and offsets
mixer.setTrackName(1, 'Drums');
mixer.setTrackName(2, 'Bass');
mixer.setTrackName(3, 'Guitar');
mixer.setTrackName(4, 'Vocals');

mixer.setTrackOffset(4, 0.5);  % Vocals start slightly later

% Add effects
mixer.addEffect(4, 'Reverb', struct('RoomSize', 0.6, 'DecayTime', 2.0));
mixer.addEffect(4, 'Compression', struct('Threshold', -12, 'Ratio', 4));

% Set levels
mixer.setTrackVolume(1, 0.8);
mixer.setTrackVolume(2, 0.7);
mixer.setTrackVolume(3, 0.6);
mixer.setTrackVolume(4, 0.9);

% 5. Process final mix
fprintf('Step 5: Processing final mix...\n');
finalMix = mixer.processMix();

% 6. Master the mix
fprintf('Step 6: Mastering...\n');
ap = AdvancedAudioProcessor();
if ap.HasAudioToolbox
    % Apply multiband compression (using parametric EQ as example)
    bands = [
        struct('frequency', 100, 'gain', 2, 'Q', 0.7), ...
        struct('frequency', 3000, 'gain', 1, 'Q', 1.0), ...
        struct('frequency', 10000, 'gain', 0.5, 'Q', 0.7)
    ];
    masteredMix = ap.parametricEQ(finalMix, fs, bands);
else
    masteredMix = finalMix;
end

% 7. Final limiting and export
fprintf('Step 7: Final export...\n');
masterEditor = AudioEditor(masteredMix, fs);
masterEditor.normalize('lufs', -14);  % Broadcast standard

% Export multiple formats
fprintf('Exporting multiple formats...\n');
masterEditor.export('final_mix_24bit.wav', 'BitDepth', 24);
masterEditor.export('final_mix_16bit.wav', 'BitDepth', 16);

fprintf('\n=== Workflow Complete ===\n');
fprintf('Final duration: %.2f seconds\n', length(masteredMix)/fs);

%% Summary
fprintf('\n===========================================\n');
fprintf('Audio Processor Enhancement Examples Complete\n');
fprintf('===========================================\n\n');

fprintf('New capabilities demonstrated:\n');
fprintf('✓ Time offset mixing\n');
fprintf('✓ Audio editing and trimming\n');
fprintf('✓ Auto-alignment\n');
fprintf('✓ Wavelet denoising\n');
fprintf('✓ Time-frequency analysis\n');
fprintf('✓ Component separation\n');
fprintf('✓ Pitch detection\n');
fprintf('✓ Onset detection\n');
fprintf('✓ Psychoacoustic analysis\n');
fprintf('✓ Feature extraction\n');
fprintf('✓ Advanced filtering\n');
fprintf('✓ Time scaling\n');
fprintf('✓ Stereo processing\n');
fprintf('✓ Crossfading\n');
fprintf('✓ Complete production workflow\n\n');

fprintf('Your audio processor now has professional-level capabilities!\n');
