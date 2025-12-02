%% CONVOLUTION REVERB EXAMPLES
% Complete examples of using convolution reverb to simulate acoustic spaces
%
% These examples show you how to:
% - Apply reverb to different audio sources
% - Use built-in impulse responses
% - Load real impulse responses
% - Control echo amount and character
% - Create professional reverb effects
%
% Prerequisites:
% - ConvolutionReverb.m must be in your path
% - Audio files for testing (or generate test signals)

%% Constants
% Audio processing constants with units and sources
SAMPLE_RATE_HZ = 44100; % [Hz] Standard CD-quality sample rate (IEC 60908)
A4_FREQUENCY_HZ = 440; % [Hz] Standard tuning frequency A4 (ISO 16:1975)
LOW_FREQ_HZ = 100; % [Hz] Low frequency cutoff for bandpass filter
HIGH_FREQ_HZ = 8000; % [Hz] High frequency cutoff for bandpass filter
DURATION_SECONDS = 2; % [s] Test signal duration
WET_DRY_MIX = 0.5; % [0-1] Wet/dry mix ratio
PAUSE_DURATION_SECONDS = 2.5; % [s] Pause duration between audio playback
CLIP_PREVENTION_SCALE = 0.5; % [0-1] Scale factor to prevent clipping
DECAY_RATE = 2; % [1/s] Exponential decay rate for test tone
BUTTERWORTH_ORDER = 4; % Filter order for Butterworth bandpass
VOCAL_FUNDAMENTAL_HZ = 200; % [Hz] Fundamental frequency for vocal simulation
VOCAL_HARMONIC_2_HZ = 400; % [Hz] Second harmonic for vocal simulation
VOCAL_HARMONIC_3_HZ = 600; % [Hz] Third harmonic for vocal simulation
VOCAL_DECAY_RATE = 4; % [1/s] Decay rate for vocal simulation
VOCAL_ENVELOPE_RISE_S = 0.2; % [s] Envelope rise time
VOCAL_ENVELOPE_FALL_S = 1.8; % [s] Envelope fall start time
PRE_DELAY_1_MS = 0.02; % [s] Pre-delay option 1 (20 ms)
PRE_DELAY_2_MS = 0.05; % [s] Pre-delay option 2 (50 ms)
PRE_DELAY_3_MS = 0.10; % [s] Pre-delay option 3 (100 ms)
EQ_CUT_LOW_DB = -8; % [dB] Low frequency cut for EQ
EQ_CUT_HIGH_DB = -8; % [dB] High frequency cut for EQ
EQ_BOOST_LOW_DB = 4; % [dB] Low frequency boost for EQ
EQ_BOOST_HIGH_DB = 4; % [dB] High frequency boost for EQ
EQ_WARM_CUT_HIGH_DB = -6; % [dB] High frequency cut for warm EQ
EQ_BRIGHT_CUT_LOW_DB = -6; % [dB] Low frequency cut for bright EQ
NOISE_DURATION_S = 0.2; % [s] White noise burst duration
NOISE_SILENCE_S = 1.8; % [s] Silence duration after noise burst
PLAYBACK_SCALE_1 = 0.7; % [0-1] Playback scale factor 1
PLAYBACK_SCALE_2 = 0.5; % [0-1] Playback scale factor 2
PLAYBACK_SCALE_3 = 0.8; % [0-1] Playback scale factor 3
PAUSE_SHORT_S = 1.5; % [s] Short pause duration
PAUSE_MEDIUM_S = 2; % [s] Medium pause duration
SPECTROGRAM_WINDOW = 512; % [samples] Spectrogram window size
SPECTROGRAM_OVERLAP = 384; % [samples] Spectrogram overlap size
SPECTROGRAM_NFFT = 512; % [samples] Spectrogram FFT size
SPECTROGRAM_WINDOW_SMALL = 256; % [samples] Small spectrogram window
SPECTROGRAM_OVERLAP_SMALL = 192; % [samples] Small spectrogram overlap
HARMONIC_COUNT = 8; % Number of harmonics for vocal simulation
VIBRATO_DEPTH = 0.02; % [0-1] Vibrato depth
VIBRATO_RATE_HZ = 5; % [Hz] Vibrato rate
VOCAL_NORMALIZE_SCALE = 0.8; % [0-1] Vocal normalization scale
WET_DRY_SUBTLE_WET = 0.25; % [0-1] Subtle wet mix
WET_DRY_SUBTLE_DRY = 0.75; % [0-1] Subtle dry mix
PRE_DELAY_VOCAL_MS = 0.05; % [s] Pre-delay for vocal (50 ms)
EQ_VOCAL_LOW_DB = -4; % [dB] Low frequency cut for vocal EQ
EQ_VOCAL_HIGH_DB = -3; % [dB] High frequency cut for vocal EQ
DAMPING_VOCAL = 0.3; % [0-1] Damping for vocal
STEREO_WIDTH_VOCAL = 1.3; % Stereo width for vocal
IMPULSE_DURATION_S = 0.1; % [s] Impulse duration for snare simulation
SNARE_NOISE_SCALE = 0.3; % [0-1] Noise scale for snare
KICK_DURATION_S = 2; % [s] Kick drum duration
KICK_LOW_FREQ_HZ = 50; % [Hz] Low frequency for kick
KICK_HIGH_FREQ_HZ = 200; % [Hz] High frequency for kick
TAIL_LENGTH_1_S = 2.0; % [s] Tail length option 1
TAIL_LENGTH_2_S = 1.0; % [s] Tail length option 2
TAIL_LENGTH_3_S = 0.5; % [s] Tail length option 3
TAIL_LENGTH_4_S = 0.25; % [s] Tail length option 4
CREATIVE_IMPULSE_DURATION_S = 0.2; % [s] Impulse duration for creative example
CREATIVE_LOW_FREQ_HZ = 100; % [Hz] Low frequency for creative example
CREATIVE_HIGH_FREQ_HZ = 4000; % [Hz] High frequency for creative example
CREATIVE_CHORD_DURATION_S = 2; % [s] Chord duration for creative example
CREATIVE_CHORD_DECAY_RATE = 2; % [1/s] Chord decay rate
CREATIVE_WET_DRY_WET = 0.7; % [0-1] Wet mix for creative example
CREATIVE_WET_DRY_DRY = 0.3; % [0-1] Dry mix for creative example
C_MAJOR_FREQ_1_HZ = 261.63; % [Hz] C note frequency
C_MAJOR_FREQ_2_HZ = 329.63; % [Hz] E note frequency
C_MAJOR_FREQ_3_HZ = 392; % [Hz] G note frequency
DAMPING_LEVEL_1 = 0.3; % [0-1] Damping level 1
DAMPING_LEVEL_2 = 0.6; % [0-1] Damping level 2
DAMPING_LEVEL_3 = 0.9; % [0-1] Damping level 3
BRIGHT_FREQ_1_HZ = 440; % [Hz] Bright frequency 1
BRIGHT_FREQ_2_HZ = 880; % [Hz] Bright frequency 2
BRIGHT_FREQ_3_HZ = 1760; % [Hz] Bright frequency 3
BRIGHT_FREQ_4_HZ = 3520; % [Hz] Bright frequency 4
BRIGHT_DECAY_RATE = 8; % [1/s] Bright decay rate
BRIGHT_DURATION_S = 0.3; % [s] Bright signal duration
MONO_DECAY_RATE = 5; % [1/s] Mono decay rate
MONO_DURATION_S = 0.5; % [s] Mono signal duration
STEREO_WIDTH_1 = 0.5; % Stereo width option 1
STEREO_WIDTH_2 = 1.0; % Stereo width option 2
STEREO_WIDTH_3 = 1.5; % Stereo width option 3
STEREO_WIDTH_4 = 2.0; % Stereo width option 4
BATCH_WET_DRY_WET = 0.35; % [0-1] Wet mix for batch processing
BATCH_WET_DRY_DRY = 0.65; % [0-1] Dry mix for batch processing
BATCH_PRE_DELAY_S = 0.03; % [s] Pre-delay for batch processing
TONE_DURATION_S = 1; % [s] Tone duration
TONE_DECAY_RATE = 3; % [1/s] Tone decay rate
TONE_FREQ_1_HZ = 554; % [Hz] Tone frequency 1 (C#)
TONE_FREQ_2_HZ = 659; % [Hz] Tone frequency 2 (E)

%% Example 1: Basic Reverb Application
% Start with the simplest case - apply reverb to audio

fprintf('\n=== Example 1: Basic Reverb ===\n');

% Create reverb
reverb = ConvolutionReverb();

% Load a built-in impulse response
reverb.loadBuiltIn('concert_hall');

% Generate test audio (or load your own)
fs = SAMPLE_RATE_HZ;
duration = DURATION_SECONDS;
t = linspace(0, duration, duration * fs)';
testAudio = sin(2*pi*A4_FREQUENCY_HZ*t) .* exp(-t*DECAY_RATE);  % Decaying tone

% Apply reverb
reverbedAudio = reverb.process(testAudio, fs);

% Compare
figure('Name', 'Example 1: Before and After');
subplot(2,1,1);
plot(t, testAudio);
title('Original (Dry)');
xlabel('Time (s)'); ylabel('Amplitude');

tReverb = (0:length(reverbedAudio)-1) / fs;
subplot(2,1,2);
plot(tReverb, reverbedAudio);
title('With Concert Hall Reverb');
xlabel('Time (s)'); ylabel('Amplitude');

% Listen
fprintf('Playing original...\n');
sound(testAudio, fs);
pause(duration + 0.5);

fprintf('Playing with reverb...\n');
sound(reverbedAudio * CLIP_PREVENTION_SCALE, fs);  % Scale down to prevent clipping
pause(length(reverbedAudio)/fs + 0.5);

%% Example 2: Compare Different Spaces
% Hear the same audio in different acoustic environments

fprintf('\n=== Example 2: Different Spaces ===\n');

% Generate percussive test signal (works well for demonstrating reverb)
fs = SAMPLE_RATE_HZ;
impulse = [1; zeros(fs*2-1, 1)];  % Single impulse
[b, a] = butter(BUTTERWORTH_ORDER, [LOW_FREQ_HZ, HIGH_FREQ_HZ]/(fs/2), 'bandpass');
clickSound = filter(b, a, impulse);
clickSound = clickSound / max(abs(clickSound));

% Create reverb processor
reverb = ConvolutionReverb();

% Try different spaces
spaces = {'small_room', 'medium_room', 'concert_hall', 'chamber', 'plate'};
results = cell(length(spaces), 1);

figure('Name', 'Example 2: Different Acoustic Spaces');
for i = 1:length(spaces)
    % Load space
    reverb.loadBuiltIn(spaces{i});

    % Process
    results{i} = reverb.process(clickSound, fs, 'WetDry', WET_DRY_MIX);

    % Plot
    subplot(length(spaces), 1, i);
    t = (0:length(results{i})-1) / fs;
    plot(t, results{i});
    title(sprintf('%s', strrep(spaces{i}, '_', ' ')));
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;

    % Play
    fprintf('Playing: %s\n', spaces{i});
    sound(results{i} * CLIP_PREVENTION_SCALE, fs);
    pause(PAUSE_DURATION_SECONDS);
end

%% Example 3: Control Echo Amount (Wet/Dry Mix)
% Adjust how much reverb vs direct sound

fprintf('\n=== Example 3: Wet/Dry Mix ===\n');

% Test signal
fs = SAMPLE_RATE_HZ;
t = linspace(0, 1, fs)';
testTone = sin(2*pi*A4_FREQUENCY_HZ*t) .* (t < 0.5);  % Short beep

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('medium_room');

% Try different wet/dry mixes
mixes = [0, 0.25, 0.5, 0.75, 1.0];  % 0% to 100% wet

figure('Name', 'Example 3: Wet/Dry Mix');
for i = 1:length(mixes)
    wetAmount = mixes(i);
    dryAmount = 1 - wetAmount;

    % Process
    result = reverb.process(testTone, fs, 'WetDry', wetAmount);

    % Plot
    subplot(length(mixes), 1, i);
    tPlot = (0:length(result)-1) / fs;
    plot(tPlot, result);
    title(sprintf('%.0f%% Wet, %.0f%% Dry', wetAmount*100, dryAmount*100));
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;

    % Play
    fprintf('Playing: %.0f%% wet\n', wetAmount*100);
    sound(result * 0.7, fs);
    pause(1.5);
end

%% Example 4: Pre-Delay for Clarity
% Add space between direct sound and reverb

fprintf('\n=== Example 4: Pre-Delay ===\n');

% Vocal-like test signal
fs = 44100;
t = linspace(0, 0.5, round(0.5*fs))';
vocalSim = sin(2*pi*200*t) + 0.5*sin(2*pi*400*t) + 0.3*sin(2*pi*600*t);
vocalSim = vocalSim .* exp(-t*4);  % Decay

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('plate');
reverb.setWetDry(0.4, 0.6);

% Try different pre-delays
preDelays = [0, 0.02, 0.05, 0.10];  % seconds

figure('Name', 'Example 4: Pre-Delay Effect');
for i = 1:length(preDelays)
    reverb.setPreDelay(preDelays(i));

    % Process
    result = reverb.process(vocalSim, fs);

    % Plot
    subplot(length(preDelays), 1, i);
    tPlot = (0:length(result)-1) / fs;
    plot(tPlot, result);
    title(sprintf('Pre-delay: %.0f ms', preDelays(i)*1000));
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;
    xlim([0, 1.5]);

    % Mark pre-delay
    if preDelays(i) > 0
        hold on;
        xline(preDelays(i), 'r--', 'Reverb Start');
        hold off;
    end

    % Play
    fprintf('Playing: %.0f ms pre-delay\n', preDelays(i)*1000);
    sound(result * 0.7, fs);
    pause(2);
end

%% Example 5: EQ the Reverb
% Shape the tone of the reverb tail

fprintf('\n=== Example 5: Reverb EQ ===\n');

% White noise burst (reveals frequency content)
fs = 44100;
noise = randn(round(0.2*fs), 1);
noise = [noise; zeros(round(1.8*fs), 1)];

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.6, 0.4);

% Try different EQ settings
eqSettings = {
    [0, 0, 0],     'Flat (No EQ)';
    [-8, 0, 0],    'Cut Lows (Thin)';
    [0, 0, -8],    'Cut Highs (Dark)';
    [+4, 0, -6],   'Warm (Boost Lows, Cut Highs)';
    [-6, 0, +4]    'Bright (Cut Lows, Boost Highs)'
};

figure('Name', 'Example 5: Reverb EQ');
for i = 1:size(eqSettings, 1)
    eq = eqSettings{i, 1};
    label = eqSettings{i, 2};

    reverb.setEQ(eq(1), eq(2), eq(3));

    % Process
    result = reverb.process(noise, fs);

    % Plot spectrogram
    subplot(size(eqSettings, 1), 1, i);
    spectrogram(result, 512, 384, 512, fs, 'yaxis');
    title(label);

    % Play
    fprintf('Playing: %s\n', label);
    sound(result * 0.5, fs);
    pause(2.5);
end

%% Example 6: Damping (Air Absorption)
% Simulate high-frequency absorption over distance

fprintf('\n=== Example 6: Damping ===\n');

% Bright test signal
fs = 44100;
t = linspace(0, 0.3, round(0.3*fs))';
bright = sum(sin(2*pi * [440, 880, 1760, 3520]' .* t'), 1)';
bright = bright .* exp(-t*8);

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.5, 0.5);

% Try different damping amounts
dampingLevels = [0, 0.3, 0.6, 0.9];

figure('Name', 'Example 6: Damping Effect');
for i = 1:length(dampingLevels)
    reverb.setDamping(dampingLevels(i));

    % Process
    result = reverb.process(bright, fs);

    % Plot
    subplot(length(dampingLevels), 1, i);
    spectrogram(result, 256, 192, 256, fs, 'yaxis');
    title(sprintf('Damping: %.0f%% (simulates air absorption)', dampingLevels(i)*100));

    % Play
    fprintf('Playing: %.0f%% damping\n', dampingLevels(i)*100);
    sound(result * 0.7, fs);
    pause(2);
end

%% Example 7: Stereo Width Control
% Adjust the stereo image of the reverb

fprintf('\n=== Example 7: Stereo Width ===\n');

% Mono test signal
fs = 44100;
t = linspace(0, 0.5, round(0.5*fs))';
monoSound = sin(2*pi*440*t) .* exp(-t*5);

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('medium_room');
reverb.setWetDry(0.6, 0.4);

% Try different stereo widths
widths = [0, 0.5, 1.0, 1.5, 2.0];

figure('Name', 'Example 7: Stereo Width');
for i = 1:length(widths)
    reverb.setStereoWidth(widths(i));

    % Process
    result = reverb.process(monoSound, fs);

    % Calculate stereo correlation
    if size(result, 2) == 2
        correlation = corrcoef(result(:,1), result(:,2));
        corr = correlation(1,2);
    else
        corr = 1;
    end

    % Plot
    subplot(length(widths), 2, i*2-1);
    plot((0:length(result)-1)/fs, result);
    title(sprintf('Width: %.1f (Correlation: %.2f)', widths(i), corr));
    xlabel('Time (s)'); ylabel('Amplitude');
    legend('Left', 'Right');
    grid on;

    % Plot stereo field
    subplot(length(widths), 2, i*2);
    if size(result, 2) == 2
        plot(result(:,1), result(:,2), '.', 'MarkerSize', 1);
        title('Stereo Field');
        xlabel('Left'); ylabel('Right');
        axis equal; grid on;
    end

    % Play
    fprintf('Playing: Width %.1f\n', widths(i));
    sound(result * 0.7, fs);
    pause(1.5);
end

%% Example 8: Reverse Reverb
% Create reverse reverb effect

fprintf('\n=== Example 8: Reverse Reverb ===\n');

% Snare-like sound
fs = 44100;
impulse = [1; zeros(round(0.1*fs)-1, 1)];
[b, a] = butter(4, [150, 8000]/(fs/2), 'bandpass');
snare = filter(b, a, impulse);
snare = snare .* (1 + randn(size(snare))*0.3);  % Add noise
snare = snare / max(abs(snare));

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.7, 0.3);

% Normal reverb
normal = reverb.process(snare, fs);

% Reverse reverb
reverb.reverseIR();
reversed = reverb.process(snare, fs);

% Plot
figure('Name', 'Example 8: Reverse Reverb');
subplot(2,1,1);
plot((0:length(normal)-1)/fs, normal);
title('Normal Reverb (decay after hit)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot((0:length(reversed)-1)/fs, reversed);
title('Reverse Reverb (swell before hit)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

% Play
fprintf('Playing: Normal reverb\n');
sound(normal * 0.7, fs);
pause(length(normal)/fs + 0.5);

fprintf('Playing: Reverse reverb (notice the swell!)\n');
sound(reversed * 0.7, fs);
pause(length(reversed)/fs + 0.5);

%% Example 9: Tail Length Control
% Truncate long reverb tails for cleaner mixes

fprintf('\n=== Example 9: Tail Length ===\n');

% Drum hit
fs = 44100;
kick = [1; zeros(round(2*fs)-1, 1)];
[b, a] = butter(4, [50, 200]/(fs/2), 'bandpass');
kick = filter(b, a, kick);
kick = kick / max(abs(kick));

% Create reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.5, 0.5);

% Try different tail lengths
tailLengths = [inf, 2.0, 1.0, 0.5, 0.25];  % seconds (inf = full)

figure('Name', 'Example 9: Tail Length');
for i = 1:length(tailLengths)
    if isinf(tailLengths(i))
        reverb.setTailLength([]);  % Full length
        label = 'Full Length';
    else
        reverb.setTailLength(tailLengths(i));
        label = sprintf('%.2f seconds', tailLengths(i));
    end

    % Process
    result = reverb.process(kick, fs);

    % Plot
    subplot(length(tailLengths), 1, i);
    plot((0:length(result)-1)/fs, result);
    title(sprintf('Tail Length: %s', label));
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;
    xlim([0, 3]);

    % Play
    fprintf('Playing: %s tail\n', label);
    sound(result * 0.7, fs);
    pause(1.5);
end

%% Example 10: Real-World Vocal Processing
% Professional vocal reverb chain

fprintf('\n=== Example 10: Professional Vocal Reverb ===\n');

% Simulate vocal phrase
fs = 44100;
t = linspace(0, 2, 2*fs)';
fundamental = 200;  % Hz

% Create harmonic-rich vocal-like sound
vocal = zeros(size(t));
for harmonic = 1:8
    vocal = vocal + (1/harmonic) * sin(2*pi * fundamental * harmonic * t);
end

% Add expression (volume envelope)
envelope = ones(size(t));
envelope(t < 0.2) = linspace(0, 1, sum(t < 0.2));
envelope(t > 1.8) = linspace(1, 0, sum(t > 1.8));
vocal = vocal .* envelope;

% Add vibrato
vibrato = 1 + 0.02*sin(2*pi*5*t);
vocalPhase = cumsum(fundamental * vibrato / fs);
vocal = sin(2*pi * vocalPhase);
vocal = vocal .* envelope;

% Normalize
vocal = vocal / max(abs(vocal)) * 0.8;

% Create professional vocal reverb
reverb = ConvolutionReverb();
reverb.loadBuiltIn('plate');  % Classic vocal reverb
reverb.setWetDry(0.25, 0.75);  % Subtle
reverb.setPreDelay(0.05);  % 50ms - keeps vocal clear
reverb.setEQ(-4, 0, -3);  % Cut lows and highs
reverb.setDamping(0.3);  % Natural rolloff
reverb.setStereoWidth(1.3);  % Slightly wide

% Process
processed = reverb.process(vocal, fs);

% Compare
figure('Name', 'Example 10: Professional Vocal Reverb');
subplot(2,1,1);
plot(t, vocal);
title('Dry Vocal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot((0:length(processed)-1)/fs, processed);
title('With Professional Reverb (Plate, Pre-delay, EQ)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

% Play
fprintf('Playing: Dry vocal\n');
sound(vocal, fs);
pause(2.5);

fprintf('Playing: With professional reverb\n');
sound(processed * 0.8, fs);
pause(length(processed)/fs + 0.5);

%% Example 11: IR Analysis and Manipulation
% Understand and modify impulse responses

fprintf('\n=== Example 11: IR Analysis ===\n');

% Create reverb and load IR
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');

% Get info
fprintf('\n--- Original IR ---\n');
reverb.getIRInfo();

% Plot IR
reverb.plotIR();

% Trim (remove silence)
fprintf('\n--- Trimming IR ---\n');
reverb.trimIR();

% Normalize
fprintf('\n--- Normalizing IR ---\n');
reverb.normalizeIR();

% Get new info
fprintf('\n--- Modified IR ---\n');
reverb.getIRInfo();

%% Example 12: Batch Processing Multiple Files
% Process multiple audio files with the same reverb

fprintf('\n=== Example 12: Batch Processing ===\n');

% Generate test files (in real use, you'd have actual files)
fs = 44100;
testSounds = {
    sin(2*pi*440*(0:fs-1)'/fs) .* exp(-(0:fs-1)'/fs*3), 'tone1';
    sin(2*pi*554*(0:fs-1)'/fs) .* exp(-(0:fs-1)'/fs*3), 'tone2';
    sin(2*pi*659*(0:fs-1)'/fs) .* exp(-(0:fs-1)'/fs*3), 'tone3'
};

% Setup reverb once
reverb = ConvolutionReverb();
reverb.loadBuiltIn('medium_room');
reverb.setWetDry(0.35, 0.65);
reverb.setPreDelay(0.03);

fprintf('Processing %d files...\n', length(testSounds));

% Process all files
results = cell(length(testSounds), 1);
for i = 1:length(testSounds)
    fprintf('  Processing %s... ', testSounds{i, 2});
    results{i} = reverb.process(testSounds{i, 1}, fs);
    fprintf('done\n');

    % In real use, save to file:
    % audiowrite(sprintf('processed_%s.wav', testSounds{i,2}), results{i}, fs);
end

fprintf('Batch processing complete!\n');

%% Example 13: Creative Sound Design
% Use convolution for creative effects

fprintf('\n=== Example 13: Creative Convolution ===\n');

% Instead of using a room IR, use a musical note as the "impulse response"
fs = 44100;
t = linspace(0, 2, 2*fs)';

% Create a chord as our "IR"
chord = sin(2*pi*261.63*t) + sin(2*pi*329.63*t) + sin(2*pi*392*t);  % C major
chord = chord .* exp(-t*2);  % Decay
chord = chord / max(abs(chord));

% Create reverb and load our chord as the IR
reverb = ConvolutionReverb();
reverb.IR = chord;
reverb.IRSampleRate = fs;
reverb.IRName = 'C_Major_Chord';

% Generate percussive test sound
impulse = [1; zeros(round(0.2*fs)-1, 1)];
[b, a] = butter(4, [100, 4000]/(fs/2), 'bandpass');
hit = filter(b, a, impulse);

% Convolve percussion with chord
reverb.setWetDry(0.7, 0.3);
weird = reverb.process(hit, fs);

% Plot
figure('Name', 'Example 13: Creative Convolution');
subplot(3,1,1);
plot((0:length(chord)-1)/fs, chord);
title('Our "IR": C Major Chord');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot((0:length(hit)-1)/fs, hit);
title('Input: Percussion Hit');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot((0:length(weird)-1)/fs, weird);
title('Result: Harmonic Resonance');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

% Play
fprintf('Playing: Original percussion\n');
sound(hit * 0.7, fs);
pause(1);

fprintf('Playing: Percussion with chord resonance (weird and cool!)\n');
sound(weird * 0.5, fs);
pause(length(weird)/fs + 0.5);

%% Summary

fprintf('\n========================================\n');
fprintf('CONVOLUTION REVERB EXAMPLES COMPLETE\n');
fprintf('========================================\n\n');

fprintf('You''ve seen:\n');
fprintf('  1. Basic reverb application\n');
fprintf('  2. Different acoustic spaces\n');
fprintf('  3. Wet/dry mix control (echo amount)\n');
fprintf('  4. Pre-delay for clarity\n');
fprintf('  5. EQ on reverb\n');
fprintf('  6. Damping (air absorption)\n');
fprintf('  7. Stereo width control\n');
fprintf('  8. Reverse reverb\n');
fprintf('  9. Tail length control\n');
fprintf(' 10. Professional vocal processing\n');
fprintf(' 11. IR analysis and manipulation\n');
fprintf(' 12. Batch processing\n');
fprintf(' 13. Creative sound design\n\n');

fprintf('Next steps:\n');
fprintf('  - Download real IRs from openair.hosted.york.ac.uk\n');
fprintf('  - Experiment with your own audio files\n');
fprintf('  - Try extreme settings for creative effects\n');
fprintf('  - Record your own impulse responses\n\n');

fprintf('See CONVOLUTION_REVERB_GUIDE.md for detailed documentation\n\n');
