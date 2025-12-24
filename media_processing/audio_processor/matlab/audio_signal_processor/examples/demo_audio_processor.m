function demo_audio_processor()
%DEMO_AUDIO_PROCESSOR Comprehensive demonstration of Audio Signal Processor
%
%   This function demonstrates the complete workflow of the Audio Signal
%   Processor, including loading audio, applying filters and effects,
%   multi-track mixing, and analysis.
%
%   Features Demonstrated:
%   ---------------------
%   - Loading audio from MATLAB built-in sounds and custom files
%   - FFT filtering with different window functions
%   - Audio effects processing (reverb, delay, EQ, compression)
%   - Multi-track mixing with effect chains
%   - Frequency analysis and spectrogram generation
%   - Audio export with different formats
%
%   Example:
%   --------
%   demo_audio_processor()
%
%   See also: launch_audio_processor, MainWindow

fprintf('Audio Signal Processor - Comprehensive Demo\n');
fprintf('==========================================\n\n');

% Initialize components
fprintf('Initializing components...\n');
libraryManager = SoundLibraryManager();
mixer = MixerCore(4, 44100); % 4-track mixer
effectsLibrary = InstrumentEffectsLibrary();

fprintf('✓ Components initialized\n\n');

% Demo 1: Load MATLAB built-in sounds
fprintf('Demo 1: Loading MATLAB Built-in Sounds\n');
fprintf('-------------------------------------\n');
demoMATLABSounds(libraryManager);

% Demo 2: FFT Filtering
fprintf('\nDemo 2: FFT Filtering\n');
fprintf('--------------------\n');
demoFFTFilters();

% Demo 3: Audio Effects
fprintf('\nDemo 3: Audio Effects\n');
fprintf('--------------------\n');
demoAudioEffects();

% Demo 4: Multi-track Mixing
fprintf('\nDemo 4: Multi-track Mixing\n');
fprintf('------------------------\n');
demoMultiTrackMixing(mixer, effectsLibrary);

% Demo 5: Frequency Analysis
fprintf('\nDemo 5: Frequency Analysis\n');
fprintf('-------------------------\n');
demoFrequencyAnalysis();

% Demo 6: Audio Export
fprintf('\nDemo 6: Audio Export\n');
fprintf('------------------\n');
demoAudioExport();

fprintf('\n🎉 Demo completed successfully!\n');
fprintf('Launch the GUI with: launch_audio_processor()\n');
end

function demoMATLABSounds(libraryManager)
% Demonstrate MATLAB built-in sound loading

fprintf('Available MATLAB sounds:\n');
matlabSounds = libraryManager.getMATLABSounds();
soundNames = fieldnames(matlabSounds);

for i = 1:min(5, length(soundNames)) % Show first 5
    soundName = soundNames{i};
    soundInfo = matlabSounds.(soundName);
    fprintf('  - %s: %s\n', soundName, soundInfo.Description);
end

if length(soundNames) > 5
    fprintf('  ... and %d more\n', length(soundNames) - 5);
end

% Load Handel's Hallelujah Chorus
try
    fprintf('\nLoading Handel''s Hallelujah Chorus...\n');
    [audioData, sampleRate, info] = libraryManager.loadMATLABSound('handel');
    fprintf('✓ Loaded: %s\n', info.Description);
    fprintf('  Duration: %.2f seconds\n', info.Duration);
    fprintf('  Sample Rate: %d Hz\n', info.SampleRate);
    fprintf('  Channels: %d\n', info.Channels);
catch ME
    fprintf('✗ Error loading MATLAB sound: %s\n', ME.message);
end
end

function demoFFTFilters()
% Demonstrate FFT filtering

% Create test signal
sampleRate = 44100;
duration = 2.0;
t = (0:round(sampleRate*duration)-1) / sampleRate;

% Multi-tone signal
testSignal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t) + 0.3*sin(2*pi*1760*t) + 0.1*randn(size(t));
testSignal = testSignal';

fprintf('Created test signal with frequencies: 440Hz, 880Hz, 1760Hz + noise\n');

% Test different filter types
filterTypes = {'Low-pass', 'High-pass', 'Band-pass', 'Band-stop'};

for i = 1:length(filterTypes)
    filterType = filterTypes{i};
    fprintf('\nTesting %s filter...\n', filterType);

    try
        % Apply FFT filter
        filteredSignal = FFTFilters(testSignal, filterType, ...
            'WindowShape', 'Gaussian', ...
            'FreqLow', 0.1, ...
            'FreqHigh', 0.3, ...
            'TransitionBW', 0.05, ...
            'ZeroPhase', true, ...
            'FreqUnit', 'normalized');

        % Calculate RMS to show effect
        originalRMS = rms(testSignal);
        filteredRMS = rms(filteredSignal);

        fprintf('  ✓ %s filter applied\n', filterType);
        fprintf('    Original RMS: %.4f\n', originalRMS);
        fprintf('    Filtered RMS: %.4f\n', filteredRMS);

    catch ME
        fprintf('  ✗ Error applying %s filter: %s\n', filterType, ME.message);
    end
end
end

function demoAudioEffects()
% Demonstrate audio effects

% Create test signal (drum-like)
sampleRate = 44100;
duration = 1.0;
t = (0:round(sampleRate*duration)-1) / sampleRate;

% Create a simple drum-like signal
testSignal = zeros(size(t));
for i = 1:4
    startIdx = round(i * sampleRate * 0.2);
    if startIdx <= length(testSignal)
        endIdx = min(startIdx + round(sampleRate * 0.1), length(testSignal));
        testSignal(startIdx:endIdx) = 0.8 * exp(-(0:endIdx-startIdx) / (sampleRate * 0.05));
    end
end
testSignal = testSignal';

fprintf('Created drum-like test signal\n');

% Test different effects
effects = {'Reverb', 'Delay', 'Compression', 'Distortion'};

for i = 1:length(effects)
    effectType = effects{i};
    fprintf('\nTesting %s effect...\n', effectType);

    try
        % Apply effect with default parameters
        processedSignal = AudioEffects(testSignal, effectType, 'SampleRate', sampleRate);

        % Calculate RMS to show effect
        originalRMS = rms(testSignal);
        processedRMS = rms(processedSignal);

        fprintf('  ✓ %s effect applied\n', effectType);
        fprintf('    Original RMS: %.4f\n', originalRMS);
        fprintf('    Processed RMS: %.4f\n', processedRMS);

    catch ME
        fprintf('  ✗ Error applying %s effect: %s\n', effectType, ME.message);
    end
end
end

function demoMultiTrackMixing(mixer, effectsLibrary)
% Demonstrate multi-track mixing

fprintf('Setting up 4-track mix...\n');

% Create different signals for each track
sampleRate = 44100;
duration = 3.0;
t = (0:round(sampleRate*duration)-1) / sampleRate;

% Track 1: Kick drum
kickSignal = zeros(size(t));
for i = 1:6
    startIdx = round(i * sampleRate * 0.5);
    if startIdx <= length(kickSignal)
        endIdx = min(startIdx + round(sampleRate * 0.1), length(kickSignal));
        kickSignal(startIdx:endIdx) = 0.9 * exp(-(0:endIdx-startIdx) / (sampleRate * 0.03));
    end
end

% Track 2: Hi-hat
hihatSignal = 0.3 * sin(2*pi*8000*t) .* (mod(t, 0.25) < 0.05);

% Track 3: Bass
bassSignal = 0.6 * sin(2*pi*60*t);

% Track 4: Lead
leadSignal = 0.4 * sin(2*pi*440*t) .* (mod(t, 1.0) < 0.5);

% Load tracks into mixer
mixer.loadTrack(1, kickSignal, sampleRate);
mixer.loadTrack(2, hihatSignal, sampleRate);
mixer.loadTrack(3, bassSignal, sampleRate);
mixer.loadTrack(4, leadSignal, sampleRate);

fprintf('✓ Loaded 4 tracks into mixer\n');

% Set track levels and pan
mixer.setTrackVolume(1, 0.8);  % Kick
mixer.setTrackVolume(2, 0.4);  % Hi-hat
mixer.setTrackVolume(3, 0.7);  % Bass
mixer.setTrackVolume(4, 0.6);  % Lead

mixer.setTrackPan(1, 0.0);     % Kick center
mixer.setTrackPan(2, -0.3);    % Hi-hat left
mixer.setTrackPan(3, 0.0);     % Bass center
mixer.setTrackPan(4, 0.3);     % Lead right

fprintf('✓ Set track levels and panning\n');

% Add effects to tracks
try
    % Add reverb to lead
    mixer.addEffect(4, 'Reverb', struct('RoomSize', 0.4, 'DecayTime', 2.0));

    % Add compression to bass
    mixer.addEffect(3, 'Compression', struct('Threshold', -6, 'Ratio', 3));

    fprintf('✓ Added effects to tracks\n');
catch ME
    fprintf('✗ Error adding effects: %s\n', ME.message);
end

% Process mix
try
    fprintf('Processing mix...\n');
    mixedAudio = mixer.processMix();

    fprintf('✓ Mix processed successfully\n');
    fprintf('  Mixed audio size: %s\n', mat2str(size(mixedAudio)));
    fprintf('  Mixed audio duration: %.2f seconds\n', size(mixedAudio, 1) / sampleRate);

catch ME
    fprintf('✗ Error processing mix: %s\n', ME.message);
end
end

function demoFrequencyAnalysis()
% Demonstrate frequency analysis

% Create test signal with known frequencies
sampleRate = 44100;
duration = 2.0;
t = (0:round(sampleRate*duration)-1) / sampleRate;

% Multi-tone signal
testSignal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t) + 0.3*sin(2*pi*1760*t);
testSignal = testSignal';

fprintf('Created test signal with frequencies: 440Hz, 880Hz, 1760Hz\n');

% Create frequency analyzer
try
    analyzer = FrequencyAnalyzer(testSignal, sampleRate);
    analyzer.analyze();

    fprintf('✓ Frequency analysis completed\n');

    % Get spectrum
    [spectrum, frequencies] = analyzer.getSpectrum();
    fprintf('  Spectrum size: %s\n', mat2str(size(spectrum)));
    fprintf('  Frequency range: %.1f Hz to %.1f Hz\n', min(frequencies), max(frequencies));

    % Detect peaks
    peaks = analyzer.detectPeaks();
    fprintf('✓ Peak detection completed\n');
    fprintf('  Detected %d peaks\n', peaks.Count);

    if ~isempty(peaks.Frequencies)
        fprintf('  Peak frequencies: ');
        for i = 1:min(5, length(peaks.Frequencies))
            fprintf('%.1f Hz ', peaks.Frequencies(i));
        end
        fprintf('\n');
    end

catch ME
    fprintf('✗ Error in frequency analysis: %s\n', ME.message);
end

% Generate spectrogram
try
    fprintf('\nGenerating spectrogram...\n');
    [S, F, T] = SpectrogramGenerator(testSignal, 'SampleRate', sampleRate);

    fprintf('✓ Spectrogram generated\n');
    fprintf('  Spectrogram size: %s\n', mat2str(size(S)));
    fprintf('  Frequency bins: %d\n', length(F));
    fprintf('  Time frames: %d\n', length(T));

catch ME
    fprintf('✗ Error generating spectrogram: %s\n', ME.message);
end
end

function demoAudioExport()
% Demonstrate audio export

% Create test signal
sampleRate = 44100;
duration = 1.0;
t = (0:round(sampleRate*duration)-1) / sampleRate;
testSignal = 0.5 * sin(2*pi*440*t) + 0.3 * sin(2*pi*880*t);
testSignal = testSignal';

fprintf('Created test signal for export\n');

% Export in different formats
formats = {'wav', 'mp3', 'flac'};

for i = 1:length(formats)
    format = formats{i};
    filename = sprintf('demo_output.%s', format);

    try
        fprintf('\nExporting as %s...\n', upper(format));

        % Create metadata
        metadata = struct();
        metadata.Title = 'Demo Audio';
        metadata.Artist = 'Audio Signal Processor';
        metadata.Comment = 'Generated for demonstration';

        % Export audio
        success = AudioExporter(testSignal, filename, ...
            'SampleRate', sampleRate, ...
            'BitsPerSample', 16, ...
            'Quality', 80, ...
            'Metadata', metadata);

        if success
            fprintf('✓ Successfully exported as %s\n', filename);

            % Check file size
            fileInfo = dir(filename);
            fprintf('  File size: %.2f KB\n', fileInfo.bytes / 1024);
        else
            fprintf('✗ Failed to export as %s\n', filename);
        end

    catch ME
        fprintf('✗ Error exporting as %s: %s\n', format, ME.message);
    end
end

% Clean up demo files
fprintf('\nCleaning up demo files...\n');
for i = 1:length(formats)
    filename = sprintf('demo_output.%s', formats{i});
    if exist(filename, 'file')
        delete(filename);
        fprintf('✓ Deleted %s\n', filename);
    end
end
end
