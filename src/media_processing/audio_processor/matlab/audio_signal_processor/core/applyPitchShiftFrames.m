function shifted = applyPitchShiftFrames(audio, semitones, fs, speed)
%APPLYPITCHSHIFTFRAMES Pitch shifting using phase vocoder
%
%   shifted = applyPitchShiftFrames(audio, semitones, fs, speed)
%
%   Uses a phase vocoder for high-quality pitch shifting. Supports both
%   constant and time-varying semitone shifts. Stereo or multi-channel audio
%   is processed channel-by-channel with the same shift curve.
%
%   Args:
%       audio    : Input audio signal [NxC] or [1xN]
%       semitones: Scalar or vector of semitone shifts (length N or 1)
%       fs       : Sample rate [Hz]
%       speed    : Reserved for compatibility
%
%   Returns:
%       shifted  : Pitch-shifted audio with the same shape as audio
%
%   Raises:
%       Error if input is invalid.
%
%   Reference:
%       "The Phase Vocoder: A Tutorial", J. Laroche, M. Dolson, IEEE Sig. Proc. Mag., 1999.

if nargin < 4
    speed = 1;
end
if ~isnumeric(audio) || isempty(audio) || ndims(audio) > 2
    error('applyPitchShiftFrames:InvalidAudio', ...
        'Input audio must be a non-empty numeric vector or 2-D matrix.');
end
if ~isnumeric(semitones) || isempty(semitones)
    error('applyPitchShiftFrames:InvalidSemitones', ...
        'Semitone shifts must be numeric and non-empty.');
end

wasRowVector = isrow(audio);
if wasRowVector
    audio = audio(:);
end

numSamples = size(audio, 1);
if ~(isscalar(semitones) || (isvector(semitones) && numel(semitones) == numSamples))
    error('applyPitchShiftFrames:InvalidSemitones', ...
        'semitones must be a scalar or a vector with one value per audio sample.');
end

shifted = zeros(size(audio));
for channel = 1:size(audio, 2)
    shifted(:, channel) = applyPitchShiftMono(audio(:, channel), semitones, fs, speed);
end

if wasRowVector
    shifted = shifted.';
end
end

function shifted = applyPitchShiftMono(audio, semitones, fs, speed)
% Pitch-shift one mono channel while preserving sample count.

audio = audio(:);
if ~isvector(audio) || isempty(audio)
    error('applyPitchShiftFrames:InvalidAudio', ...
        'Input audio must be a non-empty vector.');
end

% Parameters
N_FFT = 2048; % [samples] FFT size (power of 2, typical: 1024-4096)
HOP = N_FFT/4; % [samples] Hop size (25% overlap)

% Pad audio to fit frames
L = length(audio);
nFrames = max(1, ceil((L-N_FFT)/HOP)+1);
padLen = (nFrames-1)*HOP + N_FFT - L;
audioPadded = [audio; zeros(padLen,1)];

% STFT
win = hann(N_FFT, 'periodic');
S = zeros(N_FFT, nFrames);
for k = 1:nFrames
    idx = (1:N_FFT) + (k-1)*HOP;
    S(:,k) = fft(audioPadded(idx).*win);
end

% Pitch shift ratio per frame
if isscalar(semitones)
    ratio = 2^(semitones/12);
    ratios = ratio*ones(1,nFrames);
else
    frameIdx = round(linspace(1, numel(semitones), nFrames));
    ratios = 2.^(semitones(frameIdx)/12);
end

% Phase vocoder processing
S_shift = zeros(size(S));
phi = angle(S(:,1));
lastPhase = phi;
for k = 1:nFrames
    mag = abs(S(:,k));
    phase = angle(S(:,k));
    delta = phase - lastPhase - 2*pi*HOP*(0:N_FFT-1)'/N_FFT;
    delta = delta - 2*pi*round(delta/(2*pi));
    trueFreq = 2*pi*(0:N_FFT-1)'/N_FFT + delta/HOP;
    if ratios(k) == 1
        S_shift(:,k) = S(:,k);
    else
        phi = phi + HOP*ratios(k)*trueFreq;
        S_shift(:,k) = mag .* exp(1j*phi);
    end
    lastPhase = phase;
end

% ISTFT (overlap-add)
y = zeros((nFrames-1)*HOP + N_FFT,1);
winSum = zeros((nFrames-1)*HOP + N_FFT,1);
for k = 1:nFrames
    idx = (1:N_FFT) + (k-1)*HOP;
    frame = real(ifft(S_shift(:,k))) .* win;
    y(idx) = y(idx) + frame;
    winSum(idx) = winSum(idx) + win.^2;
end
nonzero = winSum > 1e-6;
y(nonzero) = y(nonzero) ./ winSum(nonzero);

shifted = y(1:L);

if any(isnan(shifted)) || any(isinf(shifted))
    error('applyPitchShiftFrames:InvalidOutput', ...
        'Phase vocoder produced invalid output.');
end
end
