import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// Waveform types
type WaveformType =
  | 'sinusoid'
  | 'cosine'
  | 'square'
  | 'triangle'
  | 'sawtooth'
  | 'pulse'
  | 'step'
  | 'exponential'
  | 'linear'
  | 'polynomial'
  | 'chirp'
  | 'constant';

interface WaveformParams {
  amplitude: number;
  frequency: number;
  phase: number;
  offset: number;
  dutyCycle: number;
  decayRate: number;
  slope: number;
  intercept: number;
  stepTime: number;
  pulseStart: number;
  pulseDuration: number;
  chirpF0: number;
  chirpF1: number;
  chirpMethod: 'linear' | 'exponential';
  constantValue: number;
  polyCoeffs: number[];
}

interface SignalData {
  time: number[];
  values: number[];
}

// Signal generation functions
function generateSinusoid(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, phase, offset } = params;
  return t.map(ti => amplitude * Math.sin(2 * Math.PI * frequency * ti + phase) + offset);
}

function generateCosine(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, phase, offset } = params;
  return t.map(ti => amplitude * Math.cos(2 * Math.PI * frequency * ti + phase) + offset);
}

function generateSquare(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, dutyCycle, offset } = params;
  return t.map(ti => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    return (phase < dutyCycle ? amplitude : -amplitude) + offset;
  });
}

function generateTriangle(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, offset } = params;
  return t.map(ti => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    const value = phase < 0.5
      ? 4 * amplitude * phase - amplitude
      : -4 * amplitude * phase + 3 * amplitude;
    return value + offset;
  });
}

function generateSawtooth(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, offset } = params;
  return t.map(ti => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    return 2 * amplitude * phase - amplitude + offset;
  });
}

function generatePulse(t: number[], params: WaveformParams): number[] {
  const { amplitude, pulseStart, pulseDuration, offset } = params;
  return t.map(ti =>
    ti >= pulseStart && ti < pulseStart + pulseDuration ? amplitude + offset : offset
  );
}

function generateStep(t: number[], params: WaveformParams): number[] {
  const { amplitude, stepTime, offset } = params;
  return t.map(ti => (ti >= stepTime ? amplitude : offset));
}

function generateExponential(t: number[], params: WaveformParams): number[] {
  const { amplitude, decayRate, offset } = params;
  return t.map(ti => amplitude * Math.exp(-decayRate * ti) + offset);
}

function generateLinear(t: number[], params: WaveformParams): number[] {
  const { slope, intercept } = params;
  return t.map(ti => slope * ti + intercept);
}

function generatePolynomial(t: number[], params: WaveformParams): number[] {
  const { polyCoeffs } = params;
  return t.map(ti => {
    let value = 0;
    for (let i = 0; i < polyCoeffs.length; i++) {
      value += polyCoeffs[i] * Math.pow(ti, i);
    }
    return value;
  });
}

function generateChirp(t: number[], params: WaveformParams): number[] {
  const { amplitude, chirpF0, chirpF1, chirpMethod } = params;
  const duration = t[t.length - 1];
  return t.map(ti => {
    let freq: number;
    if (chirpMethod === 'linear') {
      freq = chirpF0 + (chirpF1 - chirpF0) * ti / duration;
    } else {
      freq = chirpF0 * Math.pow(chirpF1 / chirpF0, ti / duration);
    }
    return amplitude * Math.sin(2 * Math.PI * freq * ti);
  });
}

function generateConstant(t: number[], params: WaveformParams): number[] {
  const { constantValue } = params;
  return t.map(() => constantValue);
}

// FFT implementation (simple DFT for visualization)
function computeFFT(values: number[], sampleRate: number): { freq: number[]; magnitude: number[] } {
  const n = values.length;
  const freq: number[] = [];
  const magnitude: number[] = [];

  for (let k = 0; k < n / 2; k++) {
    let real = 0;
    let imag = 0;
    for (let j = 0; j < n; j++) {
      const angle = (2 * Math.PI * k * j) / n;
      real += values[j] * Math.cos(angle);
      imag -= values[j] * Math.sin(angle);
    }
    freq.push((k * sampleRate) / n);
    magnitude.push((2 * Math.sqrt(real * real + imag * imag)) / n);
  }

  return { freq, magnitude };
}

const WAVEFORM_OPTIONS: { value: WaveformType; label: string }[] = [
  { value: 'sinusoid', label: 'Sinusoid' },
  { value: 'cosine', label: 'Cosine' },
  { value: 'square', label: 'Square Wave' },
  { value: 'triangle', label: 'Triangle Wave' },
  { value: 'sawtooth', label: 'Sawtooth' },
  { value: 'pulse', label: 'Pulse' },
  { value: 'step', label: 'Step' },
  { value: 'exponential', label: 'Exponential' },
  { value: 'linear', label: 'Linear' },
  { value: 'polynomial', label: 'Polynomial' },
  { value: 'chirp', label: 'Chirp (Sweep)' },
  { value: 'constant', label: 'Constant' },
];

export function FunctionGenerator() {
  const [activeTab, setActiveTab] = useState<'time' | 'frequency'>('time');
  const [waveformType, setWaveformType] = useState<WaveformType>('sinusoid');
  const [duration, setDuration] = useState(1);
  const [sampleRate, setSampleRate] = useState(1000);

  const [params, setParams] = useState<WaveformParams>({
    amplitude: 1,
    frequency: 5,
    phase: 0,
    offset: 0,
    dutyCycle: 0.5,
    decayRate: 2,
    slope: 1,
    intercept: 0,
    stepTime: 0.5,
    pulseStart: 0.2,
    pulseDuration: 0.3,
    chirpF0: 1,
    chirpF1: 20,
    chirpMethod: 'linear',
    constantValue: 1,
    polyCoeffs: [0, 1, -0.5],
  });

  const [polyCoeffsText, setPolyCoeffsText] = useState('0, 1, -0.5');

  // Update poly coeffs from text
  useEffect(() => {
    try {
      const coeffs = polyCoeffsText.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
      if (coeffs.length > 0) {
        setParams(p => ({ ...p, polyCoeffs: coeffs }));
      }
    } catch {
      // Invalid input, ignore
    }
  }, [polyCoeffsText]);

  // Generate signal
  const signal = useMemo((): SignalData => {
    const n = Math.floor(duration * sampleRate);
    const time = Array.from({ length: n }, (_, i) => (i / sampleRate));

    let values: number[];
    switch (waveformType) {
      case 'sinusoid':
        values = generateSinusoid(time, params);
        break;
      case 'cosine':
        values = generateCosine(time, params);
        break;
      case 'square':
        values = generateSquare(time, params);
        break;
      case 'triangle':
        values = generateTriangle(time, params);
        break;
      case 'sawtooth':
        values = generateSawtooth(time, params);
        break;
      case 'pulse':
        values = generatePulse(time, params);
        break;
      case 'step':
        values = generateStep(time, params);
        break;
      case 'exponential':
        values = generateExponential(time, params);
        break;
      case 'linear':
        values = generateLinear(time, params);
        break;
      case 'polynomial':
        values = generatePolynomial(time, params);
        break;
      case 'chirp':
        values = generateChirp(time, params);
        break;
      case 'constant':
        values = generateConstant(time, params);
        break;
      default:
        values = time.map(() => 0);
    }

    return { time, values };
  }, [waveformType, duration, sampleRate, params]);

  // Compute FFT
  const fftData = useMemo(() => {
    return computeFFT(signal.values, sampleRate);
  }, [signal, sampleRate]);

  // Chart data
  const timeChartData = useMemo(() => {
    // Downsample for performance if needed
    const maxPoints = 2000;
    const step = Math.max(1, Math.floor(signal.time.length / maxPoints));
    return signal.time
      .filter((_, i) => i % step === 0)
      .map((t, i) => ({
        time: t,
        value: signal.values[i * step],
      }));
  }, [signal]);

  const freqChartData = useMemo(() => {
    const maxFreq = Math.min(sampleRate / 2, params.frequency * 10 || 50);
    return fftData.freq
      .map((f, i) => ({ freq: f, magnitude: fftData.magnitude[i] }))
      .filter(d => d.freq <= maxFreq);
  }, [fftData, sampleRate, params.frequency]);

  // Signal statistics
  const stats = useMemo(() => {
    const vals = signal.values;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const rms = Math.sqrt(vals.reduce((a, b) => a + b * b, 0) / vals.length);
    return { min, max, mean, rms, samples: vals.length };
  }, [signal]);

  const updateParam = useCallback(<K extends keyof WaveformParams>(key: K, value: WaveformParams[K]) => {
    setParams(p => ({ ...p, [key]: value }));
  }, []);

  // Parameter inputs based on waveform type
  const renderParams = () => {
    switch (waveformType) {
      case 'sinusoid':
      case 'cosine':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Frequency (Hz)" value={params.frequency} onChange={v => updateParam('frequency', v)} min={0.01} />
            <ParamInput label="Phase (rad)" value={params.phase} onChange={v => updateParam('phase', v)} step={0.1} />
            <ParamInput label="DC Offset" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'square':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Frequency (Hz)" value={params.frequency} onChange={v => updateParam('frequency', v)} min={0.01} />
            <ParamInput label="Duty Cycle" value={params.dutyCycle} onChange={v => updateParam('dutyCycle', v)} min={0.01} max={0.99} step={0.01} />
            <ParamInput label="DC Offset" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'triangle':
      case 'sawtooth':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Frequency (Hz)" value={params.frequency} onChange={v => updateParam('frequency', v)} min={0.01} />
            <ParamInput label="DC Offset" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'pulse':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Start Time (s)" value={params.pulseStart} onChange={v => updateParam('pulseStart', v)} min={0} />
            <ParamInput label="Duration (s)" value={params.pulseDuration} onChange={v => updateParam('pulseDuration', v)} min={0.001} />
            <ParamInput label="Baseline" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'step':
        return (
          <>
            <ParamInput label="Step Value" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Step Time (s)" value={params.stepTime} onChange={v => updateParam('stepTime', v)} min={0} />
            <ParamInput label="Initial Value" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'exponential':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Decay Rate" value={params.decayRate} onChange={v => updateParam('decayRate', v)} />
            <ParamInput label="DC Offset" value={params.offset} onChange={v => updateParam('offset', v)} />
          </>
        );
      case 'linear':
        return (
          <>
            <ParamInput label="Slope" value={params.slope} onChange={v => updateParam('slope', v)} />
            <ParamInput label="Intercept" value={params.intercept} onChange={v => updateParam('intercept', v)} />
          </>
        );
      case 'polynomial':
        return (
          <div>
            <label className="block text-sm text-slate-400 mb-1">Coefficients (c₀, c₁, c₂, ...)</label>
            <input
              type="text"
              value={polyCoeffsText}
              onChange={e => setPolyCoeffsText(e.target.value)}
              placeholder="e.g., 1, 2, 0.5"
              className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-slate-500 mt-1">y = c₀ + c₁t + c₂t² + ...</p>
          </div>
        );
      case 'chirp':
        return (
          <>
            <ParamInput label="Amplitude" value={params.amplitude} onChange={v => updateParam('amplitude', v)} />
            <ParamInput label="Start Freq (Hz)" value={params.chirpF0} onChange={v => updateParam('chirpF0', v)} min={0.01} />
            <ParamInput label="End Freq (Hz)" value={params.chirpF1} onChange={v => updateParam('chirpF1', v)} min={0.01} />
            <div>
              <label className="block text-sm text-slate-400 mb-1">Sweep Method</label>
              <select
                value={params.chirpMethod}
                onChange={e => updateParam('chirpMethod', e.target.value as 'linear' | 'exponential')}
                className="w-full bg-slate-700 text-white rounded px-3 py-2"
              >
                <option value="linear">Linear</option>
                <option value="exponential">Exponential</option>
              </select>
            </div>
          </>
        );
      case 'constant':
        return (
          <ParamInput label="Value" value={params.constantValue} onChange={v => updateParam('constantValue', v)} />
        );
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Controls Panel */}
      <div className="space-y-4">
        {/* Waveform Selection */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Waveform Type</h3>
          <select
            value={waveformType}
            onChange={e => setWaveformType(e.target.value as WaveformType)}
            className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
          >
            {WAVEFORM_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Time Parameters */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Time Parameters</h3>
          <div className="space-y-3">
            <ParamInput label="Duration (s)" value={duration} onChange={setDuration} min={0.01} max={100} />
            <ParamInput label="Sample Rate (Hz)" value={sampleRate} onChange={setSampleRate} min={10} max={100000} step={10} />
          </div>
        </div>

        {/* Waveform Parameters */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Waveform Parameters</h3>
          <div className="space-y-3">
            {renderParams()}
          </div>
        </div>

        {/* Signal Info */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Signal Info</h3>
          <div className="text-sm space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">Samples:</span>
              <span className="text-white">{stats.samples}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Min:</span>
              <span className="text-white">{stats.min.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Max:</span>
              <span className="text-white">{stats.max.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Mean:</span>
              <span className="text-white">{stats.mean.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">RMS:</span>
              <span className="text-white">{stats.rms.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Panel */}
      <div className="lg:col-span-2 space-y-4">
        {/* Tabs */}
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('time')}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              activeTab === 'time'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Time Domain
          </button>
          <button
            onClick={() => setActiveTab('frequency')}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              activeTab === 'frequency'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Frequency Domain
          </button>
        </div>

        {/* Time Domain Chart */}
        {activeTab === 'time' && (
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              {WAVEFORM_OPTIONS.find(o => o.value === waveformType)?.label} - Time Domain
            </h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="time"
                    stroke="#94a3b8"
                    tickFormatter={v => v.toFixed(2)}
                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                    formatter={(value: number) => [value.toFixed(4), 'Value']}
                    labelFormatter={(label: number) => `t = ${label.toFixed(4)} s`}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Frequency Domain Chart */}
        {activeTab === 'frequency' && (
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Frequency Spectrum</h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={freqChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="freq"
                    stroke="#94a3b8"
                    tickFormatter={v => v.toFixed(1)}
                    label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    label={{ value: 'Magnitude', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                    formatter={(value: number) => [value.toFixed(4), 'Magnitude']}
                    labelFormatter={(label: number) => `f = ${label.toFixed(2)} Hz`}
                  />
                  <Line
                    type="monotone"
                    dataKey="magnitude"
                    stroke="#22c55e"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Quick Presets */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Quick Presets</h3>
          <div className="flex flex-wrap gap-2">
            <PresetButton label="1 Hz Sine" onClick={() => { setWaveformType('sinusoid'); updateParam('frequency', 1); }} />
            <PresetButton label="10 Hz Sine" onClick={() => { setWaveformType('sinusoid'); updateParam('frequency', 10); }} />
            <PresetButton label="50 Hz Square" onClick={() => { setWaveformType('square'); updateParam('frequency', 50); }} />
            <PresetButton label="Chirp 1-50 Hz" onClick={() => { setWaveformType('chirp'); updateParam('chirpF0', 1); updateParam('chirpF1', 50); }} />
            <PresetButton label="Decay τ=0.5" onClick={() => { setWaveformType('exponential'); updateParam('decayRate', 2); }} />
            <PresetButton label="Parabola" onClick={() => { setWaveformType('polynomial'); setPolyCoeffsText('0, 0, 1'); }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper components
interface ParamInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

function ParamInput({ label, value, onChange, min, max, step = 0.1 }: ParamInputProps) {
  return (
    <div>
      <label className="block text-sm text-slate-400 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
}

interface PresetButtonProps {
  label: string;
  onClick: () => void;
}

function PresetButton({ label, onClick }: PresetButtonProps) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1 bg-slate-700 text-slate-300 rounded hover:bg-slate-600 text-sm transition-colors"
    >
      {label}
    </button>
  );
}
