import { useState, useCallback, useMemo, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Waveform types
type WaveformType =
  | "sinusoid"
  | "cosine"
  | "square"
  | "triangle"
  | "sawtooth"
  | "pulse"
  | "step"
  | "exponential"
  | "linear"
  | "polynomial"
  | "chirp"
  | "constant";

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
  chirpMethod: "linear" | "exponential";
  constantValue: number;
  polyCoeffs: number[];
}

interface SignalData {
  time: number[];
  values: number[];
}

// Signal layer for stacking
interface SignalLayer {
  id: string;
  waveformType: WaveformType;
  params: WaveformParams;
  operation: "add" | "subtract";
  enabled: boolean;
  color: string;
}

const LAYER_COLORS = [
  "#3b82f6",
  "#22c55e",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
];

// Signal generation functions
function generateSinusoid(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, phase, offset } = params;
  return t.map(
    (ti) => amplitude * Math.sin(2 * Math.PI * frequency * ti + phase) + offset,
  );
}

function generateCosine(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, phase, offset } = params;
  return t.map(
    (ti) => amplitude * Math.cos(2 * Math.PI * frequency * ti + phase) + offset,
  );
}

function generateSquare(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, dutyCycle, offset } = params;
  return t.map((ti) => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    return (phase < dutyCycle ? amplitude : -amplitude) + offset;
  });
}

function generateTriangle(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, offset } = params;
  return t.map((ti) => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    const value =
      phase < 0.5
        ? 4 * amplitude * phase - amplitude
        : -4 * amplitude * phase + 3 * amplitude;
    return value + offset;
  });
}

function generateSawtooth(t: number[], params: WaveformParams): number[] {
  const { amplitude, frequency, offset } = params;
  return t.map((ti) => {
    const period = 1 / frequency;
    const phase = (ti % period) / period;
    return 2 * amplitude * phase - amplitude + offset;
  });
}

function generatePulse(t: number[], params: WaveformParams): number[] {
  const { amplitude, pulseStart, pulseDuration, offset } = params;
  return t.map((ti) =>
    ti >= pulseStart && ti < pulseStart + pulseDuration
      ? amplitude + offset
      : offset,
  );
}

function generateStep(t: number[], params: WaveformParams): number[] {
  const { amplitude, stepTime, offset } = params;
  return t.map((ti) => (ti >= stepTime ? amplitude : offset));
}

function generateExponential(t: number[], params: WaveformParams): number[] {
  const { amplitude, decayRate, offset } = params;
  return t.map((ti) => amplitude * Math.exp(-decayRate * ti) + offset);
}

function generateLinear(t: number[], params: WaveformParams): number[] {
  const { slope, intercept } = params;
  return t.map((ti) => slope * ti + intercept);
}

function generatePolynomial(t: number[], params: WaveformParams): number[] {
  const { polyCoeffs } = params;
  return t.map((ti) => {
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
  return t.map((ti) => {
    let freq: number;
    if (chirpMethod === "linear") {
      freq = chirpF0 + ((chirpF1 - chirpF0) * ti) / duration;
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

// Hanning window to reduce spectral leakage
function hanningWindow(n: number): number[] {
  const window: number[] = [];
  for (let i = 0; i < n; i++) {
    window.push(0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1))));
  }
  return window;
}

// FFT implementation with windowing for accurate frequency visualization
function computeFFT(
  values: number[],
  sampleRate: number,
): { freq: number[]; magnitude: number[] } {
  const n = values.length;
  const freq: number[] = [];
  const magnitude: number[] = [];

  // Apply Hanning window to reduce spectral leakage
  const window = hanningWindow(n);
  const windowedValues = values.map((v, i) => v * window[i]);

  // Compute window correction factor (for amplitude accuracy)
  const windowSum = window.reduce((a, b) => a + b, 0);
  const windowCorrection = n / windowSum;

  for (let k = 0; k < n / 2; k++) {
    let real = 0;
    let imag = 0;
    for (let j = 0; j < n; j++) {
      const angle = (2 * Math.PI * k * j) / n;
      real += windowedValues[j] * Math.cos(angle);
      imag -= windowedValues[j] * Math.sin(angle);
    }
    freq.push((k * sampleRate) / n);
    // Apply correct scaling: 2/n for one-sided spectrum, but DC (k=0) uses 1/n
    const scale = k === 0 ? 1 / n : 2 / n;
    magnitude.push(
      Math.sqrt(real * real + imag * imag) * scale * windowCorrection,
    );
  }

  return { freq, magnitude };
}

const WAVEFORM_OPTIONS: { value: WaveformType; label: string }[] = [
  { value: "sinusoid", label: "Sinusoid" },
  { value: "cosine", label: "Cosine" },
  { value: "square", label: "Square Wave" },
  { value: "triangle", label: "Triangle Wave" },
  { value: "sawtooth", label: "Sawtooth" },
  { value: "pulse", label: "Pulse" },
  { value: "step", label: "Step" },
  { value: "exponential", label: "Exponential" },
  { value: "linear", label: "Linear" },
  { value: "polynomial", label: "Polynomial" },
  { value: "chirp", label: "Chirp (Sweep)" },
  { value: "constant", label: "Constant" },
];

// Helper to generate signal values for a given waveform type and params
function generateSignalValues(
  time: number[],
  waveformType: WaveformType,
  params: WaveformParams,
): number[] {
  switch (waveformType) {
    case "sinusoid":
      return generateSinusoid(time, params);
    case "cosine":
      return generateCosine(time, params);
    case "square":
      return generateSquare(time, params);
    case "triangle":
      return generateTriangle(time, params);
    case "sawtooth":
      return generateSawtooth(time, params);
    case "pulse":
      return generatePulse(time, params);
    case "step":
      return generateStep(time, params);
    case "exponential":
      return generateExponential(time, params);
    case "linear":
      return generateLinear(time, params);
    case "polynomial":
      return generatePolynomial(time, params);
    case "chirp":
      return generateChirp(time, params);
    case "constant":
      return generateConstant(time, params);
    default:
      return time.map(() => 0);
  }
}

const defaultParams: WaveformParams = {
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
  chirpMethod: "linear",
  constantValue: 1,
  polyCoeffs: [0, 1, -0.5],
};

export function FunctionGenerator() {
  const [activeTab, setActiveTab] = useState<"time" | "frequency">("time");
  const [duration, setDuration] = useState(1);
  const [sampleRate, setSampleRate] = useState(1000);
  const [showLayers, setShowLayers] = useState(true);

  // Signal layers for stacking
  const [layers, setLayers] = useState<SignalLayer[]>([
    {
      id: "1",
      waveformType: "sinusoid",
      params: { ...defaultParams },
      operation: "add",
      enabled: true,
      color: LAYER_COLORS[0],
    },
  ]);

  const [selectedLayerId, setSelectedLayerId] = useState("1");

  // Get the currently selected layer
  const selectedLayer =
    layers.find((l) => l.id === selectedLayerId) || layers[0];
  const waveformType = selectedLayer?.waveformType || "sinusoid";
  const params = selectedLayer?.params || defaultParams;

  const [polyCoeffsText, setPolyCoeffsText] = useState("0, 1, -0.5");

  // Update poly coeffs from text for selected layer
  useEffect(() => {
    try {
      const coeffs = polyCoeffsText
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n));
      if (coeffs.length > 0) {
        updateLayerParams("polyCoeffs", coeffs);
      }
    } catch {
      // Invalid input, ignore
    }
  }, [polyCoeffsText]);

  // Layer management functions
  const addLayer = useCallback(() => {
    const newId = String(Date.now());
    const colorIndex = layers.length % LAYER_COLORS.length;
    setLayers((prev) => [
      ...prev,
      {
        id: newId,
        waveformType: "sinusoid",
        params: { ...defaultParams, frequency: 10 }, // Different frequency for variety
        operation: "add",
        enabled: true,
        color: LAYER_COLORS[colorIndex],
      },
    ]);
    setSelectedLayerId(newId);
  }, [layers.length]);

  const removeLayer = useCallback(
    (id: string) => {
      if (layers.length <= 1) return; // Keep at least one layer
      setLayers((prev) => prev.filter((l) => l.id !== id));
      if (selectedLayerId === id) {
        setSelectedLayerId(
          layers[0].id === id ? layers[1]?.id || layers[0].id : layers[0].id,
        );
      }
    },
    [layers, selectedLayerId],
  );

  const updateLayerWaveform = useCallback(
    (type: WaveformType) => {
      setLayers((prev) =>
        prev.map((l) =>
          l.id === selectedLayerId ? { ...l, waveformType: type } : l,
        ),
      );
    },
    [selectedLayerId],
  );

  const updateLayerParams = useCallback(
    <K extends keyof WaveformParams>(key: K, value: WaveformParams[K]) => {
      setLayers((prev) =>
        prev.map((l) =>
          l.id === selectedLayerId
            ? { ...l, params: { ...l.params, [key]: value } }
            : l,
        ),
      );
    },
    [selectedLayerId],
  );

  const updateLayerOperation = useCallback(
    (id: string, operation: "add" | "subtract") => {
      setLayers((prev) =>
        prev.map((l) => (l.id === id ? { ...l, operation } : l)),
      );
    },
    [],
  );

  const toggleLayerEnabled = useCallback((id: string) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === id ? { ...l, enabled: !l.enabled } : l)),
    );
  }, []);

  // Generate individual layer signals
  const layerSignals = useMemo(() => {
    const n = Math.floor(duration * sampleRate);
    const time = Array.from({ length: n }, (_, i) => i / sampleRate);

    return layers.map((layer) => ({
      layer,
      values: layer.enabled
        ? generateSignalValues(time, layer.waveformType, layer.params)
        : time.map(() => 0),
    }));
  }, [layers, duration, sampleRate]);

  // Generate combined signal
  const signal = useMemo((): SignalData => {
    const n = Math.floor(duration * sampleRate);
    const time = Array.from({ length: n }, (_, i) => i / sampleRate);

    // Combine all enabled layers
    const values = time.map((_, i) => {
      let sum = 0;
      for (const { layer, values: layerVals } of layerSignals) {
        if (layer.enabled) {
          if (layer.operation === "add") {
            sum += layerVals[i];
          } else {
            sum -= layerVals[i];
          }
        }
      }
      return sum;
    });

    return { time, values };
  }, [duration, sampleRate, layerSignals]);

  // Compute FFT
  const fftData = useMemo(() => {
    return computeFFT(signal.values, sampleRate);
  }, [signal, sampleRate]);

  // Chart data with layer support
  const timeChartData = useMemo(() => {
    const maxPoints = 2000;
    const step = Math.max(1, Math.floor(signal.time.length / maxPoints));

    return signal.time
      .filter((_, i) => i % step === 0)
      .map((t, idx) => {
        const i = idx * step;
        const dataPoint: Record<string, number> = {
          time: t,
          combined: signal.values[i],
        };
        // Add individual layer values
        if (showLayers && layers.length > 1) {
          layerSignals.forEach(({ layer, values }) => {
            dataPoint[`layer_${layer.id}`] = values[i];
          });
        }
        return dataPoint;
      });
  }, [signal, layerSignals, layers, showLayers]);

  const freqChartData = useMemo(() => {
    const nyquist = sampleRate / 2;
    // Use max frequency from all enabled layers
    const maxLayerFreq = Math.max(
      ...layers
        .filter((l) => l.enabled)
        .map((l) => l.params.frequency || l.params.chirpF1 || 10),
    );
    const maxFreq = Math.min(nyquist, Math.max(maxLayerFreq * 8, 50));

    return fftData.freq
      .map((f, i) => ({ freq: f, magnitude: fftData.magnitude[i] }))
      .filter((d) => d.freq <= maxFreq && d.freq >= 0);
  }, [fftData, sampleRate, layers]);

  // Signal statistics
  const stats = useMemo(() => {
    const vals = signal.values;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const rms = Math.sqrt(vals.reduce((a, b) => a + b * b, 0) / vals.length);
    return {
      min,
      max,
      mean,
      rms,
      samples: vals.length,
      layers: layers.filter((l) => l.enabled).length,
    };
  }, [signal, layers]);

  // Alias for backward compatibility in renderParams
  const updateParam = updateLayerParams;

  // Parameter inputs based on waveform type
  const renderParams = () => {
    switch (waveformType) {
      case "sinusoid":
      case "cosine":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Frequency (Hz)"
              value={params.frequency}
              onChange={(v) => updateParam("frequency", v)}
              min={0.01}
            />
            <ParamInput
              label="Phase (rad)"
              value={params.phase}
              onChange={(v) => updateParam("phase", v)}
              step={0.1}
            />
            <ParamInput
              label="DC Offset"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "square":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Frequency (Hz)"
              value={params.frequency}
              onChange={(v) => updateParam("frequency", v)}
              min={0.01}
            />
            <ParamInput
              label="Duty Cycle"
              value={params.dutyCycle}
              onChange={(v) => updateParam("dutyCycle", v)}
              min={0.01}
              max={0.99}
              step={0.01}
            />
            <ParamInput
              label="DC Offset"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "triangle":
      case "sawtooth":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Frequency (Hz)"
              value={params.frequency}
              onChange={(v) => updateParam("frequency", v)}
              min={0.01}
            />
            <ParamInput
              label="DC Offset"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "pulse":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Start Time (s)"
              value={params.pulseStart}
              onChange={(v) => updateParam("pulseStart", v)}
              min={0}
            />
            <ParamInput
              label="Duration (s)"
              value={params.pulseDuration}
              onChange={(v) => updateParam("pulseDuration", v)}
              min={0.001}
            />
            <ParamInput
              label="Baseline"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "step":
        return (
          <>
            <ParamInput
              label="Step Value"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Step Time (s)"
              value={params.stepTime}
              onChange={(v) => updateParam("stepTime", v)}
              min={0}
            />
            <ParamInput
              label="Initial Value"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "exponential":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Decay Rate"
              value={params.decayRate}
              onChange={(v) => updateParam("decayRate", v)}
            />
            <ParamInput
              label="DC Offset"
              value={params.offset}
              onChange={(v) => updateParam("offset", v)}
            />
          </>
        );
      case "linear":
        return (
          <>
            <ParamInput
              label="Slope"
              value={params.slope}
              onChange={(v) => updateParam("slope", v)}
            />
            <ParamInput
              label="Intercept"
              value={params.intercept}
              onChange={(v) => updateParam("intercept", v)}
            />
          </>
        );
      case "polynomial":
        return (
          <div>
            <label className="block text-sm text-slate-400 mb-1">
              Coefficients (c₀, c₁, c₂, ...)
            </label>
            <input
              type="text"
              value={polyCoeffsText}
              onChange={(e) => setPolyCoeffsText(e.target.value)}
              placeholder="e.g., 1, 2, 0.5"
              className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-slate-500 mt-1">
              y = c₀ + c₁t + c₂t² + ...
            </p>
          </div>
        );
      case "chirp":
        return (
          <>
            <ParamInput
              label="Amplitude"
              value={params.amplitude}
              onChange={(v) => updateParam("amplitude", v)}
            />
            <ParamInput
              label="Start Freq (Hz)"
              value={params.chirpF0}
              onChange={(v) => updateParam("chirpF0", v)}
              min={0.01}
            />
            <ParamInput
              label="End Freq (Hz)"
              value={params.chirpF1}
              onChange={(v) => updateParam("chirpF1", v)}
              min={0.01}
            />
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Sweep Method
              </label>
              <select
                value={params.chirpMethod}
                onChange={(e) =>
                  updateParam(
                    "chirpMethod",
                    e.target.value as "linear" | "exponential",
                  )
                }
                className="w-full bg-slate-700 text-white rounded px-3 py-2"
              >
                <option value="linear">Linear</option>
                <option value="exponential">Exponential</option>
              </select>
            </div>
          </>
        );
      case "constant":
        return (
          <ParamInput
            label="Value"
            value={params.constantValue}
            onChange={(v) => updateParam("constantValue", v)}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Controls Panel */}
      <div className="space-y-4">
        {/* Signal Layers */}
        <div className="bg-slate-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-lg font-semibold text-white">Signal Layers</h3>
            <button
              onClick={addLayer}
              className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-500 transition-colors"
            >
              + Add Layer
            </button>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {layers.map((layer, idx) => (
              <div
                key={layer.id}
                onClick={() => setSelectedLayerId(layer.id)}
                className={`p-2 rounded cursor-pointer border-2 transition-colors ${
                  selectedLayerId === layer.id
                    ? "border-blue-500 bg-slate-700"
                    : "border-transparent bg-slate-700/50 hover:bg-slate-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: layer.color }}
                    />
                    <span className="text-white text-sm font-medium">
                      {idx === 0 ? "" : layer.operation === "add" ? "+" : "−"}{" "}
                      {
                        WAVEFORM_OPTIONS.find(
                          (o) => o.value === layer.waveformType,
                        )?.label
                      }
                    </span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleLayerEnabled(layer.id);
                      }}
                      className={`px-2 py-0.5 rounded text-xs ${
                        layer.enabled
                          ? "bg-green-600 text-white"
                          : "bg-slate-600 text-slate-400"
                      }`}
                    >
                      {layer.enabled ? "ON" : "OFF"}
                    </button>
                    {layers.length > 1 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeLayer(layer.id);
                        }}
                        className="px-2 py-0.5 bg-red-600 text-white rounded text-xs hover:bg-red-500"
                      >
                        ×
                      </button>
                    )}
                  </div>
                </div>
                {idx > 0 && selectedLayerId === layer.id && (
                  <div className="mt-2 flex space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        updateLayerOperation(layer.id, "add");
                      }}
                      className={`flex-1 px-2 py-1 rounded text-xs ${
                        layer.operation === "add"
                          ? "bg-blue-600 text-white"
                          : "bg-slate-600 text-slate-300"
                      }`}
                    >
                      Add (+)
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        updateLayerOperation(layer.id, "subtract");
                      }}
                      className={`flex-1 px-2 py-1 rounded text-xs ${
                        layer.operation === "subtract"
                          ? "bg-blue-600 text-white"
                          : "bg-slate-600 text-slate-300"
                      }`}
                    >
                      Subtract (−)
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
          {layers.length > 1 && (
            <label className="flex items-center space-x-2 mt-3 text-sm text-slate-400">
              <input
                type="checkbox"
                checked={showLayers}
                onChange={(e) => setShowLayers(e.target.checked)}
                className="rounded bg-slate-700 border-slate-600"
              />
              <span>Show individual layers on chart</span>
            </label>
          )}
        </div>

        {/* Waveform Selection for Selected Layer */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">
            Waveform Type
            <span className="text-sm font-normal text-slate-400 ml-2">
              (Layer {layers.findIndex((l) => l.id === selectedLayerId) + 1})
            </span>
          </h3>
          <select
            value={waveformType}
            onChange={(e) =>
              updateLayerWaveform(e.target.value as WaveformType)
            }
            className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
          >
            {WAVEFORM_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Time Parameters */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">
            Time Parameters
          </h3>
          <div className="space-y-3">
            <ParamInput
              label="Duration (s)"
              value={duration}
              onChange={setDuration}
              min={0.01}
              max={100}
            />
            <ParamInput
              label="Sample Rate (Hz)"
              value={sampleRate}
              onChange={setSampleRate}
              min={10}
              max={100000}
              step={10}
            />
          </div>
        </div>

        {/* Waveform Parameters */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">
            Waveform Parameters
            <span className="text-sm font-normal text-slate-400 ml-2">
              (Layer {layers.findIndex((l) => l.id === selectedLayerId) + 1})
            </span>
          </h3>
          <div className="space-y-3">{renderParams()}</div>
        </div>

        {/* Signal Info */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">
            Combined Signal Info
          </h3>
          <div className="text-sm space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">Active Layers:</span>
              <span className="text-white">{stats.layers}</span>
            </div>
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
            onClick={() => setActiveTab("time")}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              activeTab === "time"
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            Time Domain
          </button>
          <button
            onClick={() => setActiveTab("frequency")}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              activeTab === "frequency"
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            Frequency Domain
          </button>
        </div>

        {/* Time Domain Chart */}
        {activeTab === "time" && (
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              {layers.length === 1
                ? `${WAVEFORM_OPTIONS.find((o) => o.value === waveformType)?.label} - Time Domain`
                : `Combined Signal (${layers.filter((l) => l.enabled).length} layers) - Time Domain`}
            </h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="time"
                    stroke="#94a3b8"
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{
                      value: "Time (s)",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#94a3b8",
                    }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    label={{
                      value: "Amplitude",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#94a3b8",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "none",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "#e2e8f0" }}
                    formatter={(value: number, name: string) => {
                      const displayName =
                        name === "combined"
                          ? "Combined"
                          : `Layer ${layers.findIndex((l) => `layer_${l.id}` === name) + 1}`;
                      return [value.toFixed(4), displayName];
                    }}
                    labelFormatter={(label: number) =>
                      `t = ${label.toFixed(4)} s`
                    }
                  />
                  {/* Individual layer lines (dashed, thinner) */}
                  {showLayers &&
                    layers.length > 1 &&
                    layers.map((layer) => (
                      <Line
                        key={layer.id}
                        type="monotone"
                        dataKey={`layer_${layer.id}`}
                        stroke={layer.color}
                        strokeWidth={1}
                        strokeDasharray="4 2"
                        dot={false}
                        isAnimationActive={false}
                        opacity={layer.enabled ? 0.6 : 0.2}
                      />
                    ))}
                  {/* Combined signal line (solid, thicker) */}
                  <Line
                    type="monotone"
                    dataKey="combined"
                    stroke="#ffffff"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {/* Layer Legend */}
            {showLayers && layers.length > 1 && (
              <div className="mt-3 flex flex-wrap gap-3 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-6 h-0.5 bg-white" />
                  <span className="text-slate-300">Combined</span>
                </div>
                {layers.map((layer, idx) => (
                  <div key={layer.id} className="flex items-center space-x-2">
                    <div
                      className="w-6 h-0.5"
                      style={{
                        backgroundColor: layer.color,
                        opacity: layer.enabled ? 0.8 : 0.3,
                      }}
                    />
                    <span
                      className={
                        layer.enabled ? "text-slate-300" : "text-slate-500"
                      }
                    >
                      Layer {idx + 1} (
                      {layer.operation === "add" || idx === 0 ? "+" : "−"})
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Frequency Domain Chart */}
        {activeTab === "frequency" && (
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              Frequency Spectrum
            </h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={freqChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="freq"
                    stroke="#94a3b8"
                    tickFormatter={(v) => v.toFixed(1)}
                    label={{
                      value: "Frequency (Hz)",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#94a3b8",
                    }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    label={{
                      value: "Magnitude",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#94a3b8",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "none",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "#e2e8f0" }}
                    formatter={(value: number) => [
                      value.toFixed(4),
                      "Magnitude",
                    ]}
                    labelFormatter={(label: number) =>
                      `f = ${label.toFixed(2)} Hz`
                    }
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
          <h3 className="text-lg font-semibold text-white mb-3">
            Quick Presets
          </h3>
          <p className="text-xs text-slate-500 mb-2">Single signals:</p>
          <div className="flex flex-wrap gap-2 mb-3">
            <PresetButton
              label="1 Hz Sine"
              onClick={() => {
                updateLayerWaveform("sinusoid");
                updateParam("frequency", 1);
              }}
            />
            <PresetButton
              label="10 Hz Sine"
              onClick={() => {
                updateLayerWaveform("sinusoid");
                updateParam("frequency", 10);
              }}
            />
            <PresetButton
              label="50 Hz Square"
              onClick={() => {
                updateLayerWaveform("square");
                updateParam("frequency", 50);
              }}
            />
            <PresetButton
              label="Chirp 1-50 Hz"
              onClick={() => {
                updateLayerWaveform("chirp");
                updateParam("chirpF0", 1);
                updateParam("chirpF1", 50);
              }}
            />
            <PresetButton
              label="Decay τ=0.5"
              onClick={() => {
                updateLayerWaveform("exponential");
                updateParam("decayRate", 2);
              }}
            />
            <PresetButton
              label="Parabola"
              onClick={() => {
                updateLayerWaveform("polynomial");
                setPolyCoeffsText("0, 0, 1");
              }}
            />
          </div>
          <p className="text-xs text-slate-500 mb-2">Stacked signals:</p>
          <div className="flex flex-wrap gap-2">
            <PresetButton
              label="Sine + Harmonic"
              onClick={() => {
                setLayers([
                  {
                    id: "1",
                    waveformType: "sinusoid",
                    params: { ...defaultParams, amplitude: 1, frequency: 5 },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[0],
                  },
                  {
                    id: "2",
                    waveformType: "sinusoid",
                    params: { ...defaultParams, amplitude: 0.5, frequency: 15 },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[1],
                  },
                ]);
                setSelectedLayerId("1");
              }}
            />
            <PresetButton
              label="AM Modulation"
              onClick={() => {
                setLayers([
                  {
                    id: "1",
                    waveformType: "sinusoid",
                    params: { ...defaultParams, amplitude: 1, frequency: 20 },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[0],
                  },
                  {
                    id: "2",
                    waveformType: "sinusoid",
                    params: {
                      ...defaultParams,
                      amplitude: 0.5,
                      frequency: 2,
                      offset: 0.5,
                    },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[1],
                  },
                ]);
                setSelectedLayerId("1");
              }}
            />
            <PresetButton
              label="Square − Sine"
              onClick={() => {
                setLayers([
                  {
                    id: "1",
                    waveformType: "square",
                    params: { ...defaultParams, amplitude: 1, frequency: 5 },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[0],
                  },
                  {
                    id: "2",
                    waveformType: "sinusoid",
                    params: { ...defaultParams, amplitude: 0.8, frequency: 5 },
                    operation: "subtract",
                    enabled: true,
                    color: LAYER_COLORS[1],
                  },
                ]);
                setSelectedLayerId("1");
              }}
            />
            <PresetButton
              label="3-Tone Chord"
              onClick={() => {
                setLayers([
                  {
                    id: "1",
                    waveformType: "sinusoid",
                    params: { ...defaultParams, amplitude: 1, frequency: 5 },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[0],
                  },
                  {
                    id: "2",
                    waveformType: "sinusoid",
                    params: {
                      ...defaultParams,
                      amplitude: 0.8,
                      frequency: 6.25,
                    },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[1],
                  },
                  {
                    id: "3",
                    waveformType: "sinusoid",
                    params: {
                      ...defaultParams,
                      amplitude: 0.6,
                      frequency: 7.5,
                    },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[2],
                  },
                ]);
                setSelectedLayerId("1");
              }}
            />
            <PresetButton
              label="Reset (1 Layer)"
              onClick={() => {
                setLayers([
                  {
                    id: "1",
                    waveformType: "sinusoid",
                    params: { ...defaultParams },
                    operation: "add",
                    enabled: true,
                    color: LAYER_COLORS[0],
                  },
                ]);
                setSelectedLayerId("1");
              }}
            />
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

function ParamInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.1,
}: ParamInputProps) {
  return (
    <div>
      <label className="block text-sm text-slate-400 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
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
