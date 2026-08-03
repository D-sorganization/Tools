import { useState, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// API base URL -- defaults to shared calc backend.
// Override via VITE_CALC_API_URL environment variable.
// See issue #608.
const CALC_API_BASE = import.meta.env.VITE_CALC_API_URL ?? 'http://localhost:8010';

const PIPE_SIZES = ['0.5', '0.75', '1', '1.25', '1.5', '2', '2.5', '3', '4', '6', '8', '10', '12', '14', '16', '18', '20', '24'];
const PIPE_SCHEDULES = ['5', '10', '20', '30', '40', '60', '80', '100', '120', '140', '160', 'STD', 'XS', 'XXS'];
const FLOW_UNITS = ['kg/h', 'kg/s', 'lb/hr', 'm³/h', 'SCFM', 'Nm³/h'];
const FRICTION_METHODS = ['colebrook', 'swamee-jain', 'churchill', 'haaland'];
const MATERIALS: Record<string, number> = {
  'Carbon Steel': 0.000046,
  'Stainless Steel': 0.000015,
  'Copper': 0.0000015,
  'PVC': 0.0000015,
  'HDPE': 0.000007,
  'Concrete': 0.0003,
};

// Pipe inner diameters (approximate, in meters) for common sizes and Schedule 40
const PIPE_DIAMETERS: Record<string, number> = {
  '0.5': 0.0158, '0.75': 0.0209, '1': 0.0266, '1.25': 0.0351, '1.5': 0.0409,
  '2': 0.0525, '2.5': 0.0627, '3': 0.0779, '4': 0.1023, '6': 0.1541,
  '8': 0.2027, '10': 0.2546, '12': 0.3048, '14': 0.3365, '16': 0.3810,
  '18': 0.4286, '20': 0.4780, '24': 0.5731,
};

interface GasComp { N2: number; O2: number; CO2: number; H2O: number; H2: number; CO: number; CH4: number; Ar: number; }
interface Results {
  totalPressureDrop: number;
  outletPressure: number;
  frictionPressureDrop: number;
  frictionFactor: number;
  flowRegime: string;
  velocity: number;
  reynoldsNumber: number;
  machNumber: number;
  density: number;
  erosionalVelocity: number;
  erosionRatio: number;
  warnings: string[];
  engine: string;
}

// ---------------------------------------------------------------------------
// Backend integration -- See issue #608
// ---------------------------------------------------------------------------

async function fetchFromBackend(
  diameter: number,
  pipeLength: number,
  roughness: number,
  massFlowKgS: number,
  temperature: number,
  pressurePa: number,
  mw: number,
): Promise<{ pressureDropPa: number; reynolds: number; frictionFactor: number; velocity: number; flowRegime: string; density: number; viscosity: number }> {
  const response = await fetch(`${CALC_API_BASE}/api/calc/pressure-drop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      pipe_diameter_m: diameter,
      pipe_length_m: pipeLength,
      roughness_m: roughness,
      flow_rate_kg_s: massFlowKgS,
      temperature_k: temperature,
      pressure_pa: pressurePa,
      molecular_weight_kg_mol: mw / 1000,
    }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `API error ${response.status}`);
  }

  const data = await response.json();
  return {
    pressureDropPa: data.pressure_drop_pa,
    reynolds: data.reynolds_number,
    frictionFactor: data.friction_factor,
    velocity: data.velocity_m_s,
    flowRegime: data.flow_regime,
    density: data.density_kg_m3,
    viscosity: data.viscosity_pa_s,
  };
}

// Physical constants and helper functions
const R = 8.314; // J/(mol·K)
const MW: Record<string, number> = { N2: 28.01, O2: 32.0, CO2: 44.01, H2O: 18.015, H2: 2.016, CO: 28.01, CH4: 16.04, Ar: 39.95 };

function calcMixtureMW(comp: GasComp): number {
  // ⚡ Bolt Optimization: Replace Object.entries().reduce() with a single-pass loop
  let sum = 0;
  const keys = Object.keys(comp) as (keyof GasComp)[];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    sum += comp[k] * MW[k];
  }
  return sum;
}

function calcDensity(P: number, T: number, mw: number): number {
  return (P * mw) / (R * T * 1000); // kg/m³
}

function calcViscosity(T: number): number {
  // Sutherland's formula for air-like gases (simplified)
  return 1.458e-6 * Math.pow(T, 1.5) / (T + 110.4);
}

function colebrookFrictionFactor(Re: number, roughness: number, diameter: number): number {
  if (Re < 2300) return 64 / Re; // Laminar
  const relRough = roughness / diameter;
  // Swamee-Jain explicit approximation
  const term1 = relRough / 3.7;
  const term2 = 5.74 / Math.pow(Re, 0.9);
  return 0.25 / Math.pow(Math.log10(term1 + term2), 2);
}


// ⚡ Bolt Optimization: Pre-calculate total and replace Object.values().reduce() to eliminate intermediate object creation and iteration overhead
const getTotalGasComp = (comp: GasComp) => {
  let sum = 0;
  const keys = Object.keys(comp) as (keyof GasComp)[];
  for (let i = 0; i < keys.length; i++) {
    sum += comp[keys[i]];
  }
  return sum;
};

export function PressureDropCalculator() {
  const [activeTab, setActiveTab] = useState<'input' | 'results' | 'chart'>('input');

  // Pipe parameters
  const [pipeSize, setPipeSize] = useState('4');
  const [pipeSchedule, setPipeSchedule] = useState('40');
  const [pipeLength, setPipeLength] = useState(100);
  const [material, setMaterial] = useState('Carbon Steel');
  const [elevation, setElevation] = useState(0);

  // Flow conditions
  const [flowRate, setFlowRate] = useState(1000);
  const [flowUnit, setFlowUnit] = useState('kg/h');
  const [pressure, setPressure] = useState(10);
  const [temperature, setTemperature] = useState(300);
  const [frictionMethod, setFrictionMethod] = useState('colebrook');

  // Gas composition
  const [gasComp, setGasComp] = useState<GasComp>({ N2: 78, O2: 21, CO2: 0, H2O: 0, H2: 0, CO: 0, CH4: 0, Ar: 1 });

  const [results, setResults] = useState<Results | null>(null);

  const updateGas = useCallback((key: keyof GasComp, val: number) => {
    setGasComp(prev => ({ ...prev, [key]: val }));
  }, []);

  const calculate = useCallback(async () => {
    // Normalize composition
    const total = getTotalGasComp(gasComp);
    if (Math.abs(total - 100) > 1) {
      alert(`Gas composition must sum to 100% (current: ${total.toFixed(1)}%)`);
      return;
    }
    // ⚡ Bolt Optimization: Replace Object.entries().map() and Object.fromEntries() with a single-pass loop
    const normComp = {} as GasComp;
    const keys = Object.keys(gasComp) as (keyof GasComp)[];
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      normComp[k] = gasComp[k] / 100;
    }

    // Get pipe diameter
    const diameter = PIPE_DIAMETERS[pipeSize] || 0.1;
    const area = Math.PI * diameter * diameter / 4;
    const roughness = MATERIALS[material];

    // Convert flow rate to kg/s
    let massFlowKgS = flowRate;
    if (flowUnit === 'kg/h') massFlowKgS = flowRate / 3600;
    else if (flowUnit === 'lb/hr') massFlowKgS = flowRate * 0.000126;
    else if (flowUnit === 'm³/h') {
      const mw = calcMixtureMW(normComp);
      const rho = calcDensity(pressure * 1e5, temperature, mw);
      massFlowKgS = flowRate * rho / 3600;
    }

    // Calculate properties (client-side, used as fallback and for derived metrics)
    const mw = calcMixtureMW(normComp);
    const P_Pa = pressure * 1e5;

    // Try the validated Python backend first; fall back to client-side.
    // See issue #608.
    let engine = 'client-fallback';
    let rho: number;
    let velocity: number;
    let Re: number;
    let f: number;
    let dP_friction: number;

    try {
      const backend = await fetchFromBackend(
        diameter, pipeLength, roughness, massFlowKgS, temperature, P_Pa, mw,
      );
      rho = backend.density;
      velocity = backend.velocity;
      Re = backend.reynolds;
      f = backend.frictionFactor;
      dP_friction = backend.pressureDropPa;
      engine = 'python-backend';
    } catch {
      // Fallback: local Darcy-Weisbach
      rho = calcDensity(P_Pa, temperature, mw);
      const mu = calcViscosity(temperature);
      velocity = massFlowKgS / (rho * area);
      Re = rho * velocity * diameter / mu;
      f = colebrookFrictionFactor(Re, roughness, diameter);
      dP_friction = f * (pipeLength / diameter) * 0.5 * rho * velocity * velocity;
    }

    // Elevation pressure drop
    const dP_elevation = rho * 9.81 * elevation;

    const dP_total = dP_friction + dP_elevation;
    const P_outlet = P_Pa - dP_total;

    // Speed of sound and Mach number
    const gamma = 1.4; // Approximate for air-like
    const speedOfSound = Math.sqrt(gamma * R * temperature * 1000 / mw);
    const mach = velocity / speedOfSound;

    // Erosional velocity (API RP 14E)
    const C = 100; // Typical value
    const erosionalVel = C / Math.sqrt(rho);

    // Warnings
    const warnings: string[] = [];
    if (Re < 2300) warnings.push('Laminar flow - consider larger pipe');
    if (mach > 0.3) warnings.push('High Mach number - compressibility effects significant');
    if (velocity > erosionalVel) warnings.push('Velocity exceeds erosional limit!');
    if (dP_total / P_Pa > 0.1) warnings.push('Pressure drop > 10% of inlet - consider compressible flow');

    setResults({
      totalPressureDrop: dP_total,
      outletPressure: P_outlet,
      frictionPressureDrop: dP_friction,
      frictionFactor: f,
      flowRegime: Re < 2300 ? 'Laminar' : Re < 4000 ? 'Transitional' : 'Turbulent',
      velocity,
      reynoldsNumber: Re,
      machNumber: mach,
      density: rho,
      erosionalVelocity: erosionalVel,
      erosionRatio: velocity / erosionalVel,
      warnings,
      engine,
    });
    setActiveTab('results');
  }, [pipeSize, pipeLength, material, elevation, flowRate, flowUnit, pressure, temperature, gasComp, frictionMethod]);

  const pressureProfile = useMemo(() => {
    if (!results) return [];
    const steps = 10;
    return Array.from({ length: steps + 1 }, (_, i) => ({
      distance: (i / steps) * pipeLength,
      pressure: (pressure - (i / steps) * results.totalPressureDrop / 1e5),
    }));
  }, [results, pipeLength, pressure]);

  const componentBreakdown = useMemo(() => {
    if (!results) return [];
    return [
      { name: 'Friction', value: results.frictionPressureDrop / 1000 },
      { name: 'Elevation', value: Math.abs(results.totalPressureDrop - results.frictionPressureDrop) / 1000 },
    ];
  }, [results]);

  return (
    <div className="space-y-6">
      <div className="flex space-x-4 border-b border-slate-700">
        {(['input', 'results', 'chart'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-medium capitalize ${activeTab === tab ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-400 hover:text-slate-300'}`}>
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'input' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Pipe Parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Pipe Parameters</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Nominal Size (in)</label>
                <select value={pipeSize} onChange={e => setPipeSize(e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                  {PIPE_SIZES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Schedule</label>
                <select value={pipeSchedule} onChange={e => setPipeSchedule(e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                  {PIPE_SCHEDULES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Length (m)</label>
                <input type="number" value={pipeLength} onChange={e => setPipeLength(+e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Material</label>
                <select value={material} onChange={e => setMaterial(e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                  {Object.keys(MATERIALS).map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Elevation Change (m)</label>
                <input type="number" value={elevation} onChange={e => setElevation(+e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
              </div>
            </div>
          </div>

          {/* Flow Conditions */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Flow Conditions</h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Flow Rate</label>
                  <input type="number" value={flowRate} onChange={e => setFlowRate(+e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Unit</label>
                  <select value={flowUnit} onChange={e => setFlowUnit(e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                    {FLOW_UNITS.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Inlet Pressure (bar)</label>
                <input type="number" value={pressure} onChange={e => setPressure(+e.target.value)} step="0.1" className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Temperature (K)</label>
                <input type="number" value={temperature} onChange={e => setTemperature(+e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Friction Method</label>
                <select value={frictionMethod} onChange={e => setFrictionMethod(e.target.value)} className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                  {FRICTION_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Gas Composition */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Gas Composition (mol %)</h3>
            <div className="grid grid-cols-2 gap-2">
              {(Object.keys(gasComp) as (keyof GasComp)[]).map(key => (
                <div key={key}>
                  <label className="block text-sm text-slate-400 mb-1">{key}</label>
                  <input type="number" value={gasComp[key]} onChange={e => updateGas(key, +e.target.value)} min="0" max="100" step="0.1" className="w-full bg-slate-700 text-white rounded px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" />
                </div>
              ))}
            </div>
            <div className="mt-3 text-sm text-slate-400">
              Total: <span className={Math.abs(getTotalGasComp(gasComp) - 100) < 1 ? 'text-green-400' : 'text-yellow-400'}>
                {getTotalGasComp(gasComp).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="lg:col-span-3">
            <button onClick={calculate} className="w-full py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-800">
              Calculate Pressure Drop
            </button>
          </div>
        </div>
      )}

      {activeTab === 'results' && results && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Total Pressure Drop</div>
              <div className="text-2xl font-bold text-blue-400">{(results.totalPressureDrop / 1000).toFixed(2)} kPa</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Outlet Pressure</div>
              <div className="text-2xl font-bold text-green-400">{(results.outletPressure / 1e5).toFixed(3)} bar</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Flow Regime</div>
              <div className="text-2xl font-bold text-purple-400">{results.flowRegime}</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Friction Factor</div>
              <div className="text-2xl font-bold text-yellow-400">{results.frictionFactor.toFixed(6)}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Flow Properties</h3>
              <table className="w-full text-sm">
                <tbody>
                  {[
                    ['Velocity', `${results.velocity.toFixed(3)} m/s`],
                    ['Reynolds Number', results.reynoldsNumber.toFixed(0)],
                    ['Mach Number', results.machNumber.toFixed(4)],
                    ['Density', `${results.density.toFixed(3)} kg/m³`],
                    ['Erosional Velocity', `${results.erosionalVelocity.toFixed(2)} m/s`],
                    ['Erosion Ratio', results.erosionRatio.toFixed(3)],
                  ].map(([label, value]) => (
                    <tr key={label} className="border-b border-slate-700">
                      <td className="py-2 text-slate-400">{label}</td>
                      <td className="py-2 text-white text-right">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="bg-slate-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Warnings</h3>
              {results.warnings.length > 0 ? (
                <ul className="space-y-2">
                  {results.warnings.map((w, i) => (
                    <li key={i} className="text-yellow-400 text-sm flex items-start">
                      <span className="mr-2">⚠</span>{w}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-green-400 text-sm">No warnings. All parameters within acceptable ranges.</p>
              )}
            </div>
          </div>

          {/* Engine indicator -- See issue #608 */}
          <div className="text-right">
            <span className="text-xs text-slate-500">Engine: {results.engine}</span>
          </div>
        </div>
      )}

      {activeTab === 'chart' && results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Pressure Profile</h3>
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={pressureProfile}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="distance" stroke="#94a3b8" label={{ value: 'Distance (m)', position: 'bottom', fill: '#94a3b8' }} />
                  <YAxis stroke="#94a3b8" label={{ value: 'Pressure (bar)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="pressure" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Pressure Drop Components</h3>
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={componentBreakdown}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" label={{ value: 'kPa', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {(activeTab === 'results' || activeTab === 'chart') && !results && (
        <div className="bg-slate-800 rounded-lg p-12 text-center">
          <p className="text-slate-400">No results yet. Configure inputs and click "Calculate Pressure Drop".</p>
        </div>
      )}
    </div>
  );
}
