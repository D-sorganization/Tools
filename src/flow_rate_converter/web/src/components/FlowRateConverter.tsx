import { useState, useCallback } from 'react';

// API base URL -- defaults to shared calc backend.
// Override via VITE_CALC_API_URL environment variable.
// See issue #608.
const CALC_API_BASE = import.meta.env.VITE_CALC_API_URL ?? 'http://localhost:8010';

// Supported units per category
const UNITS: Record<string, string[]> = {
  mass: ['kg/s', 'kg/h', 'kg/min', 'g/s', 'g/h', 'lb/s', 'lb/h', 'lb/min', 'ton/h'],
  molar: ['mol/s', 'mol/h', 'mol/min', 'kmol/s', 'kmol/h', 'kmol/min', 'lbmol/s', 'lbmol/h', 'lbmol/min'],
  volumetric: ['m3/s', 'm3/h', 'm3/min', 'L/s', 'L/min', 'L/h', 'ft3/s', 'ft3/min', 'ft3/h', 'CFM', 'GPM'],
};

// Client-side conversion factors (SI base: kg/s, mol/s, m3/s)
// See issue #608 for context on why the Python backend is preferred.
const MASS_TO_KG_S: Record<string, number> = {
  'kg/s': 1, 'kg/h': 1 / 3600, 'kg/min': 1 / 60, 'g/s': 1e-3, 'g/h': 1e-3 / 3600,
  'lb/s': 0.45359237, 'lb/h': 0.45359237 / 3600, 'lb/min': 0.45359237 / 60, 'ton/h': 1000 / 3600,
};

const MOLAR_TO_MOL_S: Record<string, number> = {
  'mol/s': 1, 'mol/h': 1 / 3600, 'mol/min': 1 / 60, 'kmol/s': 1e3, 'kmol/h': 1e3 / 3600,
  'kmol/min': 1e3 / 60, 'lbmol/s': 453.59237, 'lbmol/h': 453.59237 / 3600, 'lbmol/min': 453.59237 / 60,
};

const VOLUMETRIC_TO_M3_S: Record<string, number> = {
  'm3/s': 1, 'm3/h': 1 / 3600, 'm3/min': 1 / 60, 'L/s': 1e-3, 'L/min': 1e-3 / 60,
  'L/h': 1e-3 / 3600, 'ft3/s': 0.028316846592, 'ft3/min': 0.028316846592 / 60,
  'ft3/h': 0.028316846592 / 3600, 'CFM': 0.028316846592 / 60, 'GPM': 6.30902e-5,
};

const TABLES: Record<string, Record<string, number>> = {
  mass: MASS_TO_KG_S,
  molar: MOLAR_TO_MOL_S,
  volumetric: VOLUMETRIC_TO_M3_S,
};

interface ConversionResult {
  result: number;
  fromUnit: string;
  toUnit: string;
  category: string;
  engine: string;
}

function convertFallback(value: number, fromUnit: string, toUnit: string, category: string): ConversionResult {
  const table = TABLES[category];
  if (!table || !table[fromUnit] || !table[toUnit]) {
    throw new Error(`Unknown unit or category: ${fromUnit} -> ${toUnit} (${category})`);
  }
  const base = value * table[fromUnit];
  const result = base / table[toUnit];
  return { result, fromUnit, toUnit, category, engine: 'client-fallback' };
}

async function fetchFromBackend(
  value: number, fromUnit: string, toUnit: string, category: string,
): Promise<ConversionResult> {
  const response = await fetch(`${CALC_API_BASE}/api/calc/flow-rate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value, from_unit: fromUnit, to_unit: toUnit, category }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `API error ${response.status}`);
  }

  const data = await response.json();
  return {
    result: data.result,
    fromUnit: data.from_unit,
    toUnit: data.to_unit,
    category: data.category,
    engine: 'python-backend',
  };
}

export function FlowRateConverter() {
  const [category, setCategory] = useState<string>('mass');
  const [value, setValue] = useState<number>(1000);
  const [fromUnit, setFromUnit] = useState<string>('kg/h');
  const [toUnit, setToUnit] = useState<string>('lb/h');
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCategoryChange = useCallback((newCategory: string) => {
    setCategory(newCategory);
    const units = UNITS[newCategory];
    setFromUnit(units[0]);
    setToUnit(units[1]);
    setResult(null);
    setError(null);
  }, []);

  const swapUnits = useCallback(() => {
    setFromUnit(toUnit);
    setToUnit(fromUnit);
    setResult(null);
  }, [fromUnit, toUnit]);

  const convert = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let res: ConversionResult;
      try {
        res = await fetchFromBackend(value, fromUnit, toUnit, category);
      } catch {
        res = convertFallback(value, fromUnit, toUnit, category);
      }
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Conversion failed');
    } finally {
      setLoading(false);
    }
  }, [value, fromUnit, toUnit, category]);

  const currentUnits = UNITS[category] || [];

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Category Selector */}
      <div className="flex border-b border-slate-700">
        {Object.keys(UNITS).map((cat) => (
          <button
            key={cat}
            onClick={() => handleCategoryChange(cat)}
            className={`px-6 py-3 font-medium capitalize transition-colors ${
              category === cat
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            {cat} Flow
          </button>
        ))}
      </div>

      {/* Conversion Panel */}
      <div className="bg-slate-800 rounded-lg p-6 space-y-5">
        {/* Input Value */}
        <div>
          <label className="block text-sm text-slate-300 mb-2">Value</label>
          <input
            type="number"
            value={value}
            onChange={(e) => setValue(Number(e.target.value))}
            className="w-full bg-slate-700 text-white text-lg rounded px-4 py-3 border border-slate-600 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          />
        </div>

        {/* From / Swap / To */}
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-end">
          <div>
            <label className="block text-sm text-slate-300 mb-2">From</label>
            <select
              value={fromUnit}
              onChange={(e) => setFromUnit(e.target.value)}
              className="w-full bg-slate-700 text-white rounded px-3 py-3 border border-slate-600 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {currentUnits.map((u) => (
                <option key={u} value={u}>{u}</option>
              ))}
            </select>
          </div>

          <button
            onClick={swapUnits}
            className="px-3 py-3 text-slate-400 hover:text-white transition-colors"
            title="Swap units"
            aria-label="Swap units"
          >
            &#8646;
          </button>

          <div>
            <label className="block text-sm text-slate-300 mb-2">To</label>
            <select
              value={toUnit}
              onChange={(e) => setToUnit(e.target.value)}
              className="w-full bg-slate-700 text-white rounded px-3 py-3 border border-slate-600 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {currentUnits.map((u) => (
                <option key={u} value={u}>{u}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Convert Button */}
        <button
          onClick={convert}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Converting...' : 'Convert'}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div className="bg-green-900/30 border border-green-700 rounded-lg p-6 text-center">
          <p className="text-slate-400 text-sm mb-2">Result</p>
          <p className="text-3xl font-bold text-green-400">
            {result.result.toLocaleString(undefined, { maximumSignificantDigits: 8 })} {result.toUnit}
          </p>
          <p className="text-slate-500 text-sm mt-3">
            {value.toLocaleString()} {result.fromUnit} = {result.result.toLocaleString(undefined, { maximumSignificantDigits: 8 })} {result.toUnit}
          </p>
          {/* Engine indicator -- See issue #608 */}
          <p className="text-xs text-slate-600 mt-2">Engine: {result.engine}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
}

export default FlowRateConverter;
