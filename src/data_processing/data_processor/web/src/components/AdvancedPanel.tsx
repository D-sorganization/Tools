import { useState, memo } from 'react';
import { Calculator, TrendingUp, TrendingDown } from 'lucide-react';
import type {
  IntegrationMethod,
  DifferentiationMethod,
  IntegrationConfig,
  DifferentiationConfig,
  FormulaConfig,
} from '../types';

interface AdvancedPanelProps {
  signals: string[];
  selectedSignals: string[];
  timeColumn: string | null;
  disabled: boolean;
  onIntegrate: (config: IntegrationConfig) => void;
  onDifferentiate: (config: DifferentiationConfig) => void;
  onApplyFormula: (config: FormulaConfig) => void;
}

// ⚡ Bolt: Wrapped AdvancedPanel in React.memo() to prevent unnecessary O(N) re-render
// cascades when parent (App.tsx) UI state changes (like switching tabs).
// Performance impact: Eliminates UI stuttering during tab navigation.
export const AdvancedPanel = memo(function AdvancedPanel({
  selectedSignals,
  timeColumn,
  disabled,
  onIntegrate,
  onDifferentiate,
  onApplyFormula,
}: AdvancedPanelProps) {
  const [intMethod, setIntMethod] = useState<IntegrationMethod>('trapezoidal');
  const [diffMethod, setDiffMethod] = useState<DifferentiationMethod>('spline');
  const [diffOrder, setDiffOrder] = useState(1);
  const [diffWindowSize, setDiffWindowSize] = useState(11);
  const [diffPolyOrder, setDiffPolyOrder] = useState(3);
  const [formulaName, setFormulaName] = useState('');
  const [formula, setFormula] = useState('');

  const handleIntegrate = () => {
    if (!timeColumn || selectedSignals.length === 0) return;
    onIntegrate({
      method: intMethod,
      signals: selectedSignals,
      timeColumn,
    });
  };

  const handleDifferentiate = () => {
    if (!timeColumn || selectedSignals.length === 0) return;
    onDifferentiate({
      method: diffMethod,
      signals: selectedSignals,
      timeColumn,
      order: diffOrder,
      windowSize: diffWindowSize,
      polyOrder: diffPolyOrder,
    });
  };

  const handleApplyFormula = () => {
    if (!formulaName || !formula) return;
    onApplyFormula({ name: formulaName, formula });
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Calculator className="w-4 h-4" />
        Advanced Operations
      </div>
      <div className="card-body space-y-6">
        {/* Integration Section */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-dark-300 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Integration
          </h4>
          <div className="space-y-2">
            <label className="block text-xs text-dark-400">Method</label>
            <select
              className="select w-full"
              value={intMethod}
              onChange={(e) => setIntMethod(e.target.value as IntegrationMethod)}
              disabled={disabled}
            >
              <option value="trapezoidal">Trapezoidal</option>
              <option value="simpson">Simpson's Rule</option>
              <option value="rectangular">Rectangular</option>
            </select>
          </div>
          <button
            className="btn btn-primary w-full"
            onClick={handleIntegrate}
            disabled={disabled || !timeColumn || selectedSignals.length === 0}
          >
            Integrate Selected Signals
          </button>
        </div>

        {/* Differentiation Section */}
        <div className="space-y-3 pt-4 border-t border-dark-700">
          <h4 className="text-sm font-medium text-dark-300 flex items-center gap-2">
            <TrendingDown className="w-4 h-4" />
            Differentiation
          </h4>
          <div className="space-y-2">
            <label className="block text-xs text-dark-400">Method</label>
            <select
              className="select w-full"
              value={diffMethod}
              onChange={(e) => setDiffMethod(e.target.value as DifferentiationMethod)}
              disabled={disabled}
            >
              <option value="spline">Spline (Acausal)</option>
              <option value="rolling_polynomial">Rolling Polynomial (Causal)</option>
            </select>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">Order</label>
              <input
                type="number"
                className="input w-full"
                value={diffOrder}
                onChange={(e) => setDiffOrder(parseInt(e.target.value) || 1)}
                min={1}
                max={3}
                disabled={disabled}
              />
            </div>
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">Window</label>
              <input
                type="number"
                className="input w-full"
                value={diffWindowSize}
                onChange={(e) => setDiffWindowSize(parseInt(e.target.value) || 11)}
                min={3}
                max={51}
                step={2}
                disabled={disabled}
              />
            </div>
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">Poly Order</label>
              <input
                type="number"
                className="input w-full"
                value={diffPolyOrder}
                onChange={(e) => setDiffPolyOrder(parseInt(e.target.value) || 3)}
                min={2}
                max={6}
                disabled={disabled}
              />
            </div>
          </div>
          <button
            className="btn btn-primary w-full"
            onClick={handleDifferentiate}
            disabled={disabled || !timeColumn || selectedSignals.length === 0}
          >
            Differentiate Selected Signals
          </button>
        </div>

        {/* Custom Formula Section */}
        <div className="space-y-3 pt-4 border-t border-dark-700">
          <h4 className="text-sm font-medium text-dark-300 flex items-center gap-2">
            <Calculator className="w-4 h-4" />
            Custom Formula
          </h4>
          <div className="space-y-2">
            <label className="block text-xs text-dark-400">New Signal Name</label>
            <input
              type="text"
              className="input w-full"
              value={formulaName}
              onChange={(e) => setFormulaName(e.target.value)}
              placeholder="e.g., velocity"
              disabled={disabled}
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs text-dark-400">Formula</label>
            <input
              type="text"
              className="input w-full"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
              placeholder="e.g., signal1 + signal2 * 2"
              disabled={disabled}
            />
            <p className="text-xs text-dark-500">
              Available: +, -, *, /, **, sqrt, sin, cos, abs, log, exp
            </p>
          </div>
          <button
            className="btn btn-primary w-full"
            onClick={handleApplyFormula}
            disabled={disabled || !formulaName || !formula}
          >
            Apply Formula
          </button>
        </div>
      </div>
    </div>
  );
});

export default AdvancedPanel;
