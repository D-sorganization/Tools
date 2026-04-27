import { memo } from 'react';
import { Calculator, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { Statistics } from '../types';

interface StatisticsPanelProps {
  statistics: Statistics;
  selectedSignals: string[];
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 0.001 || Math.abs(value) > 10000) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

// ⚡ Bolt: Wrapped StatisticsPanel in React.memo() to prevent expensive table re-renders
// when unrelated parent components update.
// Performance impact: Eliminates layout thrashing during main UI interactions.
export const StatisticsPanel = memo(function StatisticsPanel({ statistics, selectedSignals }: StatisticsPanelProps) {
  if (selectedSignals.length === 0 || Object.keys(statistics).length === 0) {
    return (
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Calculator className="w-4 h-4" />
          Statistics
        </div>
        <div className="card-body text-dark-400 text-center py-8">
          Select signals to view statistics
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Calculator className="w-4 h-4" />
        Statistics
      </div>
      <div className="card-body overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-dark-400 border-b border-dark-700">
              <th className="text-left py-2 px-2">Signal</th>
              <th className="text-right py-2 px-2">Mean</th>
              <th className="text-right py-2 px-2">Std</th>
              <th className="text-right py-2 px-2">Min</th>
              <th className="text-right py-2 px-2">Max</th>
              <th className="text-right py-2 px-2">Median</th>
            </tr>
          </thead>
          <tbody>
            {selectedSignals.map((signal) => {
              const stats = statistics[signal];
              if (!stats) return null;

              return (
                <tr key={signal} className="border-b border-dark-800 hover:bg-dark-700/30">
                  <td className="py-2 px-2 font-medium text-dark-100">{signal}</td>
                  <td className="text-right py-2 px-2 text-dark-300 font-mono">
                    {formatNumber(stats.mean)}
                  </td>
                  <td className="text-right py-2 px-2 text-dark-300 font-mono">
                    {formatNumber(stats.std)}
                  </td>
                  <td className="text-right py-2 px-2 text-blue-400 font-mono">
                    <span className="inline-flex items-center gap-1">
                      <TrendingDown className="w-3 h-3" />
                      {formatNumber(stats.min)}
                    </span>
                  </td>
                  <td className="text-right py-2 px-2 text-green-400 font-mono">
                    <span className="inline-flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      {formatNumber(stats.max)}
                    </span>
                  </td>
                  <td className="text-right py-2 px-2 text-amber-400 font-mono">
                    <span className="inline-flex items-center gap-1">
                      <Minus className="w-3 h-3" />
                      {formatNumber(stats.median)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
});

export default StatisticsPanel;
