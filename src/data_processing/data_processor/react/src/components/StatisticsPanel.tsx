/**
 * Statistics display panel component.
 */

import React, { useCallback } from 'react';
import type { SignalStatistics } from '../types';

interface StatisticsPanelProps {
  statistics: SignalStatistics[];
  onCalculate: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export function StatisticsPanel({
  statistics,
  onCalculate,
  isLoading,
  disabled,
}: StatisticsPanelProps) {
  const formatValue = useCallback((value: number | null): string => {
    if (value === null || value === undefined) {
      return 'N/A';
    }
    return value.toFixed(4);
  }, []);

  return (
    <div className="panel">
      <h3 className="panel-title">Signal Statistics</h3>

      <button
        className="btn"
        onClick={onCalculate}
        disabled={disabled || isLoading}
        style={{ marginBottom: 12 }}
      >
        {isLoading ? 'Calculating...' : 'Calculate Statistics'}
      </button>

      {statistics.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
          Click Calculate to view statistics
        </div>
      ) : (
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          {statistics.map((stat) => (
            <div
              key={stat.name}
              style={{
                marginBottom: 16,
                padding: 12,
                backgroundColor: 'var(--bg-tertiary)',
                borderRadius: 4,
              }}
            >
              <div style={{ fontWeight: 'bold', marginBottom: 8 }}>{stat.name}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12 }}>
                <div>Count: {stat.count}</div>
                <div>Mean: {formatValue(stat.mean)}</div>
                <div>Std: {formatValue(stat.std)}</div>
                <div>Median: {formatValue(stat.median)}</div>
                <div>Min: {formatValue(stat.min)}</div>
                <div>Max: {formatValue(stat.max)}</div>
                <div>Q25: {formatValue(stat.q25)}</div>
                <div>Q75: {formatValue(stat.q75)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default StatisticsPanel;
