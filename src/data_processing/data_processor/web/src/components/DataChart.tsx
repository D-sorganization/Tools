import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { BarChart2 } from 'lucide-react';
import type { DataRow } from '../types';

interface DataChartProps {
  data: DataRow[];
  selectedSignals: string[];
  title?: string;
}

const CHART_COLORS = [
  '#3b82f6', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
];

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  label?: string | number;
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="custom-tooltip">
      <p className="text-dark-300 text-xs mb-1">Index: {label}</p>
      {payload.map((entry, index) => (
        <p key={index} className="text-sm" style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

export function DataChart({ data, selectedSignals, title = 'Signal Plot' }: DataChartProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) return [];

    // Downsample for performance if needed
    const maxPoints = 1000;
    const step = Math.ceil(data.length / maxPoints);

    return data.filter((_, i) => i % step === 0).map((row, index) => ({
      index: index * step,
      ...row,
    }));
  }, [data]);

  if (data.length === 0) {
    return (
      <div className="card h-full">
        <div className="card-header flex items-center gap-2">
          <BarChart2 className="w-4 h-4" />
          {title}
        </div>
        <div className="card-body flex items-center justify-center h-64 text-dark-400">
          Load data to visualize signals
        </div>
      </div>
    );
  }

  if (selectedSignals.length === 0) {
    return (
      <div className="card h-full">
        <div className="card-header flex items-center gap-2">
          <BarChart2 className="w-4 h-4" />
          {title}
        </div>
        <div className="card-body flex items-center justify-center h-64 text-dark-400">
          Select signals to display
        </div>
      </div>
    );
  }

  return (
    <div className="card h-full">
      <div className="card-header flex items-center gap-2">
        <BarChart2 className="w-4 h-4" />
        {title}
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#343541" />
            <XAxis
              dataKey="index"
              tick={{ fill: '#8e8ea0', fontSize: 12 }}
              axisLine={{ stroke: '#4a4a5a' }}
            />
            <YAxis
              tick={{ fill: '#8e8ea0', fontSize: 12 }}
              axisLine={{ stroke: '#4a4a5a' }}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: 10 }}
              iconType="line"
            />
            {selectedSignals.map((signal, index) => (
              <Line
                key={signal}
                type="monotone"
                dataKey={signal}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default DataChart;
