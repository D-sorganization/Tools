/**
 * Electrode Advisor Calculator
 *
 * Provides electrode positioning guidance, wear analysis, remaining life display,
 * replacement scheduling, and visual position indicator for plasma gasification systems.
 *
 * Consolidated from Gasification_Model to Tools repository.
 */

import { useState, useMemo } from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area,
} from 'recharts';

/** Electrode types with typical properties */
const ELECTRODE_TYPES: Record<
  string,
  {
    name: string;
    material: string;
    maxCurrent: number;
    typicalWearRate: number;
    minLength: number;
    maxLength: number;
    costPerUnit: number;
    description: string;
  }
> = {
  graphite_standard: {
    name: 'Standard Graphite',
    material: 'High-density graphite',
    maxCurrent: 3000,
    typicalWearRate: 0.8,
    minLength: 200,
    maxLength: 1500,
    costPerUnit: 2500,
    description: 'General purpose graphite electrodes for most applications',
  },
  graphite_hd: {
    name: 'High-Density Graphite',
    material: 'Ultra-high density graphite',
    maxCurrent: 4000,
    typicalWearRate: 0.5,
    minLength: 200,
    maxLength: 1800,
    costPerUnit: 4500,
    description: 'Premium electrodes for extended service life',
  },
  tungsten: {
    name: 'Tungsten',
    material: 'Tungsten alloy',
    maxCurrent: 2000,
    typicalWearRate: 0.15,
    minLength: 100,
    maxLength: 500,
    costPerUnit: 8000,
    description: 'High-temperature applications with minimal wear',
  },
  copper_graphite: {
    name: 'Copper-Graphite Composite',
    material: 'Cu-C composite',
    maxCurrent: 5000,
    typicalWearRate: 1.2,
    minLength: 150,
    maxLength: 1200,
    costPerUnit: 3200,
    description: 'High current capacity with moderate wear',
  },
};

/** Calculate wear rate based on operating conditions */
function calculateWearRate(electrodeType: string, current: number, temperature: number): number {
  const electrode = ELECTRODE_TYPES[electrodeType];
  if (!electrode) return 0;

  const baseRate = electrode.typicalWearRate;
  const currentRatio = current / electrode.maxCurrent;
  const currentFactor = Math.pow(currentRatio, 2);
  const tempFactor = 1 + 0.001 * (temperature - 1500);

  return baseRate * currentFactor * Math.max(0.5, tempFactor);
}

/** Calculate optimal electrode position based on gap and wear */
function calculateOptimalPosition(
  currentPosition: number,
  gapDistance: number,
  currentLength: number,
  wornLength: number,
  minLength: number
): { optimal: number; adjustment: number; status: string } {
  const tipPosition = currentPosition + currentLength - wornLength;
  const idealTipPosition = gapDistance;
  const adjustment = idealTipPosition - tipPosition;

  let status = 'Good';
  const remainingUsable = currentLength - wornLength - minLength;

  if (remainingUsable < 0) {
    status = 'Replace Required';
  } else if (remainingUsable < 100) {
    status = 'Replace Soon';
  } else if (Math.abs(adjustment) > 50) {
    status = 'Adjustment Needed';
  }

  return {
    optimal: currentPosition + adjustment,
    adjustment,
    status,
  };
}

/** Calculate remaining life and replacement schedule */
function calculateRemainingLife(
  currentLength: number,
  wornLength: number,
  minLength: number,
  wearRate: number,
  operatingHoursPerDay: number
): {
  remainingMm: number;
  remainingHours: number;
  remainingDays: number;
  replacementDate: Date;
} {
  const remainingMm = currentLength - wornLength - minLength;
  const remainingHours = wearRate > 0 ? remainingMm / wearRate : Infinity;
  const remainingDays = remainingHours / operatingHoursPerDay;

  const replacementDate = new Date();
  replacementDate.setDate(replacementDate.getDate() + Math.floor(remainingDays));

  return {
    remainingMm: Math.max(0, remainingMm),
    remainingHours: Math.max(0, remainingHours),
    remainingDays: Math.max(0, remainingDays),
    replacementDate,
  };
}

/** Generate wear projection data */
function generateWearProjection(
  currentLength: number,
  wornLength: number,
  minLength: number,
  wearRate: number,
  days: number
): Array<{ day: number; length: number; usable: number }> {
  const data = [];
  const hoursPerDay = 20;

  for (let day = 0; day <= days; day++) {
    const wearAmount = wornLength + day * hoursPerDay * wearRate;
    const remainingLength = Math.max(minLength, currentLength - wearAmount);
    const usableLength = Math.max(0, remainingLength - minLength);

    data.push({
      day,
      length: remainingLength,
      usable: usableLength,
    });
  }

  return data;
}

// Simple UI Components (standalone versions)
const Card = ({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) => <div className={`bg-white rounded-lg shadow border ${className}`}>{children}</div>;

const CardHeader = ({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) => <div className={`px-4 py-3 border-b ${className}`}>{children}</div>;

const CardTitle = ({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) => <h3 className={`font-semibold ${className}`}>{children}</h3>;

const CardContent = ({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) => <div className={`p-4 ${className}`}>{children}</div>;

const Label = ({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) => <label className={`text-sm font-medium text-gray-700 ${className}`}>{children}</label>;

const Input = ({
  type = 'text',
  value,
  onChange,
  min,
  max,
  step,
  className = '',
}: {
  type?: string;
  value: number | string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}) => (
  <input
    type={type}
    value={value}
    onChange={onChange}
    min={min}
    max={max}
    step={step}
    className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`}
  />
);

const Select = ({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}) => (
  <select
    value={value}
    onChange={(e) => onChange(e.target.value)}
    className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
  >
    {children}
  </select>
);

type TabValue = 'status' | 'position' | 'schedule';

export function ElectrodeAdvisorCalculator() {
  // Tab state
  const [activeTab, setActiveTab] = useState<TabValue>('status');

  // Electrode selection
  const [electrodeType, setElectrodeType] = useState('graphite_standard');

  // Current state
  const [currentLength, setCurrentLength] = useState(1500);
  const [wornLength, setWornLength] = useState(150);
  const [currentPosition, setCurrentPosition] = useState(500);

  // Operating conditions
  const [operatingCurrent, setOperatingCurrent] = useState(2500);
  const [plasmaTemp, setPlasmaTemp] = useState(1500);
  const [gapDistance, setGapDistance] = useState(300);
  const [hoursPerDay, setHoursPerDay] = useState(20);

  // Historical wear data
  const [historicalWearRate, setHistoricalWearRate] = useState(0.7);

  const electrode = ELECTRODE_TYPES[electrodeType];

  // Calculate current wear rate
  const calculatedWearRate = useMemo(
    () => calculateWearRate(electrodeType, operatingCurrent, plasmaTemp),
    [electrodeType, operatingCurrent, plasmaTemp]
  );

  // Use average of calculated and historical
  const effectiveWearRate = (calculatedWearRate + historicalWearRate) / 2;

  // Calculate optimal position
  const positionResult = useMemo(
    () =>
      calculateOptimalPosition(
        currentPosition,
        gapDistance,
        currentLength,
        wornLength,
        electrode.minLength
      ),
    [currentPosition, gapDistance, currentLength, wornLength, electrode.minLength]
  );

  // Calculate remaining life
  const lifeResult = useMemo(
    () =>
      calculateRemainingLife(
        currentLength,
        wornLength,
        electrode.minLength,
        effectiveWearRate,
        hoursPerDay
      ),
    [currentLength, wornLength, electrode.minLength, effectiveWearRate, hoursPerDay]
  );

  // Generate wear projection
  const wearProjection = useMemo(
    () =>
      generateWearProjection(currentLength, wornLength, electrode.minLength, effectiveWearRate, 30),
    [currentLength, wornLength, electrode.minLength, effectiveWearRate]
  );

  // Status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Good':
        return 'bg-green-100 border-green-500 text-green-700';
      case 'Adjustment Needed':
        return 'bg-amber-100 border-amber-500 text-amber-700';
      case 'Replace Soon':
        return 'bg-orange-100 border-orange-500 text-orange-700';
      case 'Replace Required':
        return 'bg-red-100 border-red-500 text-red-700';
      default:
        return 'bg-gray-100 border-gray-500 text-gray-700';
    }
  };

  // Remaining life percentage
  const lifePercent =
    ((currentLength - wornLength - electrode.minLength) /
      (electrode.maxLength - electrode.minLength)) *
    100;

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
        {(['status', 'position', 'schedule'] as TabValue[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 px-4 rounded-md capitalize transition-colors ${
              activeTab === tab
                ? 'bg-white shadow text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Status Tab */}
      {activeTab === 'status' && (
        <div className="space-y-6">
          {/* Electrode Selection */}
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Electrode Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Electrode Type</Label>
                <Select value={electrodeType} onChange={setElectrodeType}>
                  {Object.entries(ELECTRODE_TYPES).map(([key, value]) => (
                    <option key={key} value={key}>
                      {value.name}
                    </option>
                  ))}
                </Select>
                <p className="text-sm text-gray-500">{electrode.description}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label className="text-sm">Current Length (mm)</Label>
                  <Input
                    type="number"
                    value={currentLength}
                    onChange={(e) => setCurrentLength(parseFloat(e.target.value) || 0)}
                    min={electrode.minLength}
                    max={electrode.maxLength}
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-sm">Worn Length (mm)</Label>
                  <Input
                    type="number"
                    value={wornLength}
                    onChange={(e) => setWornLength(parseFloat(e.target.value) || 0)}
                    min={0}
                    max={currentLength - electrode.minLength}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Plasma Temperature (C)</Label>
                  <Input
                    type="number"
                    value={plasmaTemp}
                    onChange={(e) => setPlasmaTemp(parseFloat(e.target.value) || 0)}
                    min={1000}
                    max={3000}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Operating Current (A)</Label>
                  <Input
                    type="number"
                    value={operatingCurrent}
                    onChange={(e) => setOperatingCurrent(parseFloat(e.target.value) || 0)}
                    min={100}
                    max={electrode.maxCurrent}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Status Summary */}
          <Card className={`border-2 ${getStatusColor(positionResult.status)}`}>
            <CardContent className="py-6">
              <div className="text-center">
                <div className="text-3xl font-bold">{positionResult.status}</div>
                <div className="text-sm mt-2">
                  {positionResult.status === 'Good' && 'Electrode is operating within parameters'}
                  {positionResult.status === 'Adjustment Needed' &&
                    `Adjust position by ${positionResult.adjustment.toFixed(0)} mm`}
                  {positionResult.status === 'Replace Soon' &&
                    `Only ${lifeResult.remainingMm.toFixed(0)} mm usable length remaining`}
                  {positionResult.status === 'Replace Required' &&
                    'Electrode has reached end of life'}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Life Indicator */}
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Remaining Life</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Usable: {lifeResult.remainingMm.toFixed(0)} mm</span>
                  <span>{Math.max(0, lifePercent).toFixed(0)}%</span>
                </div>
                <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      lifePercent > 50
                        ? 'bg-green-500'
                        : lifePercent > 25
                        ? 'bg-amber-500'
                        : lifePercent > 10
                        ? 'bg-orange-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.max(0, Math.min(100, lifePercent))}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">{lifeResult.remainingHours.toFixed(0)}</div>
                  <div className="text-sm text-gray-500">Hours</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{lifeResult.remainingDays.toFixed(1)}</div>
                  <div className="text-sm text-gray-500">Days</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{effectiveWearRate.toFixed(2)}</div>
                  <div className="text-sm text-gray-500">mm/hr wear</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Wear Rates */}
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Wear Rate Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-100 rounded-lg">
                  <div className="text-sm text-gray-500">Calculated Rate</div>
                  <div className="text-xl font-bold">{calculatedWearRate.toFixed(3)} mm/hr</div>
                  <div className="text-xs text-gray-500">Based on current & temperature</div>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-sm">Historical Wear Rate (mm/hr)</Label>
                  <Input
                    type="number"
                    value={historicalWearRate}
                    onChange={(e) => setHistoricalWearRate(parseFloat(e.target.value) || 0)}
                    min={0.01}
                    max={5}
                    step={0.01}
                  />
                </div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg text-center">
                <div className="text-sm text-blue-600">Effective Wear Rate (Average)</div>
                <div className="text-2xl font-bold text-blue-700">
                  {effectiveWearRate.toFixed(3)} mm/hr
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Position Tab */}
      {activeTab === 'position' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Position Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Current Position (mm)</Label>
                  <Input
                    type="number"
                    value={currentPosition}
                    onChange={(e) => setCurrentPosition(parseFloat(e.target.value) || 0)}
                    min={0}
                    max={2000}
                  />
                  <p className="text-xs text-gray-500">Distance from reference point</p>
                </div>
                <div className="space-y-1.5">
                  <Label>Target Gap Distance (mm)</Label>
                  <Input
                    type="number"
                    value={gapDistance}
                    onChange={(e) => setGapDistance(parseFloat(e.target.value) || 0)}
                    min={100}
                    max={1000}
                  />
                  <p className="text-xs text-gray-500">Optimal electrode tip to slag distance</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Position Recommendation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-center mb-6">
                <div className="p-3 bg-gray-100 rounded-lg">
                  <div className="text-sm text-gray-500">Current</div>
                  <div className="text-xl font-bold">{currentPosition} mm</div>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="text-sm text-green-600">Optimal</div>
                  <div className="text-xl font-bold text-green-700">
                    {positionResult.optimal.toFixed(0)} mm
                  </div>
                </div>
                <div
                  className={`p-3 rounded-lg ${
                    Math.abs(positionResult.adjustment) < 10
                      ? 'bg-green-50'
                      : Math.abs(positionResult.adjustment) < 50
                      ? 'bg-amber-50'
                      : 'bg-red-50'
                  }`}
                >
                  <div className="text-sm text-gray-500">Adjustment</div>
                  <div
                    className={`text-xl font-bold ${
                      Math.abs(positionResult.adjustment) < 10
                        ? 'text-green-700'
                        : Math.abs(positionResult.adjustment) < 50
                        ? 'text-amber-700'
                        : 'text-red-700'
                    }`}
                  >
                    {positionResult.adjustment > 0 ? '+' : ''}
                    {positionResult.adjustment.toFixed(0)} mm
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Visual Position Indicator */}
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Visual Position Indicator</CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 400 300" className="w-full h-72">
                <rect x="0" y="0" width="400" height="300" fill="#f8fafc" />
                <rect
                  x="100"
                  y="40"
                  width="200"
                  height="220"
                  fill="#e2e8f0"
                  stroke="#64748b"
                  strokeWidth={2}
                />
                <rect x="100" y="200" width="200" height="60" fill="#78350f" opacity={0.6} />
                <text x="200" y="235" textAnchor="middle" fill="#fff" fontSize="12">
                  Slag Bath
                </text>
                <rect
                  x="180"
                  y="20"
                  width="40"
                  height="30"
                  fill="#475569"
                  stroke="#1e293b"
                  strokeWidth={2}
                />
                <rect
                  x="185"
                  y="50"
                  width="30"
                  height={Math.min(150, (currentLength - wornLength) * 0.1)}
                  fill="#374151"
                  stroke="#1f2937"
                  strokeWidth={2}
                />
                <rect
                  x="185"
                  y={50 + Math.min(150, (currentLength - wornLength) * 0.1) - 5}
                  width="30"
                  height="5"
                  fill="#ef4444"
                />
                <ellipse
                  cx="200"
                  cy={60 + Math.min(150, (currentLength - wornLength) * 0.1)}
                  rx="15"
                  ry="20"
                  fill="#fbbf24"
                  opacity={0.7}
                />
                <ellipse
                  cx="200"
                  cy={60 + Math.min(150, (currentLength - wornLength) * 0.1)}
                  rx="8"
                  ry="12"
                  fill="#fff"
                  opacity={0.8}
                />
                <text x="200" y="15" textAnchor="middle" fill="#1e293b" fontSize="12" fontWeight="bold">
                  Electrode Assembly
                </text>
              </svg>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Schedule Tab */}
      {activeTab === 'schedule' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Operating Schedule</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1.5">
                <Label>Operating Hours per Day</Label>
                <Input
                  type="number"
                  value={hoursPerDay}
                  onChange={(e) => setHoursPerDay(parseFloat(e.target.value) || 1)}
                  min={1}
                  max={24}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-blue-50 border-blue-200">
            <CardHeader className="py-4">
              <CardTitle className="text-base text-blue-800">Replacement Schedule</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="p-4 bg-white rounded-lg">
                  <div className="text-sm text-gray-500">Estimated Replacement Date</div>
                  <div className="text-xl font-bold text-blue-700">
                    {lifeResult.replacementDate.toLocaleDateString()}
                  </div>
                </div>
                <div className="p-4 bg-white rounded-lg">
                  <div className="text-sm text-gray-500">Days Until Replacement</div>
                  <div className="text-xl font-bold text-blue-700">
                    {Math.floor(lifeResult.remainingDays)} days
                  </div>
                </div>
              </div>

              <div className="p-4 bg-white rounded-lg">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="font-medium">Electrode Cost</div>
                    <div className="text-sm text-gray-500">{electrode.name}</div>
                  </div>
                  <div className="text-xl font-bold">${electrode.costPerUnit.toLocaleString()}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Wear Projection Chart */}
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">30-Day Wear Projection</CardTitle>
            </CardHeader>
            <CardContent className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={wearProjection}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="day"
                    label={{ value: 'Days', position: 'bottom', offset: -5 }}
                  />
                  <YAxis label={{ value: 'Length (mm)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(0)} mm`,
                      name === 'length' ? 'Total Length' : 'Usable Length',
                    ]}
                  />
                  <ReferenceLine
                    y={electrode.minLength}
                    stroke="#ef4444"
                    strokeDasharray="3 3"
                    label={{ value: 'Min Length', fill: '#ef4444', fontSize: 10 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="length"
                    stroke="#3b82f6"
                    fill="#bfdbfe"
                    name="Total Length"
                  />
                  <Area
                    type="monotone"
                    dataKey="usable"
                    stroke="#22c55e"
                    fill="#bbf7d0"
                    name="Usable Length"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="text-xs text-gray-500">
        Note: Wear rates are estimates based on typical operating conditions. Actual wear may vary
        based on feedstock composition, plasma conditions, and electrode quality.
      </div>
    </div>
  );
}

export default ElectrodeAdvisorCalculator;
