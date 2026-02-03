/**
 * TRC Vessel Designer Calculator
 *
 * Interactive TRC (Thermal Reaction Chamber) vessel design tool with
 * SVG visualization, volume calculations, and residence time estimation.
 *
 * Consolidated from Gasification_Model to Tools repository.
 */

import { useState, useMemo } from 'react';

/** Refractory layer */
interface RefractoryLayer {
  material: string;
  thickness: number; // mm
  maxTemp: number; // C
  conductivity: number; // W/m-K
}

/** Burner configuration */
interface BurnerConfig {
  id: number;
  position: number;
  angle: number;
  type: 'plasma' | 'oxy-fuel' | 'auxiliary';
  power: number;
}

/** Calculate vessel volumes */
function calculateVesselVolumes(
  length: number,
  innerDiameter: number,
  refractoryThickness: number
): {
  grossVolume: number;
  netVolume: number;
  refractoryVolume: number;
  surfaceArea: number;
} {
  const outerRadius = innerDiameter / 2 + refractoryThickness;
  const innerRadius = innerDiameter / 2;

  const grossVolume = Math.PI * Math.pow(outerRadius, 2) * length;
  const netVolume = Math.PI * Math.pow(innerRadius, 2) * length;
  const refractoryVolume = grossVolume - netVolume;
  const surfaceArea = 2 * Math.PI * innerRadius * length + 2 * Math.PI * Math.pow(innerRadius, 2);

  return { grossVolume, netVolume, refractoryVolume, surfaceArea };
}

/** Calculate residence time */
function calculateResidenceTime(
  netVolume: number,
  volumetricFlow: number,
  temperature: number,
  pressure: number
): {
  residenceTime: number;
  actualFlow: number;
  standardFlow: number;
} {
  const T_standard = 273.15;
  const P_standard = 101.325;
  const T_actual = temperature + 273.15;

  const expansionFactor = (T_actual / T_standard) * (P_standard / pressure);
  const standardFlow = volumetricFlow / expansionFactor;
  const residenceTime = (netVolume / volumetricFlow) * 3600;

  return { residenceTime, actualFlow: volumetricFlow, standardFlow };
}

/** Calculate heat loss */
function calculateHeatLoss(
  surfaceArea: number,
  innerTemp: number,
  outerTemp: number,
  layers: RefractoryLayer[]
): number {
  let totalResistance = 0;

  layers.forEach((layer) => {
    totalResistance += layer.thickness / 1000 / layer.conductivity;
  });

  const h_conv = 10;
  totalResistance += 1 / h_conv;

  const deltaT = innerTemp - outerTemp;
  const heatFlux = deltaT / totalResistance;

  return (heatFlux * surfaceArea) / 1000;
}

/** Refractory presets */
const REFRACTORY_PRESETS: Record<string, RefractoryLayer[]> = {
  standard: [
    { material: 'High-Alumina Working Lining', thickness: 150, maxTemp: 1800, conductivity: 2.5 },
    { material: 'Insulating Firebrick', thickness: 115, maxTemp: 1400, conductivity: 0.4 },
    { material: 'Microporous Insulation', thickness: 25, maxTemp: 1000, conductivity: 0.025 },
  ],
  high_temp: [
    { material: 'Chrome-Alumina Working', thickness: 200, maxTemp: 1900, conductivity: 3.0 },
    { material: 'High-Alumina Backup', thickness: 100, maxTemp: 1700, conductivity: 2.0 },
    { material: 'Insulating Firebrick', thickness: 115, maxTemp: 1400, conductivity: 0.4 },
    { material: 'Calcium Silicate Board', thickness: 50, maxTemp: 1000, conductivity: 0.08 },
  ],
  economy: [
    { material: 'Castable Refractory', thickness: 200, maxTemp: 1600, conductivity: 1.8 },
    { material: 'Ceramic Fiber Blanket', thickness: 50, maxTemp: 1200, conductivity: 0.15 },
  ],
};

// Simple UI Components
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white rounded-lg shadow border ${className}`}>{children}</div>
);

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

const Label = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <label className={`text-sm font-medium text-gray-700 ${className}`}>{children}</label>
);

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

const Button = ({
  children,
  onClick,
  variant = 'default',
  size = 'default',
  className = '',
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm';
  className?: string;
}) => {
  const baseClasses = 'font-medium rounded-md transition-colors';
  const sizeClasses = size === 'sm' ? 'px-3 py-1 text-sm' : 'px-4 py-2';
  const variantClasses = {
    default: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border border-gray-300 hover:bg-gray-50',
    ghost: 'hover:bg-gray-100',
  };

  return (
    <button
      onClick={onClick}
      className={`${baseClasses} ${sizeClasses} ${variantClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

type TabValue = 'dimensions' | 'refractory' | 'equipment' | 'results';

export function TRCVesselDesignerCalculator() {
  const [activeTab, setActiveTab] = useState<TabValue>('dimensions');

  // Vessel dimensions
  const [vesselLength, setVesselLength] = useState(6);
  const [innerDiameter, setInnerDiameter] = useState(2);

  // Refractory
  const [refractoryPreset, setRefractoryPreset] = useState('standard');
  const [refractoryLayers, setRefractoryLayers] = useState<RefractoryLayer[]>(
    REFRACTORY_PRESETS.standard
  );

  // Burner configuration
  const [burners, setBurners] = useState<BurnerConfig[]>([
    { id: 1, position: 0.15, angle: 15, type: 'plasma', power: 1500 },
    { id: 2, position: 0.4, angle: 0, type: 'oxy-fuel', power: 500 },
  ]);

  // Operating conditions
  const [operatingTemp, setOperatingTemp] = useState(1400);
  const [operatingPressure, setOperatingPressure] = useState(101.325);
  const [volumetricFlow, setVolumetricFlow] = useState(2000);
  const [ambientTemp, setAmbientTemp] = useState(25);

  // Calculations
  const totalRefractoryThickness = useMemo(
    () => refractoryLayers.reduce((sum, layer) => sum + layer.thickness, 0) / 1000,
    [refractoryLayers]
  );

  const volumes = useMemo(
    () => calculateVesselVolumes(vesselLength, innerDiameter, totalRefractoryThickness),
    [vesselLength, innerDiameter, totalRefractoryThickness]
  );

  const residenceResults = useMemo(
    () =>
      calculateResidenceTime(volumes.netVolume, volumetricFlow, operatingTemp, operatingPressure),
    [volumes.netVolume, volumetricFlow, operatingTemp, operatingPressure]
  );

  const heatLoss = useMemo(
    () => calculateHeatLoss(volumes.surfaceArea, operatingTemp, ambientTemp, refractoryLayers),
    [volumes.surfaceArea, operatingTemp, ambientTemp, refractoryLayers]
  );

  const totalBurnerPower = burners.reduce((sum, b) => sum + b.power, 0);
  const outerDiameter = innerDiameter + 2 * totalRefractoryThickness;

  const handlePresetChange = (preset: string) => {
    setRefractoryPreset(preset);
    if (REFRACTORY_PRESETS[preset]) {
      setRefractoryLayers([...REFRACTORY_PRESETS[preset]]);
    }
  };

  const addBurner = () => {
    const newId = Math.max(...burners.map((b) => b.id), 0) + 1;
    setBurners([...burners, { id: newId, position: 0.5, angle: 0, type: 'auxiliary', power: 200 }]);
  };

  const removeBurner = (id: number) => {
    setBurners(burners.filter((b) => b.id !== id));
  };

  // SVG dimensions
  const svgWidth = 700;
  const svgHeight = 350;
  const vesselStartX = 80;
  const vesselStartY = 80;
  const vesselDrawLength = 500;
  const vesselDrawHeight = 150;

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
        {(['dimensions', 'refractory', 'equipment', 'results'] as TabValue[]).map((tab) => (
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

      {/* Dimensions Tab */}
      {activeTab === 'dimensions' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Vessel Dimensions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Vessel Length (m)</Label>
                  <Input
                    type="number"
                    value={vesselLength}
                    onChange={(e) => setVesselLength(parseFloat(e.target.value) || 1)}
                    min={1}
                    max={20}
                    step={0.1}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Inner Diameter (m)</Label>
                  <Input
                    type="number"
                    value={innerDiameter}
                    onChange={(e) => setInnerDiameter(parseFloat(e.target.value) || 0.5)}
                    min={0.5}
                    max={5}
                    step={0.1}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-100 rounded-lg">
                  <div className="text-sm text-gray-500">Outer Diameter</div>
                  <div className="text-xl font-bold">{outerDiameter.toFixed(3)} m</div>
                </div>
                <div className="p-3 bg-gray-100 rounded-lg">
                  <div className="text-sm text-gray-500">L/D Ratio</div>
                  <div className="text-xl font-bold">{(vesselLength / innerDiameter).toFixed(2)}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Operating Conditions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Operating Temperature (C)</Label>
                  <Input
                    type="number"
                    value={operatingTemp}
                    onChange={(e) => setOperatingTemp(parseFloat(e.target.value) || 500)}
                    min={500}
                    max={2000}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Operating Pressure (kPa)</Label>
                  <Input
                    type="number"
                    value={operatingPressure}
                    onChange={(e) => setOperatingPressure(parseFloat(e.target.value) || 50)}
                    min={50}
                    max={500}
                    step={0.1}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Gas Flow Rate (m3/hr)</Label>
                  <Input
                    type="number"
                    value={volumetricFlow}
                    onChange={(e) => setVolumetricFlow(parseFloat(e.target.value) || 100)}
                    min={100}
                    max={50000}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Ambient Temperature (C)</Label>
                  <Input
                    type="number"
                    value={ambientTemp}
                    onChange={(e) => setAmbientTemp(parseFloat(e.target.value) || -20)}
                    min={-20}
                    max={50}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Refractory Tab */}
      {activeTab === 'refractory' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Refractory Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Refractory Preset</Label>
                <select
                  value={refractoryPreset}
                  onChange={(e) => handlePresetChange(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md"
                >
                  <option value="standard">Standard (3-layer)</option>
                  <option value="high_temp">High Temperature (4-layer)</option>
                  <option value="economy">Economy (2-layer)</option>
                </select>
              </div>

              <div className="space-y-2">
                <Label className="text-sm">Refractory Layers</Label>
                <div className="space-y-2">
                  {refractoryLayers.map((layer, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-gray-100 rounded-lg flex items-center justify-between"
                    >
                      <div>
                        <div className="font-medium text-sm">{layer.material}</div>
                        <div className="text-xs text-gray-500">
                          Max: {layer.maxTemp}C | k: {layer.conductivity} W/m-K
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold">{layer.thickness} mm</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-3 bg-blue-50 rounded-lg flex justify-between items-center">
                  <span className="font-medium">Total Thickness</span>
                  <span className="text-xl font-bold text-blue-700">
                    {(totalRefractoryThickness * 1000).toFixed(0)} mm
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Equipment Tab */}
      {activeTab === 'equipment' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4 flex flex-row items-center justify-between">
              <CardTitle className="text-base">Burner Configuration</CardTitle>
              <Button variant="outline" size="sm" onClick={addBurner}>
                Add Burner
              </Button>
            </CardHeader>
            <CardContent className="space-y-3">
              {burners.map((burner) => (
                <div key={burner.id} className="p-3 bg-gray-100 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Burner {burner.id}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeBurner(burner.id)}
                      className="text-red-500"
                    >
                      Remove
                    </Button>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <div>
                      <Label className="text-xs">Position</Label>
                      <Input
                        type="number"
                        value={burner.position}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value) || 0;
                          setBurners(
                            burners.map((b) =>
                              b.id === burner.id
                                ? { ...b, position: Math.min(1, Math.max(0, val)) }
                                : b
                            )
                          );
                        }}
                        min={0}
                        max={1}
                        step={0.05}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Type</Label>
                      <select
                        value={burner.type}
                        onChange={(e) =>
                          setBurners(
                            burners.map((b) =>
                              b.id === burner.id
                                ? { ...b, type: e.target.value as BurnerConfig['type'] }
                                : b
                            )
                          )
                        }
                        className="w-full px-2 py-1 border rounded-md text-sm"
                      >
                        <option value="plasma">Plasma</option>
                        <option value="oxy-fuel">Oxy-Fuel</option>
                        <option value="auxiliary">Auxiliary</option>
                      </select>
                    </div>
                    <div>
                      <Label className="text-xs">Angle (deg)</Label>
                      <Input
                        type="number"
                        value={burner.angle}
                        onChange={(e) =>
                          setBurners(
                            burners.map((b) =>
                              b.id === burner.id
                                ? { ...b, angle: parseFloat(e.target.value) || 0 }
                                : b
                            )
                          )
                        }
                        min={-45}
                        max={45}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Power (kW)</Label>
                      <Input
                        type="number"
                        value={burner.power}
                        onChange={(e) =>
                          setBurners(
                            burners.map((b) =>
                              b.id === burner.id
                                ? { ...b, power: parseFloat(e.target.value) || 0 }
                                : b
                            )
                          )
                        }
                        min={0}
                        max={5000}
                      />
                    </div>
                  </div>
                </div>
              ))}

              <div className="p-3 bg-orange-50 rounded-lg text-center">
                <div className="text-sm text-orange-600">Total Burner Power</div>
                <div className="text-2xl font-bold text-orange-700">
                  {totalBurnerPower.toFixed(0)} kW
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Volume Calculations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg text-center">
                  <div className="text-sm text-blue-600">Net Internal Volume</div>
                  <div className="text-2xl font-bold text-blue-700">
                    {volumes.netVolume.toFixed(2)} m3
                  </div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-sm text-gray-600">Gross Volume</div>
                  <div className="text-2xl font-bold text-gray-700">
                    {volumes.grossVolume.toFixed(2)} m3
                  </div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-sm text-gray-600">Refractory Volume</div>
                  <div className="text-2xl font-bold text-gray-700">
                    {volumes.refractoryVolume.toFixed(2)} m3
                  </div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-sm text-gray-600">Inner Surface Area</div>
                  <div className="text-2xl font-bold text-gray-700">
                    {volumes.surfaceArea.toFixed(1)} m2
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-green-50 border-green-200">
            <CardHeader className="py-4">
              <CardTitle className="text-base text-green-800">Residence Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-white rounded-lg">
                  <div className="text-sm text-green-600">Residence Time</div>
                  <div className="text-2xl font-bold text-green-700">
                    {residenceResults.residenceTime.toFixed(1)} s
                  </div>
                </div>
                <div className="p-3 bg-white rounded-lg">
                  <div className="text-sm text-green-600">Actual Flow</div>
                  <div className="text-2xl font-bold text-green-700">
                    {residenceResults.actualFlow.toFixed(0)} m3/hr
                  </div>
                </div>
                <div className="p-3 bg-white rounded-lg">
                  <div className="text-sm text-green-600">Standard Flow</div>
                  <div className="text-2xl font-bold text-green-700">
                    {residenceResults.standardFlow.toFixed(0)} Nm3/hr
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="py-4">
              <CardTitle className="text-base">Heat Balance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-red-50 rounded-lg text-center">
                  <div className="text-sm text-red-600">Estimated Heat Loss</div>
                  <div className="text-2xl font-bold text-red-700">{heatLoss.toFixed(0)} kW</div>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg text-center">
                  <div className="text-sm text-orange-600">Total Burner Power</div>
                  <div className="text-2xl font-bold text-orange-700">
                    {totalBurnerPower.toFixed(0)} kW
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-gray-100 rounded-lg">
                <div className="flex justify-between items-center">
                  <span>Heat Loss / Burner Power Ratio:</span>
                  <span
                    className={`font-bold ${
                      heatLoss / totalBurnerPower < 0.1
                        ? 'text-green-600'
                        : heatLoss / totalBurnerPower < 0.2
                        ? 'text-amber-600'
                        : 'text-red-600'
                    }`}
                  >
                    {((heatLoss / totalBurnerPower) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Vessel Diagram - Always visible */}
      <Card>
        <CardHeader className="py-4">
          <CardTitle className="text-base">Vessel Diagram</CardTitle>
        </CardHeader>
        <CardContent>
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-64">
            <rect x="0" y="0" width={svgWidth} height={svgHeight} fill="#f8fafc" />
            <rect
              x={vesselStartX}
              y={vesselStartY - 10}
              width={vesselDrawLength}
              height={vesselDrawHeight + 20}
              fill="#64748b"
              stroke="#1e293b"
              strokeWidth={2}
              rx={10}
            />
            <rect
              x={vesselStartX + 10}
              y={vesselStartY}
              width={vesselDrawLength - 20}
              height={vesselDrawHeight}
              fill="#cbd5e1"
              stroke="#94a3b8"
              strokeWidth={1}
              rx={5}
            />
            <rect
              x={vesselStartX + 30}
              y={vesselStartY + 20}
              width={vesselDrawLength - 60}
              height={vesselDrawHeight - 40}
              fill="#fef3c7"
              stroke="#f59e0b"
              strokeWidth={2}
              rx={3}
            />

            {/* Burners */}
            {burners.map((burner) => {
              const x = vesselStartX + burner.position * vesselDrawLength;
              const burnerColor =
                burner.type === 'plasma'
                  ? '#8b5cf6'
                  : burner.type === 'oxy-fuel'
                  ? '#ef4444'
                  : '#22c55e';

              return (
                <g key={burner.id}>
                  <rect
                    x={x - 15}
                    y={vesselStartY - 35}
                    width={30}
                    height={25}
                    fill={burnerColor}
                    stroke="#1e293b"
                    strokeWidth={1}
                    rx={3}
                  />
                  <polygon
                    points={`${x},${vesselStartY + 20} ${x - 8},${vesselStartY + 40} ${x + 8},${vesselStartY + 40}`}
                    fill="#fbbf24"
                    opacity={0.8}
                  />
                </g>
              );
            })}

            {/* Dimensions */}
            <text
              x={vesselStartX + vesselDrawLength / 2}
              y={vesselStartY + vesselDrawHeight + 65}
              textAnchor="middle"
              fill="#64748b"
              fontSize="11"
            >
              L = {vesselLength.toFixed(1)} m
            </text>
            <text
              x={vesselStartX + vesselDrawLength + 45}
              y={vesselStartY + vesselDrawHeight / 2 + 4}
              textAnchor="start"
              fill="#64748b"
              fontSize="10"
            >
              ID = {(innerDiameter * 1000).toFixed(0)} mm
            </text>
          </svg>
        </CardContent>
      </Card>

      <div className="text-xs text-gray-500">
        Note: This is a preliminary design tool. Final vessel design should be verified by a
        qualified pressure vessel engineer according to applicable codes (ASME, EN, etc.).
      </div>
    </div>
  );
}

export default TRCVesselDesignerCalculator;
