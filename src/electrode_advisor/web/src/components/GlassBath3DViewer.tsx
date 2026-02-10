/**
 * Glass Bath 3D Visualization for Electrode Advisor
 *
 * Provides interactive 3D rendering of glass bath geometry with:
 * - Rectangular/cylindrical bath geometry
 * - Electrode placement in 3D space
 * - Conductive path visualization
 * - Transparency controls
 * - Camera view presets (top, side, front, perspective)
 * - Current density heatmap overlay
 *
 * See issue #606.
 */

import { useState, useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import {
  OrbitControls,
  PerspectiveCamera,
  Text,
  Line,
  Html,
} from '@react-three/drei';
import * as THREE from 'three';

// ---- Types ----

interface ElectrodeConfig {
  type: string;
  position: [number, number, number];
  length: number;
  diameter: number;
  wornLength: number;
  current: number;
}

interface BathConfig {
  shape: 'rectangular' | 'cylindrical';
  width: number;
  depth: number;
  height: number;
  glassLevel: number;
}

interface ViewPreset {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
}

interface GlassBath3DViewerProps {
  electrodeType: string;
  currentLength: number;
  wornLength: number;
  operatingCurrent: number;
  plasmaTemp: number;
}

// ---- Constants ----

const VIEW_PRESETS: ViewPreset[] = [
  { name: 'Perspective', position: [4, 3, 4], target: [0, 0, 0] },
  { name: 'Top', position: [0, 6, 0.01], target: [0, 0, 0] },
  { name: 'Front', position: [0, 0, 6], target: [0, 0, 0] },
  { name: 'Side', position: [6, 0, 0], target: [0, 0, 0] },
  { name: 'Bottom', position: [0, -6, 0.01], target: [0, 0, 0] },
];

const HEATMAP_COLORS = [
  new THREE.Color(0x0000ff), // cold - blue
  new THREE.Color(0x00ffff), // cool - cyan
  new THREE.Color(0x00ff00), // warm - green
  new THREE.Color(0xffff00), // hot - yellow
  new THREE.Color(0xff0000), // very hot - red
];

// ---- Helper functions ----

function getHeatmapColor(intensity: number): THREE.Color {
  const t = Math.max(0, Math.min(1, intensity));
  const idx = t * (HEATMAP_COLORS.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.min(lower + 1, HEATMAP_COLORS.length - 1);
  const frac = idx - lower;
  return HEATMAP_COLORS[lower].clone().lerp(HEATMAP_COLORS[upper], frac);
}

function computeCurrentDensity(
  point: THREE.Vector3,
  electrodes: ElectrodeConfig[]
): number {
  let totalDensity = 0;
  for (const electrode of electrodes) {
    const tipPos = new THREE.Vector3(
      electrode.position[0],
      electrode.position[1] - (electrode.length - electrode.wornLength) * 0.001,
      electrode.position[2]
    );
    const dist = point.distanceTo(tipPos);
    if (dist < 0.01) continue;
    const density = (electrode.current * 0.001) / (4 * Math.PI * dist * dist);
    totalDensity += density;
  }
  return totalDensity;
}

// ---- 3D Components ----

function GlassBathMesh({
  config,
  opacity,
}: {
  config: BathConfig;
  opacity: number;
}) {
  const { width, depth, height, glassLevel } = config;

  return (
    <group>
      {/* Bath walls - wireframe for structure */}
      <mesh position={[0, -height / 2, 0]}>
        {config.shape === 'rectangular' ? (
          <boxGeometry args={[width, height, depth]} />
        ) : (
          <cylinderGeometry args={[width / 2, width / 2, height, 32]} />
        )}
        <meshStandardMaterial
          color="#8b8b8b"
          transparent
          opacity={opacity * 0.3}
          side={THREE.DoubleSide}
          wireframe={false}
        />
      </mesh>

      {/* Wireframe overlay */}
      <mesh position={[0, -height / 2, 0]}>
        {config.shape === 'rectangular' ? (
          <boxGeometry args={[width, height, depth]} />
        ) : (
          <cylinderGeometry args={[width / 2, width / 2, height, 32]} />
        )}
        <meshStandardMaterial
          color="#cccccc"
          wireframe
          transparent
          opacity={opacity * 0.5}
        />
      </mesh>

      {/* Glass/slag level */}
      <mesh position={[0, -height + glassLevel / 2, 0]}>
        {config.shape === 'rectangular' ? (
          <boxGeometry args={[width * 0.99, glassLevel, depth * 0.99]} />
        ) : (
          <cylinderGeometry
            args={[width / 2 * 0.99, width / 2 * 0.99, glassLevel, 32]}
          />
        )}
        <meshStandardMaterial
          color="#8B4513"
          transparent
          opacity={opacity * 0.6}
        />
      </mesh>

      {/* Glass surface indicator */}
      <mesh
        position={[0, -height + glassLevel, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        {config.shape === 'rectangular' ? (
          <planeGeometry args={[width * 0.99, depth * 0.99]} />
        ) : (
          <circleGeometry args={[width / 2 * 0.99, 32]} />
        )}
        <meshStandardMaterial
          color="#D4A017"
          transparent
          opacity={opacity * 0.7}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

function ElectrodeMesh({
  config,
  showArc,
}: {
  config: ElectrodeConfig;
  showArc: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const arcRef = useRef<THREE.PointLight>(null);

  const effectiveLength = (config.length - config.wornLength) * 0.001; // mm to m
  const radius = config.diameter * 0.0005; // mm to m, radius

  useFrame(({ clock }) => {
    if (arcRef.current && showArc) {
      arcRef.current.intensity = 2 + Math.sin(clock.elapsedTime * 10) * 0.5;
    }
  });

  const tipY = config.position[1] - effectiveLength;

  return (
    <group position={config.position}>
      {/* Electrode holder (top clamp) */}
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[radius * 2, radius * 2, 0.1, 16]} />
        <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Electrode body */}
      <mesh
        ref={meshRef}
        position={[0, -effectiveLength / 2, 0]}
      >
        <cylinderGeometry args={[radius, radius, effectiveLength, 16]} />
        <meshStandardMaterial
          color="#444444"
          metalness={0.6}
          roughness={0.4}
        />
      </mesh>

      {/* Worn tip indicator (red band) */}
      <mesh position={[0, -effectiveLength + 0.01, 0]}>
        <cylinderGeometry args={[radius * 1.01, radius * 1.01, 0.02, 16]} />
        <meshStandardMaterial color="#ff4444" emissive="#ff2222" emissiveIntensity={0.3} />
      </mesh>

      {/* Plasma arc glow */}
      {showArc && (
        <>
          <pointLight
            ref={arcRef}
            position={[0, -effectiveLength - 0.05, 0]}
            color="#FFA500"
            intensity={3}
            distance={0.5}
          />
          <mesh position={[0, -effectiveLength - 0.03, 0]}>
            <sphereGeometry args={[0.04, 16, 16]} />
            <meshStandardMaterial
              color="#FFD700"
              emissive="#FFA500"
              emissiveIntensity={2}
              transparent
              opacity={0.8}
            />
          </mesh>
        </>
      )}

      {/* Label */}
      <Html position={[0, 0.15, 0]} center>
        <div
          style={{
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '2px 6px',
            borderRadius: 4,
            fontSize: 10,
            whiteSpace: 'nowrap',
          }}
        >
          {config.type} ({config.current}A)
        </div>
      </Html>
    </group>
  );
}

function ConductivePaths({
  electrodes,
  bathConfig,
}: {
  electrodes: ElectrodeConfig[];
  bathConfig: BathConfig;
}) {
  const paths = useMemo(() => {
    const result: Array<{ points: [number, number, number][]; color: string }> = [];

    for (let i = 0; i < electrodes.length; i++) {
      for (let j = i + 1; j < electrodes.length; j++) {
        const e1 = electrodes[i];
        const e2 = electrodes[j];
        const tip1Y = e1.position[1] - (e1.length - e1.wornLength) * 0.001;
        const tip2Y = e2.position[1] - (e2.length - e2.wornLength) * 0.001;

        // Create curved conductive path through glass
        const glassY = -bathConfig.height + bathConfig.glassLevel * 0.5;
        const midX = (e1.position[0] + e2.position[0]) / 2;
        const midZ = (e1.position[2] + e2.position[2]) / 2;

        const points: [number, number, number][] = [
          [e1.position[0], tip1Y, e1.position[2]],
          [
            (e1.position[0] + midX) / 2,
            glassY,
            (e1.position[2] + midZ) / 2,
          ],
          [midX, glassY - 0.2, midZ],
          [
            (e2.position[0] + midX) / 2,
            glassY,
            (e2.position[2] + midZ) / 2,
          ],
          [e2.position[0], tip2Y, e2.position[2]],
        ];

        const avgCurrent = (e1.current + e2.current) / 2;
        const intensity = Math.min(1, avgCurrent / 5000);
        const color = getHeatmapColor(intensity);
        result.push({
          points,
          color: `#${color.getHexString()}`,
        });
      }
    }

    return result;
  }, [electrodes, bathConfig]);

  return (
    <group>
      {paths.map((path, idx) => (
        <Line
          key={idx}
          points={path.points}
          color={path.color}
          lineWidth={2}
          dashed
          dashScale={10}
          dashSize={0.1}
          gapSize={0.05}
        />
      ))}
    </group>
  );
}

function HeatmapOverlay({
  electrodes,
  bathConfig,
  visible,
}: {
  electrodes: ElectrodeConfig[];
  bathConfig: BathConfig;
  visible: boolean;
}) {
  const geometry = useMemo(() => {
    if (!visible) return null;

    const resolution = 20;
    const { width, depth, height, glassLevel } = bathConfig;
    const glassY = -height + glassLevel * 0.5;

    const positions: number[] = [];
    const colors: number[] = [];

    // Create a grid of points at the glass level
    for (let ix = 0; ix < resolution; ix++) {
      for (let iz = 0; iz < resolution; iz++) {
        const x = (ix / (resolution - 1) - 0.5) * width * 0.9;
        const z = (iz / (resolution - 1) - 0.5) * depth * 0.9;
        const point = new THREE.Vector3(x, glassY, z);
        const density = computeCurrentDensity(point, electrodes);
        const normalizedDensity = Math.min(1, density / 50);
        const color = getHeatmapColor(normalizedDensity);

        positions.push(x, glassY + 0.01, z);
        colors.push(color.r, color.g, color.b);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3)
    );
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // Create indices for triangle strip
    const indices: number[] = [];
    for (let ix = 0; ix < resolution - 1; ix++) {
      for (let iz = 0; iz < resolution - 1; iz++) {
        const a = ix * resolution + iz;
        const b = (ix + 1) * resolution + iz;
        const c = ix * resolution + (iz + 1);
        const d = (ix + 1) * resolution + (iz + 1);
        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }
    geo.setIndex(indices);
    geo.computeVertexNormals();

    return geo;
  }, [electrodes, bathConfig, visible]);

  if (!visible || !geometry) return null;

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        transparent
        opacity={0.6}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function ScaleIndicator() {
  return (
    <group position={[-2.5, -2.5, -2.5]}>
      {/* X axis */}
      <Line points={[[0, 0, 0], [1, 0, 0]]} color="red" lineWidth={2} />
      <Text position={[1.15, 0, 0]} fontSize={0.12} color="red">
        X
      </Text>
      {/* Y axis */}
      <Line points={[[0, 0, 0], [0, 1, 0]]} color="green" lineWidth={2} />
      <Text position={[0, 1.15, 0]} fontSize={0.12} color="green">
        Y
      </Text>
      {/* Z axis */}
      <Line points={[[0, 0, 0], [0, 0, 1]]} color="blue" lineWidth={2} />
      <Text position={[0, 0, 1.15]} fontSize={0.12} color="blue">
        Z
      </Text>
      {/* Scale label */}
      <Text position={[0.5, -0.15, 0]} fontSize={0.08} color="#999">
        1m
      </Text>
    </group>
  );
}

function Scene({
  bathConfig,
  electrodes,
  showHeatmap,
  showPaths,
  showArc,
  opacity,
  viewPreset,
}: {
  bathConfig: BathConfig;
  electrodes: ElectrodeConfig[];
  showHeatmap: boolean;
  showPaths: boolean;
  showArc: boolean;
  opacity: number;
  viewPreset: ViewPreset;
}) {
  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={viewPreset.position}
        fov={50}
      />
      <OrbitControls target={viewPreset.target} enableDamping />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} castShadow />
      <directionalLight position={[-3, 3, -3]} intensity={0.3} />

      {/* Grid floor */}
      <gridHelper args={[6, 12, '#555555', '#333333']} position={[0, -3, 0]} />

      {/* Glass bath */}
      <GlassBathMesh config={bathConfig} opacity={opacity} />

      {/* Electrodes */}
      {electrodes.map((electrode, idx) => (
        <ElectrodeMesh key={idx} config={electrode} showArc={showArc} />
      ))}

      {/* Conductive paths */}
      {showPaths && (
        <ConductivePaths electrodes={electrodes} bathConfig={bathConfig} />
      )}

      {/* Heatmap */}
      <HeatmapOverlay
        electrodes={electrodes}
        bathConfig={bathConfig}
        visible={showHeatmap}
      />

      {/* Scale indicator */}
      <ScaleIndicator />
    </>
  );
}

// ---- Main Component ----

export function GlassBath3DViewer({
  electrodeType,
  currentLength,
  wornLength,
  operatingCurrent,
  plasmaTemp,
}: GlassBath3DViewerProps) {
  const [bathShape, setBathShape] = useState<'rectangular' | 'cylindrical'>(
    'rectangular'
  );
  const [bathWidth, setBathWidth] = useState(3.0);
  const [bathDepth, setBathDepth] = useState(2.0);
  const [bathHeight, setBathHeight] = useState(2.5);
  const [glassLevel, setGlassLevel] = useState(1.5);

  const [numElectrodes, setNumElectrodes] = useState(3);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showPaths, setShowPaths] = useState(true);
  const [showArc, setShowArc] = useState(true);
  const [opacity, setOpacity] = useState(0.7);
  const [activePreset, setActivePreset] = useState(0);

  const bathConfig: BathConfig = useMemo(
    () => ({
      shape: bathShape,
      width: bathWidth,
      depth: bathDepth,
      height: bathHeight,
      glassLevel,
    }),
    [bathShape, bathWidth, bathDepth, bathHeight, glassLevel]
  );

  // Generate electrode positions evenly spaced
  const electrodes: ElectrodeConfig[] = useMemo(() => {
    const result: ElectrodeConfig[] = [];
    const spacing = bathWidth * 0.6;

    for (let i = 0; i < numElectrodes; i++) {
      const angle = (i / numElectrodes) * Math.PI * 2;
      const x =
        numElectrodes === 1
          ? 0
          : Math.cos(angle) * spacing * 0.4;
      const z =
        numElectrodes === 1
          ? 0
          : Math.sin(angle) * spacing * 0.4;

      result.push({
        type: electrodeType,
        position: [x, 0.1, z],
        length: currentLength,
        diameter: 150, // mm
        wornLength,
        current: operatingCurrent / numElectrodes,
      });
    }

    return result;
  }, [
    numElectrodes,
    electrodeType,
    currentLength,
    wornLength,
    operatingCurrent,
    bathWidth,
  ]);

  const handlePresetChange = useCallback((idx: number) => {
    setActivePreset(idx);
  }, []);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Bath Configuration */}
        <div className="bg-white rounded-lg shadow border p-4">
          <h3 className="font-semibold text-sm mb-3">Bath Configuration</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Shape</label>
              <select
                value={bathShape}
                onChange={(e) =>
                  setBathShape(e.target.value as 'rectangular' | 'cylindrical')
                }
                className="flex-1 text-sm border rounded px-2 py-1"
              >
                <option value="rectangular">Rectangular</option>
                <option value="cylindrical">Cylindrical</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Width (m)</label>
              <input
                type="range"
                min={1}
                max={6}
                step={0.1}
                value={bathWidth}
                onChange={(e) => setBathWidth(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">{bathWidth.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Depth (m)</label>
              <input
                type="range"
                min={1}
                max={6}
                step={0.1}
                value={bathDepth}
                onChange={(e) => setBathDepth(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">{bathDepth.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Height (m)</label>
              <input
                type="range"
                min={1}
                max={5}
                step={0.1}
                value={bathHeight}
                onChange={(e) => setBathHeight(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">{bathHeight.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Glass Level (m)</label>
              <input
                type="range"
                min={0.1}
                max={bathHeight}
                step={0.1}
                value={glassLevel}
                onChange={(e) => setGlassLevel(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">{glassLevel.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Electrodes</label>
              <input
                type="range"
                min={1}
                max={6}
                step={1}
                value={numElectrodes}
                onChange={(e) => setNumElectrodes(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">{numElectrodes}</span>
            </div>
          </div>
        </div>

        {/* Visualization Controls */}
        <div className="bg-white rounded-lg shadow border p-4">
          <h3 className="font-semibold text-sm mb-3">Visualization Controls</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-24">Transparency</label>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs w-10 text-right">
                {(opacity * 100).toFixed(0)}%
              </span>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                  className="w-3.5 h-3.5"
                />
                Heatmap
              </label>
              <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={showPaths}
                  onChange={(e) => setShowPaths(e.target.checked)}
                  className="w-3.5 h-3.5"
                />
                Conductive Paths
              </label>
              <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={showArc}
                  onChange={(e) => setShowArc(e.target.checked)}
                  className="w-3.5 h-3.5"
                />
                Plasma Arc
              </label>
            </div>

            {/* Camera Presets */}
            <div className="pt-1">
              <label className="text-xs text-gray-600 block mb-1">Camera View</label>
              <div className="flex flex-wrap gap-1">
                {VIEW_PRESETS.map((preset, idx) => (
                  <button
                    key={preset.name}
                    onClick={() => handlePresetChange(idx)}
                    className={`px-2 py-1 text-xs rounded ${
                      activePreset === idx
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Legend */}
            <div className="pt-1">
              <label className="text-xs text-gray-600 block mb-1">
                Current Density Legend
              </label>
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-500">Low</span>
                <div
                  className="flex-1 h-3 rounded"
                  style={{
                    background:
                      'linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)',
                  }}
                />
                <span className="text-xs text-gray-500">High</span>
              </div>
            </div>

            {/* Info */}
            <div className="text-xs text-gray-500 pt-1">
              <p>Electrode: {electrodeType} @ {operatingCurrent}A</p>
              <p>
                Effective length:{' '}
                {(currentLength - wornLength).toFixed(0)} mm |
                Temp: {plasmaTemp} C
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="bg-gray-900 rounded-lg overflow-hidden" style={{ height: 500 }}>
        <Canvas shadows>
          <Scene
            bathConfig={bathConfig}
            electrodes={electrodes}
            showHeatmap={showHeatmap}
            showPaths={showPaths}
            showArc={showArc}
            opacity={opacity}
            viewPreset={VIEW_PRESETS[activePreset]}
          />
        </Canvas>
      </div>

      <p className="text-xs text-gray-500">
        Drag to rotate, scroll to zoom, right-click drag to pan. Conductive
        paths show estimated current flow between electrodes through the glass
        melt. Heatmap shows relative current density distribution at the glass
        surface level.
      </p>
    </div>
  );
}

export default GlassBath3DViewer;
