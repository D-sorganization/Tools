import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { RotatingBaseRun } from "../rotatingBaseRunCatalog";

interface ChartDefinition {
  title: string;
  yLabel: string;
  lines: Array<{ key: string; label: string; color: string }>;
}

const CHARTS: ChartDefinition[] = [
  {
    title: "Club Contact Power",
    yLabel: "Power (W)",
    lines: [{ key: "contactPower", label: "Contact Power", color: "#f5c451" }],
  },
  {
    title: "Force-Generated Couple",
    yLabel: "Couple (N·m)",
    lines: [{ key: "forceCouple", label: "Force Couple", color: "#f27d52" }],
  },
  {
    title: "Torso and Club Rates",
    yLabel: "Rate (rad/s)",
    lines: [
      { key: "torsoRate", label: "Torso Rate", color: "#4fc3f7" },
      { key: "clubRate", label: "Club Rate", color: "#ba9cff" },
    ],
  },
  {
    title: "Distal Segment Kinetic Energy",
    yLabel: "Energy (J)",
    lines: [{ key: "distalEnergy", label: "Distal Energy", color: "#69d49f" }],
  },
  {
    title: "Independent Grip-Force Magnitudes",
    yLabel: "Force (N)",
    lines: [
      { key: "leadGrip", label: "Lead Grip", color: "#66a5ff" },
      { key: "trailGrip", label: "Trail Grip", color: "#ff8db3" },
    ],
  },
];

export function RotatingBaseTraceCharts({ run }: { run: RotatingBaseRun }) {
  const trace = run.trace;
  const data = trace.time_s.map((time, index) => ({
    time,
    contactPower: trace.contact_power_on_club_w[index],
    forceCouple: trace.force_generated_couple_nm[index],
    torsoRate: trace.torso_rate_rad_s[index],
    clubRate: trace.club_rate_rad_s[index],
    distalEnergy: trace.distal_segment_kinetic_energy_j[index],
    leadGrip: Math.hypot(...trace.force_on_club_n[index][0]),
    trailGrip: Math.hypot(...trace.force_on_club_n[index][1]),
  }));

  return (
    <section className="rotating-base-study__traces" aria-labelledby="rotating-base-traces-title">
      <h3 id="rotating-base-traces-title">Time-Resolved Registered Evidence</h3>
      <p>
        Full-resolution traces from the canonical Python provider. Invalid and
        adverse rows remain selectable and are not removed from these views.
      </p>
      <div className="rotating-base-study__trace-grid">
        {CHARTS.map((chart) => (
          <figure key={chart.title} aria-label={chart.title}>
            <figcaption>{chart.title}</figcaption>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 8 }}>
                <CartesianGrid stroke="#314766" strokeDasharray="3 3" />
                <XAxis
                  dataKey="time"
                  label={{ value: "Time (s)", position: "insideBottom", offset: -12 }}
                  stroke="#b8c9e2"
                  tickFormatter={(value: number) => value.toFixed(3)}
                />
                <YAxis
                  label={{ value: chart.yLabel, angle: -90, position: "insideLeft" }}
                  stroke="#b8c9e2"
                  width={72}
                />
                <Tooltip
                  formatter={(value) =>
                    typeof value === "number" ? value.toPrecision(6) : String(value)
                  }
                  labelFormatter={(value) => `Time ${Number(value).toFixed(4)} s`}
                />
                <Legend verticalAlign="top" />
                {chart.lines.map((line) => (
                  <Line
                    key={line.key}
                    dataKey={line.key}
                    name={line.label}
                    stroke={line.color}
                    dot={false}
                    isAnimationActive={false}
                    strokeWidth={2}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </figure>
        ))}
      </div>
    </section>
  );
}
