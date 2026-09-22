/**
 * Landing-dispersion canvas for the web Variation tab (#4120 V3):
 * top-down scatter (x = lateral, + right; y = carry) of every
 * successful run with the 2-sigma dispersion ellipse overlaid.
 */

import { useEffect, useMemo, useRef } from "react";

import {
  signedDistance,
  type TargetRegionTs,
} from "../model/targets";
import { courseColors } from "../model/course";
import { type VariationDatasetTs } from "../model/variation";
import {
  dispersionEllipse,
  pairedLandingPoints,
} from "../model/variationAnalysis";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";

export function LandingCanvas({
  dataset,
  target,
  ensemble,
}: {
  dataset: VariationDatasetTs;
  /** Target region (#4125 H7b): dashed overlay + hold-% headline. */
  target?: TargetRegionTs;
  ensemble?: SwingVariationResultTs | null;
}): JSX.Element {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const landingPoints = useMemo(() => pairedLandingPoints(dataset), [dataset]);
  const ellipse = useMemo(() => dispersionEllipse(dataset), [dataset]);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const points: Array<[number, number]> = landingPoints.map(
      (point) => [point.lateralM, point.carryM],
    );
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    if (points.length === 0) return;
    const pad = 2.0;
    const reach = ellipse ? ellipse.semiMajorM : 0;
    // Window includes the target region so its boundary never clips.
    const tx = target
      ? target.kind === "green"
        ? [target.lateralM - target.radiusM, target.lateralM + target.radiusM]
        : [-target.halfWidthM, target.halfWidthM]
      : [];
    const ty = target
      ? target.kind === "green"
        ? [target.distanceM - target.radiusM, target.distanceM + target.radiusM]
        : [
            target.distanceM - target.bandHalfLengthM,
            target.distanceM + target.bandHalfLengthM,
          ]
      : [];
    // ⚡ Bolt Optimization: Calculate bounds with a single pass instead of spreading large arrays
    // This avoids "Maximum call stack size exceeded" errors and O(N) garbage collection on hot renders
    const ellipseCenterLat = ellipse?.centerLateralM ?? 0;
    const ellipseCenterCarry = ellipse?.centerCarryM ?? 0;
    let minXRaw = Math.min(...tx, ellipseCenterLat - reach);
    let maxXRaw = Math.max(...tx, ellipseCenterLat + reach);
    let minYRaw = Math.min(...ty, ellipseCenterCarry - reach);
    let maxYRaw = Math.max(...ty, ellipseCenterCarry + reach);
    for (let i = 0; i < points.length; i++) {
      const pX = points[i][0];
      const pY = points[i][1];
      if (pX < minXRaw) minXRaw = pX;
      if (pX > maxXRaw) maxXRaw = pX;
      if (pY < minYRaw) minYRaw = pY;
      if (pY > maxYRaw) maxYRaw = pY;
    }
    const minX = minXRaw - pad;
    const maxX = maxXRaw + pad;
    const minY = minYRaw - pad;
    const maxY = maxYRaw + pad;
    const scale = Math.min(
      (width - 40) / (maxX - minX || 1),
      (height - 40) / (maxY - minY || 1),
    );
    const px = (x: number) => 20 + (x - minX) * scale;
    const py = (y: number) => height - 20 - (y - minY) * scale;

    const course = courseColors();
    for (const [x, y] of points) {
      // Landing scatter: color by target containment when a target is
      // set (#4125 H7b) — holding shots in the green tone.
      ctx.fillStyle = target
        ? signedDistance(target, y, x) <= 0
          ? course.green
          : "rgba(56, 189, 248, 0.65)"
        : "rgba(56, 189, 248, 0.65)";
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3, 0, 2 * Math.PI);
      ctx.fill();
    }
    if (target) {
      // Dashed target boundary (canvas x = lateral, y = carry).
      ctx.strokeStyle = course.flag;
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      if (target.kind === "green") {
        ctx.ellipse(
          px(target.lateralM),
          py(target.distanceM),
          target.radiusM * scale,
          target.radiusM * scale,
          0,
          0,
          2 * Math.PI,
        );
      } else {
        ctx.rect(
          px(-target.halfWidthM),
          py(target.distanceM + target.bandHalfLengthM),
          2 * target.halfWidthM * scale,
          2 * target.bandHalfLengthM * scale,
        );
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Hold-% headline: fraction of shots inside the target.
      // ⚡ Bolt Optimization: Replace Math.max(...spread) and chained maps
      // with a single-pass loop to eliminate array allocations on render
      let held = 0;
      let total = 0;
      for (let i = 0; i < points.length; i++) {
        const lateral = points[i][0];
        const carry = points[i][1];
        if (!Number.isFinite(carry) || !Number.isFinite(lateral)) continue;
        total += 1;
        if (signedDistance(target, carry, lateral) <= 0) held += 1;
      }
      const pct = total ? ((100 * held) / total).toFixed(0) : "–";
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px sans-serif";
      ctx.fillText(`${held}/${total} shots hold the target (${pct}%)`, 8, 14);
    }
    if (ellipse) {
      if (
        ellipse.diagnostic &&
        !ellipse.diagnostic.isNormal &&
        ellipse.convexHull &&
        ellipse.convexHull.length > 2
      ) {
        // Non-normal bivariate distribution fallback: draw convex hull envelope
        ctx.strokeStyle = "#eab308";
        ctx.fillStyle = "rgba(234, 179, 8, 0.08)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        const first = ellipse.convexHull[0];
        ctx.moveTo(px(first[0]), py(first[1]));
        for (let i = 1; i < ellipse.convexHull.length; i++) {
          ctx.lineTo(px(ellipse.convexHull[i][0]), py(ellipse.convexHull[i][1]));
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = "#eb6a3c";
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        // Engine angle is CCW from the carry axis; canvas x = lateral.
        ctx.ellipse(
          px(ellipse.centerLateralM),
          py(ellipse.centerCarryM),
          ellipse.semiMajorM * scale,
          ellipse.semiMinorM * scale,
          -((90.0 - ellipse.angleDeg) * Math.PI) / 180.0,
          0,
          2 * Math.PI,
        );
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText("lateral [m] (+ right) →", width / 2 - 50, height - 4);
    ctx.save();
    ctx.translate(12, height / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("carry [m] →", 0, 0);
    ctx.restore();
  }, [dataset, ellipse, landingPoints, target]);
  const counts = ensemble && {
    hits: ensemble.runs.filter((run) => run.status === "evaluated_hit").length,
    misses: ensemble.runs.filter((run) => run.status === "evaluated_no_impact").length,
    failures: ensemble.runs.filter((run) => run.status === "numerical_failure").length,
  };
  const landingCount = landingPoints.length;
  return (
    <div className="space-y-2">
      <canvas
        ref={ref}
        width={560}
        height={420}
        className="w-full rounded-lg border border-slate-800 bg-slate-950/60"
        title={
          ellipse?.diagnostic && !ellipse.diagnostic.isNormal
            ? "Landing positions with convex hull envelope fallback (Mardia non-normal distribution)."
            : "Landing positions of every evaluated hit, viewed from above; the dashed ellipse is the 2-sigma dispersion fit."
        }
      />
      <p className="text-xs text-slate-400" role="status">
        {counts
          ? `Hits: ${counts.hits} · No impact: ${counts.misses} · Numerical failures: ${counts.failures} · Plotted landings: ${landingCount}. Misses and failures have no fabricated landing coordinates.`
          : `Evaluated landings: ${landingCount}/${dataset.plan.nRuns}. Scalar studies do not expose a geometric no-impact cohort.`}
        {ellipse?.diagnostic && !ellipse.diagnostic.isNormal && (
          <span className="ml-2 font-medium text-amber-400">
            ⚠ Mardia non-normal (p_skew={ellipse.diagnostic.skewnessPValue.toFixed(3)}, p_kurt={ellipse.diagnostic.kurtosisPValue.toFixed(3)}): displaying convex hull envelope fallback.
          </span>
        )}
      </p>
    </div>
  );
}
