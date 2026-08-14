/**
 * Unit systems — TypeScript mirror of `src/rate_of_closure/units.py`.
 *
 * The model is always canonical (mph, deg/s, mm, µs); the UI converts
 * at the edge via these factor tables: canonical = displayed * factor.
 * Also mirrors the per-field hover guidance (suggested golf-swing range
 * plus the source of the suggestion).
 */

export const SPEED_UNITS: Record<string, number> = {
  mph: 1.0,
  "m/s": 2.236936292054402,
  "km/h": 0.621371192237334,
  "ft/s": 0.681818181818182,
};

export const ROTATION_UNITS: Record<string, number> = {
  "deg/s": 1.0,
  "rad/s": 57.29577951308232,
  rpm: 6.0,
};

export const LENGTH_UNITS: Record<string, number> = {
  mm: 1.0,
  cm: 10.0,
  in: 25.4,
};

/**
 * Ball-flight distances (#4125 H6): displayed * factor = canonical
 * metres — internal physics stays SI. Yards listed FIRST: the
 * drop-downs default to the first entry, so distances read in yards
 * out of the box (user direction).
 */
export const DISTANCE_UNITS: Record<string, number> = {
  yd: 0.9144,
  m: 1.0,
};

export type Quantity = "speed" | "rotation" | "length" | "distance";

export const QUANTITY_UNITS: Record<Quantity, Record<string, number>> = {
  speed: SPEED_UNITS,
  rotation: ROTATION_UNITS,
  length: LENGTH_UNITS,
  distance: DISTANCE_UNITS,
};

/** A canonical-metres distance formatted in the given display unit. */
export function formatDistanceM(
  valueM: number,
  unit: string,
  decimals = 1,
): string {
  return `${(valueM / DISTANCE_UNITS[unit]).toFixed(decimals)} ${unit}`;
}

export function toCanonical(
  quantity: Quantity,
  unit: string,
  value: number,
): number {
  return value * QUANTITY_UNITS[quantity][unit];
}

export function fromCanonical(
  quantity: Quantity,
  unit: string,
  value: number,
): number {
  return value / QUANTITY_UNITS[quantity][unit];
}

/** Hover guidance per scenario field (canonical-unit ranges + source). */
export const FIELD_GUIDANCE: Record<string, string> = {
  clubheadSpeedMph:
    "Suggested range: 80-130 mph driver clubhead speed (tour average " +
    "near 113 mph; strong amateurs 90-105). Source: openly published " +
    "tour launch-monitor averages. Reference frame: speed magnitude of " +
    "the clubhead reference point; +x is down the target line.",
  omegaPlaneDps:
    "Suggested range: 1,800-2,400 deg/s swing-plane rotation at impact " +
    "for skilled players. Source: 3-D motion-capture studies collected " +
    "in the AffineDrift closure-rate dossier. Reference frame: right-hand " +
    "rotation about the oriented swing-plane normal (SPV).",
  omegaShaftDps:
    "Suggested range: 652-2,432 deg/s about the shaft (tour driver mean " +
    "1,307 +/- 304, n = 94). Source: Cheetham 2014, via the AffineDrift " +
    "closure-rate dossier. Reference frame: right-hand rotation about " +
    "the shaft axis from grip toward clubhead (HTV).",
  lieAngleDeg:
    "Suggested range: 55-62 deg for a driver delivered near its static " +
    "lie; 90 deg makes the shaft vertical to isolate pure horizontal " +
    "closure. Source: published driver spec sheets.",
  comToFaceMm:
    "Suggested range: 25-50 mm from the geometric center forward to the " +
    "face center for modern drivers; 40 mm is the AffineDrift " +
    "worked-example value. Source: openly published launch-monitor " +
    "material.",
  impactOffsetToeMm:
    "Suggested range: within +/-15 mm of face center for reasonable " +
    "strikes; gear-effect studies use up to +/-20 mm. Source: published " +
    "robot-test impact maps.",
  impactOffsetHighMm:
    "Suggested range: within +/-10 mm of face center vertically. " +
    "Source: published robot-test impact maps.",
  contactDurationUs:
    "Suggested range: 400-500 microseconds of ball-face contact for a " +
    "driver. Source: openly published high-speed impact studies.",
  clubSelection:
    "Suggested range: pick the club closest to yours; selecting one " +
    "drives GC-to-face and lie from its spec (your overrides are kept). " +
    "Source: typical published manufacturer specs, normalized to SI in " +
    "the club library.",
  clubLoftDeg:
    "Suggested range: 8-13 deg drivers, 15-19 deg fairway woods and " +
    "hybrids, 21-45 deg irons, 46-64 deg wedges. Source: typical " +
    "published manufacturer spec sheets.",
  faceCurvatureEnabled:
    "Suggested range: on for drivers, fairway woods, and hybrids " +
    "(curved faces); off for irons, wedges, and putters (flat faces). " +
    "Source: typical published fitting references.",
  faceBulgeRadiusMm:
    "Suggested range: 250-330 mm horizontal (heel-toe) face radius for " +
    "a modern driver (10-13 in). Source: typical published fitting " +
    "values.",
  faceRollRadiusMm:
    "Suggested range: 250-330 mm vertical (crown-sole) face radius, " +
    "usually similar to bulge on drivers. Source: typical published " +
    "fitting values.",
  swingSource:
    "Suggested range: Manual Scenario replays the explorer's " +
    "constant-twist delivery; Double and Triple Pendulum generate gravity-driven " +
    "swing on the oriented plane. Source: classic double-pendulum golf " +
    "models (Cochran & Stobbs; Jorgensen, The Physics of Golf).",
  planeYawDeg:
    "Suggested range: -20 to +20 deg rotation of the swing plane about " +
    "the vertical (aim left/right of the target line). Source: 3-D " +
    "swing-plane studies collected in the AffineDrift closure-rate " +
    "dossier. Reference frame: rotate the plane about world +y (up); " +
    "+x is the target line and +z is right of target.",
  planeSideTiltDeg:
    "Suggested range: -60 to -35 deg side tilt for a driver (a vertical " +
    "plane is 0; tour driver swing planes lean roughly 45-55 deg from " +
    "vertical). Source: published 3-D swing-plane measurements. Reference " +
    "frame: roll about the plane's downrange axis after yaw; negative " +
    "leans a right-handed driver's plane toward the golfer.",
  planeForwardTiltDeg:
    "Suggested range: -10 to +10 deg forward/back tilt of the in-plane " +
    "upright axis. Source: published 3-D swing-plane measurements. " +
    "Reference frame: pitch about the yawed-and-side-tilted plane's local " +
    "lateral axis; positive tips the upright axis downrange.",
  impactTimeScrub:
    "Suggested range: anywhere inside the swing; the default is the " +
    "instant of maximum clubhead speed. Scrubbing moves the swing " +
    "relative to the fixed ball so the clubhead meets it at the chosen " +
    "instant. Source: launch-monitor impact-timing convention " +
    "(maximum-compression reference).",
  ballVisible:
    "Suggested range: on to show the ball at its fixed impact position. " +
    "Source: launch-monitor convention of a fixed ball and a swing " +
    "delivered to it.",
  groundVisible:
    "Suggested range: on to show the ground line for spatial reference. " +
    "Source: standard golf-scene convention.",
  courseVisible:
    "Suggested range: on to render the course furniture — fairway strip " +
    "along the target line, putting green with hole and flag at the " +
    "configurable green distance, tee marker at the origin. Source: " +
    "standard golf-course presentation; tones derived from the theme " +
    "palette.",
  screwAxisVisible:
    "Suggested range: on for swing-scale engineering analysis. The magenta " +
    "line is the directed instantaneous screw axis, the orange wrapped curve " +
    "shows rotational handedness, and the cyan radius is R_ISA. Pure " +
    "translation is shown as an arrow because its axis is at infinity. " +
    "Source: Lynch and Park, Modern Robotics, Chapter 3.3.2 (twists). " +
    "Reference frame: app/world x target, y up, z right.",
  swingFlightToggle:
    "Off by default: the flight envelope (100+ m) dwarfs the swing " +
    "envelope (~3 m), collapsing the swing to a dot when both share one " +
    "scale. Turn on only to see the full trajectory in context. Source: " +
    "typical driver carry (Penner 2003) vs. club length (published " +
    "manufacturer specs).",
  strikeVectorsVisible:
    "Delivered club path, face normal, and attack-angle directions " +
    "projected into the face plane. Source: standard launch-monitor " +
    "D-plane presentation (TrackMan literature; Jorgensen, The Physics " +
    "of Golf).",
  showCgMarker:
      "Suggested range: on to mark the active display reference. Generated " +
      "representative heads use their uniform-density geometric centroid; " +
      "procedural and imported STL views use the scenario reference datum. " +
      "An imported STL does not encode density, physical registration, or mass CG. " +
      "Source: generated-head geometry or current scenario reference authority.",
  fxBallSpeed:
    "Suggested range: 120-190 mph ball speed (tour driver average near " +
    "167 mph; strong amateurs 140-160). Source: openly published tour " +
    "launch-monitor averages.",
  fxLaunchAngle:
    "Suggested range: 8-16 deg launch for drivers (tour average near " +
    "10.9 deg); higher for irons and wedges. Source: openly published " +
    "tour launch-monitor averages.",
  fxLaunchDirection:
    "Suggested range: within +/-10 deg of the target line; positive values " +
    "start right of the target line, negative values start left, and zero " +
    "starts on target. Source: TrackMan's published Launch Direction " +
    "definition. Reference frame: horizontal ball-CG motion relative to " +
    "the target line immediately after separation.",
  // Compatibility alias for older UI/plugin callers.
  fxAzimuth:
    "Suggested range: within +/-10 deg of the target line; positive values " +
    "start right of the target line, negative values start left, and zero " +
    "starts on target. Source: TrackMan's published Launch Direction " +
    "definition. Reference frame: horizontal ball-CG motion relative to " +
    "the target line immediately after separation.",
  fxSpinRpm:
    "Suggested range: 2,000-3,500 rpm total spin for drivers (tour " +
    "average near 2,686 rpm); 4,000-10,000+ for irons and wedges. " +
    "Source: openly published tour launch-monitor averages.",
  fxSpinAxisTilt:
    "Suggested range: within +/-20 deg spin-axis tilt; + = fade/slice " +
    "side (curves right for a right-handed player), - = draw/hook side. " +
    "Source: TrackMan D-plane literature. Reference frame: viewed from " +
    "behind the ball toward the target; 0 deg is pure backspin, positive " +
    "tilts the axis toward fade/slice curvature.",
  fxSpeedUnit:
    "Suggested range: mph for launch-monitor style entry, m/s for SI " +
    "work; the model always computes in SI. Source: launch-monitor " +
    "display convention.",
};
