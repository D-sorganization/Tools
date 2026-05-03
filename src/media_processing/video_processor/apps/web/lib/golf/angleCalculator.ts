/**
 * Angle Calculator for Golf Swing Analysis
 * Provides precise biomechanical angle measurements from pose landmarks
 */

import {
  BodyAngles,
  Landmark,
  PoseLandmark,
  StanceDirection,
} from './types';

/**
 * Calculate angle between three points (in degrees)
 * The angle is measured at point B (the middle point)
 */
export function calculateAngle(a: Landmark, b: Landmark, c: Landmark): number {
  const vectorBA = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z,
  };

  const vectorBC = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: c.z - b.z,
  };

  const dotProduct =
    vectorBA.x * vectorBC.x + vectorBA.y * vectorBC.y + vectorBA.z * vectorBC.z;

  const magnitudeBA = Math.sqrt(
    vectorBA.x ** 2 + vectorBA.y ** 2 + vectorBA.z ** 2
  );
  const magnitudeBC = Math.sqrt(
    vectorBC.x ** 2 + vectorBC.y ** 2 + vectorBC.z ** 2
  );

  if (magnitudeBA === 0 || magnitudeBC === 0) {
    return 0;
  }

  const cosAngle = Math.max(-1, Math.min(1, dotProduct / (magnitudeBA * magnitudeBC)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/**
 * Calculate 2D angle from horizontal (for measuring tilts)
 */
export function calculateAngleFromHorizontal(
  start: Landmark,
  end: Landmark
): number {
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
}

/**
 * Calculate rotation angle in the horizontal plane (bird's eye view)
 * Uses z-coordinates for depth perception
 */
export function calculateHorizontalRotation(
  left: Landmark,
  right: Landmark
): number {
  const deltaX = right.x - left.x;
  const deltaZ = right.z - left.z;
  return Math.atan2(deltaZ, deltaX) * (180 / Math.PI);
}

/**
 * Get midpoint between two landmarks
 */
export function getMidpoint(a: Landmark, b: Landmark): Landmark {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
  };
}

/**
 * Calculate distance between two landmarks
 */
export function calculateDistance(a: Landmark, b: Landmark): number {
  return Math.sqrt(
    (b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2
  );
}

/**
 * Calculate all body angles from pose landmarks
 */
export function calculateBodyAngles(
  landmarks: Landmark[],
  stance: StanceDirection = StanceDirection.RIGHT_HANDED
): BodyAngles {
  // Key landmarks
  const leftShoulder = landmarks[PoseLandmark.LEFT_SHOULDER];
  const rightShoulder = landmarks[PoseLandmark.RIGHT_SHOULDER];
  const leftHip = landmarks[PoseLandmark.LEFT_HIP];
  const rightHip = landmarks[PoseLandmark.RIGHT_HIP];
  const leftElbow = landmarks[PoseLandmark.LEFT_ELBOW];
  const rightElbow = landmarks[PoseLandmark.RIGHT_ELBOW];
  const leftWrist = landmarks[PoseLandmark.LEFT_WRIST];
  const rightWrist = landmarks[PoseLandmark.RIGHT_WRIST];
  const leftKnee = landmarks[PoseLandmark.LEFT_KNEE];
  const rightKnee = landmarks[PoseLandmark.RIGHT_KNEE];
  const leftAnkle = landmarks[PoseLandmark.LEFT_ANKLE];
  const rightAnkle = landmarks[PoseLandmark.RIGHT_ANKLE];
  const nose = landmarks[PoseLandmark.NOSE];

  // Calculate midpoints
  const shoulderMid = getMidpoint(leftShoulder, rightShoulder);
  const hipMid = getMidpoint(leftHip, rightHip);

  // Spine angle (forward tilt from vertical)
  const spineAngle = 90 - calculateAngleFromHorizontal(hipMid, shoulderMid);

  // Spine lateral bend
  const spineLateral = calculateAngleFromHorizontal(leftHip, leftShoulder) -
    calculateAngleFromHorizontal(rightHip, rightShoulder);

  // Spine rotation (using z-depth)
  const spineRotation = calculateHorizontalRotation(leftShoulder, rightShoulder);

  // Hip rotation
  const hipRotation = calculateHorizontalRotation(leftHip, rightHip);

  // Hip tilt (lateral)
  const hipTilt = calculateAngleFromHorizontal(leftHip, rightHip);

  // Hip slide (horizontal displacement from address)
  const hipSlide = (hipMid.x - 0.5) * 100; // Normalized to cm (assuming 1 unit = 100cm)

  // Shoulder rotation and tilt
  const shoulderRotation = calculateHorizontalRotation(leftShoulder, rightShoulder);
  const shoulderTilt = calculateAngleFromHorizontal(leftShoulder, rightShoulder);

  // X-Factor: differential between shoulder and hip rotation
  const xFactor = Math.abs(shoulderRotation - hipRotation);
  const xFactorStretch = xFactor; // This gets updated during transition analysis

  // Elbow angles
  const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
  const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);

  // Wrist angles (relative to forearm)
  const leftPinky = landmarks[PoseLandmark.LEFT_PINKY];
  const rightPinky = landmarks[PoseLandmark.RIGHT_PINKY];
  const leftWristAngle = calculateAngle(leftElbow, leftWrist, leftPinky);
  const rightWristAngle = calculateAngle(rightElbow, rightWrist, rightPinky);

  // Knee flexion angles
  const leftKneeFlexion = 180 - calculateAngle(leftHip, leftKnee, leftAnkle);
  const rightKneeFlexion = 180 - calculateAngle(rightHip, rightKnee, rightAnkle);

  return {
    spineAngle,
    spineLateral,
    spineRotation,
    hipRotation,
    hipTilt,
    hipSlide,
    shoulderRotation,
    shoulderTilt,
    leftElbowAngle,
    rightElbowAngle,
    leftWristAngle,
    rightWristAngle,
    leftKneeFlexion,
    rightKneeFlexion,
    xFactor,
    xFactorStretch,
  };
}

/**
 * Calculate ideal angle ranges for different swing phases
 */
export function getIdealAngleRanges(phase: string): Partial<Record<keyof BodyAngles, [number, number]>> {
  const ranges: Record<string, Partial<Record<keyof BodyAngles, [number, number]>>> = {
    address: {
      spineAngle: [25, 45],
      hipRotation: [-5, 5],
      shoulderRotation: [-5, 5],
      leftKneeFlexion: [15, 30],
      rightKneeFlexion: [15, 30],
      xFactor: [0, 10],
    },
    top_of_backswing: {
      spineAngle: [20, 45],
      shoulderRotation: [75, 105],
      hipRotation: [35, 55],
      xFactor: [40, 60],
      leftElbowAngle: [160, 180],
    },
    impact: {
      spineAngle: [25, 45],
      hipRotation: [30, 50],
      shoulderRotation: [-10, 20],
      xFactor: [35, 55],
      leftElbowAngle: [165, 180],
    },
    finish: {
      spineAngle: [0, 25],
      shoulderRotation: [85, 110],
      hipRotation: [75, 95],
    },
  };

  return ranges[phase] || {};
}

/**
 * Check if an angle is within the ideal range
 */
export function isAngleInRange(
  angle: number,
  idealRange: [number, number]
): boolean {
  return angle >= idealRange[0] && angle <= idealRange[1];
}

/**
 * Calculate how far an angle deviates from ideal
 */
export function calculateAngleDeviation(
  angle: number,
  idealRange: [number, number]
): number {
  if (angle < idealRange[0]) {
    return idealRange[0] - angle;
  }
  if (angle > idealRange[1]) {
    return angle - idealRange[1];
  }
  return 0;
}

/**
 * Smooth angle data over time using moving average
 */
export function smoothAngles(
  angleHistory: number[],
  windowSize: number = 5
): number[] {
  const len = angleHistory.length;
  if (len < windowSize || windowSize <= 1) {
    return angleHistory;
  }

  // ⚡ Bolt Optimization: Use a pre-allocated array instead of pushing to an empty array.
  // We also split the loop into three parts (left edge, middle hot-path, right edge)
  // to avoid Math.min() and Math.max() checks on every iteration.
  // And we track the sum manually to avoid the O(N) allocation of .slice() and overhead of .reduce().
  const smoothed = new Array<number>(len);
  const leftHalf = Math.floor(windowSize / 2);
  const rightHalf = Math.ceil(windowSize / 2);
  const middleWindowSpan = leftHalf + rightHalf;

  // Left edge (bounds checking needed for start)
  for (let i = 0; i < leftHalf; i++) {
    const end = Math.min(len, i + rightHalf);
    let sum = 0;
    for (let j = 0; j < end; j++) {
      sum += angleHistory[j];
    }
    smoothed[i] = sum / end;
  }

  // Middle section (hot path, no bounds checking)
  const endMiddle = Math.max(leftHalf, len - rightHalf + 1);
  for (let i = leftHalf; i < endMiddle; i++) {
    const start = i - leftHalf;
    const end = i + rightHalf;
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += angleHistory[j];
    }
    smoothed[i] = sum / middleWindowSpan;
  }

  // Right edge (bounds checking needed for end)
  for (let i = endMiddle; i < len; i++) {
    const start = i - leftHalf;
    let sum = 0;
    for (let j = start; j < len; j++) {
      sum += angleHistory[j];
    }
    smoothed[i] = sum / (len - start);
  }

  return smoothed;
}

/**
 * Calculate angular velocity (degrees per second)
 */
export function calculateAngularVelocity(
  angle1: number,
  angle2: number,
  timeDeltaMs: number
): number {
  if (timeDeltaMs === 0) return 0;
  return ((angle2 - angle1) / timeDeltaMs) * 1000;
}

/**
 * Detect stance direction (right-handed or left-handed)
 */
export function detectStanceDirection(
  landmarks: Landmark[],
  targetLineAngle: number = 0
): StanceDirection {
  const leftShoulder = landmarks[PoseLandmark.LEFT_SHOULDER];
  const rightShoulder = landmarks[PoseLandmark.RIGHT_SHOULDER];
  const leftHip = landmarks[PoseLandmark.LEFT_HIP];
  const rightHip = landmarks[PoseLandmark.RIGHT_HIP];

  // Calculate body orientation
  const shoulderAngle = calculateHorizontalRotation(leftShoulder, rightShoulder);
  const hipAngle = calculateHorizontalRotation(leftHip, rightHip);

  // Average body rotation
  const bodyRotation = (shoulderAngle + hipAngle) / 2;

  // Adjust for target line
  const relativeRotation = bodyRotation - targetLineAngle;

  // Right-handed golfers face left of target (positive rotation)
  // Left-handed golfers face right of target (negative rotation)
  if (relativeRotation > 20) {
    return StanceDirection.RIGHT_HANDED;
  } else if (relativeRotation < -20) {
    return StanceDirection.LEFT_HANDED;
  }

  return StanceDirection.UNKNOWN;
}
