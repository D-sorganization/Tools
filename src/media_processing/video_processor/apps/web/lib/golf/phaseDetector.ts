/**
 * Golf Swing Phase Detection
 * Automatically identifies swing phases using pose landmark analysis
 */

import {
  BodyAngles,
  Landmark,
  PhaseTransition,
  PoseFrame,
  PoseLandmark,
  StanceDirection,
  SwingPhase,
} from './types';
import {
  calculateBodyAngles,
  calculateAngularVelocity,
  getMidpoint,
  calculateDistance,
} from './angleCalculator';

export type { PhaseTransition } from './types';

export interface PhaseDetectionResult {
  phases: PhaseTransition[];
  keyFrames: {
    address?: number;
    takeaway?: number;
    midBackswing?: number;
    top?: number;
    transition?: number;
    impact?: number;
    followThrough?: number;
    finish?: number;
  };
  analysisConfidence: number;
}

/**
 * Detect all swing phases from a sequence of pose frames
 */
export function detectSwingPhases(
  poseFrames: PoseFrame[],
  fps: number = 30,
  stance: StanceDirection = StanceDirection.RIGHT_HANDED
): PhaseDetectionResult {
  if (poseFrames.length < 10) {
    return {
      phases: [],
      keyFrames: {},
      analysisConfidence: 0,
    };
  }

  // Calculate angles for all frames
  const angleHistory: { frame: number; angles: BodyAngles }[] = poseFrames.map(
    (frame) => ({
      frame: frame.frameNumber,
      angles: calculateBodyAngles(frame.landmarks, stance),
    })
  );

  // Calculate velocities
  const velocityHistory = calculateVelocityProfile(angleHistory, fps);

  // Find key frame indices
  const keyFrames = findKeyFrames(angleHistory, velocityHistory, stance);

  // Build phase transitions
  const phases = buildPhaseTransitions(keyFrames, poseFrames, fps);

  // Calculate overall confidence
  const analysisConfidence = calculatePhaseConfidence(keyFrames, poseFrames);

  return {
    phases,
    keyFrames,
    analysisConfidence,
  };
}

/**
 * Calculate velocity profile for shoulder and hip rotation
 */
function calculateVelocityProfile(
  angleHistory: { frame: number; angles: BodyAngles }[],
  fps: number
): { frame: number; shoulderVelocity: number; hipVelocity: number; handVelocity: number }[] {
  const frameDuration = 1000 / fps;
  const velocities: { frame: number; shoulderVelocity: number; hipVelocity: number; handVelocity: number }[] = [];

  for (let i = 1; i < angleHistory.length; i++) {
    const prev = angleHistory[i - 1];
    const curr = angleHistory[i];

    const shoulderVelocity = calculateAngularVelocity(
      prev.angles.shoulderRotation,
      curr.angles.shoulderRotation,
      frameDuration
    );

    const hipVelocity = calculateAngularVelocity(
      prev.angles.hipRotation,
      curr.angles.hipRotation,
      frameDuration
    );

    // Approximate hand velocity using wrist angle changes
    const handVelocity = calculateAngularVelocity(
      prev.angles.leftWristAngle + prev.angles.rightWristAngle,
      curr.angles.leftWristAngle + curr.angles.rightWristAngle,
      frameDuration
    );

    velocities.push({
      frame: curr.frame,
      shoulderVelocity,
      hipVelocity,
      handVelocity,
    });
  }

  return velocities;
}

/**
 * Find key frame indices in the swing
 */
function findKeyFrames(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[],
  stance: StanceDirection
): {
  address?: number;
  takeaway?: number;
  midBackswing?: number;
  top?: number;
  transition?: number;
  impact?: number;
  followThrough?: number;
  finish?: number;
} {
  const keyFrames: {
    address?: number;
    takeaway?: number;
    midBackswing?: number;
    top?: number;
    transition?: number;
    impact?: number;
    followThrough?: number;
    finish?: number;
  } = {};

  if (angleHistory.length === 0) return keyFrames;

  // Find address: low velocity, minimal rotation
  keyFrames.address = findAddressFrame(angleHistory, velocityHistory);

  // Find top of backswing: maximum shoulder rotation in backswing direction
  keyFrames.top = findTopOfBackswing(angleHistory, stance);

  // Find transition: where hip starts moving toward target before shoulders
  if (keyFrames.top !== undefined) {
    keyFrames.transition = findTransitionFrame(angleHistory, velocityHistory, keyFrames.top);
  }

  // Find impact: maximum downswing velocity and return to address-like angles
  if (keyFrames.top !== undefined) {
    keyFrames.impact = findImpactFrame(angleHistory, velocityHistory, keyFrames.top);
  }

  // Find takeaway: first significant movement from address
  if (keyFrames.address !== undefined && keyFrames.top !== undefined) {
    keyFrames.takeaway = findTakeawayFrame(angleHistory, velocityHistory, keyFrames.address);
  }

  // Find mid-backswing
  if (keyFrames.takeaway !== undefined && keyFrames.top !== undefined) {
    keyFrames.midBackswing = Math.floor((keyFrames.takeaway + keyFrames.top) / 2);
  }

  // Find follow-through: after impact, shoulder rotation continues
  if (keyFrames.impact !== undefined) {
    keyFrames.followThrough = findFollowThroughFrame(angleHistory, keyFrames.impact);
  }

  // Find finish: velocity near zero, rotated past impact
  if (keyFrames.followThrough !== undefined) {
    keyFrames.finish = findFinishFrame(angleHistory, velocityHistory, keyFrames.followThrough);
  }

  return keyFrames;
}

/**
 * Find the address frame (setup position)
 */
function findAddressFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[]
): number | undefined {
  // Look for frames with minimal velocity in the first 30% of the video
  const searchEndIdx = Math.floor(angleHistory.length * 0.3);

  let minVelocitySum = Infinity;
  let addressIdx = 0;

  for (let i = 0; i < Math.min(searchEndIdx, velocityHistory.length); i++) {
    const vel = velocityHistory[i];
    const velocitySum = Math.abs(vel.shoulderVelocity) + Math.abs(vel.hipVelocity);

    if (velocitySum < minVelocitySum) {
      minVelocitySum = velocitySum;
      addressIdx = i;
    }
  }

  // Verify address has reasonable spine angle (20-50 degrees)
  const addressAngles = angleHistory[addressIdx]?.angles;
  if (addressAngles && addressAngles.spineAngle >= 15 && addressAngles.spineAngle <= 55) {
    return angleHistory[addressIdx].frame;
  }

  return angleHistory[0]?.frame;
}

/**
 * Find the top of backswing frame
 */
function findTopOfBackswing(
  angleHistory: { frame: number; angles: BodyAngles }[],
  stance: StanceDirection
): number | undefined {
  let maxRotation = -Infinity;
  let topIdx = -1;

  // For right-handed: look for maximum positive shoulder rotation
  // For left-handed: look for maximum negative shoulder rotation
  const rotationSign = stance === StanceDirection.RIGHT_HANDED ? 1 : -1;

  for (let i = 0; i < angleHistory.length; i++) {
    const rotation = angleHistory[i].angles.shoulderRotation * rotationSign;
    if (rotation > maxRotation) {
      maxRotation = rotation;
      topIdx = i;
    }
  }

  if (topIdx >= 0 && maxRotation > 30) {
    return angleHistory[topIdx].frame;
  }

  return undefined;
}

/**
 * Find the transition frame (hip starts before shoulders)
 */
function findTransitionFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[],
  topFrame: number
): number | undefined {
  // Find the index of the top frame
  const topIdx = angleHistory.findIndex((a) => a.frame === topFrame);
  if (topIdx < 0) return undefined;

  // Look for when hip velocity reverses before shoulder velocity
  for (let i = topIdx; i < velocityHistory.length - 1; i++) {
    const current = velocityHistory[i];
    const next = velocityHistory[i + 1];

    // Hip velocity becoming more negative (toward target)
    if (current.hipVelocity > 0 && next.hipVelocity < 0) {
      return velocityHistory[i + 1].frame;
    }
  }

  // Fallback: just after top
  if (topIdx + 3 < angleHistory.length) {
    return angleHistory[topIdx + 3].frame;
  }

  return undefined;
}

/**
 * Find the impact frame
 */
function findImpactFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[],
  topFrame: number
): number | undefined {
  const topIdx = angleHistory.findIndex((a) => a.frame === topFrame);
  if (topIdx < 0) return undefined;

  // Impact is typically 60-75% of the way through downswing
  // Look for frame where shoulders return near address position

  let maxDownswingVelocity = 0;
  let impactIdx = -1;

  for (let i = topIdx; i < velocityHistory.length; i++) {
    const vel = velocityHistory[i];
    const angles = angleHistory[i]?.angles;

    // Look for peak negative velocity (rotation toward target)
    if (Math.abs(vel.shoulderVelocity) > maxDownswingVelocity) {
      maxDownswingVelocity = Math.abs(vel.shoulderVelocity);
    }

    // Impact occurs when shoulders are near square (-10 to 20 degrees)
    if (
      angles &&
      angles.shoulderRotation >= -15 &&
      angles.shoulderRotation <= 30 &&
      i > topIdx + 3
    ) {
      impactIdx = i;
      break;
    }
  }

  if (impactIdx >= 0) {
    return angleHistory[impactIdx].frame;
  }

  // Fallback: 70% through remaining frames after top
  const remainingFrames = angleHistory.length - topIdx;
  const estimatedImpactIdx = topIdx + Math.floor(remainingFrames * 0.35);
  if (estimatedImpactIdx < angleHistory.length) {
    return angleHistory[estimatedImpactIdx].frame;
  }

  return undefined;
}

/**
 * Find the takeaway frame (first significant movement)
 */
function findTakeawayFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[],
  addressFrame: number
): number | undefined {
  const addressIdx = angleHistory.findIndex((a) => a.frame === addressFrame);
  if (addressIdx < 0) return undefined;

  // Look for first frame with significant rotational velocity
  const velocityThreshold = 20; // degrees per second

  for (let i = addressIdx; i < velocityHistory.length; i++) {
    const vel = velocityHistory[i];
    if (Math.abs(vel.shoulderVelocity) > velocityThreshold) {
      return velocityHistory[i].frame;
    }
  }

  return undefined;
}

/**
 * Find the follow-through frame
 */
function findFollowThroughFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  impactFrame: number
): number | undefined {
  const impactIdx = angleHistory.findIndex((a) => a.frame === impactFrame);
  if (impactIdx < 0) return undefined;

  // Follow-through is when shoulders have rotated 45+ degrees past impact
  for (let i = impactIdx; i < angleHistory.length; i++) {
    const angles = angleHistory[i].angles;
    // Shoulders rotated well past impact (toward target)
    if (angles.shoulderRotation > 60 || angles.shoulderRotation < -60) {
      return angleHistory[i].frame;
    }
  }

  // Fallback: 50% through remaining frames
  const remainingFrames = angleHistory.length - impactIdx;
  const estimatedIdx = impactIdx + Math.floor(remainingFrames * 0.5);
  if (estimatedIdx < angleHistory.length) {
    return angleHistory[estimatedIdx].frame;
  }

  return undefined;
}

/**
 * Find the finish frame
 */
function findFinishFrame(
  angleHistory: { frame: number; angles: BodyAngles }[],
  velocityHistory: { frame: number; shoulderVelocity: number; hipVelocity: number }[],
  followThroughFrame: number
): number | undefined {
  const ftIdx = angleHistory.findIndex((a) => a.frame === followThroughFrame);
  if (ftIdx < 0) return undefined;

  // Finish is when velocity drops below threshold and rotation is complete
  const velocityThreshold = 15;

  for (let i = ftIdx; i < velocityHistory.length; i++) {
    const vel = velocityHistory[i];
    if (
      Math.abs(vel.shoulderVelocity) < velocityThreshold &&
      Math.abs(vel.hipVelocity) < velocityThreshold
    ) {
      return velocityHistory[i].frame;
    }
  }

  // Fallback: last frame
  return angleHistory[angleHistory.length - 1]?.frame;
}

/**
 * Build phase transitions from key frames
 */
function buildPhaseTransitions(
  keyFrames: {
    address?: number;
    takeaway?: number;
    midBackswing?: number;
    top?: number;
    transition?: number;
    impact?: number;
    followThrough?: number;
    finish?: number;
  },
  poseFrames: PoseFrame[],
  fps: number
): PhaseTransition[] {
  const phases: PhaseTransition[] = [];
  const frameDuration = 1000 / fps;

  const addPhase = (
    phase: SwingPhase,
    startFrame: number | undefined,
    endFrame: number | undefined
  ) => {
    if (startFrame === undefined || endFrame === undefined) return;
    if (startFrame > endFrame) return;

    phases.push({
      phase,
      startFrame,
      endFrame,
      duration: (endFrame - startFrame + 1) * frameDuration,
      confidence: 0.8, // Default confidence
    });
  };

  // Build phases in order
  addPhase(SwingPhase.ADDRESS, keyFrames.address, keyFrames.takeaway);
  addPhase(SwingPhase.TAKEAWAY, keyFrames.takeaway, keyFrames.midBackswing);
  addPhase(SwingPhase.BACKSWING, keyFrames.midBackswing, keyFrames.top);
  addPhase(SwingPhase.TOP_OF_BACKSWING, keyFrames.top, keyFrames.transition);
  addPhase(SwingPhase.TRANSITION, keyFrames.transition, keyFrames.transition !== undefined ? keyFrames.transition + 2 : undefined);
  addPhase(SwingPhase.DOWNSWING, keyFrames.transition, keyFrames.impact);
  addPhase(SwingPhase.IMPACT, keyFrames.impact, keyFrames.impact !== undefined ? keyFrames.impact + 1 : undefined);
  addPhase(SwingPhase.FOLLOW_THROUGH, keyFrames.impact, keyFrames.followThrough);
  addPhase(SwingPhase.FINISH, keyFrames.followThrough, keyFrames.finish);

  return phases;
}

/**
 * Calculate overall phase detection confidence
 */
function calculatePhaseConfidence(
  keyFrames: {
    address?: number;
    top?: number;
    impact?: number;
    finish?: number;
  },
  poseFrames: PoseFrame[]
): number {
  let confidence = 0;
  let weights = 0;

  // Weight key frames by importance
  if (keyFrames.address !== undefined) {
    confidence += 0.2;
    weights += 0.2;
  }
  if (keyFrames.top !== undefined) {
    confidence += 0.3;
    weights += 0.3;
  }
  if (keyFrames.impact !== undefined) {
    confidence += 0.35;
    weights += 0.35;
  }
  if (keyFrames.finish !== undefined) {
    confidence += 0.15;
    weights += 0.15;
  }

  // Factor in pose confidence
  const avgPoseConfidence =
    poseFrames.reduce((sum, f) => sum + f.confidence, 0) / poseFrames.length;

  return (confidence / weights) * avgPoseConfidence;
}

/**
 * Get the current swing phase for a given frame
 */
export function getPhaseAtFrame(
  phases: PhaseTransition[],
  frameNumber: number
): SwingPhase {
  for (const phase of phases) {
    if (frameNumber >= phase.startFrame && frameNumber <= phase.endFrame) {
      return phase.phase;
    }
  }
  return SwingPhase.UNKNOWN;
}

/**
 * Calculate phase timing quality (tempo score)
 */
export function calculateTempoQuality(phases: PhaseTransition[]): number {
  const backswingPhases = phases.filter((p) =>
    [SwingPhase.TAKEAWAY, SwingPhase.BACKSWING, SwingPhase.TOP_OF_BACKSWING].includes(p.phase)
  );
  const downswingPhases = phases.filter((p) =>
    [SwingPhase.TRANSITION, SwingPhase.DOWNSWING, SwingPhase.IMPACT].includes(p.phase)
  );

  const backswingDuration = backswingPhases.reduce((sum, p) => sum + p.duration, 0);
  const downswingDuration = downswingPhases.reduce((sum, p) => sum + p.duration, 0);

  if (downswingDuration === 0) return 0;

  const ratio = backswingDuration / downswingDuration;

  // Ideal tempo ratio is around 3:1
  const idealRatio = 3.0;
  const deviation = Math.abs(ratio - idealRatio);

  // Score: 100 if perfect, decreasing with deviation
  return Math.max(0, 100 - deviation * 25);
}
