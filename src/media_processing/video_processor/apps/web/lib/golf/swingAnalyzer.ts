/**
 * Golf Swing Analyzer
 * Main analysis engine that combines all metrics and generates comprehensive reports
 */

import { v4 as uuidv4 } from "uuid";
import {
  AnalysisConfig,
  BalanceMetrics,
  BodyAngles,
  BodyVelocities,
  Landmark,
  PlaneMetrics,
  PoseFrame,
  PoseLandmark,
  PostureMetrics,
  StanceDirection,
  SwingAnalysis,
  SwingIssue,
  SwingPhase,
  SwingPositionMetrics,
  SwingScores,
  SwingType,
  TempoMetrics,
} from "./types";
import {
  calculateBodyAngles,
  calculateAngularVelocity,
  getMidpoint,
  calculateDistance,
  detectStanceDirection,
  getIdealAngleRanges,
  calculateAngleDeviation,
} from "./angleCalculator";
import { detectSwingPhases, PhaseDetectionResult } from "./phaseDetector";

const DEFAULT_CONFIG: AnalysisConfig = {
  detectStance: true,
  calculateVelocities: true,
  detectPhases: true,
  generateRecommendations: true,
  minConfidenceThreshold: 0.5,
  smoothingWindow: 5,
};

/**
 * Main swing analysis function
 */
export function analyzeSwing(
  poseFrames: PoseFrame[],
  fps: number = 30,
  videoId?: string,
  config: Partial<AnalysisConfig> = {},
): SwingAnalysis {
  const fullConfig = { ...DEFAULT_CONFIG, ...config };

  // Filter frames by confidence
  const validFrames = poseFrames.filter(
    (f) => f.confidence >= fullConfig.minConfidenceThreshold,
  );

  if (validFrames.length < 10) {
    throw new Error("Insufficient valid pose frames for analysis");
  }

  // Detect golfer stance
  const stance = fullConfig.detectStance
    ? detectStanceDirection(
        validFrames[0].landmarks,
        fullConfig.targetLineAngle,
      )
    : StanceDirection.RIGHT_HANDED;

  // Detect swing phases
  const phaseResult = fullConfig.detectPhases
    ? detectSwingPhases(validFrames, fps, stance)
    : { phases: [], keyFrames: {}, analysisConfidence: 0 };

  // Get metrics at key positions
  const keyPositions = extractKeyPositionMetrics(
    validFrames,
    phaseResult,
    fps,
    stance,
  );

  // Calculate tempo metrics
  const tempo = calculateTempoMetrics(phaseResult, fps);

  // Calculate balance metrics
  const balance = calculateBalanceMetrics(validFrames, phaseResult, stance);

  // Calculate plane metrics
  const plane = calculatePlaneMetrics(validFrames, phaseResult, stance);

  // Calculate posture metrics
  const posture = calculatePostureMetrics(validFrames, phaseResult, stance);

  // Identify issues
  const issues = identifySwingIssues(
    keyPositions,
    tempo,
    balance,
    plane,
    posture,
  );

  // Generate recommendations
  const recommendations = fullConfig.generateRecommendations
    ? generateRecommendations(issues)
    : [];

  // Calculate scores
  const scores = calculateSwingScores(tempo, balance, plane, posture, issues);

  return {
    sessionId: uuidv4(),
    videoId: videoId || uuidv4(),
    analysisTimestamp: Date.now(),
    golferStance: stance,
    swingType: SwingType.UNKNOWN, // DEFERRED: Implement swing type detection
    totalFrames: poseFrames.length,
    fps,
    poseFrames: validFrames,
    phases: phaseResult.phases,
    keyPositions,
    tempo,
    balance,
    plane,
    posture,
    scores,
    issues,
    recommendations,
  };
}

/**
 * Extract metrics at key swing positions
 */
function extractKeyPositionMetrics(
  frames: PoseFrame[],
  phaseResult: PhaseDetectionResult,
  fps: number,
  stance: StanceDirection,
): SwingAnalysis["keyPositions"] {
  const keyPositions: SwingAnalysis["keyPositions"] = {};

  const getMetricsAtFrame = (
    frameNumber: number | undefined,
  ): SwingPositionMetrics | undefined => {
    if (frameNumber === undefined) return undefined;

    const frame = frames.find((f) => f.frameNumber === frameNumber);
    if (!frame) return undefined;

    const angles = calculateBodyAngles(frame.landmarks, stance);

    // Find previous and next frames for velocity calculation
    const frameIdx = frames.findIndex((f) => f.frameNumber === frameNumber);
    let velocities: BodyVelocities | undefined;

    if (frameIdx > 0 && frameIdx < frames.length - 1) {
      const prevFrame = frames[frameIdx - 1];
      const nextFrame = frames[frameIdx + 1];
      const prevAngles = calculateBodyAngles(prevFrame.landmarks, stance);
      const nextAngles = calculateBodyAngles(nextFrame.landmarks, stance);
      const frameDuration = 2000 / fps; // Time between prev and next

      velocities = {
        hipRotationalVelocity: calculateAngularVelocity(
          prevAngles.hipRotation,
          nextAngles.hipRotation,
          frameDuration,
        ),
        shoulderRotationalVelocity: calculateAngularVelocity(
          prevAngles.shoulderRotation,
          nextAngles.shoulderRotation,
          frameDuration,
        ),
        handSpeed: calculateHandSpeed(
          prevFrame.landmarks,
          nextFrame.landmarks,
          frameDuration,
        ),
        headMovement: calculateHeadMovement(
          frames[0].landmarks,
          frame.landmarks,
        ),
      };
    }

    return {
      frameNumber,
      timestamp: frame.timestamp,
      angles,
      velocities,
      confidence: frame.confidence,
    };
  };

  keyPositions.address = getMetricsAtFrame(phaseResult.keyFrames.address);
  keyPositions.top = getMetricsAtFrame(phaseResult.keyFrames.top);
  keyPositions.impact = getMetricsAtFrame(phaseResult.keyFrames.impact);
  keyPositions.finish = getMetricsAtFrame(phaseResult.keyFrames.finish);

  return keyPositions;
}

/**
 * Calculate hand speed from landmark movement
 */
function calculateHandSpeed(
  prevLandmarks: Landmark[],
  nextLandmarks: Landmark[],
  timeDeltaMs: number,
): number {
  const prevLeftWrist = prevLandmarks[PoseLandmark.LEFT_WRIST];
  const prevRightWrist = prevLandmarks[PoseLandmark.RIGHT_WRIST];
  const nextLeftWrist = nextLandmarks[PoseLandmark.LEFT_WRIST];
  const nextRightWrist = nextLandmarks[PoseLandmark.RIGHT_WRIST];

  const prevMid = getMidpoint(prevLeftWrist, prevRightWrist);
  const nextMid = getMidpoint(nextLeftWrist, nextRightWrist);

  const distance = calculateDistance(prevMid, nextMid);
  // Convert to m/s (assuming normalized coordinates with 2m reference)
  return (distance * 2) / (timeDeltaMs / 1000);
}

/**
 * Calculate head movement from address position
 */
function calculateHeadMovement(
  addressLandmarks: Landmark[],
  currentLandmarks: Landmark[],
): number {
  const addressNose = addressLandmarks[PoseLandmark.NOSE];
  const currentNose = currentLandmarks[PoseLandmark.NOSE];

  // Return distance in cm (assuming 200cm reference frame)
  return calculateDistance(addressNose, currentNose) * 200;
}

/**
 * Calculate tempo metrics
 */
function calculateTempoMetrics(
  phaseResult: PhaseDetectionResult,
  fps: number,
): TempoMetrics {
  const phases = phaseResult.phases;

  // Find backswing phases
  const backswingPhases = phases.filter((p) =>
    [
      SwingPhase.TAKEAWAY,
      SwingPhase.BACKSWING,
      SwingPhase.TOP_OF_BACKSWING,
    ].includes(p.phase),
  );

  // Find downswing phases
  const downswingPhases = phases.filter((p) =>
    [SwingPhase.TRANSITION, SwingPhase.DOWNSWING, SwingPhase.IMPACT].includes(
      p.phase,
    ),
  );

  const backswingDuration = backswingPhases.reduce(
    (sum, p) => sum + p.duration,
    0,
  );
  const downswingDuration = downswingPhases.reduce(
    (sum, p) => sum + p.duration,
    0,
  );
  const totalSwingDuration = backswingDuration + downswingDuration;

  // Find transition phase for pause calculation
  const transitionPhase = phases.find((p) => p.phase === SwingPhase.TRANSITION);
  const transitionPause = transitionPhase?.duration || 0;

  // Calculate tempo ratio
  const tempoRatio =
    downswingDuration > 0 ? backswingDuration / downswingDuration : 0;

  // Categorize rhythm
  let rhythm: "smooth" | "quick" | "slow" | "uneven" = "smooth";
  if (tempoRatio < 2) {
    rhythm = "quick";
  } else if (tempoRatio > 4) {
    rhythm = "slow";
  } else if (Math.abs(tempoRatio - 3) > 1) {
    rhythm = "uneven";
  }

  return {
    backswingDuration,
    downswingDuration,
    totalSwingDuration,
    tempoRatio,
    transitionPause,
    rhythm,
  };
}

/**
 * Calculate balance and weight shift metrics
 */
function calculateBalanceMetrics(
  frames: PoseFrame[],
  phaseResult: PhaseDetectionResult,
  stance: StanceDirection,
): BalanceMetrics {
  const getWeightDistribution = (
    landmarks: Landmark[],
  ): { left: number; right: number } => {
    const leftHip = landmarks[PoseLandmark.LEFT_HIP];
    const rightHip = landmarks[PoseLandmark.RIGHT_HIP];
    const hipMid = getMidpoint(leftHip, rightHip);

    // Weight distribution based on hip position relative to feet
    const leftAnkle = landmarks[PoseLandmark.LEFT_ANKLE];
    const rightAnkle = landmarks[PoseLandmark.RIGHT_ANKLE];
    const stanceWidth = Math.abs(rightAnkle.x - leftAnkle.x);

    if (stanceWidth === 0) {
      return { left: 50, right: 50 };
    }

    const hipPositionRatio = (hipMid.x - leftAnkle.x) / stanceWidth;
    const rightWeight = Math.min(100, Math.max(0, hipPositionRatio * 100));
    const leftWeight = 100 - rightWeight;

    return { left: leftWeight, right: rightWeight };
  };

  // Get frames at key positions
  const addressFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.address,
  );
  const topFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.top,
  );
  const impactFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.impact,
  );
  const finishFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.finish,
  );

  // Calculate weight at each position
  const addressWeight = addressFrame
    ? getWeightDistribution(addressFrame.landmarks)
    : { left: 50, right: 50 };
  const topWeight = topFrame
    ? getWeightDistribution(topFrame.landmarks)
    : { left: 30, right: 70 };
  const impactWeight = impactFrame
    ? getWeightDistribution(impactFrame.landmarks)
    : { left: 70, right: 30 };
  const finishWeight = finishFrame
    ? getWeightDistribution(finishFrame.landmarks)
    : { left: 90, right: 10 };

  // Calculate sway and slide
  let swayAmount = 0;
  let slideAmount = 0;
  let hipBump = 0;

  if (addressFrame && topFrame && impactFrame) {
    const addressHipMid = getMidpoint(
      addressFrame.landmarks[PoseLandmark.LEFT_HIP],
      addressFrame.landmarks[PoseLandmark.RIGHT_HIP],
    );
    const topHipMid = getMidpoint(
      topFrame.landmarks[PoseLandmark.LEFT_HIP],
      topFrame.landmarks[PoseLandmark.RIGHT_HIP],
    );
    const impactHipMid = getMidpoint(
      impactFrame.landmarks[PoseLandmark.LEFT_HIP],
      impactFrame.landmarks[PoseLandmark.RIGHT_HIP],
    );

    // Sway: lateral hip movement in backswing (in cm, assuming 200cm reference)
    swayAmount = Math.abs(topHipMid.x - addressHipMid.x) * 200;

    // Slide: lateral hip movement through impact
    slideAmount = Math.abs(impactHipMid.x - topHipMid.x) * 200;

    // Hip bump: forward hip movement toward target in downswing
    hipBump = (impactHipMid.x - topHipMid.x) * 200;
  }

  return {
    addressWeight,
    topWeight,
    impactWeight,
    finishWeight,
    swayAmount,
    slideAmount,
    hipBump,
  };
}

/**
 * Calculate swing plane metrics
 */
function calculatePlaneMetrics(
  frames: PoseFrame[],
  phaseResult: PhaseDetectionResult,
  stance: StanceDirection,
): PlaneMetrics {
  // Calculate shaft angles at key positions
  const calculateShaftAngle = (landmarks: Landmark[]): number => {
    const leftWrist = landmarks[PoseLandmark.LEFT_WRIST];
    const rightWrist = landmarks[PoseLandmark.RIGHT_WRIST];
    const handMid = getMidpoint(leftWrist, rightWrist);

    // Approximate club shaft using hand position
    // Shaft angle is relative to horizontal
    const deltaY = handMid.y - 0.5; // Relative to center
    const deltaX = Math.abs(handMid.z); // Depth

    return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
  };

  const addressFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.address,
  );
  const topFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.top,
  );
  const impactFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.impact,
  );

  const shaftAngleAtAddress = addressFrame
    ? calculateShaftAngle(addressFrame.landmarks)
    : 45;
  const shaftAngleAtTop = topFrame
    ? calculateShaftAngle(topFrame.landmarks)
    : 60;
  const shaftAngleAtImpact = impactFrame
    ? calculateShaftAngle(impactFrame.landmarks)
    : 45;

  // Estimate backswing and downswing plane angles
  const backswingPlaneAngle = 60; // Typical backswing plane
  const downswingPlaneAngle = 55; // Typically shallower

  const planeDifferential = Math.abs(backswingPlaneAngle - downswingPlaneAngle);
  const onPlane = planeDifferential < 10;

  return {
    backswingPlaneAngle,
    downswingPlaneAngle,
    planeDifferential,
    onPlane,
    shaftAngleAtAddress,
    shaftAngleAtTop,
    shaftAngleAtImpact,
  };
}

/**
 * Calculate posture metrics
 */
function calculatePostureMetrics(
  frames: PoseFrame[],
  phaseResult: PhaseDetectionResult,
  stance: StanceDirection,
): PostureMetrics {
  const addressFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.address,
  );
  const topFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.top,
  );
  const impactFrame = frames.find(
    (f) => f.frameNumber === phaseResult.keyFrames.impact,
  );

  // Address posture
  let addressPosture: PostureMetrics["addressPosture"] = {
    spineAngle: 35,
    kneeFlexion: 25,
    armHang: "good",
  };

  if (addressFrame) {
    const angles = calculateBodyAngles(addressFrame.landmarks, stance);
    addressPosture = {
      spineAngle: angles.spineAngle,
      kneeFlexion: (angles.leftKneeFlexion + angles.rightKneeFlexion) / 2,
      armHang: "good", // DEFERRED: Implement arm hang detection
    };
  }

  // Head stability (compare head position throughout swing)
  let headStability = 100;
  if (addressFrame) {
    const addressNose = addressFrame.landmarks[PoseLandmark.NOSE];
    let maxMovement = 0;

    for (const frame of frames) {
      const currentNose = frame.landmarks[PoseLandmark.NOSE];
      const movement = calculateDistance(addressNose, currentNose) * 200; // cm
      maxMovement = Math.max(maxMovement, movement);
    }

    // Score decreases as head movement increases (10cm max = 0 score)
    headStability = Math.max(0, 100 - maxMovement * 10);
  }

  // Detect common faults
  let earlyExtension = false;
  let lossOfPosture = false;
  let reverseSpineTilt = false;

  if (addressFrame && impactFrame) {
    const addressAngles = calculateBodyAngles(addressFrame.landmarks, stance);
    const impactAngles = calculateBodyAngles(impactFrame.landmarks, stance);

    // Early extension: hips move toward ball, spine becomes more upright
    earlyExtension = impactAngles.spineAngle < addressAngles.spineAngle - 10;

    // Loss of posture: significant change in spine angle
    lossOfPosture =
      Math.abs(impactAngles.spineAngle - addressAngles.spineAngle) > 15;

    // Reverse spine tilt: spine tilts toward target in backswing (wrong direction)
    if (topFrame) {
      const topAngles = calculateBodyAngles(topFrame.landmarks, stance);
      reverseSpineTilt = topAngles.spineLateral < -10;
    }
  }

  return {
    addressPosture,
    headStability,
    earlyExtension,
    lossOfPosture,
    reverseSpineTilt,
  };
}

/**
 * Identify swing issues and faults
 */
function identifySwingIssues(
  keyPositions: SwingAnalysis["keyPositions"],
  tempo: TempoMetrics,
  balance: BalanceMetrics,
  plane: PlaneMetrics,
  posture: PostureMetrics,
): SwingIssue[] {
  const issues: SwingIssue[] = [];

  // Tempo issues
  if (tempo.tempoRatio < 2) {
    issues.push({
      id: uuidv4(),
      name: "Quick Tempo",
      severity: "moderate",
      phase: SwingPhase.BACKSWING,
      description: "Backswing is too fast relative to downswing",
      detectedAt: 0,
      measurementValue: tempo.tempoRatio,
      expectedRange: [2.5, 3.5],
      drillRecommendation:
        'Practice counting "1-2-3" during backswing, "1" during downswing',
    });
  } else if (tempo.tempoRatio > 4) {
    issues.push({
      id: uuidv4(),
      name: "Slow Tempo",
      severity: "minor",
      phase: SwingPhase.BACKSWING,
      description: "Backswing is too slow, may cause timing issues",
      detectedAt: 0,
      measurementValue: tempo.tempoRatio,
      expectedRange: [2.5, 3.5],
      drillRecommendation: "Use a metronome at 80 BPM to practice rhythm",
    });
  }

  // Balance issues
  if (balance.swayAmount > 15) {
    issues.push({
      id: uuidv4(),
      name: "Excessive Sway",
      severity: "major",
      phase: SwingPhase.BACKSWING,
      description: `Lateral hip movement of ${balance.swayAmount.toFixed(1)}cm during backswing`,
      detectedAt: keyPositions.top?.frameNumber || 0,
      measurementValue: balance.swayAmount,
      expectedRange: [0, 10],
      drillRecommendation:
        "Place a club against your trail hip and practice rotating without moving it",
    });
  }

  if (balance.slideAmount > 20) {
    issues.push({
      id: uuidv4(),
      name: "Excessive Slide",
      severity: "moderate",
      phase: SwingPhase.DOWNSWING,
      description: `Lateral movement of ${balance.slideAmount.toFixed(1)}cm during downswing`,
      detectedAt: keyPositions.impact?.frameNumber || 0,
      measurementValue: balance.slideAmount,
      expectedRange: [5, 15],
      drillRecommendation:
        "Practice with a stability ball between your lead knee and a wall",
    });
  }

  // Posture issues
  if (posture.earlyExtension) {
    issues.push({
      id: uuidv4(),
      name: "Early Extension",
      severity: "major",
      phase: SwingPhase.DOWNSWING,
      description:
        "Hips move toward the ball and spine becomes more upright before impact",
      detectedAt: keyPositions.impact?.frameNumber || 0,
      measurementValue: 0,
      expectedRange: [0, 0],
      drillRecommendation:
        "Practice with your glutes touching a wall throughout the swing",
    });
  }

  if (posture.reverseSpineTilt) {
    issues.push({
      id: uuidv4(),
      name: "Reverse Spine Tilt",
      severity: "major",
      phase: SwingPhase.BACKSWING,
      description: "Spine tilts toward target during backswing instead of away",
      detectedAt: keyPositions.top?.frameNumber || 0,
      measurementValue: 0,
      expectedRange: [0, 0],
      drillRecommendation:
        "Keep your head behind the ball and feel your weight move into your trail side",
    });
  }

  if (posture.headStability < 60) {
    issues.push({
      id: uuidv4(),
      name: "Head Movement",
      severity: "moderate",
      phase: SwingPhase.BACKSWING,
      description: "Excessive head movement during swing",
      detectedAt: 0,
      measurementValue: 100 - posture.headStability,
      expectedRange: [0, 40],
      drillRecommendation:
        "Practice swinging with a mirror or video to monitor head position",
    });
  }

  // Plane issues
  if (!plane.onPlane && plane.planeDifferential > 10) {
    issues.push({
      id: uuidv4(),
      name: "Swing Plane Inconsistency",
      severity: "moderate",
      phase: SwingPhase.DOWNSWING,
      description: `Downswing plane differs from backswing by ${plane.planeDifferential.toFixed(1)}°`,
      detectedAt: 0,
      measurementValue: plane.planeDifferential,
      expectedRange: [0, 8],
      drillRecommendation:
        "Use alignment sticks to visualize and practice consistent plane",
    });
  }

  // X-Factor issues
  if (keyPositions.top?.angles.xFactor !== undefined) {
    const xFactor = keyPositions.top.angles.xFactor;
    if (xFactor < 30) {
      issues.push({
        id: uuidv4(),
        name: "Insufficient X-Factor",
        severity: "minor",
        phase: SwingPhase.TOP_OF_BACKSWING,
        description: `X-Factor of ${xFactor.toFixed(1)}° is below optimal range`,
        detectedAt: keyPositions.top.frameNumber,
        measurementValue: xFactor,
        expectedRange: [40, 60],
        drillRecommendation:
          "Practice shoulder rotation with hips stable to increase separation",
      });
    }
  }

  return issues;
}

/**
 * Generate recommendations based on identified issues
 */
function generateRecommendations(issues: SwingIssue[]): string[] {
  const recommendations: string[] = [];

  // Prioritize major issues
  const majorIssues = issues.filter((i) => i.severity === "major");
  const moderateIssues = issues.filter((i) => i.severity === "moderate");

  if (majorIssues.length > 0) {
    recommendations.push(
      `Focus on fixing ${majorIssues.length} major issue${majorIssues.length > 1 ? "s" : ""}: ${majorIssues.map((i) => i.name).join(", ")}`,
    );
  }

  // Add specific drill recommendations
  for (const issue of [...majorIssues, ...moderateIssues].slice(0, 3)) {
    if (issue.drillRecommendation) {
      recommendations.push(`For ${issue.name}: ${issue.drillRecommendation}`);
    }
  }

  // Add general recommendations
  if (issues.length === 0) {
    recommendations.push(
      "Great swing! Focus on consistency and maintaining good fundamentals.",
    );
  } else if (issues.length <= 2) {
    recommendations.push(
      "Good overall swing mechanics. Work on the identified areas for improvement.",
    );
  } else {
    recommendations.push(
      "Consider working with a PGA professional to address multiple swing issues systematically.",
    );
  }

  return recommendations;
}

/**
 * Calculate swing scores
 */
function calculateSwingScores(
  tempo: TempoMetrics,
  balance: BalanceMetrics,
  plane: PlaneMetrics,
  posture: PostureMetrics,
  issues: SwingIssue[],
): SwingScores {
  // Tempo score (ideal ratio ~3:1)
  const tempoDeviation = Math.abs(tempo.tempoRatio - 3);
  const tempoScore = Math.max(0, 100 - tempoDeviation * 20);

  // Balance score
  const swayPenalty = Math.min(30, balance.swayAmount * 2);
  const slidePenalty = Math.min(20, balance.slideAmount);
  const balanceScore = Math.max(0, 100 - swayPenalty - slidePenalty);

  // Plane score
  const planeScore = Math.max(0, 100 - plane.planeDifferential * 5);

  // Posture score
  let postureScore = posture.headStability;
  if (posture.earlyExtension) postureScore -= 20;
  if (posture.lossOfPosture) postureScore -= 15;
  if (posture.reverseSpineTilt) postureScore -= 25;
  postureScore = Math.max(0, postureScore);

  // Rotation score (based on X-factor and shoulder turn)
  const rotationScore = 75; // Default, would be calculated from actual measurements

  // Timing score (based on sequence)
  const timingScore =
    tempo.rhythm === "smooth" ? 90 : tempo.rhythm === "uneven" ? 50 : 70;

  // Consistency score (based on variance in metrics)
  const consistencyScore = 80; // Would need multiple swings to calculate

  // Issue penalty
  const issuePenalty =
    issues.filter((i) => i.severity === "major").length * 10 +
    issues.filter((i) => i.severity === "moderate").length * 5 +
    issues.filter((i) => i.severity === "minor").length * 2;

  // Overall score
  const componentScores = [
    tempoScore,
    balanceScore,
    planeScore,
    postureScore,
    rotationScore,
    timingScore,
  ];
  const avgScore =
    componentScores.reduce((sum, s) => sum + s, 0) / componentScores.length;
  const overall = Math.max(0, Math.min(100, avgScore - issuePenalty / 2));

  return {
    overall,
    tempo: tempoScore,
    balance: balanceScore,
    plane: planeScore,
    posture: postureScore,
    rotation: rotationScore,
    timing: timingScore,
    consistency: consistencyScore,
  };
}

/**
 * Quick analysis for real-time feedback
 */
export function quickAnalyze(
  landmarks: Landmark[],
  previousLandmarks?: Landmark[],
  stance: StanceDirection = StanceDirection.RIGHT_HANDED,
): {
  angles: BodyAngles;
  currentPhase: SwingPhase;
  issues: string[];
} {
  const angles = calculateBodyAngles(landmarks, stance);

  // Simple phase detection based on current angles
  let currentPhase = SwingPhase.UNKNOWN;
  if (
    Math.abs(angles.shoulderRotation) < 15 &&
    Math.abs(angles.hipRotation) < 10
  ) {
    currentPhase = SwingPhase.ADDRESS;
  } else if (angles.shoulderRotation > 60) {
    currentPhase = SwingPhase.BACKSWING;
  } else if (angles.shoulderRotation > 80) {
    currentPhase = SwingPhase.TOP_OF_BACKSWING;
  }

  // Quick issue detection
  const issues: string[] = [];
  if (angles.spineAngle < 20 || angles.spineAngle > 50) {
    issues.push("Check spine angle");
  }
  if (Math.abs(angles.spineLateral) > 15) {
    issues.push("Excessive lateral bend");
  }

  return { angles, currentPhase, issues };
}
