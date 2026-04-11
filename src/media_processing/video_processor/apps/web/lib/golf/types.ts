/**
 * Golf Swing Analysis Type Definitions
 * Professional-grade types for biomechanical analysis
 */

// MediaPipe Pose Landmark indices
export enum PoseLandmark {
  NOSE = 0,
  LEFT_EYE_INNER = 1,
  LEFT_EYE = 2,
  LEFT_EYE_OUTER = 3,
  RIGHT_EYE_INNER = 4,
  RIGHT_EYE = 5,
  RIGHT_EYE_OUTER = 6,
  LEFT_EAR = 7,
  RIGHT_EAR = 8,
  MOUTH_LEFT = 9,
  MOUTH_RIGHT = 10,
  LEFT_SHOULDER = 11,
  RIGHT_SHOULDER = 12,
  LEFT_ELBOW = 13,
  RIGHT_ELBOW = 14,
  LEFT_WRIST = 15,
  RIGHT_WRIST = 16,
  LEFT_PINKY = 17,
  RIGHT_PINKY = 18,
  LEFT_INDEX = 19,
  RIGHT_INDEX = 20,
  LEFT_THUMB = 21,
  RIGHT_THUMB = 22,
  LEFT_HIP = 23,
  RIGHT_HIP = 24,
  LEFT_KNEE = 25,
  RIGHT_KNEE = 26,
  LEFT_ANKLE = 27,
  RIGHT_ANKLE = 28,
  LEFT_HEEL = 29,
  RIGHT_HEEL = 30,
  LEFT_FOOT_INDEX = 31,
  RIGHT_FOOT_INDEX = 32,
}

// Basic 3D point with visibility confidence
export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

// Complete pose for a single frame
export interface PoseFrame {
  frameNumber: number;
  timestamp: number; // milliseconds
  landmarks: Landmark[];
  confidence: number;
}

// Swing phases based on professional golf instruction
export enum SwingPhase {
  ADDRESS = "address",
  TAKEAWAY = "takeaway",
  BACKSWING = "backswing",
  TOP_OF_BACKSWING = "top_of_backswing",
  TRANSITION = "transition",
  DOWNSWING = "downswing",
  IMPACT = "impact",
  FOLLOW_THROUGH = "follow_through",
  FINISH = "finish",
  UNKNOWN = "unknown",
}

export interface PhaseTransition {
  phase: SwingPhase;
  startFrame: number;
  endFrame: number;
  duration: number; // milliseconds
  confidence: number;
}

// Golfer's stance direction
export enum StanceDirection {
  RIGHT_HANDED = "right_handed",
  LEFT_HANDED = "left_handed",
  UNKNOWN = "unknown",
}

// Swing type categorization
export enum SwingType {
  DRIVER = "driver",
  IRON = "iron",
  WEDGE = "wedge",
  PUTTER = "putter",
  HYBRID = "hybrid",
  UNKNOWN = "unknown",
}

// Angle measurements in degrees
export interface BodyAngles {
  // Spine angles
  spineAngle: number; // Forward tilt
  spineLateral: number; // Side bend
  spineRotation: number; // Rotation around vertical axis

  // Hip angles
  hipRotation: number; // Rotation relative to target line
  hipTilt: number; // Lateral tilt
  hipSlide: number; // Lateral movement in cm

  // Shoulder angles
  shoulderRotation: number; // Rotation relative to target line
  shoulderTilt: number; // Plane angle

  // Arm angles
  leftElbowAngle: number;
  rightElbowAngle: number;
  leftWristAngle: number;
  rightWristAngle: number;

  // Knee angles
  leftKneeFlexion: number;
  rightKneeFlexion: number;

  // X-Factor (shoulder-hip differential)
  xFactor: number;
  xFactorStretch: number; // Maximum during transition
}

// Velocity measurements
export interface BodyVelocities {
  hipRotationalVelocity: number; // degrees/second
  shoulderRotationalVelocity: number;
  handSpeed: number; // meters/second
  headMovement: number; // cm of movement from address
}

// Tempo and timing metrics
export interface TempoMetrics {
  backswingDuration: number; // milliseconds
  downswingDuration: number;
  totalSwingDuration: number;
  tempoRatio: number; // backswing:downswing (e.g., 3:1 = 3.0)
  transitionPause: number; // milliseconds at top
  rhythm: "smooth" | "quick" | "slow" | "uneven";
}

// Balance and weight shift
export interface BalanceMetrics {
  addressWeight: {
    left: number; // percentage 0-100
    right: number;
  };
  topWeight: {
    left: number;
    right: number;
  };
  impactWeight: {
    left: number;
    right: number;
  };
  finishWeight: {
    left: number;
    right: number;
  };
  swayAmount: number; // cm
  slideAmount: number; // cm
  hipBump: number; // cm of lateral hip movement in downswing
}

// Plane analysis
export interface PlaneMetrics {
  backswingPlaneAngle: number; // degrees from horizontal
  downswingPlaneAngle: number;
  planeDifferential: number; // deviation between planes
  onPlane: boolean;
  shaftAngleAtAddress: number;
  shaftAngleAtTop: number;
  shaftAngleAtImpact: number;
}

// Posture analysis
export interface PostureMetrics {
  addressPosture: {
    spineAngle: number;
    kneeFlexion: number;
    armHang: "good" | "too_far" | "too_close";
  };
  headStability: number; // 0-100, higher is more stable
  earlyExtension: boolean;
  lossOfPosture: boolean;
  reverseSpineTilt: boolean;
}

// Complete swing analysis result
export interface SwingAnalysis {
  // Session info
  sessionId: string;
  videoId: string;
  analysisTimestamp: number;
  golferStance: StanceDirection;
  swingType: SwingType;

  // Frame data
  totalFrames: number;
  fps: number;
  poseFrames: PoseFrame[];

  // Phase detection
  phases: PhaseTransition[];

  // Metrics at key positions
  keyPositions: {
    address?: SwingPositionMetrics;
    top?: SwingPositionMetrics;
    impact?: SwingPositionMetrics;
    finish?: SwingPositionMetrics;
  };

  // Dynamic metrics
  tempo: TempoMetrics;
  balance: BalanceMetrics;
  plane: PlaneMetrics;
  posture: PostureMetrics;

  // Overall scores (0-100)
  scores: SwingScores;

  // Identified issues and recommendations
  issues: SwingIssue[];
  recommendations: string[];
}

// Metrics at a specific position
export interface SwingPositionMetrics {
  frameNumber: number;
  timestamp: number;
  angles: BodyAngles;
  velocities?: BodyVelocities;
  confidence: number;
}

// Scoring breakdown
export interface SwingScores {
  overall: number;
  tempo: number;
  balance: number;
  plane: number;
  posture: number;
  rotation: number;
  timing: number;
  consistency: number;
}

// Identified swing faults
export interface SwingIssue {
  id: string;
  name: string;
  severity: "minor" | "moderate" | "major";
  phase: SwingPhase;
  description: string;
  detectedAt: number; // frame number
  measurementValue: number;
  expectedRange: [number, number];
  drillRecommendation?: string;
}

// Session history for tracking improvement
export interface SwingSession {
  id: string;
  userId?: string;
  timestamp: number;
  videoFileName: string;
  videoDuration: number;
  swingCount: number;
  analysis: SwingAnalysis;
  notes?: string;
  tags?: string[];
}

// Comparison between two swings
export interface SwingComparison {
  swing1: SwingAnalysis;
  swing2: SwingAnalysis;
  differences: {
    metric: string;
    value1: number;
    value2: number;
    delta: number;
    improvement: boolean;
  }[];
  overallImprovement: number; // percentage
}

// Export format for reports
export interface SwingReport {
  generatedAt: number;
  session: SwingSession;
  analysis: SwingAnalysis;
  comparison?: SwingComparison;
  charts: {
    type: "tempo" | "angles" | "velocity" | "balance" | "trajectory";
    data: unknown;
  }[];
  summary: string;
}

// Real-time analysis state
export interface RealTimeAnalysisState {
  isAnalyzing: boolean;
  currentPhase: SwingPhase;
  frameCount: number;
  lastPoseConfidence: number;
  liveMetrics: Partial<BodyAngles>;
  issues: SwingIssue[];
}

// Configuration for analysis
export interface AnalysisConfig {
  detectStance: boolean;
  calculateVelocities: boolean;
  detectPhases: boolean;
  generateRecommendations: boolean;
  minConfidenceThreshold: number; // 0-1
  smoothingWindow: number; // frames
  targetLineAngle?: number; // degrees from camera view
}
