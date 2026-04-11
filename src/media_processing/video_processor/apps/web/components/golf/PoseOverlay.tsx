'use client';

import { useRef, useEffect, useCallback } from 'react';
import { Landmark, PoseLandmark, SwingPhase } from '@/lib/golf/types';
import { getPhaseAtFrame, PhaseTransition } from '@/lib/golf/phaseDetector';

interface PoseOverlayProps {
  landmarks: Landmark[];
  referenceLandmarks?: Landmark[];
  width: number;
  height: number;
  showAngles?: boolean;
  showPhase?: boolean;
  currentPhase?: SwingPhase;
  highlightDifferences?: boolean;
  primaryColor?: string;
  referenceColor?: string;
  referenceOpacity?: number;
}

// Pose connections for skeleton drawing
const POSE_CONNECTIONS: [PoseLandmark, PoseLandmark][] = [
  // Face
  [PoseLandmark.LEFT_EAR, PoseLandmark.LEFT_EYE],
  [PoseLandmark.RIGHT_EAR, PoseLandmark.RIGHT_EYE],
  [PoseLandmark.LEFT_EYE, PoseLandmark.NOSE],
  [PoseLandmark.RIGHT_EYE, PoseLandmark.NOSE],
  // Torso
  [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
  [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP],
  [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP],
  [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
  // Left arm
  [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW],
  [PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST],
  [PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY],
  [PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX],
  [PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB],
  // Right arm
  [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW],
  [PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST],
  [PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY],
  [PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX],
  [PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB],
  // Left leg
  [PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE],
  [PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE],
  [PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL],
  [PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX],
  [PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX],
  // Right leg
  [PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE],
  [PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE],
  [PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL],
  [PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX],
  [PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX],
];

// Key joints to highlight
const KEY_JOINTS = [
  PoseLandmark.LEFT_SHOULDER,
  PoseLandmark.RIGHT_SHOULDER,
  PoseLandmark.LEFT_HIP,
  PoseLandmark.RIGHT_HIP,
  PoseLandmark.LEFT_ELBOW,
  PoseLandmark.RIGHT_ELBOW,
  PoseLandmark.LEFT_WRIST,
  PoseLandmark.RIGHT_WRIST,
  PoseLandmark.LEFT_KNEE,
  PoseLandmark.RIGHT_KNEE,
];

export default function PoseOverlay({
  landmarks,
  referenceLandmarks,
  width,
  height,
  showAngles = false,
  showPhase = false,
  currentPhase,
  highlightDifferences = false,
  primaryColor = '#00FF00',
  referenceColor = '#FF6600',
  referenceOpacity = 0.5,
}: PoseOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // Draw reference pose first (underneath)
    if (referenceLandmarks) {
      drawSkeleton(ctx, referenceLandmarks, referenceColor, referenceOpacity, 2);
      drawJoints(ctx, referenceLandmarks, referenceColor, referenceOpacity, 4);
    }

    // Draw current pose
    drawSkeleton(ctx, landmarks, primaryColor, 1, 3);
    drawJoints(ctx, landmarks, primaryColor, 1, 6);

    // Draw difference indicators if enabled
    if (highlightDifferences && referenceLandmarks) {
      drawDifferenceIndicators(ctx, landmarks, referenceLandmarks);
    }

    // Draw angles if enabled
    if (showAngles) {
      drawAngleIndicators(ctx, landmarks);
    }

    // Draw phase indicator
    if (showPhase && currentPhase) {
      drawPhaseIndicator(ctx, currentPhase);
    }
  }, [
    landmarks,
    referenceLandmarks,
    width,
    height,
    showAngles,
    showPhase,
    currentPhase,
    highlightDifferences,
    primaryColor,
    referenceColor,
    referenceOpacity,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  const drawSkeleton = (
    ctx: CanvasRenderingContext2D,
    lm: Landmark[],
    color: string,
    opacity: number,
    lineWidth: number
  ) => {
    ctx.globalAlpha = opacity;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    POSE_CONNECTIONS.forEach(([start, end]) => {
      const startPoint = lm[start];
      const endPoint = lm[end];

      if (
        startPoint &&
        endPoint &&
        (startPoint.visibility || 0) > 0.3 &&
        (endPoint.visibility || 0) > 0.3
      ) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
      }
    });

    ctx.globalAlpha = 1;
  };

  const drawJoints = (
    ctx: CanvasRenderingContext2D,
    lm: Landmark[],
    color: string,
    opacity: number,
    radius: number
  ) => {
    ctx.globalAlpha = opacity;
    ctx.fillStyle = color;

    KEY_JOINTS.forEach((joint) => {
      const point = lm[joint];
      if (point && (point.visibility || 0) > 0.3) {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, radius, 0, 2 * Math.PI);
        ctx.fill();

        // White center for visibility
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, radius * 0.4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = color;
      }
    });

    ctx.globalAlpha = 1;
  };

  const drawDifferenceIndicators = (
    ctx: CanvasRenderingContext2D,
    current: Landmark[],
    reference: Landmark[]
  ) => {
    const threshold = 0.05; // 5% difference threshold

    KEY_JOINTS.forEach((joint) => {
      const curr = current[joint];
      const ref = reference[joint];

      if (!curr || !ref) return;
      if ((curr.visibility || 0) < 0.3 || (ref.visibility || 0) < 0.3) return;

      const dx = curr.x - ref.x;
      const dy = curr.y - ref.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance > threshold) {
        // Draw arrow from reference to current
        const startX = ref.x * width;
        const startY = ref.y * height;
        const endX = curr.x * width;
        const endY = curr.y * height;

        // Color based on magnitude
        const intensity = Math.min(1, distance / 0.15);
        ctx.strokeStyle = `rgba(255, ${Math.floor(255 * (1 - intensity))}, 0, 0.8)`;
        ctx.lineWidth = 2;

        // Draw line
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Draw arrowhead
        const angle = Math.atan2(endY - startY, endX - startX);
        const arrowLength = 10;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - arrowLength * Math.cos(angle - Math.PI / 6),
          endY - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - arrowLength * Math.cos(angle + Math.PI / 6),
          endY - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }
    });
  };

  const drawAngleIndicators = (ctx: CanvasRenderingContext2D, lm: Landmark[]) => {
    const drawAngle = (
      joint: PoseLandmark,
      point1: PoseLandmark,
      point2: PoseLandmark,
      label: string
    ) => {
      const j = lm[joint];
      const p1 = lm[point1];
      const p2 = lm[point2];

      if (!j || !p1 || !p2) return;
      if ((j.visibility || 0) < 0.3) return;

      const jx = j.x * width;
      const jy = j.y * height;

      // Calculate angle
      const angle1 = Math.atan2(p1.y - j.y, p1.x - j.x);
      const angle2 = Math.atan2(p2.y - j.y, p2.x - j.x);
      let angleDiff = (angle2 - angle1) * (180 / Math.PI);
      if (angleDiff < 0) angleDiff += 360;
      if (angleDiff > 180) angleDiff = 360 - angleDiff;

      // Draw arc
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(jx, jy, 20, angle1, angle2);
      ctx.stroke();

      // Draw label
      ctx.font = '12px sans-serif';
      ctx.fillStyle = 'yellow';
      ctx.textAlign = 'center';
      const labelX = jx + 35 * Math.cos((angle1 + angle2) / 2);
      const labelY = jy + 35 * Math.sin((angle1 + angle2) / 2);
      ctx.fillText(`${angleDiff.toFixed(0)}°`, labelX, labelY);
    };

    // Draw key angles
    drawAngle(
      PoseLandmark.LEFT_ELBOW,
      PoseLandmark.LEFT_SHOULDER,
      PoseLandmark.LEFT_WRIST,
      'L Elbow'
    );
    drawAngle(
      PoseLandmark.RIGHT_ELBOW,
      PoseLandmark.RIGHT_SHOULDER,
      PoseLandmark.RIGHT_WRIST,
      'R Elbow'
    );
    drawAngle(
      PoseLandmark.LEFT_KNEE,
      PoseLandmark.LEFT_HIP,
      PoseLandmark.LEFT_ANKLE,
      'L Knee'
    );
    drawAngle(
      PoseLandmark.RIGHT_KNEE,
      PoseLandmark.RIGHT_HIP,
      PoseLandmark.RIGHT_ANKLE,
      'R Knee'
    );
  };

  const drawPhaseIndicator = (ctx: CanvasRenderingContext2D, phase: SwingPhase) => {
    const phaseLabels: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: 'Address',
      [SwingPhase.TAKEAWAY]: 'Takeaway',
      [SwingPhase.BACKSWING]: 'Backswing',
      [SwingPhase.TOP_OF_BACKSWING]: 'Top',
      [SwingPhase.TRANSITION]: 'Transition',
      [SwingPhase.DOWNSWING]: 'Downswing',
      [SwingPhase.IMPACT]: 'Impact',
      [SwingPhase.FOLLOW_THROUGH]: 'Follow Through',
      [SwingPhase.FINISH]: 'Finish',
      [SwingPhase.UNKNOWN]: '',
    };

    const phaseColors: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: '#64748B',
      [SwingPhase.TAKEAWAY]: '#3B82F6',
      [SwingPhase.BACKSWING]: '#6366F1',
      [SwingPhase.TOP_OF_BACKSWING]: '#8B5CF6',
      [SwingPhase.TRANSITION]: '#A855F7',
      [SwingPhase.DOWNSWING]: '#F97316',
      [SwingPhase.IMPACT]: '#EF4444',
      [SwingPhase.FOLLOW_THROUGH]: '#22C55E',
      [SwingPhase.FINISH]: '#10B981',
      [SwingPhase.UNKNOWN]: '#9CA3AF',
    };

    const label = phaseLabels[phase];
    if (!label) return;

    // Draw phase badge
    ctx.font = 'bold 16px sans-serif';
    const textWidth = ctx.measureText(label).width;
    const padding = 12;
    const badgeWidth = textWidth + padding * 2;
    const badgeHeight = 28;
    const x = width - badgeWidth - 10;
    const y = 10;

    // Badge background
    ctx.fillStyle = phaseColors[phase];
    ctx.beginPath();
    ctx.roundRect(x, y, badgeWidth, badgeHeight, 6);
    ctx.fill();

    // Badge text
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + badgeWidth / 2, y + badgeHeight / 2);
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute top-0 left-0 pointer-events-none"
      style={{ zIndex: 30 }}
    />
  );
}
