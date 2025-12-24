'use client';

import { useEffect, useRef, useState } from 'react';

interface PoseDetectorProps {
  videoElement: HTMLVideoElement | null;
  enabled: boolean;
  onPoseDetected?: (landmarks: unknown) => void;
  disabled?: boolean;
}

export default function PoseDetector({
  videoElement,
  enabled,
  onPoseDetected,
  disabled = false,
}: PoseDetectorProps) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const poseRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled || disabled) return;

    const initializeMediaPipe = async () => {
      try {
        const { Pose } = await import('@mediapipe/pose');
        const { drawConnectors, drawLandmarks } = await import('@mediapipe/drawing_utils');
        const { POSE_CONNECTIONS } = await import('@mediapipe/pose');

        const pose = new Pose({
          locateFile: (file: string) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
          },
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults((results: any) => {
          if (canvasRef.current && videoElement) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

            if (results.poseLandmarks) {
              setDetectionCount((prev) => prev + 1);
              if (onPoseDetected) {
                onPoseDetected(results.poseLandmarks);
              }

              drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 2,
              });
              drawLandmarks(ctx, results.poseLandmarks, {
                color: '#FF0000',
                lineWidth: 1,
                radius: 3,
              });
            }
          }
        });

        poseRef.current = pose;
        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize MediaPipe:', error);
      }
    };

    initializeMediaPipe();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, disabled, videoElement, onPoseDetected]);

  useEffect(() => {
    if (!isInitialized || !enabled || !videoElement || disabled) {
      setIsDetecting(false);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    setIsDetecting(true);

    const detectPose = async () => {
      if (!videoElement || !poseRef.current || videoElement.readyState < 2) {
        animationFrameRef.current = requestAnimationFrame(detectPose);
        return;
      }

      try {
        await poseRef.current.send({ image: videoElement });
      } catch (error) {
        console.error('Pose detection error:', error);
      }

      animationFrameRef.current = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isInitialized, enabled, videoElement, disabled]);

  if (!enabled || disabled) {
    return null;
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-700">Pose Detection</h3>
        <div className="flex items-center space-x-2">
          {isDetecting && (
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          )}
          <span className="text-xs text-gray-500">{detectionCount} detections</span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 z-20 pointer-events-none"
        style={{ display: enabled ? 'block' : 'none' }}
      />
      <p className="text-xs text-gray-500">
        {isInitialized
          ? 'Pose detection active - landmarks shown on video'
          : 'Initializing MediaPipe...'}
      </p>
    </div>
  );
}
