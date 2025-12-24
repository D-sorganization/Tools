'use client';

import { useVideoFrame } from '@/hooks/useVideoFrame';

interface FrameNavigatorProps {
  videoElement: HTMLVideoElement | null;
  currentFrame: number;
  totalFrames: number;
  fps?: number;
  onFrameChange?: (frame: number) => void;
  disabled?: boolean;
}

export default function FrameNavigator({
  videoElement,
  currentFrame,
  totalFrames,
  fps = 30,
  onFrameChange,
  disabled = false,
}: FrameNavigatorProps) {
  const { goToFrame, nextFrame, previousFrame } = useVideoFrame({
    videoElement,
    fps,
  });

  const handleFrameInput = (frame: number) => {
    const clampedFrame = Math.max(0, Math.min(frame, totalFrames - 1));
    goToFrame(clampedFrame);
    if (onFrameChange) {
      onFrameChange(clampedFrame);
    }
  };

  const handlePreviousFrame = () => {
    previousFrame();
    if (onFrameChange) {
      onFrameChange(currentFrame - 1);
    }
  };

  const handleNextFrame = () => {
    nextFrame();
    if (onFrameChange) {
      onFrameChange(currentFrame + 1);
    }
  };

  const formatFrameTime = (frame: number): string => {
    const seconds = frame / fps;
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center space-x-2">
        <button
          onClick={handlePreviousFrame}
          disabled={disabled || currentFrame <= 0}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          title="Previous Frame"
        >
          ‹
        </button>

        <div className="flex-1 flex items-center space-x-2">
          <input
            type="number"
            min="0"
            max={totalFrames - 1}
            value={currentFrame}
            onChange={(e) => handleFrameInput(parseInt(e.target.value) || 0)}
            disabled={disabled}
            className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <span className="text-sm text-gray-600">/ {totalFrames - 1}</span>
        </div>

        <button
          onClick={handleNextFrame}
          disabled={disabled || currentFrame >= totalFrames - 1}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          title="Next Frame"
        >
          ›
        </button>
      </div>

      <div className="text-xs text-gray-500 text-center">
        Frame {currentFrame} • {formatFrameTime(currentFrame)} • {fps} fps
      </div>
    </div>
  );
}
