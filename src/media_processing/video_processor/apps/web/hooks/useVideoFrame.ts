import { useCallback, useRef } from 'react';

interface UseVideoFrameOptions {
  videoElement: HTMLVideoElement | null;
  fps?: number;
}

export function useVideoFrame({ videoElement, fps = 30 }: UseVideoFrameOptions) {
  const frameCountRef = useRef(0);

  const goToFrame = useCallback(
    (frameNumber: number) => {
      if (!videoElement) return;
      const frameTime = frameNumber / fps;
      videoElement.currentTime = Math.max(0, Math.min(frameTime, videoElement.duration));
    },
    [videoElement, fps]
  );

  const getCurrentFrame = useCallback((): number => {
    if (!videoElement) return 0;
    return Math.max(0, Math.floor(videoElement.currentTime * fps));
  }, [videoElement, fps]);

  const getTotalFrames = useCallback((): number => {
    if (!videoElement || !Number.isFinite(videoElement.duration)) return 0;
    return Math.floor(videoElement.duration * fps);
  }, [videoElement, fps]);

  const nextFrame = useCallback(() => {
    if (!videoElement) return;
    const currentFrame = getCurrentFrame();
    goToFrame(currentFrame + 1);
  }, [videoElement, getCurrentFrame, goToFrame]);

  const previousFrame = useCallback(() => {
    if (!videoElement) return;
    const currentFrame = getCurrentFrame();
    goToFrame(currentFrame - 1);
  }, [videoElement, getCurrentFrame, goToFrame]);

  const extractFrame = useCallback(
    async (frameNumber?: number): Promise<Blob | null> => {
      if (!videoElement) return null;

      const targetFrame = frameNumber ?? getCurrentFrame();
      const frameTime = targetFrame / fps;
      const originalTime = videoElement.currentTime;

      videoElement.currentTime = Math.max(0, Math.min(frameTime, videoElement.duration));

      return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          videoElement.currentTime = originalTime;
          resolve(null);
          return;
        }

        const handleSeeked = () => {
          canvas.width = videoElement!.videoWidth;
          canvas.height = videoElement!.videoHeight;
          ctx.drawImage(videoElement!, 0, 0, canvas.width, canvas.height);

          canvas.toBlob((blob) => {
            videoElement!.currentTime = originalTime;
            videoElement!.removeEventListener('seeked', handleSeeked);
            resolve(blob);
          }, 'image/png');
        };

        videoElement.addEventListener('seeked', handleSeeked);
      });
    },
    [videoElement, fps, getCurrentFrame]
  );

  return {
    goToFrame,
    getCurrentFrame,
    getTotalFrames,
    nextFrame,
    previousFrame,
    extractFrame,
  };
}
