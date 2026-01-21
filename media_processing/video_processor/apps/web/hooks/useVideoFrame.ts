import { useCallback, useRef, useMemo } from 'react';

interface UseVideoFrameOptions {
  videoElement: HTMLVideoElement | null;
  fps?: number;
  cacheSize?: number;
}

// LRU Cache for frame blobs
class FrameCache {
  private cache: Map<number, Blob> = new Map();
  private maxSize: number;

  constructor(maxSize: number = 30) {
    this.maxSize = maxSize;
  }

  get(frameNumber: number): Blob | undefined {
    const blob = this.cache.get(frameNumber);
    if (blob) {
      // Move to end (most recently used)
      this.cache.delete(frameNumber);
      this.cache.set(frameNumber, blob);
    }
    return blob;
  }

  set(frameNumber: number, blob: Blob): void {
    // If key exists, delete it first to update order
    if (this.cache.has(frameNumber)) {
      this.cache.delete(frameNumber);
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry (first in map)
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(frameNumber, blob);
  }

  clear(): void {
    this.cache.clear();
  }
}

export function useVideoFrame({ videoElement, fps = 30, cacheSize = 30 }: UseVideoFrameOptions) {
  const frameCountRef = useRef(0);
  const frameCacheRef = useRef<FrameCache>(new FrameCache(cacheSize));

  // Reusable canvas for frame extraction
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  // Initialize canvas lazily
  const getCanvas = useCallback(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
      ctxRef.current = canvasRef.current.getContext('2d');
    }
    return { canvas: canvasRef.current, ctx: ctxRef.current };
  }, []);

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

      // Check cache first
      const cachedBlob = frameCacheRef.current.get(targetFrame);
      if (cachedBlob) {
        return cachedBlob;
      }

      const frameTime = targetFrame / fps;
      const originalTime = videoElement.currentTime;

      videoElement.currentTime = Math.max(0, Math.min(frameTime, videoElement.duration));

      return new Promise((resolve) => {
        const { canvas, ctx } = getCanvas();
        if (!ctx) {
          videoElement.currentTime = originalTime;
          resolve(null);
          return;
        }

        const handleSeeked = () => {
          // Resize canvas only if dimensions changed
          if (canvas.width !== videoElement!.videoWidth || canvas.height !== videoElement!.videoHeight) {
            canvas.width = videoElement!.videoWidth;
            canvas.height = videoElement!.videoHeight;
          }
          ctx.drawImage(videoElement!, 0, 0, canvas.width, canvas.height);

          canvas.toBlob((blob) => {
            videoElement!.currentTime = originalTime;
            videoElement!.removeEventListener('seeked', handleSeeked);

            // Cache the blob
            if (blob) {
              frameCacheRef.current.set(targetFrame, blob);
            }
            resolve(blob);
          }, 'image/png');
        };

        videoElement.addEventListener('seeked', handleSeeked);
      });
    },
    [videoElement, fps, getCurrentFrame, getCanvas]
  );

  const clearCache = useCallback(() => {
    frameCacheRef.current.clear();
  }, []);

  return {
    goToFrame,
    getCurrentFrame,
    getTotalFrames,
    nextFrame,
    previousFrame,
    extractFrame,
    clearCache,
  };
}
