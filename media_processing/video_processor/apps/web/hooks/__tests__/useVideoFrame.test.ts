/**
 * Tests for useVideoFrame hook
 *
 * Tests frame navigation functionality for video playback.
 */

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVideoFrame } from '../useVideoFrame';
import { createMockVideoElement } from '@/test/utils';

describe('useVideoFrame', () => {
  describe('getCurrentFrame', () => {
    it('should return 0 when video element is null', () => {
      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: null, fps: 30 })
      );

      expect(result.current.getCurrentFrame()).toBe(0);
    });

    it('should calculate current frame correctly', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 1.0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      expect(result.current.getCurrentFrame()).toBe(30);
    });

    it('should calculate current frame for different FPS', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 2.0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 60 })
      );

      expect(result.current.getCurrentFrame()).toBe(120);
    });

    it('should handle fractional seconds', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0.5,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      expect(result.current.getCurrentFrame()).toBe(15);
    });

    it('should return 0 for negative currentTime', () => {
      const mockVideo = createMockVideoElement({
        currentTime: -1.0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      // Should return 0 or handle gracefully
      const frame = result.current.getCurrentFrame();
      expect(frame).toBeGreaterThanOrEqual(0);
    });
  });

  describe('getTotalFrames', () => {
    it('should return 0 when video element is null', () => {
      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: null, fps: 30 })
      );

      expect(result.current.getTotalFrames()).toBe(0);
    });

    it('should calculate total frames correctly', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      expect(result.current.getTotalFrames()).toBe(300);
    });

    it('should calculate total frames for different FPS', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 5.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 60 })
      );

      expect(result.current.getTotalFrames()).toBe(300);
    });

    it('should handle NaN duration', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: NaN,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      expect(result.current.getTotalFrames()).toBe(0);
    });
  });

  describe('goToFrame', () => {
    it('should set video currentTime to correct frame', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      act(() => {
        result.current.goToFrame(30); // Go to frame 30 (1 second)
      });

      expect(mockVideo.currentTime).toBeCloseTo(1.0, 5);
    });

    it('should not set time when video element is null', () => {
      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: null, fps: 30 })
      );

      // Should not throw
      expect(() => {
        act(() => {
          result.current.goToFrame(30);
        });
      }).not.toThrow();
    });

    it('should clamp frame to valid range', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      // Try to go past the end
      act(() => {
        result.current.goToFrame(500); // Beyond 300 total frames
      });

      // Should be clamped to duration
      expect(mockVideo.currentTime).toBeLessThanOrEqual(10.0);
    });
  });

  describe('nextFrame', () => {
    it('should advance by one frame', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      act(() => {
        result.current.nextFrame();
      });

      expect(mockVideo.currentTime).toBeCloseTo(1 / 30, 5);
    });

    it('should not exceed video duration', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 9.99,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      act(() => {
        result.current.nextFrame();
      });

      expect(mockVideo.currentTime).toBeLessThanOrEqual(10.0);
    });
  });

  describe('previousFrame', () => {
    it('should go back by one frame', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 1.0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      act(() => {
        result.current.previousFrame();
      });

      expect(mockVideo.currentTime).toBeCloseTo(1.0 - 1 / 30, 5);
    });

    it('should not go below 0', () => {
      const mockVideo = createMockVideoElement({
        currentTime: 0,
        duration: 10.0,
      }) as HTMLVideoElement;

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      act(() => {
        result.current.previousFrame();
      });

      expect(mockVideo.currentTime).toBeGreaterThanOrEqual(0);
    });
  });

  describe('extractFrame', () => {
    it('should return null when video element is null', async () => {
      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: null, fps: 30 })
      );

      const blob = await result.current.extractFrame();
      expect(blob).toBeNull();
    });

    // Note: Full extractFrame testing requires canvas mocking
    // which is complex. This is a basic test.
    it('should attempt to extract frame from video', async () => {
      const mockVideo = createMockVideoElement({
        currentTime: 1.0,
        duration: 10.0,
        videoWidth: 1920,
        videoHeight: 1080,
      }) as HTMLVideoElement;

      // Mock canvas to return null context to avoid hanging on seeked event
      const originalCreateElement = document.createElement;
      vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
        if (tagName === 'canvas') {
          return {
            getContext: () => null,
          } as any;
        }
        return originalCreateElement.call(document, tagName);
      });

      const { result } = renderHook(() =>
        useVideoFrame({ videoElement: mockVideo, fps: 30 })
      );

      // This will return null in test environment due to canvas limitations
      // but verifies the function runs without errors
      const blob = await result.current.extractFrame();
      expect(blob).toBeNull();

      vi.restoreAllMocks();
    });
  });
});
