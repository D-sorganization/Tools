/**
 * Tests for VideoPlayer component
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VideoPlayer from '../VideoPlayer';

describe('VideoPlayer', () => {
  const mockOnClear = vi.fn();
  const mockOnTimeUpdate = vi.fn();
  const mockOnVideoElementReady = vi.fn();

  const defaultProps = {
    videoUrl: 'blob:test-video.mp4',
    onClear: mockOnClear,
    onTimeUpdate: mockOnTimeUpdate,
    onVideoElementReady: mockOnVideoElementReady,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render video element with correct src', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;
      expect(video).toBeInTheDocument();
      expect(video.src).toContain('test-video.mp4');
    });

    it('should render play/pause button', () => {
      render(<VideoPlayer {...defaultProps} />);

      const playButton = screen.getByLabelText('Play');
      expect(playButton).toBeInTheDocument();
    });

    it('should render clear button', () => {
      render(<VideoPlayer {...defaultProps} />);

      const clearButton = screen.getByText('Upload New Video');
      expect(clearButton).toBeInTheDocument();
    });

    it('should render volume controls', () => {
      render(<VideoPlayer {...defaultProps} />);

      const muteButton = screen.getByLabelText('Mute');
      expect(muteButton).toBeInTheDocument();
    });

    it('should render time display', () => {
      render(<VideoPlayer {...defaultProps} />);

      // Should show 0:00 / 0:00 initially
      expect(screen.getByText(/0:00.*0:00/)).toBeInTheDocument();
    });
  });

  describe('Video Controls', () => {
    it('should toggle play state when play button clicked', async () => {
      const user = userEvent.setup();
      render(<VideoPlayer {...defaultProps} />);

      const playButton = screen.getByLabelText('Play');
      await user.click(playButton);

      // Should now show pause button
      await waitFor(() => {
        expect(screen.getByLabelText('Pause')).toBeInTheDocument();
      });
    });

    it('should call onClear when clear button clicked', async () => {
      const user = userEvent.setup();
      render(<VideoPlayer {...defaultProps} />);

      const clearButton = screen.getByText('Upload New Video');
      await user.click(clearButton);

      expect(mockOnClear).toHaveBeenCalledTimes(1);
    });

    it('should update time when timeline is seeked', async () => {
      render(<VideoPlayer {...defaultProps} />);

      const seekBar = screen.getAllByRole('slider')[0]; // First slider is timeline
      fireEvent.change(seekBar, { target: { value: '5' } });

      const video = screen.getByRole('video') as HTMLVideoElement;
      expect(video.currentTime).toBe(5);
    });

    it('should toggle mute when mute button clicked', async () => {
      const user = userEvent.setup();
      render(<VideoPlayer {...defaultProps} />);

      const muteButton = screen.getByLabelText('Mute');
      await user.click(muteButton);

      // Should now show unmute button
      await waitFor(() => {
        expect(screen.getByLabelText('Unmute')).toBeInTheDocument();
      });
    });

    it('should change volume when volume slider is moved', () => {
      render(<VideoPlayer {...defaultProps} />);

      const volumeSlider = screen.getAllByRole('slider')[1]; // Second slider is volume
      fireEvent.change(volumeSlider, { target: { value: '0.5' } });

      const video = screen.getByRole('video') as HTMLVideoElement;
      expect(video.volume).toBe(0.5);
    });
  });

  describe('Video Events', () => {
    it('should call onVideoElementReady when metadata is loaded', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Trigger loadedmetadata event
      fireEvent.loadedMetadata(video);

      expect(mockOnVideoElementReady).toHaveBeenCalledWith(video);
    });

    it('should call onTimeUpdate when video time updates', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;
      video.currentTime = 5;

      // Trigger timeupdate event
      fireEvent.timeUpdate(video);

      expect(mockOnTimeUpdate).toHaveBeenCalledWith(5);
    });

    it('should reset to beginning when video ends', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Trigger ended event
      fireEvent.ended(video);

      // Should show play button again
      expect(screen.getByLabelText('Play')).toBeInTheDocument();
    });
  });

  describe('Time Formatting', () => {
    it('should format seconds correctly', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Set duration to 125 seconds (2:05)
      Object.defineProperty(video, 'duration', { value: 125, writable: true });
      fireEvent.loadedMetadata(video);

      // Should display 2:05
      expect(screen.getByText(/2:05/)).toBeInTheDocument();
    });

    it('should pad seconds with leading zero', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Set duration to 65 seconds (1:05)
      Object.defineProperty(video, 'duration', { value: 65, writable: true });
      fireEvent.loadedMetadata(video);

      // Should display 1:05, not 1:5
      expect(screen.getByText(/1:05/)).toBeInTheDocument();
    });

    it('should show progress percentage', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Set duration and current time
      Object.defineProperty(video, 'duration', { value: 100, writable: true });
      video.currentTime = 50;

      fireEvent.loadedMetadata(video);
      fireEvent.timeUpdate(video);

      // Should show 50.0%
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels for play button', () => {
      render(<VideoPlayer {...defaultProps} />);

      const playButton = screen.getByLabelText('Play');
      expect(playButton).toHaveAttribute('aria-label', 'Play');
    });

    it('should have proper ARIA labels for mute button', () => {
      render(<VideoPlayer {...defaultProps} />);

      const muteButton = screen.getByLabelText('Mute');
      expect(muteButton).toHaveAttribute('aria-label', 'Mute');
    });

    it('should update ARIA label when state changes', async () => {
      const user = userEvent.setup();
      render(<VideoPlayer {...defaultProps} />);

      const playButton = screen.getByLabelText('Play');
      await user.click(playButton);

      await waitFor(() => {
        const pauseButton = screen.getByLabelText('Pause');
        expect(pauseButton).toHaveAttribute('aria-label', 'Pause');
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle missing video element gracefully', () => {
      const { container } = render(<VideoPlayer {...defaultProps} />);

      // Remove video element
      const video = container.querySelector('video');
      video?.remove();

      // Should not throw when clicking play
      const playButton = screen.getAllByRole('button')[1]; // Bottom play button
      expect(() => fireEvent.click(playButton)).not.toThrow();
    });

    it('should handle volume of 0', () => {
      render(<VideoPlayer {...defaultProps} />);

      const volumeSlider = screen.getAllByRole('slider')[1];
      fireEvent.change(volumeSlider, { target: { value: '0' } });

      // Should show muted icon
      expect(screen.getByLabelText('Unmute')).toBeInTheDocument();
    });

    it('should restore previous volume when unmuting', async () => {
      const user = userEvent.setup();
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;

      // Set volume to 0.8
      const volumeSlider = screen.getAllByRole('slider')[1];
      fireEvent.change(volumeSlider, { target: { value: '0.8' } });

      // Mute
      const muteButton = screen.getByLabelText('Mute');
      await user.click(muteButton);

      expect(video.volume).toBe(0);

      // Unmute
      const unmuteButton = screen.getByLabelText('Unmute');
      await user.click(unmuteButton);

      // Should restore to 0.8
      expect(video.volume).toBe(0.8);
    });

    it('should handle duration of 0', () => {
      render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;
      Object.defineProperty(video, 'duration', { value: 0, writable: true });
      fireEvent.loadedMetadata(video);

      // Should not crash and show 0:00
      expect(screen.getByText(/0:00.*0:00/)).toBeInTheDocument();
    });
  });

  describe('Cleanup', () => {
    it('should remove event listeners on unmount', () => {
      const { unmount } = render(<VideoPlayer {...defaultProps} />);

      const video = screen.getByRole('video') as HTMLVideoElement;
      const removeEventListenerSpy = vi.spyOn(video, 'removeEventListener');

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('timeupdate', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('loadedmetadata', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('ended', expect.any(Function));
    });
  });
});
