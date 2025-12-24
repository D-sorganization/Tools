/**
 * Tests for VideoUploader component
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VideoUploader from '../VideoUploader';
import { createMockFile } from '@/test/utils';

describe('VideoUploader', () => {
  let onVideoUpload: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onVideoUpload = vi.fn();
  });

  describe('Rendering', () => {
    it('should render upload area', () => {
      render(<VideoUploader onVideoUpload={onVideoUpload} />);

      expect(screen.getByText(/Drop your video here, or click to browse/i)).toBeInTheDocument();
    });

    it('should display supported formats', () => {
      render(<VideoUploader onVideoUpload={onVideoUpload} />);

      expect(screen.getByText(/Supported formats: MP4, WebM, OGG, MOV, AVI, MKV/i)).toBeInTheDocument();
    });

    it('should display maximum file size', () => {
      render(<VideoUploader onVideoUpload={onVideoUpload} />);

      expect(screen.getByText(/Maximum file size: 500MB/i)).toBeInTheDocument();
    });

    it('should render upload icon', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);

      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
      expect(svg).toHaveClass('w-16', 'h-16', 'text-gray-400');
    });

    it('should render hidden file input', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);

      const input = container.querySelector('input[type="file"]');
      expect(input).toBeInTheDocument();
      expect(input).toHaveClass('sr-only');
      expect(input).toHaveAttribute('accept', 'video/*');
    });

    it('should not display error initially', () => {
      render(<VideoUploader onVideoUpload={onVideoUpload} />);

      const errorElement = screen.queryByText(/File type not supported/i);
      expect(errorElement).not.toBeInTheDocument();
    });
  });

  describe('File Validation', () => {
    it('should accept valid MP4 file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
      expect(screen.queryByText(/File type not supported/i)).not.toBeInTheDocument();
    });

    it('should accept valid WebM file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.webm', 10 * 1024 * 1024, 'video/webm');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should accept valid OGG file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.ogg', 10 * 1024 * 1024, 'video/ogg');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should accept valid MOV file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mov', 10 * 1024 * 1024, 'video/quicktime');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should accept valid AVI file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.avi', 10 * 1024 * 1024, 'video/x-msvideo');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should accept valid MKV file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mkv', 10 * 1024 * 1024, 'video/x-matroska');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should reject non-video file type', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');

      const user = userEvent.setup({ applyAccept: false });
      await user.upload(input, file);

      expect(onVideoUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/File type not supported/i)).toBeInTheDocument();
    });

    it('should reject PDF file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.pdf', 1 * 1024 * 1024, 'application/pdf');

      const user = userEvent.setup({ applyAccept: false });
      await user.upload(input, file);

      expect(onVideoUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/File type not supported/i)).toBeInTheDocument();
    });

    it('should reject file larger than 500MB', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 501 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(onVideoUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/File size too large/i)).toBeInTheDocument();
      expect(screen.getByText(/Maximum size: 500\.00 MB/i)).toBeInTheDocument();
    });

    it('should accept file exactly at 500MB limit', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 500 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
      expect(screen.queryByText(/File size too large/i)).not.toBeInTheDocument();
    });

    it('should accept very small video file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 1024, 'video/mp4'); // 1KB

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });
  });

  describe('Drag and Drop', () => {
    it('should change styling when dragging over', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      fireEvent.dragOver(dropArea);

      expect(dropArea.className).toContain('border-blue-500');
      expect(dropArea.className).toContain('bg-blue-50');
    });

    it('should revert styling when drag leaves', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      fireEvent.dragOver(dropArea);
      fireEvent.dragLeave(dropArea);

      expect(dropArea.className).not.toContain('border-blue-500');
      expect(dropArea.className).not.toContain('bg-blue-50');
    });

    it('should handle valid video file drop', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');

      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [file],
        },
      });

      await waitFor(() => {
        expect(onVideoUpload).toHaveBeenCalledWith(file);
      });
    });

    it('should revert dragging state after drop', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');

      fireEvent.dragOver(dropArea);
      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [file],
        },
      });

      expect(dropArea.className).not.toContain('border-blue-500');
      expect(dropArea.className).not.toContain('bg-blue-50');
    });

    it('should show error when dropping non-video file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');

      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [file],
        },
      });

      await waitFor(() => {
        expect(screen.getByText(/Please drop a video file/i)).toBeInTheDocument();
      });
      expect(onVideoUpload).not.toHaveBeenCalled();
    });

    it('should find first video file when multiple files dropped', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const imageFile = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');
      const videoFile = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');

      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [imageFile, videoFile],
        },
      });

      await waitFor(() => {
        expect(onVideoUpload).toHaveBeenCalledWith(videoFile);
      });
    });

    it('should show error when no video files in drop', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file1 = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');
      const file2 = createMockFile('test.pdf', 1 * 1024 * 1024, 'application/pdf');

      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [file1, file2],
        },
      });

      await waitFor(() => {
        expect(screen.getByText(/Please drop a video file/i)).toBeInTheDocument();
      });
      expect(onVideoUpload).not.toHaveBeenCalled();
    });

    it('should prevent default drag over behavior', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const event = new Event('dragover', { bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, 'preventDefault');
      const stopPropagationSpy = vi.spyOn(event, 'stopPropagation');

      dropArea.dispatchEvent(event);

      expect(preventDefaultSpy).toHaveBeenCalled();
      expect(stopPropagationSpy).toHaveBeenCalled();
    });

    it('should prevent default drop behavior', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');
      const event = new Event('drop', { bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(event, 'preventDefault');

      Object.defineProperty(event, 'dataTransfer', {
        value: { files: [file] },
      });

      dropArea.dispatchEvent(event);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });
  });

  describe('Click to Upload', () => {
    it('should trigger file input click when area is clicked', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;
      const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

      const clickSpy = vi.spyOn(fileInput, 'click');

      fireEvent.click(dropArea);

      expect(clickSpy).toHaveBeenCalled();
    });

    it('should have cursor-pointer class on drop area', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      expect(dropArea.className).toContain('cursor-pointer');
    });
  });

  describe('Error Display', () => {
    it('should display error message when validation fails', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');

      const user = userEvent.setup({ applyAccept: false });
      await user.upload(input, file);

      const errorElement = screen.getByText(/File type not supported/i);
      expect(errorElement).toBeInTheDocument();
      expect(errorElement.closest('div')).toHaveClass('bg-red-50');
    });

    it('should apply error styling to drop area', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      const file = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');

      const user = userEvent.setup({ applyAccept: false });
      await user.upload(input, file);

      expect(dropArea.className).toContain('border-red-300');
      expect(dropArea.className).toContain('bg-red-50');
    });

    it('should clear error when valid file is uploaded after error', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      // First, upload invalid file
      const invalidFile = createMockFile('test.jpg', 1 * 1024 * 1024, 'image/jpeg');
      const user = userEvent.setup({ applyAccept: false });
      await user.upload(input, invalidFile);

      expect(screen.getByText(/File type not supported/i)).toBeInTheDocument();

      // Then, upload valid file
      const validFile = createMockFile('test.mp4', 10 * 1024 * 1024, 'video/mp4');
      await userEvent.upload(input, validFile);

      expect(screen.queryByText(/File type not supported/i)).not.toBeInTheDocument();
      expect(onVideoUpload).toHaveBeenCalledWith(validFile);
    });

    it('should show file size error with maximum size', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 600 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(screen.getByText(/File size too large/i)).toBeInTheDocument();
      expect(screen.getByText(/Maximum size: 500\.00 MB/i)).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty file input change', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      fireEvent.change(input, { target: { files: [] } });

      expect(onVideoUpload).not.toHaveBeenCalled();
    });

    it('should handle null file input', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      fireEvent.change(input, { target: { files: null } });

      expect(onVideoUpload).not.toHaveBeenCalled();
    });

    it('should handle drag events with no files', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      fireEvent.drop(dropArea, {
        dataTransfer: {
          files: [],
        },
      });

      expect(screen.getByText(/Please drop a video file/i)).toBeInTheDocument();
    });

    it('should only process first file from input', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file1 = createMockFile('test1.mp4', 10 * 1024 * 1024, 'video/mp4');
      const file2 = createMockFile('test2.mp4', 20 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, [file1, file2]);

      expect(onVideoUpload).toHaveBeenCalledTimes(1);
      expect(onVideoUpload).toHaveBeenCalledWith(file1);
    });

    it('should handle file with special characters in name', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test (1) [final].mp4', 10 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should handle file with unicode characters in name', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('测试视频.mp4', 10 * 1024 * 1024, 'video/mp4');

      await userEvent.upload(input, file);

      expect(onVideoUpload).toHaveBeenCalledWith(file);
    });

    it('should reject zero-byte file', async () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      const file = createMockFile('test.mp4', 0, 'video/mp4');

      await userEvent.upload(input, file);

      // Zero-byte files should be rejected by validation
      expect(onVideoUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/Invalid video file/i)).toBeInTheDocument();
    });
  });

  describe('User Feedback', () => {
    it('should show hover state on drop area', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      expect(dropArea.className).toContain('hover:border-gray-400');
      expect(dropArea.className).toContain('hover:bg-gray-50');
    });

    it('should have transition animation', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const dropArea = container.querySelector('div[class*="border-dashed"]') as HTMLElement;

      expect(dropArea.className).toContain('transition-colors');
      expect(dropArea.className).toContain('duration-200');
    });

    it('should display all allowed file types in UI', () => {
      render(<VideoUploader onVideoUpload={onVideoUpload} />);

      const formatsText = screen.getByText(/Supported formats:/i);
      expect(formatsText.textContent).toContain('MP4');
      expect(formatsText.textContent).toContain('WebM');
      expect(formatsText.textContent).toContain('OGG');
      expect(formatsText.textContent).toContain('MOV');
      expect(formatsText.textContent).toContain('AVI');
      expect(formatsText.textContent).toContain('MKV');
    });
  });

  describe('Accessibility', () => {
    it('should have proper file input accept attribute', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      expect(input).toHaveAttribute('accept', 'video/*');
    });

    it('should be keyboard accessible via file input', () => {
      const { container } = render(<VideoUploader onVideoUpload={onVideoUpload} />);
      const input = container.querySelector('input[type="file"]') as HTMLInputElement;

      expect(input).toBeInTheDocument();
      expect(input.type).toBe('file');
    });
  });
});
