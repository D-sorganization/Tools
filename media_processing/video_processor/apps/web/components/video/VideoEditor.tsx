'use client';

import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import { useRef, useState } from 'react';

interface VideoEditorProps {
  videoFile: File | null;
  onExport: (blob: Blob) => void;
  disabled?: boolean;
}

interface TrimRange {
  start: number;
  end: number;
}

interface CropArea {
  x: number;
  y: number;
  width: number;
  height: number;
}

type RotationAngle = 0 | 90 | 180 | 270;

export default function VideoEditor({
  videoFile,
  onExport,
  disabled = false,
}: VideoEditorProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [trimRange, setTrimRange] = useState<TrimRange>({ start: 0, end: 0 });
  const [cropArea, _setCropArea] = useState<CropArea | null>(null);
  const [rotation, setRotation] = useState<RotationAngle>(0);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const ffmpegRef = useRef<FFmpeg | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);

  const loadFFmpeg = async () => {
    if (ffmpegRef.current) return;

    const ffmpeg = new FFmpeg();
    ffmpegRef.current = ffmpeg;

    ffmpeg.on('log', ({ message }) => {
      // eslint-disable-next-line no-console
      console.log(message);
    });

    await ffmpeg.load();
  };

  const handleFileLoad = (file: File) => {
    if (!videoPreviewRef.current) return;

    const url = URL.createObjectURL(file);
    videoPreviewRef.current.src = url;

    videoPreviewRef.current.onloadedmetadata = () => {
      if (videoPreviewRef.current) {
        const duration = videoPreviewRef.current.duration;
        setVideoDuration(duration);
        setTrimRange({ start: 0, end: duration });
      }
    };
  };

  const handleExport = async () => {
    if (!videoFile || disabled || isProcessing) return;

    setIsProcessing(true);

    try {
      await loadFFmpeg();
      const ffmpeg = ffmpegRef.current;
      if (!ffmpeg) throw new Error('FFmpeg not loaded');

      await ffmpeg.writeFile('input.mp4', await fetchFile(videoFile));

      const outputArgs: string[] = ['-i', 'input.mp4'];

      if (trimRange.start > 0 || trimRange.end < videoDuration) {
        outputArgs.push('-ss', trimRange.start.toString());
        outputArgs.push('-t', (trimRange.end - trimRange.start).toString());
      }

      if (cropArea) {
        outputArgs.push(
          '-vf',
          `crop=${cropArea.width}:${cropArea.height}:${cropArea.x}:${cropArea.y}`
        );
      }

      if (rotation !== 0) {
        const rotationFilter =
          rotation === 90
            ? 'transpose=1'
            : rotation === 180
            ? 'transpose=1,transpose=1'
            : rotation === 270
            ? 'transpose=2'
            : '';
        if (rotationFilter) {
          if (cropArea) {
            const currentFilter = outputArgs[outputArgs.indexOf('-vf') + 1];
            outputArgs[outputArgs.indexOf('-vf') + 1] = `${currentFilter},${rotationFilter}`;
          } else {
            outputArgs.push('-vf', rotationFilter);
          }
        }
      }

      outputArgs.push('-c:v', 'libx264');
      outputArgs.push('-c:a', 'copy');
      outputArgs.push('output.mp4');

      await ffmpeg.exec(outputArgs);

      const data = await ffmpeg.readFile('output.mp4');
      const blob = new Blob([data as any], { type: 'video/mp4' });

      await ffmpeg.deleteFile('input.mp4');
      await ffmpeg.deleteFile('output.mp4');

      onExport(blob);
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export video. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (videoFile) {
    handleFileLoad(videoFile);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-700">Video Editing</h3>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Trim Video
        </label>
        <div className="space-y-2">
          <div>
            <label className="text-xs text-gray-600">Start: {formatTime(trimRange.start)}</label>
            <input
              type="range"
              min="0"
              max={videoDuration}
              step="0.1"
              value={trimRange.start}
              onChange={(e) =>
                setTrimRange({
                  ...trimRange,
                  start: parseFloat(e.target.value),
                })
              }
              disabled={disabled || videoDuration === 0}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
            />
          </div>
          <div>
            <label className="text-xs text-gray-600">
              End: {formatTime(trimRange.end)}
            </label>
            <input
              type="range"
              min="0"
              max={videoDuration}
              step="0.1"
              value={trimRange.end}
              onChange={(e) =>
                setTrimRange({
                  ...trimRange,
                  end: parseFloat(e.target.value),
                })
              }
              disabled={disabled || videoDuration === 0}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
            />
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Rotate Video
        </label>
        <div className="grid grid-cols-4 gap-2">
          {([0, 90, 180, 270] as RotationAngle[]).map((angle) => (
            <button
              key={angle}
              onClick={() => setRotation(angle)}
              disabled={disabled}
              className={`
                px-3 py-2 text-sm font-medium rounded-md transition-colors
                ${
                  rotation === angle
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              {angle}°
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={handleExport}
        disabled={disabled || isProcessing || !videoFile}
        className="w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isProcessing ? 'Processing...' : 'Export Video'}
      </button>

      <video
        ref={videoPreviewRef}
        className="hidden"
        preload="metadata"
      />
    </div>
  );
}
