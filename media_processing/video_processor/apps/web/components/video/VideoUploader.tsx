'use client';

import { ChangeEvent, DragEvent, useRef, useState } from 'react';
import { quickValidateVideoFile, MAX_FILE_SIZE } from '@/lib/validation/video';
import { ValidationError } from '@/lib/errors';

interface VideoUploaderProps {
  onVideoUpload: (file: File) => void;
}

const MAX_FILE_SIZE_MB = MAX_FILE_SIZE / (1024 * 1024);

export default function VideoUploader({ onVideoUpload }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);

    try {
      // Use comprehensive validation with proper error handling
      quickValidateVideoFile(file);

      // If validation passes, notify parent component
      onVideoUpload(file);
    } catch (err) {
      // Display user-friendly error message
      if (err instanceof ValidationError) {
        setError(err.getUserMessage());
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Invalid video file. Please try another file.');
      }
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files) as File[];
    const videoFile = files.find((file: File) => file.type.startsWith('video/'));
    if (videoFile) {
      handleFile(videoFile);
    } else {
      setError('Please drop a video file');
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
          transition-colors duration-200
          ${
            isDragging
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
          }
          ${error ? 'border-red-300 bg-red-50' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileInputChange}
          className="sr-only"
          aria-label="Drop your video here, or click to browse"
        />

        <div className="flex flex-col items-center justify-center space-y-4">
          <svg
            className="w-16 h-16 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>

          <div>
            <p className="text-lg font-medium text-gray-900 mb-1">
              Drop your video here, or click to browse
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: MP4, WebM, OGG, MOV, AVI, MKV
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Maximum file size: {MAX_FILE_SIZE_MB}MB
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}
    </div>
  );
}
