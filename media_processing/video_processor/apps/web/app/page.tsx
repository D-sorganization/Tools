'use client';

import PoseDetector from '@/components/ai/PoseDetector';
import AnnotationExportComponent from '@/components/annotations/AnnotationExport';
import AudioRecorder from '@/components/audio/AudioRecorder';
import ToolsPanel from '@/components/tools/ToolsPanel';
import EditorCanvas, { DrawingTool, EditorCanvasHandle } from '@/components/video/EditorCanvas';
import FrameNavigator from '@/components/video/FrameNavigator';
import VideoEditor from '@/components/video/VideoEditor';
import VideoPlayer from '@/components/video/VideoPlayer';
import VideoUploader from '@/components/video/VideoUploader';
import { useVideoFrame } from '@/hooks/useVideoFrame';
import { logger } from '@/lib/logger';
import { AudioRecordingError, VideoProcessingError, getUserMessage } from '@/lib/errors';
import { fabric } from 'fabric';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

export default function HomePage() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [currentTool, setCurrentTool] = useState<DrawingTool>('select');
  const [currentColor, setCurrentColor] = useState('#FF0000');
  const [strokeWidth, setStrokeWidth] = useState(2);
  const [annotations, setAnnotations] = useState<fabric.Object[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [poseDetectionEnabled, setPoseDetectionEnabled] = useState(false);
  const [fabricCanvas, _setFabricCanvas] = useState<fabric.Canvas | null>(null);
  const canvasRef = useRef<EditorCanvasHandle>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Note: config is not imported here to avoid client-side issues
  // TODO: Move fps to client-side config or use from video metadata
  const DEFAULT_FPS = 30; // Could be from process.env.NEXT_PUBLIC_DEFAULT_VIDEO_FPS

  const { getCurrentFrame, getTotalFrames } = useVideoFrame({
    videoElement,
    fps: DEFAULT_FPS,
  });

  const handleVideoUpload = (file: File) => {
    try {
      // Note: File validation is performed by VideoUploader component
      // before this handler is called (quickValidateVideoFile)
      if (!file) {
        throw new VideoProcessingError('No file provided');
      }

      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);

      logger.info('Video uploaded successfully', {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
      });

      toast.success(`Video loaded: ${file.name}`);
    } catch (error) {
      logger.error('Failed to upload video', { error, file });
      toast.error(getUserMessage(error));
    }
  };

  const handleClearVideo = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoFile(null);
    setVideoUrl(null);
    setVideoElement(null);
    setAnnotations([]);
    setCurrentFrame(0);
    setTotalFrames(0);
    setPoseDetectionEnabled(false);
  };

  const handleTimeUpdate = (time: number) => {
    setCurrentTime(time);
    const frame = getCurrentFrame();
    setCurrentFrame(frame);
  };

  const handleVideoElementReady = (element: HTMLVideoElement | null) => {
    setVideoElement(element);
    videoRef.current = element;
    if (element) {
      const frames = getTotalFrames();
      setTotalFrames(frames);
    }
  };

  const handleAnnotationChange = (newAnnotations: fabric.Object[]) => {
    setAnnotations(newAnnotations);
  };

  const handleAudioRecorded = (audioBlob: Blob, startTime: number) => {
    try {
      // Validate input
      if (!audioBlob || audioBlob.size === 0) {
        throw new AudioRecordingError('Empty audio blob received', {
          startTime,
          blobSize: audioBlob?.size,
        });
      }

      const audioUrl = URL.createObjectURL(audioBlob);

      logger.info('Audio recorded successfully', {
        audioUrl,
        startTime,
        duration: audioBlob.size,
        videoId: videoFile?.name,
      });

      toast.success('Audio commentary recorded successfully');

      // TODO: Save to database when backend is ready
      // await saveAudioTrack({ url: audioUrl, startTime, duration: audioBlob.size });
    } catch (error) {
      logger.error('Failed to process audio recording', { error, audioBlob, startTime });
      toast.error(getUserMessage(error));
    }
  };

  const handleVideoExport = (blob: Blob) => {
    try {
      // Validate input
      if (!blob || blob.size === 0) {
        throw new VideoProcessingError('Empty video blob received', {
          blobSize: blob?.size,
        });
      }

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `edited-${videoFile?.name || 'video.mp4'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      logger.info('Video exported successfully', {
        fileName: a.download,
        fileSize: blob.size,
      });

      toast.success(`Video exported: ${a.download}`);
    } catch (error) {
      logger.error('Failed to export video', { error, blob });
      toast.error(getUserMessage(error));
    }
  };

  const handlePoseDetected = (landmarks: unknown) => {
    try {
      logger.debug('Pose detected', {
        landmarks,
        videoId: videoFile?.name,
        currentTime,
        currentFrame,
      });

      // TODO: Save pose data to state or database when ready
      // For now, just log for debugging
    } catch (error) {
      logger.error('Failed to process pose detection', { error, landmarks });
      // Don't show toast for pose detection errors (they're continuous)
      // toast.error(getUserMessage(error));
    }
  };

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  useEffect(() => {
    if (canvasRef.current) {
      canvasRef.current.setTool(currentTool);
      canvasRef.current.setColor(currentColor);
      canvasRef.current.setStrokeWidth(strokeWidth);
    }
  }, [currentTool, currentColor, strokeWidth]);

  useEffect(() => {
    if (videoElement && canvasRef.current) {
      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
          return;
        }

        switch (e.key) {
          case 'ArrowLeft':
            e.preventDefault();
            if (videoElement) {
              videoElement.currentTime = Math.max(0, videoElement.currentTime - 1 / 30);
            }
            break;
          case 'ArrowRight':
            e.preventDefault();
            if (videoElement) {
              videoElement.currentTime = Math.min(
                videoElement.duration,
                videoElement.currentTime + 1 / 30
              );
            }
            break;
          case ' ':
            e.preventDefault();
            if (videoElement) {
              if (videoElement.paused) {
                videoElement.play();
              } else {
                videoElement.pause();
              }
            }
            break;
          case 'Delete':
          case 'Backspace':
            e.preventDefault();
            canvasRef.current?.deleteSelected();
            break;
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [videoElement]);

  useEffect(() => {
    if (videoElement && totalFrames === 0) {
      const frames = getTotalFrames();
      setTotalFrames(frames);
    }
  }, [videoElement, totalFrames, getTotalFrames]);

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Golf Swing Video Analyzer
          </h1>
          <p className="text-gray-600">
            Upload and analyze your golf swing videos with AI-powered tools
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Keyboard shortcuts: Space (play/pause), ← → (frame navigation), Delete (remove annotation)
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              {videoUrl ? (
                <div className="space-y-4">
                  <div className="relative">
                    <VideoPlayer
                      videoUrl={videoUrl}
                      onClear={handleClearVideo}
                      onTimeUpdate={handleTimeUpdate}
                      onVideoElementReady={handleVideoElementReady}
                    />
                    <EditorCanvas
                      ref={canvasRef}
                      videoElement={videoElement}
                      currentTime={currentTime}
                      onAnnotationChange={handleAnnotationChange}
                    />
                    {poseDetectionEnabled && (
                      <PoseDetector
                        videoElement={videoElement}
                        enabled={poseDetectionEnabled}
                        onPoseDetected={handlePoseDetected}
                        disabled={!videoUrl}
                      />
                    )}
                  </div>
                  {videoElement && totalFrames > 0 && (
                    <FrameNavigator
                      videoElement={videoElement}
                      currentFrame={currentFrame}
                      totalFrames={totalFrames}
                      fps={30}
                      onFrameChange={setCurrentFrame}
                      disabled={!videoUrl}
                    />
                  )}
                </div>
              ) : (
                <VideoUploader onVideoUpload={handleVideoUpload} />
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <ToolsPanel
                currentTool={currentTool}
                onToolChange={setCurrentTool}
                currentColor={currentColor}
                onColorChange={setCurrentColor}
                strokeWidth={strokeWidth}
                onStrokeWidthChange={setStrokeWidth}
                onClear={() => canvasRef.current?.clearCanvas()}
                onDeleteSelected={() => canvasRef.current?.deleteSelected()}
                disabled={!videoUrl}
              />
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-700">AI Features</h3>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={poseDetectionEnabled}
                    onChange={(e) => setPoseDetectionEnabled(e.target.checked)}
                    disabled={!videoUrl}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"></div>
                </label>
              </div>
              {poseDetectionEnabled && (
                <PoseDetector
                  videoElement={videoElement}
                  enabled={poseDetectionEnabled}
                  onPoseDetected={handlePoseDetected}
                  disabled={!videoUrl}
                />
              )}
              {!poseDetectionEnabled && (
                <p className="text-xs text-gray-500">
                  Enable pose detection to analyze body movements in your swing video
                </p>
              )}
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <AudioRecorder
                videoElement={videoElement}
                onRecordingComplete={handleAudioRecorded}
                disabled={!videoUrl}
              />
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <VideoEditor
                videoFile={videoFile}
                onExport={handleVideoExport}
                disabled={!videoUrl}
              />
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <AnnotationExportComponent
                annotations={annotations}
                currentFrame={currentFrame}
                totalFrames={totalFrames}
                videoId={videoFile?.name}
                canvas={fabricCanvas}
                disabled={!videoUrl}
              />
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Video Info
              </h2>
              {videoFile ? (
                <div className="space-y-2 text-sm text-gray-600">
                  <p>
                    <span className="font-medium">Name:</span> {videoFile.name}
                  </p>
                  <p>
                    <span className="font-medium">Size:</span>{' '}
                    {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                  <p>
                    <span className="font-medium">Type:</span> {videoFile.type}
                  </p>
                  <p>
                    <span className="font-medium">Annotations:</span>{' '}
                    {annotations.length}
                  </p>
                  {totalFrames > 0 && (
                    <p>
                      <span className="font-medium">Frames:</span> {totalFrames} (
                      {totalFrames / 30}s @ 30fps)
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No video uploaded</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
