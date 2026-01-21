'use client';

import { fabric } from 'fabric';
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';

interface EditorCanvasProps {
  videoElement: HTMLVideoElement | null;
  currentTime: number;
  onAnnotationChange?: (annotations: fabric.Object[]) => void;
}

export type DrawingTool = 'select' | 'line' | 'arrow' | 'text' | 'freehand';
export type DrawingColor = string;
export type DrawingStrokeWidth = number;

export interface EditorCanvasHandle {
  setTool: (tool: DrawingTool) => void;
  setColor: (color: string) => void;
  setStrokeWidth: (width: number) => void;
  clearCanvas: () => void;
  deleteSelected: () => void;
  getTool: () => DrawingTool;
  getColor: () => string;
  getStrokeWidth: () => number;
}

// Debounce utility function
function debounce<T extends (...args: unknown[]) => void>(fn: T, delay: number): T {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return ((...args: unknown[]) => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  }) as T;
}

const EditorCanvas = forwardRef<EditorCanvasHandle, EditorCanvasProps>(
  ({ videoElement, currentTime: _currentTime, onAnnotationChange }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fabricCanvasRef = useRef<fabric.Canvas | null>(null);
    const [currentTool, setCurrentTool] = useState<DrawingTool>('select');
    const [currentColor, setCurrentColor] = useState<DrawingColor>('#FF0000');
    const [strokeWidth, setStrokeWidth] = useState<DrawingStrokeWidth>(2);
    const [isDrawing, setIsDrawing] = useState(false);

    // Memoized annotation change handler to avoid recreating on every render
    const notifyAnnotationChange = useCallback(() => {
      const canvas = fabricCanvasRef.current;
      if (canvas && onAnnotationChange) {
        onAnnotationChange(canvas.getObjects());
      }
    }, [onAnnotationChange]);

    // Debounced version for frequent events
    const debouncedNotifyAnnotationChange = useCallback(
      debounce(notifyAnnotationChange, 16), // ~60fps
      [notifyAnnotationChange]
    );

    useImperativeHandle(ref, () => ({
      setTool: (tool: DrawingTool) => setCurrentTool(tool),
      setColor: (color: string) => setCurrentColor(color),
      setStrokeWidth: (width: number) => setStrokeWidth(width),
      clearCanvas: () => {
        const canvas = fabricCanvasRef.current;
        if (!canvas) return;
        canvas.clear();
        if (onAnnotationChange) {
          onAnnotationChange([]);
        }
      },
      deleteSelected: () => {
        const canvas = fabricCanvasRef.current;
        if (!canvas) return;
        const activeObjects = canvas.getActiveObjects();
        activeObjects.forEach((obj: fabric.Object) => canvas.remove(obj));
        canvas.discardActiveObject();
        canvas.renderAll();
        notifyAnnotationChange();
      },
      getTool: () => currentTool,
      getColor: () => currentColor,
      getStrokeWidth: () => strokeWidth,
    }));

    useEffect(() => {
      if (!canvasRef.current) return;

      const canvas = new fabric.Canvas(canvasRef.current, {
        width: 800,
        height: 600,
        backgroundColor: 'transparent',
      });

      fabricCanvasRef.current = canvas;

      if (onAnnotationChange) {
        onAnnotationChange(canvas.getObjects());
      }

      // Debounced resize handler to prevent excessive renders
      const resizeCanvas = debounce(() => {
        if (!videoElement || !canvas) return;
        const container = canvas.getElement().parentElement;
        if (!container) return;

        const containerWidth = container.clientWidth;
        const videoAspectRatio = videoElement.videoWidth / videoElement.videoHeight;
        const canvasHeight = containerWidth / videoAspectRatio;

        canvas.setDimensions({
          width: containerWidth,
          height: canvasHeight,
        });
        canvas.renderAll();
      }, 100); // 100ms debounce for resize

      if (videoElement) {
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
      }

      return () => {
        window.removeEventListener('resize', resizeCanvas);
        canvas.dispose();
      };
    }, [videoElement]);

    useEffect(() => {
      const canvas = fabricCanvasRef.current;
      if (!canvas) return;

      // Unified handler for all annotation changes
      const handleAnnotationChange = () => {
        debouncedNotifyAnnotationChange();
      };

      canvas.on('object:added', handleAnnotationChange);
      canvas.on('object:removed', handleAnnotationChange);
      canvas.on('object:modified', handleAnnotationChange);

      return () => {
        canvas.off('object:added', handleAnnotationChange);
        canvas.off('object:removed', handleAnnotationChange);
        canvas.off('object:modified', handleAnnotationChange);
      };
    }, [debouncedNotifyAnnotationChange]);

    useEffect(() => {
      const canvas = fabricCanvasRef.current;
      if (!canvas) return;

      canvas.isDrawingMode = currentTool === 'freehand';
      canvas.freeDrawingBrush.width = strokeWidth;
      canvas.freeDrawingBrush.color = currentColor;

      if (currentTool === 'select') {
        canvas.selection = true;
        canvas.defaultCursor = 'default';
      } else if (currentTool === 'line') {
        canvas.selection = false;
        canvas.defaultCursor = 'crosshair';
      } else if (currentTool === 'arrow') {
        canvas.selection = false;
        canvas.defaultCursor = 'crosshair';
      } else if (currentTool === 'text') {
        canvas.selection = false;
        canvas.defaultCursor = 'text';
      } else if (currentTool === 'freehand') {
        canvas.selection = false;
        canvas.defaultCursor = 'crosshair';
      }
    }, [currentTool, currentColor, strokeWidth]);

    useEffect(() => {
      const canvas = fabricCanvasRef.current;
      if (!canvas) return;

      let startPoint: { x: number; y: number } | null = null;

      const handleMouseDown = (options: fabric.IEvent) => {
        if (currentTool === 'line' || currentTool === 'arrow') {
          const pointer = canvas.getPointer(options.e);
          startPoint = { x: pointer.x, y: pointer.y };
          setIsDrawing(true);
        }
      };

      const handleMouseMove = (options: fabric.IEvent) => {
        if (
          (currentTool === 'line' || currentTool === 'arrow') &&
          isDrawing &&
          startPoint
        ) {
          const pointer = canvas.getPointer(options.e);
          const objects = canvas.getObjects();
          const existingLine = objects[objects.length - 1];

          if (existingLine && existingLine.type === 'line') {
            canvas.remove(existingLine);
          }

          if (currentTool === 'arrow') {
            const angle = Math.atan2(
              pointer.y - startPoint.y,
              pointer.x - startPoint.x
            );
            const arrowHeadLength = 15;
            const arrowHeadAngle = Math.PI / 6;

            const arrowPath = `L ${pointer.x - arrowHeadLength * Math.cos(angle - arrowHeadAngle)} ${pointer.y - arrowHeadLength * Math.sin(angle - arrowHeadAngle)} M ${pointer.x} ${pointer.y} L ${pointer.x - arrowHeadLength * Math.cos(angle + arrowHeadAngle)} ${pointer.y - arrowHeadLength * Math.sin(angle + arrowHeadAngle)}`;

            const pathData = `M ${startPoint.x} ${startPoint.y} L ${pointer.x} ${pointer.y} ${arrowPath}`;
            const arrow = new fabric.Path(pathData, {
              stroke: currentColor,
              strokeWidth: strokeWidth,
              fill: '',
              selectable: true,
              evented: true,
            });
            canvas.add(arrow);
          } else {
            const line = new fabric.Line(
              [startPoint.x, startPoint.y, pointer.x, pointer.y],
              {
                stroke: currentColor,
                strokeWidth: strokeWidth,
                selectable: true,
                evented: true,
              }
            );
            canvas.add(line);
          }
        }
      };

      const handleMouseUp = () => {
        if (currentTool === 'line' || currentTool === 'arrow') {
          setIsDrawing(false);
          startPoint = null;
        }
      };

      canvas.on('mouse:down', handleMouseDown);
      canvas.on('mouse:move', handleMouseMove);
      canvas.on('mouse:up', handleMouseUp);

      const handleTextClick = (options: fabric.IEvent) => {
        if (currentTool === 'text') {
          const pointer = canvas.getPointer(options.e);
          const text = new fabric.IText('Double click to edit', {
            left: pointer.x,
            top: pointer.y,
            fontFamily: 'Arial',
            fontSize: 20,
            fill: currentColor,
            selectable: true,
            evented: true,
          });
          canvas.add(text);
          canvas.setActiveObject(text);
          canvas.renderAll();
        }
      };

      if (currentTool === 'text') {
        canvas.on('mouse:down', handleTextClick);
      }

      return () => {
        canvas.off('mouse:down', handleMouseDown);
        canvas.off('mouse:move', handleMouseMove);
        canvas.off('mouse:up', handleMouseUp);
        if (currentTool === 'text') {
          canvas.off('mouse:down', handleTextClick);
        }
      };
    }, [currentTool, currentColor, strokeWidth, isDrawing]);

    return <canvas ref={canvasRef} className="absolute top-0 left-0 z-10" />;
  }
);

EditorCanvas.displayName = 'EditorCanvas';

export default EditorCanvas;
