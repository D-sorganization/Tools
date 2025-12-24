'use client';

import {
    AnnotationExport,
    downloadAnnotationJSON,
    exportAnnotationsToJSON,
    importAnnotationsFromJSON,
} from '@/lib/video/annotationExporter';
import { fabric } from 'fabric';
import { useRef } from 'react';

interface AnnotationExportProps {
  annotations: fabric.Object[];
  currentFrame: number;
  totalFrames?: number;
  videoId?: string;
  canvas: fabric.Canvas | null;
  disabled?: boolean;
}

export default function AnnotationExportComponent({
  annotations,
  currentFrame,
  totalFrames = 0,
  videoId,
  canvas,
  disabled = false,
}: AnnotationExportProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    if (!canvas) return;

    const exportData = exportAnnotationsToJSON(
      annotations,
      currentFrame,
      videoId,
      totalFrames
    );
    downloadAnnotationJSON(exportData);
  };

  const handleImport = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !canvas) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target?.result as string) as AnnotationExport;
        importAnnotationsFromJSON(jsonData, canvas);
      } catch (error) {
        console.error('Failed to import annotations:', error);
        alert('Failed to import annotations. Please check the file format.');
      }
    };
    reader.readAsText(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-gray-700">Annotations</h3>
      <div className="flex flex-col space-y-2">
        <button
          onClick={handleExport}
          disabled={disabled || annotations.length === 0}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Export Annotations
        </button>
        <button
          onClick={handleImport}
          disabled={disabled || !canvas}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Import Annotations
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
      <div className="text-xs text-gray-500">
        {annotations.length} annotation{annotations.length !== 1 ? 's' : ''} on current frame
      </div>
    </div>
  );
}
