import { fabric } from 'fabric';

export interface AnnotationData {
  id: string;
  type: string;
  frame: number;
  data: unknown;
  style?: Record<string, unknown>;
}

export interface AnnotationExport {
  version: string;
  videoId?: string;
  annotations: AnnotationData[];
  metadata: {
    exportDate: string;
    totalFrames: number;
    annotationCount: number;
  };
}

export function exportAnnotationsToJSON(
  annotations: fabric.Object[],
  currentFrame: number,
  videoId?: string,
  totalFrames?: number
): AnnotationExport {
  const annotationData: AnnotationData[] = annotations.map((obj, index) => ({
    id: obj.name || `annotation-${index}`,
    type: obj.type || 'unknown',
    frame: currentFrame,
    data: obj.toObject(['name', 'selectable', 'evented']),
    style: {
      stroke: (obj as fabric.Object & { stroke?: string }).stroke,
      strokeWidth: (obj as fabric.Object & { strokeWidth?: number }).strokeWidth,
      fill: (obj as fabric.Object & { fill?: string }).fill,
    },
  }));

  return {
    version: '1.0.0',
    videoId,
    annotations: annotationData,
    metadata: {
      exportDate: new Date().toISOString(),
      totalFrames: totalFrames || 0,
      annotationCount: annotationData.length,
    },
  };
}

export function importAnnotationsFromJSON(
  exportData: AnnotationExport,
  canvas: fabric.Canvas
): void {
  canvas.clear();

  exportData.annotations.forEach((annotation) => {
    try {
      fabric.util.enlivenObjects([annotation.data], (objects: fabric.Object[]) => {
        objects.forEach((obj: fabric.Object) => {
          if (annotation.style) {
            if (annotation.style.stroke) {
              (obj as fabric.Object & { stroke?: string }).stroke = annotation.style.stroke as string;
            }
            if (annotation.style.strokeWidth) {
              (obj as fabric.Object & { strokeWidth?: number }).strokeWidth = annotation.style.strokeWidth as number;
            }
            if (annotation.style.fill) {
              (obj as fabric.Object & { fill?: string }).fill = annotation.style.fill as string;
            }
          }
          canvas.add(obj);
        });
        canvas.renderAll();
      });
    } catch (error) {
      console.error(`Failed to import annotation ${annotation.id}:`, error);
    }
  });
}

export function downloadAnnotationJSON(exportData: AnnotationExport, filename?: string): void {
  const jsonStr = JSON.stringify(exportData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `annotations-${new Date().toISOString()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
