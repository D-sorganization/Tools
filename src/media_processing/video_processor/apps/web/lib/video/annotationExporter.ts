import * as fabric from 'fabric';

type AnnotationObject = fabric.Object & {
  fill?: string;
  name?: string;
  stroke?: string;
  strokeWidth?: number;
};

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
  const annotationData: AnnotationData[] = annotations.map((obj, index) => {
    const annotationObject = obj as AnnotationObject;

    return {
      id: annotationObject.name || `annotation-${index}`,
      type: annotationObject.type || 'unknown',
      frame: currentFrame,
      data: annotationObject.toObject(['name', 'selectable', 'evented']),
      style: {
        stroke: annotationObject.stroke,
        strokeWidth: annotationObject.strokeWidth,
        fill: annotationObject.fill,
      },
    };
  });

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

export async function importAnnotationsFromJSON(
  exportData: AnnotationExport,
  canvas: fabric.Canvas
): Promise<void> {
  canvas.clear();

  for (const annotation of exportData.annotations) {
    try {
      const objects = await fabric.util.enlivenObjects<fabric.Object>([annotation.data]);
      objects.forEach((obj: fabric.Object) => {
        const annotationObject = obj as AnnotationObject;
        if (annotation.style) {
          if (annotation.style.stroke) {
            annotationObject.stroke = annotation.style.stroke as string;
          }
          if (annotation.style.strokeWidth) {
            annotationObject.strokeWidth = annotation.style.strokeWidth as number;
          }
          if (annotation.style.fill) {
            annotationObject.fill = annotation.style.fill as string;
          }
        }
        canvas.add(obj);
      });
    } catch (error) {
      console.error(`Failed to import annotation ${annotation.id}:`, error);
    }
  }

  canvas.renderAll();
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
