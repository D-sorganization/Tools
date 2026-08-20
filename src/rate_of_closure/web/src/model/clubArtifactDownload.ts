/** Shared browser download boundary for deterministic club artifacts. */

/** Browser operations injected at the download boundary for deterministic tests. */
export interface ClubArtifactDownloadRuntime {
  createObjectUrl: (blob: Blob) => string;
  clickDownload: (url: string, filename: string) => void;
  revokeObjectUrl: (url: string) => void;
}

/** Return the browser implementation of the artifact download operations. */
export function browserArtifactDownloadRuntime(): ClubArtifactDownloadRuntime {
  return {
    createObjectUrl: (blob) => URL.createObjectURL(blob),
    clickDownload: (url, filename) => {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.append(anchor);
      try {
        anchor.click();
      } finally {
        anchor.remove();
      }
    },
    revokeObjectUrl: (url) => URL.revokeObjectURL(url),
  };
}

/** Download one artifact and release its object URL after every click outcome. */
export function downloadClubArtifact(
  payload: BlobPart,
  mediaType: string,
  filename: string,
  runtime: ClubArtifactDownloadRuntime,
): void {
  if (!mediaType) throw new Error("artifact media type must be non-empty");
  if (!filename) throw new Error("artifact filename must be non-empty");
  const url = runtime.createObjectUrl(new Blob([payload], { type: mediaType }));
  if (!url) throw new Error("browser did not create an artifact download URL");
  try {
    runtime.clickDownload(url, filename);
  } finally {
    runtime.revokeObjectUrl(url);
  }
}
