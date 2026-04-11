import { useCallback, useRef } from "react";
import { Upload, FileText, X } from "lucide-react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  fileName: string | null;
  onClear: () => void;
  isLoading: boolean;
}

export function FileUpload({
  onFileSelect,
  fileName,
  onClear,
  isLoading,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith(".csv")) {
        onFileSelect(file);
      }
    },
    [onFileSelect],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleClick();
      }
    },
    [handleClick],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect],
  );

  if (fileName) {
    return (
      <div className="flex items-center justify-between p-4 bg-dark-800 rounded-lg border border-dark-600">
        <div className="flex items-center gap-3">
          <FileText className="w-5 h-5 text-blue-500" />
          <span className="text-dark-100">{fileName}</span>
        </div>
        <button
          onClick={onClear}
          className="p-1 hover:bg-dark-700 rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label="Clear file"
        >
          <X className="w-4 h-4 text-dark-400 hover:text-dark-100" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={isLoading ? -1 : 0}
      aria-label="Upload CSV file"
      className={`
        flex flex-col items-center justify-center p-8
        border-2 border-dashed border-dark-600 rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
        bg-dark-800/50 hover:bg-dark-800 hover:border-blue-500
        cursor-pointer transition-all duration-200
        ${isLoading ? "opacity-50 cursor-wait" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="hidden"
        disabled={isLoading}
      />
      <Upload className="w-12 h-12 text-dark-500 mb-4" />
      <p className="text-dark-200 font-medium mb-1">
        {isLoading ? "Loading..." : "Drop your CSV file here"}
      </p>
      <p className="text-dark-400 text-sm">or click to browse</p>
    </div>
  );
}

export default FileUpload;
