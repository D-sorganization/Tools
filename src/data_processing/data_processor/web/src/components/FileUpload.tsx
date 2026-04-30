import { useCallback, useRef, memo, useState } from 'react';
import { Upload, FileText, X, AlertCircle } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  fileName: string | null;
  onClear: () => void;
  isLoading: boolean;
}

// ⚡ Bolt Optimization: Wrap component in React.memo to prevent unnecessary re-renders when unrelated parent state changes
export const FileUpload = memo(function FileUpload({ onFileSelect, fileName, onClear, isLoading }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validateFile = useCallback((file: File): string | null => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const validExtensions = ['.csv'];

    if (!validExtensions.some((ext) => file.name.endsWith(ext))) {
      return 'Invalid file type. Please upload a CSV file.';
    }

    if (file.size > maxSize) {
      return `File is too large. Maximum size is 50MB. Your file is ${(file.size / 1024 / 1024).toFixed(2)}MB.`;
    }

    if (file.size === 0) {
      return 'File is empty. Please select a file with data.';
    }

    return null;
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setValidationError(null);
      const file = e.dataTransfer.files[0];
      if (file) {
        const error = validateFile(file);
        if (error) {
          setValidationError(error);
        } else {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect, validateFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }, [handleClick]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setValidationError(null);
      const file = e.target.files?.[0];
      if (file) {
        const error = validateFile(file);
        if (error) {
          setValidationError(error);
        } else {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect, validateFile]
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
    <div className="space-y-2">
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
          border-2 border-dashed rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
          bg-dark-800/50 cursor-pointer transition-all duration-200
          ${validationError ? 'border-red-500 hover:border-red-500 bg-red-900/10' : 'border-dark-600 hover:bg-dark-800 hover:border-blue-500'}
          ${isLoading ? 'opacity-50 cursor-wait' : ''}
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
        <Upload className={`w-12 h-12 mb-4 ${validationError ? 'text-red-500' : 'text-dark-500'}`} />
        <p className="text-dark-200 font-medium mb-1">
          {isLoading ? 'Loading...' : 'Drop your CSV file here'}
        </p>
        <p className="text-dark-400 text-sm">or click to browse</p>
      </div>

      {/* Validation Error Display */}
      {validationError && (
        <div className="flex items-start gap-2 p-3 bg-red-900/20 border border-red-500/50 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-300">{validationError}</p>
        </div>
      )}

      {/* File Size Info */}
      <p className="text-xs text-dark-500 text-center">
        Maximum file size: 50MB | Supported format: CSV
      </p>
    </div>
  );
});

export default FileUpload;
