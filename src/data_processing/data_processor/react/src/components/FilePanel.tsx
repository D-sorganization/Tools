/**
 * File selection and loading panel component.
 */

import React, { useState, useCallback } from 'react';

interface FilePanelProps {
  onLoad: (path: string) => void;
  isLoading: boolean;
}

export function FilePanel({ onLoad, isLoading }: FilePanelProps) {
  const [filePath, setFilePath] = useState('');

  const handlePathChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFilePath(e.target.value);
    },
    []
  );

  const handleLoad = useCallback(() => {
    if (filePath.trim()) {
      onLoad(filePath.trim());
    }
  }, [filePath, onLoad]);

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleLoad();
      }
    },
    [handleLoad]
  );

  return (
    <div className="panel">
      <h3 className="panel-title">File Selection</h3>

      <div className="form-group">
        <label className="form-label">File Path</label>
        <input
          type="text"
          className="input"
          placeholder="Enter CSV file path..."
          value={filePath}
          onChange={handlePathChange}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
      </div>

      <button
        className="btn"
        onClick={handleLoad}
        disabled={isLoading || !filePath.trim()}
      >
        {isLoading ? (
          <>
            <span className="spinner" /> Loading...
          </>
        ) : (
          'Load File'
        )}
      </button>
    </div>
  );
}

export default FilePanel;
