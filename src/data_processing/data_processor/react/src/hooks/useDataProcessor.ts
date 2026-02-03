/**
 * Custom hook for data processor operations.
 */

import { useState, useCallback } from 'react';
import { dataProcessorApi } from '../api/client';
import type {
  DataPreviewResponse,
  ExportFormat,
  FilterParameters,
  FilterType,
  SignalInfo,
  SignalStatistics,
} from '../types';

interface DataProcessorState {
  fileId: string | null;
  signals: SignalInfo[];
  preview: DataPreviewResponse | null;
  statistics: SignalStatistics[];
  isLoading: boolean;
  error: string | null;
}

const initialState: DataProcessorState = {
  fileId: null,
  signals: [],
  preview: null,
  statistics: [],
  isLoading: false,
  error: null,
};

export function useDataProcessor() {
  const [state, setState] = useState<DataProcessorState>(initialState);

  const setLoading = useCallback((isLoading: boolean) => {
    setState((prev) => ({ ...prev, isLoading, error: null }));
  }, []);

  const setError = useCallback((error: string) => {
    setState((prev) => ({ ...prev, isLoading: false, error }));
  }, []);

  const loadFile = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const response = await dataProcessorApi.loadFile(path);
      if (response.success && response.file_id) {
        const signalsResponse = await dataProcessorApi.getSignals(response.file_id);
        const previewResponse = await dataProcessorApi.getPreview(response.file_id);
        setState((prev) => ({
          ...prev,
          fileId: response.file_id,
          signals: signalsResponse.signals,
          preview: previewResponse,
          isLoading: false,
        }));
      } else {
        setError(response.error || 'Failed to load file');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [setLoading, setError]);

  const applyFilter = useCallback(
    async (
      filterType: FilterType,
      selectedSignals: string[],
      parameters: FilterParameters
    ) => {
      if (!state.fileId) {
        setError('No file loaded');
        return;
      }

      setLoading(true);
      try {
        const response = await dataProcessorApi.applyFilter(
          state.fileId,
          filterType,
          selectedSignals,
          parameters
        );
        if (response.success) {
          // Refresh preview after filtering
          const previewResponse = await dataProcessorApi.getPreview(state.fileId);
          setState((prev) => ({
            ...prev,
            preview: previewResponse,
            isLoading: false,
          }));
        } else {
          setError(response.error || 'Failed to apply filter');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    },
    [state.fileId, setLoading, setError]
  );

  const calculateStatistics = useCallback(
    async (selectedSignals: string[]) => {
      if (!state.fileId) {
        setError('No file loaded');
        return;
      }

      setLoading(true);
      try {
        const response = await dataProcessorApi.getStatistics(
          state.fileId,
          selectedSignals
        );
        setState((prev) => ({
          ...prev,
          statistics: response.statistics,
          isLoading: false,
        }));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    },
    [state.fileId, setLoading, setError]
  );

  const exportData = useCallback(
    async (format: ExportFormat, selectedSignals: string[]) => {
      if (!state.fileId) {
        setError('No file loaded');
        return null;
      }

      setLoading(true);
      try {
        const response = await dataProcessorApi.exportData(
          state.fileId,
          format,
          selectedSignals
        );
        setState((prev) => ({ ...prev, isLoading: false }));
        return response;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        return null;
      }
    },
    [state.fileId, setLoading, setError]
  );

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  return {
    ...state,
    loadFile,
    applyFilter,
    calculateStatistics,
    exportData,
    clearError,
    reset,
  };
}

export default useDataProcessor;
