'use client';

import { DrawingTool } from '@/components/video/EditorCanvas';

interface ToolsPanelProps {
  currentTool: DrawingTool;
  onToolChange: (tool: DrawingTool) => void;
  currentColor: string;
  onColorChange: (color: string) => void;
  strokeWidth: number;
  onStrokeWidthChange: (width: number) => void;
  onClear: () => void;
  onDeleteSelected: () => void;
  disabled?: boolean;
}

const TOOLS: Array<{ id: DrawingTool; label: string; icon: string }> = [
  { id: 'select', label: 'Select', icon: '↖' },
  { id: 'line', label: 'Line', icon: '─' },
  { id: 'arrow', label: 'Arrow', icon: '→' },
  { id: 'text', label: 'Text', icon: 'A' },
  { id: 'freehand', label: 'Freehand', icon: '✎' },
];

const COLORS = [
  '#FF0000',
  '#00FF00',
  '#0000FF',
  '#FFFF00',
  '#FF00FF',
  '#00FFFF',
  '#FFFFFF',
  '#000000',
];

export default function ToolsPanel({
  currentTool,
  onToolChange,
  currentColor,
  onColorChange,
  strokeWidth,
  onStrokeWidthChange,
  onClear,
  onDeleteSelected,
  disabled = false,
}: ToolsPanelProps) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Drawing Tools</h3>
        <div className="grid grid-cols-5 gap-2">
          {TOOLS.map((tool) => (
            <button
              key={tool.id}
              onClick={() => !disabled && onToolChange(tool.id)}
              disabled={disabled}
              className={`
                px-3 py-2 text-sm font-medium rounded-md transition-colors
                ${
                  currentTool === tool.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
              title={tool.label}
            >
              <div className="text-lg">{tool.icon}</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Color</h3>
        <div className="grid grid-cols-4 gap-2">
          {COLORS.map((color) => (
            <button
              key={color}
              onClick={() => !disabled && onColorChange(color)}
              disabled={disabled}
              className={`
                w-full h-10 rounded-md border-2 transition-all
                ${
                  currentColor === color
                    ? 'border-gray-900 ring-2 ring-offset-2 ring-blue-500'
                    : 'border-gray-300 hover:border-gray-400'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
              style={{ backgroundColor: color }}
              title={color}
            />
          ))}
        </div>
        <input
          type="color"
          value={currentColor}
          onChange={(e) => !disabled && onColorChange(e.target.value)}
          disabled={disabled}
          className="mt-2 w-full h-10 rounded-md border border-gray-300 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Stroke Width: {strokeWidth}px
        </label>
        <input
          type="range"
          min="1"
          max="20"
          value={strokeWidth}
          onChange={(e) =>
            !disabled && onStrokeWidthChange(parseInt(e.target.value))
          }
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>

      <div className="pt-4 border-t border-gray-200 space-y-2">
        <button
          onClick={onDeleteSelected}
          disabled={disabled || currentTool !== 'select'}
          className="w-full px-4 py-2 text-sm font-medium text-red-700 bg-red-50 rounded-md hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Delete Selected
        </button>
        <button
          onClick={onClear}
          disabled={disabled}
          className="w-full px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Clear All
        </button>
      </div>
    </div>
  );
}
