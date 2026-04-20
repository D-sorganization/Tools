import { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronRight } from 'lucide-react';

interface HelpSection {
  title: string;
  content: string;
}

const helpSections: HelpSection[] = [
  {
    title: 'Getting Started',
    content: `
      1. **Load Data**: Click "Select CSV File" or drag a file to load your data.
      2. **Select Signals**: Use the signal list to choose which signals to analyze.
      3. **Apply Filters**: Use the filter panel to smooth or process signals.
      4. **View Results**: Switch between chart and table views to inspect your data.
      5. **Export**: Download processed data in your preferred format.
    `,
  },
  {
    title: 'Filters',
    content: `
      **Moving Average**: Smooths data by averaging nearby points.

      **Median Filter**: Removes spikes by taking the median of nearby points.

      **Gaussian Filter**: Smooths using a Gaussian-weighted average.

      **Z-Score Filter**: Removes outliers beyond a threshold number of standard deviations.

      **Savitzky-Golay**: Polynomial smoothing that preserves peaks and valleys.
    `,
  },
  {
    title: 'Integration',
    content: `
      Calculate the cumulative integral of signals over time.

      **Methods:**
      - **Trapezoidal**: Standard numerical integration
      - **Simpson's Rule**: Higher accuracy for smooth signals
      - **Rectangular**: Simple sum (left Riemann sum)
    `,
  },
  {
    title: 'Differentiation',
    content: `
      Calculate derivatives (rate of change) of signals.

      **Methods:**
      - **Spline (Acausal)**: Uses cubic spline interpolation for smooth derivatives
      - **Rolling Polynomial (Causal)**: Uses Savitzky-Golay filter for real-time applications

      **Parameters:**
      - **Order**: 1st, 2nd, or 3rd derivative
      - **Window Size**: Points used for rolling polynomial (odd number)
      - **Poly Order**: Polynomial degree for Savitzky-Golay
    `,
  },
  {
    title: 'Resampling',
    content: `
      Change the time resolution of your data.

      **Target Frequency**: Choose preset (1s, 100ms, etc.) or enter custom.

      **Aggregation Methods:**
      - **Mean**: Average of values in each interval
      - **Median**: Middle value in each interval
      - **First/Last**: First or last value in each interval
      - **Min/Max**: Minimum or maximum value
      - **Sum**: Total of all values

      **Interpolate**: Fill gaps between resampled points.
    `,
  },
  {
    title: 'Time Range',
    content: `
      Trim data to a specific time window.

      Enter start and/or end times to keep only data within that range.
      Supports numeric values and datetime strings.
    `,
  },
  {
    title: 'Trendlines',
    content: `
      Fit mathematical models to your data.

      **Types:**
      - **Linear**: y = mx + b
      - **Polynomial**: y = aₙxⁿ + ... + a₁x + a₀
      - **Exponential**: y = a·eᵇˣ
      - **Power**: y = axᵇ

      **R²** (R-squared): Indicates how well the trendline fits. 1.0 = perfect fit.
    `,
  },
  {
    title: 'Custom Formulas',
    content: `
      Create new signals using mathematical expressions.

      **Examples:**
      - \`velocity = position / time\`
      - \`power = voltage * current\`
      - \`magnitude = sqrt(x**2 + y**2)\`

      **Available Functions:**
      sin, cos, tan, sqrt, abs, log, log10, exp, min, max

      **Operators:**
      +, -, *, /, ** (power)
    `,
  },
  {
    title: 'Export Formats',
    content: `
      **CSV**: Universal compatibility, opens in any spreadsheet.

      **JSON**: Structured data format, good for web applications.

      **Excel**: Native Excel format with formatting preserved.
    `,
  },
  {
    title: 'Keyboard Shortcuts',
    content: `
      - **Ctrl+O**: Open file
      - **Ctrl+S**: Export data
      - **Ctrl+F**: Focus signal search
      - **Escape**: Clear selection
    `,
  },
];

export function HelpPanel() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const toggleSection = (title: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(title)) {
      newExpanded.delete(title);
    } else {
      newExpanded.add(title);
    }
    setExpandedSections(newExpanded);
  };

  const expandAll = () => {
    setExpandedSections(new Set(helpSections.map((s) => s.title)));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
  };

  return (
    <div className="card max-h-[calc(100vh-200px)] overflow-y-auto">
      <div className="card-header flex items-center justify-between sticky top-0 bg-dark-800 z-10">
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4" />
          Help & Documentation
        </div>
        <div className="flex gap-2">
          <button
            className="text-xs text-dark-400 hover:text-dark-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            onClick={expandAll}
          >
            Expand All
          </button>
          <span className="text-dark-600">|</span>
          <button
            className="text-xs text-dark-400 hover:text-dark-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            onClick={collapseAll}
          >
            Collapse All
          </button>
        </div>
      </div>
      <div className="card-body space-y-2">
        {helpSections.map((section) => {
          const isExpanded = expandedSections.has(section.title);
          const sectionId = `sect-${section.title.replace(/\s+/g, '-')}`;
          return (
            <div key={section.title} className="border border-dark-700 rounded-lg">
              <button
                className="w-full p-3 flex items-center justify-between text-left hover:bg-dark-700/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-lg"
                onClick={() => toggleSection(section.title)}
                aria-expanded={isExpanded}
                aria-controls={sectionId}
              >
                <span className="font-medium text-dark-200">{section.title}</span>
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-dark-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-dark-400" />
                )}
              </button>
              {isExpanded && (
                <div
                  id={sectionId}
                  className="px-3 pb-3 text-sm text-dark-300 whitespace-pre-line"
                >
                  {section.content.trim()}
                </div>
              )}
            </div>
          );
        })}

        {/* Version Info */}
        <div className="pt-4 mt-4 border-t border-dark-700 text-center text-xs text-dark-500">
          Data Processor v2.0
          <br />
          Built with React + TypeScript + Tauri
        </div>
      </div>
    </div>
  );
}

export default HelpPanel;
