'use client';

import { SwingPhase } from '@/lib/golf/types';
import { PhaseTransition } from '@/lib/golf/phaseDetector';

interface PhaseTimelineProps {
  phases: PhaseTransition[];
  totalDuration: number;
  currentFrame?: number;
}

export default function PhaseTimeline({
  phases,
  totalDuration,
  currentFrame,
}: PhaseTimelineProps) {
  const getPhaseColor = (phase: SwingPhase): string => {
    const colors: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: 'bg-slate-400',
      [SwingPhase.TAKEAWAY]: 'bg-blue-400',
      [SwingPhase.BACKSWING]: 'bg-blue-500',
      [SwingPhase.TOP_OF_BACKSWING]: 'bg-indigo-500',
      [SwingPhase.TRANSITION]: 'bg-purple-500',
      [SwingPhase.DOWNSWING]: 'bg-orange-500',
      [SwingPhase.IMPACT]: 'bg-red-500',
      [SwingPhase.FOLLOW_THROUGH]: 'bg-green-500',
      [SwingPhase.FINISH]: 'bg-emerald-500',
      [SwingPhase.UNKNOWN]: 'bg-gray-300',
    };
    return colors[phase];
  };

  const getPhaseLabel = (phase: SwingPhase): string => {
    const labels: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: 'Address',
      [SwingPhase.TAKEAWAY]: 'Takeaway',
      [SwingPhase.BACKSWING]: 'Backswing',
      [SwingPhase.TOP_OF_BACKSWING]: 'Top',
      [SwingPhase.TRANSITION]: 'Transition',
      [SwingPhase.DOWNSWING]: 'Downswing',
      [SwingPhase.IMPACT]: 'Impact',
      [SwingPhase.FOLLOW_THROUGH]: 'Follow-Through',
      [SwingPhase.FINISH]: 'Finish',
      [SwingPhase.UNKNOWN]: '?',
    };
    return labels[phase];
  };

  const getPhaseIcon = (phase: SwingPhase): string => {
    const icons: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: '🏌️',
      [SwingPhase.TAKEAWAY]: '↗️',
      [SwingPhase.BACKSWING]: '🔄',
      [SwingPhase.TOP_OF_BACKSWING]: '⬆️',
      [SwingPhase.TRANSITION]: '⚡',
      [SwingPhase.DOWNSWING]: '⬇️',
      [SwingPhase.IMPACT]: '💥',
      [SwingPhase.FOLLOW_THROUGH]: '↪️',
      [SwingPhase.FINISH]: '✅',
      [SwingPhase.UNKNOWN]: '❓',
    };
    return icons[phase];
  };

  // Calculate current phase based on frame
  const getCurrentPhase = (): SwingPhase | null => {
    if (currentFrame === undefined) return null;
    for (const phase of phases) {
      if (currentFrame >= phase.startFrame && currentFrame <= phase.endFrame) {
        return phase.phase;
      }
    }
    return null;
  };

  const activePhase = getCurrentPhase();

  // Key phases for simplified view
  const keyPhases = phases.filter((p) =>
    [
      SwingPhase.ADDRESS,
      SwingPhase.TOP_OF_BACKSWING,
      SwingPhase.IMPACT,
      SwingPhase.FINISH,
    ].includes(p.phase)
  );

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Swing Phases</h3>

      {/* Timeline Bar */}
      <div className="mb-6">
        <div className="flex h-8 rounded-lg overflow-hidden">
          {phases.map((phase, index) => {
            const widthPercent = (phase.duration / totalDuration) * 100;
            const isActive = phase.phase === activePhase;

            return (
              <div
                key={index}
                className={`${getPhaseColor(phase.phase)} flex items-center justify-center text-white text-xs font-medium transition-all ${
                  isActive ? 'ring-2 ring-offset-1 ring-blue-600' : ''
                }`}
                style={{ width: `${widthPercent}%`, minWidth: '20px' }}
                title={`${getPhaseLabel(phase.phase)}: ${phase.duration.toFixed(0)}ms`}
              >
                {widthPercent > 10 && (
                  <span className="truncate px-1">{getPhaseIcon(phase.phase)}</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Timeline Labels */}
        <div className="flex justify-between mt-2">
          <span className="text-xs text-gray-500">0ms</span>
          <span className="text-xs text-gray-500">{totalDuration.toFixed(0)}ms</span>
        </div>
      </div>

      {/* Phase Legend */}
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 mb-6">
        {phases.map((phase, index) => (
          <div
            key={index}
            className={`flex items-center space-x-2 p-2 rounded-lg ${
              phase.phase === activePhase ? 'bg-blue-50' : 'bg-gray-50'
            }`}
          >
            <div className={`w-3 h-3 rounded-full ${getPhaseColor(phase.phase)}`} />
            <span className="text-xs font-medium text-gray-700 truncate">
              {getPhaseLabel(phase.phase)}
            </span>
          </div>
        ))}
      </div>

      {/* Key Positions Timeline */}
      <div className="relative">
        <h4 className="text-sm font-medium text-gray-700 mb-4">Key Positions</h4>

        <div className="flex items-center justify-between relative">
          {/* Connection line */}
          <div className="absolute left-0 right-0 top-8 h-0.5 bg-gray-200" />

          {keyPhases.length > 0 ? (
            keyPhases.map((phase, index) => (
              <div key={index} className="relative z-10 text-center flex-1">
                <div
                  className={`w-16 h-16 mx-auto rounded-full ${getPhaseColor(
                    phase.phase
                  )} flex items-center justify-center text-white text-2xl shadow-md ${
                    phase.phase === activePhase ? 'ring-4 ring-blue-300' : ''
                  }`}
                >
                  {getPhaseIcon(phase.phase)}
                </div>
                <p className="text-sm font-medium text-gray-900 mt-2">
                  {getPhaseLabel(phase.phase)}
                </p>
                <p className="text-xs text-gray-500">{phase.duration.toFixed(0)}ms</p>
                <p className="text-xs text-gray-400">Frame {phase.startFrame}</p>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 text-center w-full">
              No key positions detected
            </p>
          )}
        </div>
      </div>

      {/* Phase Durations Table */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Phase Breakdown</h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {phases.map((phase, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
            >
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${getPhaseColor(phase.phase)}`} />
                <span className="text-sm text-gray-600">{getPhaseLabel(phase.phase)}</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {phase.duration.toFixed(0)}ms
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
