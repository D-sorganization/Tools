'use client';

import { SwingScores } from '@/lib/golf/types';

interface ScoreCardProps {
  scores: SwingScores;
}

export default function ScoreCard({ scores }: ScoreCardProps) {
  const getScoreColor = (score: number): string => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreGradient = (score: number): string => {
    if (score >= 80) return 'from-green-500 to-emerald-600';
    if (score >= 60) return 'from-yellow-500 to-amber-600';
    return 'from-red-500 to-rose-600';
  };

  const getScoreLabel = (score: number): string => {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Very Good';
    if (score >= 70) return 'Good';
    if (score >= 60) return 'Fair';
    if (score >= 50) return 'Needs Work';
    return 'Poor';
  };

  const scoreItems = [
    { key: 'tempo', label: 'Tempo', icon: '⏱' },
    { key: 'balance', label: 'Balance', icon: '⚖' },
    { key: 'plane', label: 'Plane', icon: '📐' },
    { key: 'posture', label: 'Posture', icon: '🧍' },
    { key: 'rotation', label: 'Rotation', icon: '🔄' },
    { key: 'timing', label: 'Timing', icon: '🎯' },
    { key: 'consistency', label: 'Consistency', icon: '📊' },
  ] as const;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Overall Score Header */}
      <div className={`bg-gradient-to-r ${getScoreGradient(scores.overall)} p-6 text-white`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium opacity-90">Overall Score</p>
            <p className="text-4xl font-bold">{Math.round(scores.overall)}</p>
            <p className="text-sm opacity-75">{getScoreLabel(scores.overall)}</p>
          </div>
          <div className="w-24 h-24 rounded-full border-4 border-white border-opacity-30 flex items-center justify-center">
            <div className="text-center">
              <span className="text-3xl font-bold">{Math.round(scores.overall)}</span>
              <span className="text-sm opacity-75">/100</span>
            </div>
          </div>
        </div>
      </div>

      {/* Component Scores */}
      <div className="p-6">
        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">
          Component Breakdown
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {scoreItems.map(({ key, label, icon }) => {
            const score = scores[key];
            return (
              <div
                key={key}
                className="bg-gray-50 rounded-lg p-4 text-center hover:bg-gray-100 transition-colors"
              >
                <span className="text-2xl mb-2 block">{icon}</span>
                <p className={`text-2xl font-bold ${getScoreColor(score)}`}>
                  {Math.round(score)}
                </p>
                <p className="text-sm text-gray-500">{label}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Score Bars */}
      <div className="px-6 pb-6">
        <div className="space-y-3">
          {scoreItems.map(({ key, label }) => {
            const score = scores[key];
            return (
              <div key={key} className="flex items-center">
                <span className="w-24 text-sm text-gray-600">{label}</span>
                <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden mx-3">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${getScoreGradient(score)} transition-all duration-500`}
                    style={{ width: `${score}%` }}
                  />
                </div>
                <span className={`w-10 text-right text-sm font-medium ${getScoreColor(score)}`}>
                  {Math.round(score)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
