import { describe, expect, it } from 'vitest';

import { smoothAngles } from '../angleCalculator';

function smoothAnglesBaseline(angleHistory: number[], windowSize: number): number[] {
  const smoothed: number[] = [];
  for (let i = 0; i < angleHistory.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(angleHistory.length, i + Math.ceil(windowSize / 2));
    const window = angleHistory.slice(start, end);
    const average = window.reduce((sum, value) => sum + value, 0) / window.length;
    smoothed.push(average);
  }
  return smoothed;
}

describe('smoothAngles', () => {
  it('matches the baseline averaging logic for fractional window sizes', () => {
    const angleHistory = [10, 20, 30, 40, 50, 60, 70];
    const windowSize = 4.5;

    expect(smoothAngles(angleHistory, windowSize)).toEqual(
      smoothAnglesBaseline(angleHistory, windowSize)
    );
  });
});
