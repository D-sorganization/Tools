import type { PendulumParams } from './physics';

export type AnimationAlignment = 'fixed_hub' | 'impact_aligned';

export interface ThumbnailOrigin {
    x: number;
    y: number;
}

export interface PendulumGeometry {
    originX: number;
    originY: number;
    wristX: number;
    wristY: number;
    tipX: number;
    tipY: number;
}

const FIXED_HUB: ThumbnailOrigin = { x: 96, y: 88 };
const SHARED_IMPACT = { x: 150, y: 148 };

export function interpolateSeries(
    time: number[],
    values: number[],
    target: number,
): number {
    if (target <= time[0]) return values[0];
    const last = time.length - 1;
    if (target >= time[last]) return values[last];

    let left = 0;
    let right = last;
    while (right - left > 1) {
        const middle = Math.floor((left + right) / 2);
        if (time[middle] <= target) left = middle;
        else right = middle;
    }
    const interval = time[right] - time[left];
    const fraction = interval > 0 ? (target - time[left]) / interval : 0;
    return values[left] + fraction * (values[right] - values[left]);
}

export function pendulumThumbnailGeometry(
    arm: number,
    wrist: number,
    params: Pick<PendulumParams, 'L1' | 'L2'>,
    origin: ThumbnailOrigin = FIXED_HUB,
): PendulumGeometry {
    const scale = 74 / (params.L1 + params.L2);
    const originX = origin.x;
    const originY = origin.y;
    const wristX = originX + params.L1 * scale * Math.sin(arm);
    const wristY = originY + params.L1 * scale * Math.cos(arm);
    const clubAngle = arm + wrist;
    const tipX = wristX + params.L2 * scale * Math.sin(clubAngle);
    const tipY = wristY + params.L2 * scale * Math.cos(clubAngle);
    return { originX, originY, wristX, wristY, tipX, tipY };
}

/** Return the camera origin for a declared comparison reference frame. */
export function thumbnailOrigin(
    alignment: AnimationAlignment,
    impact: PendulumGeometry,
): ThumbnailOrigin {
    if (alignment === 'fixed_hub') return { ...FIXED_HUB };
    if (alignment !== 'impact_aligned') {
        throw new RangeError(`Unsupported animation alignment: ${String(alignment)}`);
    }
    return {
        x: FIXED_HUB.x + SHARED_IMPACT.x - impact.tipX,
        y: FIXED_HUB.y + SHARED_IMPACT.y - impact.tipY,
    };
}

export function wrappedDegrees(radians: number): number {
    const value = radians * 180 / Math.PI;
    return (((value + 180) % 360) + 360) % 360 - 180;
}
