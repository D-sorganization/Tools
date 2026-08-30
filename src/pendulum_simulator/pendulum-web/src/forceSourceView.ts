import type { PendulumParams } from './physics';

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
) {
    const scale = 74 / (params.L1 + params.L2);
    const originX = 96;
    const originY = 88;
    const wristX = originX + params.L1 * scale * Math.sin(arm);
    const wristY = originY + params.L1 * scale * Math.cos(arm);
    const clubAngle = arm + wrist;
    const tipX = wristX + params.L2 * scale * Math.sin(clubAngle);
    const tipY = wristY + params.L2 * scale * Math.cos(clubAngle);
    return { originX, originY, wristX, wristY, tipX, tipY };
}

export function wrappedDegrees(radians: number): number {
    const value = radians * 180 / Math.PI;
    return (((value + 180) % 360) + 360) % 360 - 180;
}
