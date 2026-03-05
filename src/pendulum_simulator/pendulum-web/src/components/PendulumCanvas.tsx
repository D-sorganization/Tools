import React, { useRef, useEffect, useCallback } from 'react';
import { forwardKinematics, PendulumParams } from '../physics';
import { State } from '../physics';

interface PendulumCanvasProps {
    states: State[];
    params: PendulumParams;
    currentIdx: number;
    trailLength?: number;
    width?: number;
    height?: number;
}

const COLORS = {
    arm: '#60aaff',
    club: '#ff9955',
    joint: '#ffffff',
    shoulder: '#ddddff',
    trail: 'rgba(255,153,85,0.45)',
    grid: 'rgba(80,80,160,0.18)',
    bg: '#1a1a28',
};

export const PendulumCanvas: React.FC<PendulumCanvasProps> = ({
    states,
    params,
    currentIdx,
    trailLength = 100,
    width = 400,
    height = 450,
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const toCanvas = useCallback((x: number, y: number, scale: number): [number, number] => {
        return [width / 2 + x * scale, height * 0.18 - y * scale];
    }, [width, height]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || states.length === 0) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const scale = Math.min(width, height) / (2.5 * (params.L1 + params.L2));

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, width, height);

        // Grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        for (let gx = 0; gx < width; gx += 40) {
            ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, height); ctx.stroke();
        }
        for (let gy = 0; gy < height; gy += 40) {
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke();
        }

        // Trail
        const trailStart = Math.max(0, currentIdx - trailLength);
        if (trailStart < currentIdx) {
            ctx.beginPath();
            for (let i = trailStart; i <= currentIdx && i < states.length; i++) {
                const pos = forwardKinematics(states[i][0], states[i][1], params);
                const [tx, ty] = toCanvas(pos.tip[0], pos.tip[1], scale);
                const alpha = (i - trailStart) / (currentIdx - trailStart);
                ctx.strokeStyle = `rgba(255,153,85,${alpha * 0.5})`;
                if (i === trailStart) { ctx.moveTo(tx, ty); } else { ctx.lineTo(tx, ty); }
            }
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Current frame
        const idx = Math.min(currentIdx, states.length - 1);
        const pos = forwardKinematics(states[idx][0], states[idx][1], params);
        const [sx, sy] = toCanvas(pos.shoulder[0], pos.shoulder[1], scale);
        const [wx, wy] = toCanvas(pos.wrist[0], pos.wrist[1], scale);
        const [tx, ty] = toCanvas(pos.tip[0], pos.tip[1], scale);

        // Segment 1 (arm)
        ctx.beginPath();
        ctx.moveTo(sx, sy); ctx.lineTo(wx, wy);
        ctx.strokeStyle = COLORS.arm;
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Segment 2 (club)
        ctx.beginPath();
        ctx.moveTo(wx, wy); ctx.lineTo(tx, ty);
        ctx.strokeStyle = COLORS.club;
        ctx.lineWidth = 4;
        ctx.stroke();

        // Joints
        [[sx, sy, 8, COLORS.shoulder], [wx, wy, 7, '#aaaaff'], [tx, ty, 5, '#ffa040']].forEach(
            ([x, y, r, color]) => {
                ctx.beginPath();
                ctx.arc(x as number, y as number, r as number, 0, 2 * Math.PI);
                ctx.fillStyle = color as string;
                ctx.fill();
            }
        );

        // Pivot cross-hair
        ctx.strokeStyle = '#555577';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx - 10, sy); ctx.lineTo(sx + 10, sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(sx, sy - 10); ctx.lineTo(sx, sy + 10); ctx.stroke();

    }, [states, params, currentIdx, trailLength, width, height, toCanvas]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{ borderRadius: 8, border: '1px solid #303048' }}
        />
    );
};
