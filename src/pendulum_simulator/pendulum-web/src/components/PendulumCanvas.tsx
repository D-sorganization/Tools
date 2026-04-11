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
    shaft: '#ff9955',
    clubhead: '#ffcc44',
    joint: '#ffffff',
    shoulder: '#ddddff',
    trail: 'rgba(255,153,85,0.45)',
    grid: 'rgba(80,80,160,0.18)',
    bg: '#1a1a28',
};

/** Draw the clubhead sphere at the tip. */
function drawClubhead(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    mClub: number, _scale: number,
): void {
    // Radius proportional to clubhead mass, min 6px, max 18px
    const radius = Math.max(6, Math.min(18, 6 + mClub * 30));
    const gradient = ctx.createRadialGradient(x - 2, y - 2, 1, x, y, radius);
    gradient.addColorStop(0, '#ffe080');
    gradient.addColorStop(0.7, COLORS.clubhead);
    gradient.addColorStop(1, '#cc9900');
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,200,60,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

/** Draw trail of tip positions (fading). */
function drawTrail(
    ctx: CanvasRenderingContext2D,
    states: State[], params: PendulumParams,
    currentIdx: number, trailLength: number,
    toCanvas: (x: number, y: number, s: number) => [number, number],
    scale: number,
): void {
    const trailStart = Math.max(0, currentIdx - trailLength);
    if (trailStart >= currentIdx) return;

    for (let i = trailStart; i < currentIdx && i < states.length; i++) {
        const pos = forwardKinematics(states[i][0], states[i][1], params);
        const [tx, ty] = toCanvas(pos.tip[0], pos.tip[1], scale);
        const alpha = (i - trailStart) / (currentIdx - trailStart) * 0.5;
        ctx.beginPath();
        ctx.arc(tx, ty, 1.5, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(255,153,85,${alpha})`;
        ctx.fill();
    }
}

/** Draw a single pendulum segment. */
function drawSegment(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number, x2: number, y2: number,
    color: string, width: number,
): void {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.stroke();
}

/** Draw a joint dot. */
function drawJoint(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, r: number, color: string,
): void {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

/** Draw the pivot crosshair. */
function drawCrosshair(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
): void {
    ctx.strokeStyle = '#555577';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
}

/** Draw background grid. */
function drawGrid(
    ctx: CanvasRenderingContext2D,
    w: number, h: number,
): void {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let gx = 0; gx < w; gx += 40) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }
    for (let gy = 0; gy < h; gy += 40) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
}

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

        const totalLen = params.L1 + params.L2;
        const scale = Math.min(width, height) / (2.5 * totalLen);

        // Background
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, width, height);

        drawGrid(ctx, width, height);
        drawTrail(ctx, states, params, currentIdx, trailLength, toCanvas, scale);

        // Current frame
        const idx = Math.min(currentIdx, states.length - 1);
        const pos = forwardKinematics(states[idx][0], states[idx][1], params);
        const [sx, sy] = toCanvas(pos.shoulder[0], pos.shoulder[1], scale);
        const [wx, wy] = toCanvas(pos.wrist[0], pos.wrist[1], scale);
        const [tx, ty] = toCanvas(pos.tip[0], pos.tip[1], scale);

        // Arm segment (segment 1)
        drawSegment(ctx, sx, sy, wx, wy, COLORS.arm, 5);

        // Shaft segment (segment 2)
        drawSegment(ctx, wx, wy, tx, ty, COLORS.shaft, 4);

        // Joints
        drawJoint(ctx, sx, sy, 8, COLORS.shoulder);
        drawJoint(ctx, wx, wy, 7, '#aaaaff');

        // Clubhead sphere at tip
        if (params.mClub > 0) {
            drawClubhead(ctx, tx, ty, params.mClub, scale);
        } else {
            drawJoint(ctx, tx, ty, 5, '#ffa040');
        }

        drawCrosshair(ctx, sx, sy);

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
