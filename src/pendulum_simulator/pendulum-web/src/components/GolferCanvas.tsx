import React, { useRef, useEffect, useCallback } from 'react';
import { forwardKinematics_golfer, GolferParams } from '../physics_golfer';
import { StateGolfer } from '../physics_golfer';

interface GolferCanvasProps {
    states: StateGolfer[];
    params: GolferParams;
    currentIdx: number;
    trailLength?: number;
    width?: number;
    height?: number;
}

const COLORS = {
    hub: '#999999',
    right_arm: '#ff6666',
    left_arm: '#6666ff',
    club: '#66dd66',
    clubhead: '#ffcc44',
    joint: '#ffffff',
    grip: '#ddaaff',
    trail: 'rgba(102,221,102,0.45)',
    grid: 'rgba(80,80,160,0.18)',
    bg: '#1a1a28',
};

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

/** Draw a segment. */
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

/** Draw the clubhead sphere at the tip. */
function drawClubhead(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    mClubhead: number,
): void {
    const radius = Math.max(8, Math.min(20, 8 + mClubhead * 40));
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

/** Draw trail of club tip positions. */
function drawTrail(
    ctx: CanvasRenderingContext2D,
    states: StateGolfer[],
    params: GolferParams,
    currentIdx: number,
    trailLength: number,
    toCanvas: (x: number, y: number) => [number, number],
): void {
    const trailStart = Math.max(0, currentIdx - trailLength);
    if (trailStart >= currentIdx) return;

    for (let i = trailStart; i < currentIdx && i < states.length; i++) {
        const q = [states[i][0], states[i][1], states[i][2], states[i][3],
            states[i][4], states[i][5], states[i][6], states[i][7]] as any;
        const pos = forwardKinematics_golfer(q, params);
        const [tx, ty] = toCanvas(pos.club_tip[0], pos.club_tip[1]);
        const alpha = (i - trailStart) / (currentIdx - trailStart) * 0.5;
        ctx.beginPath();
        ctx.arc(tx, ty, 1.5, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(102,221,102,${alpha})`;
        ctx.fill();
    }
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

/** Draw pivot crosshair. */
function drawCrosshair(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
): void {
    ctx.strokeStyle = '#555577';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
}

export const GolferCanvas: React.FC<GolferCanvasProps> = ({
    states,
    params,
    currentIdx,
    trailLength = 100,
    width = 500,
    height = 500,
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const toCanvas = useCallback((x: number, y: number): [number, number] => {
        const scale = Math.min(width, height) / 4;
        return [width / 2 + x * scale, height / 2 - y * scale];
    }, [width, height]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || states.length === 0) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Background
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, width, height);

        drawGrid(ctx, width, height);

        // Current frame
        const idx = Math.min(currentIdx, states.length - 1);
        const q = [
            states[idx][0], states[idx][1], states[idx][2], states[idx][3],
            states[idx][4], states[idx][5], states[idx][6], states[idx][7]
        ] as [number, number, number, number, number, number, number, number];

        const pos = forwardKinematics_golfer(q, params);
        drawTrail(ctx, states, params, currentIdx, trailLength, toCanvas);

        // Origin reference
        const [ox, oy] = toCanvas(0, 0);
        drawCrosshair(ctx, ox, oy);

        // Hub to hub tip (vertical standoff)
        const [hub_x, hub_y] = toCanvas(pos.hub[0], pos.hub[1]);
        drawSegment(ctx, ox, oy, hub_x, hub_y, COLORS.hub, 6);
        drawJoint(ctx, hub_x, hub_y, 7, COLORS.hub);

        // Shoulder line (perpendicular to hub)
        const [rs_x, rs_y] = toCanvas(pos.rs[0], pos.rs[1]);
        const [ls_x, ls_y] = toCanvas(pos.ls[0], pos.ls[1]);
        drawSegment(ctx, rs_x, rs_y, ls_x, ls_y, COLORS.hub, 4);

        // Right arm: RS → RE → RH
        const [re_x, re_y] = toCanvas(pos.re[0], pos.re[1]);
        const [rh_x, rh_y] = toCanvas(pos.rh[0], pos.rh[1]);
        drawSegment(ctx, rs_x, rs_y, re_x, re_y, COLORS.right_arm, 5);
        drawSegment(ctx, re_x, re_y, rh_x, rh_y, COLORS.right_arm, 4);
        drawJoint(ctx, rs_x, rs_y, 6, COLORS.right_arm);
        drawJoint(ctx, re_x, re_y, 6, COLORS.right_arm);

        // Left arm: LS → LE → LH
        const [le_x, le_y] = toCanvas(pos.le[0], pos.le[1]);
        const [lh_x, lh_y] = toCanvas(pos.lh[0], pos.lh[1]);
        drawSegment(ctx, ls_x, ls_y, le_x, le_y, COLORS.left_arm, 5);
        drawSegment(ctx, le_x, le_y, lh_x, lh_y, COLORS.left_arm, 4);
        drawJoint(ctx, ls_x, ls_y, 6, COLORS.left_arm);
        drawJoint(ctx, le_x, le_y, 6, COLORS.left_arm);

        // Club: club_base → club_tip
        const [club_base_x, club_base_y] = toCanvas(pos.club_base[0], pos.club_base[1]);
        const [club_tip_x, club_tip_y] = toCanvas(pos.club_tip[0], pos.club_tip[1]);
        drawSegment(ctx, club_base_x, club_base_y, club_tip_x, club_tip_y, COLORS.club, 6);

        // Hands grip the club
        drawJoint(ctx, rh_x, rh_y, 6, COLORS.grip);
        drawJoint(ctx, lh_x, lh_y, 6, COLORS.grip);
        drawJoint(ctx, club_base_x, club_base_y, 5, COLORS.club);

        // Clubhead at tip
        drawClubhead(ctx, club_tip_x, club_tip_y, params.m_clubhead);

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
