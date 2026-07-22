import React, { useMemo, useRef, useState, useEffect } from "react";
import { GripVertical } from "lucide-react";

import { usePanelLayout } from "../hooks/usePanelLayout";
import { clampPanelHeight } from "../lib/panelLayout";

/** One panel in a stack: a stable id, an optional header title, and its body. */
export interface PanelItem {
  id: string;
  title?: React.ReactNode;
  node: React.ReactNode;
}

/**
 * A vertical stack of dashboard panels the operator can drag to reorder and
 * resize, with the arrangement persisted per `regionId` (see
 * {@link file://../hooks/usePanelLayout.ts}). Reorder uses native HTML5 drag from
 * a grip handle (the same approach the tab bar uses); resize uses a bottom
 * pointer-drag gutter. The panel content is passed in as `node`, so a stack
 * doesn't care what a panel renders (LOD).
 */
export const PanelStack: React.FC<{
  regionId: string;
  panels: PanelItem[];
  /** Gap between panels (CSS length). */
  gap?: string;
}> = ({ regionId, panels, gap = "1.25rem" }) => {
  const ids = useMemo(() => panels.map((p) => p.id), [panels]);
  const byId = useMemo(() => new Map(panels.map((p) => [p.id, p])), [panels]);
  const layout = usePanelLayout(regionId, ids);
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);

  const ordered = layout.order
    .map((id) => byId.get(id))
    .filter((p): p is PanelItem => p !== undefined);

  const endDrag = (): void => {
    setDragId(null);
    setOverId(null);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap }}>
      {ordered.map((p) => (
        <PanelChrome
          key={p.id}
          id={p.id}
          title={p.title}
          height={layout.heights[p.id]}
          dragging={dragId === p.id}
          isDropTarget={overId === p.id && dragId !== null && dragId !== p.id}
          onGrab={() => setDragId(p.id)}
          onDragEnterPanel={() => dragId && setOverId(p.id)}
          onDropOnPanel={() => {
            if (dragId && dragId !== p.id) layout.move(dragId, p.id);
            endDrag();
          }}
          onDragEnd={endDrag}
          onResizeCommit={(px) => layout.setHeight(p.id, px)}
        >
          {p.node}
        </PanelChrome>
      ))}
    </div>
  );
};

const PanelChrome: React.FC<{
  id: string;
  title?: React.ReactNode;
  height?: number;
  dragging: boolean;
  isDropTarget: boolean;
  onGrab: () => void;
  onDragEnterPanel: () => void;
  onDropOnPanel: () => void;
  onDragEnd: () => void;
  onResizeCommit: (px: number) => void;
  children: React.ReactNode;
}> = ({
  id,
  title,
  height,
  dragging,
  isDropTarget,
  onGrab,
  onDragEnterPanel,
  onDropOnPanel,
  onDragEnd,
  onResizeCommit,
  children,
}) => {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const resize = useRef<{ startY: number; startH: number } | null>(null);
  // Live height while dragging the gutter, committed to the persisted layout on
  // release so the whole (10 Hz) panel tree isn't re-rendered every pointer move.
  const [liveHeight, setLiveHeight] = useState<number | undefined>(height);
  useEffect(() => setLiveHeight(height), [height]);

  const onResizeDown = (e: React.PointerEvent): void => {
    const el = bodyRef.current;
    if (!el) return;
    e.preventDefault();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    resize.current = { startY: e.clientY, startH: el.getBoundingClientRect().height };
  };
  const onResizeMove = (e: React.PointerEvent): void => {
    if (!resize.current) return;
    setLiveHeight(clampPanelHeight(resize.current.startH + (e.clientY - resize.current.startY)));
  };
  const onResizeUp = (e: React.PointerEvent): void => {
    if (!resize.current) return;
    resize.current = null;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    if (liveHeight !== undefined) onResizeCommit(liveHeight);
  };

  return (
    <section
      className="panel-stack-item"
      onDragOver={(e) => {
        if (!dragging) e.preventDefault(); // allow drop
        onDragEnterPanel();
      }}
      onDrop={(e) => {
        e.preventDefault();
        onDropOnPanel();
      }}
      style={{
        // Transparent chrome: the panel's own content brings its card, so this
        // wrapper only adds the grip + resize affordances (no nested card).
        opacity: dragging ? 0.55 : 1,
        outline: isDropTarget ? "2px dashed var(--accent-cyan)" : undefined,
        outlineOffset: "4px",
        borderRadius: "8px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.4rem" }}>
        <span
          className="panel-grip"
          draggable
          onDragStart={(e) => {
            // dataTransfer is always present in a browser but may be absent in
            // jsdom; reorder is driven by component state, not the payload.
            if (e.dataTransfer) {
              e.dataTransfer.effectAllowed = "move";
              try {
                e.dataTransfer.setData("text/plain", id);
              } catch {
                /* restricted dataTransfer — ignore */
              }
            }
            onGrab();
          }}
          onDragEnd={onDragEnd}
          title="Drag to reorder"
          aria-label={
            typeof title === "string" ? `Reorder ${title}` : "Reorder panel"
          }
          style={{
            display: "inline-flex",
            alignItems: "center",
            cursor: "grab",
            color: "var(--text-muted)",
            flexShrink: 0,
            touchAction: "none",
          }}
        >
          <GripVertical size={14} aria-hidden />
        </span>
        {title !== undefined && (
          <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-primary)" }}>
            {title}
          </div>
        )}
      </div>

      <div
        ref={bodyRef}
        style={liveHeight !== undefined ? { height: liveHeight, overflow: "auto" } : undefined}
      >
        {children}
      </div>

      <div
        className="panel-resize-gutter"
        onPointerDown={onResizeDown}
        onPointerMove={onResizeMove}
        onPointerUp={onResizeUp}
        role="separator"
        aria-orientation="horizontal"
        aria-label={
          typeof title === "string" ? `Resize ${title}` : "Resize panel"
        }
        title="Drag to resize"
        style={{
          height: 8,
          marginTop: "0.3rem",
          cursor: "ns-resize",
          borderRadius: "3px",
          background:
            "repeating-linear-gradient(90deg, var(--panel-border) 0 6px, transparent 6px 12px)",
          opacity: 0.5,
          touchAction: "none",
        }}
      />
    </section>
  );
};
