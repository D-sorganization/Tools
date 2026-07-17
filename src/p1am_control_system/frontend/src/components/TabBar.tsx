import React, { useEffect, useMemo, useState } from "react";
import { TABS, type TabId, defaultTabOrder } from "../lib/tabs";

/**
 * Dashboard tab bar (#3543 / #3546).
 *
 * Renders the centralized {@link TABS} in a caller-controlled order. Tabs can be
 * reordered by drag-and-drop (or the right-click menu's Move Left/Right, which
 * also keeps reordering reachable without a pointer drag) and hidden via the
 * right-click menu. Order/visibility state is owned by the host (App) so it can
 * be persisted; this component only emits change callbacks.
 */

const TAB_BY_ID: Record<TabId, (typeof TABS)[number]> = Object.fromEntries(
  TABS.map((tab) => [tab.id, tab]),
) as Record<TabId, (typeof TABS)[number]>;

interface ContextMenuState {
  id: TabId;
  x: number;
  y: number;
}

export const TabBar: React.FC<{
  activeTab: TabId;
  visibleTabs: Record<TabId, boolean>;
  onSelect: (id: TabId) => void;
  /** Display order of tab ids. Defaults to the declared {@link TABS} order. */
  order?: TabId[];
  /** Emitted with the new full id order after a drag or Move Left/Right. */
  onReorder?: (order: TabId[]) => void;
  /** Emitted when the operator hides a tab from the right-click menu. */
  onHide?: (id: TabId) => void;
}> = ({ activeTab, visibleTabs, onSelect, order, onReorder, onHide }) => {
  const [dragId, setDragId] = useState<TabId | null>(null);
  const [menu, setMenu] = useState<ContextMenuState | null>(null);

  // Effective order: caller's order (known ids only) + any tabs missing from it
  // (a tab added in a new release must never be silently dropped).
  const effectiveOrder = useMemo<TabId[]>(() => {
    const base = order && order.length ? order : defaultTabOrder();
    const kept = base.filter((id) => TAB_BY_ID[id]);
    // ⚡ Bolt Optimization: Replace O(N^2) chained .filter().includes() with a Set and single-pass loop
    const keptSet = new Set(kept);
    const missing: TabId[] = [];
    for (const id of defaultTabOrder()) {
      if (!keptSet.has(id)) {
        missing.push(id);
      }
    }
    return [...kept, ...missing];
  }, [order]);

  const visibleOrdered = effectiveOrder.filter((id) => visibleTabs[id]);

  // Dismiss the context menu on any outside click or Escape.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  const dropBefore = (dragged: TabId, target: TabId) => {
    if (dragged === target || !onReorder) return;
    const next = effectiveOrder.filter((id) => id !== dragged);
    next.splice(next.indexOf(target), 0, dragged);
    onReorder(next);
  };

  const move = (id: TabId, dir: -1 | 1) => {
    if (!onReorder) return;
    const idx = effectiveOrder.indexOf(id);
    const swap = idx + dir;
    if (swap < 0 || swap >= effectiveOrder.length) return;
    const next = [...effectiveOrder];
    [next[idx], next[swap]] = [next[swap], next[idx]];
    onReorder(next);
  };

  const menuTab = menu ? TAB_BY_ID[menu.id] : null;

  return (
    <div
      role="tablist"
      style={{
        display: "flex",
        borderBottom: "1px solid var(--panel-border)",
        gap: "0.25rem",
        paddingBottom: "0.25rem",
        marginBottom: "0.5rem",
      }}
    >
      {visibleOrdered.map((id) => {
        const tab = TAB_BY_ID[id];
        const isActive = activeTab === id;
        return (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={isActive}
            draggable
            className={`tab-btn ${isActive ? "active" : ""} ${
              dragId === id ? "dragging" : ""
            }`}
            onClick={() => onSelect(id)}
            onContextMenu={(e) => {
              e.preventDefault();
              setMenu({ id, x: e.clientX, y: e.clientY });
            }}
            onDragStart={(e) => {
              setDragId(id);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (dragId) dropBefore(dragId, id);
              setDragId(null);
            }}
            onDragEnd={() => setDragId(null)}
            title="Drag to reorder · right-click for options"
            style={{
              color: isActive ? tab.accentVar : "var(--text-secondary)",
              borderBottomColor: isActive ? tab.accentVar : "transparent",
              cursor: "grab",
              opacity: dragId === id ? 0.4 : 1,
            }}
          >
            {tab.label}
          </button>
        );
      })}

      {menu && menuTab && (
        <div
          role="menu"
          aria-label={`${menuTab.label} tab options`}
          onClick={(e) => e.stopPropagation()}
          style={{
            position: "fixed",
            top: menu.y,
            left: menu.x,
            zIndex: 1000,
            minWidth: "9rem",
            background: "var(--panel-bg)",
            border: "1px solid var(--panel-border)",
            borderRadius: "8px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
            padding: "0.25rem",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              fontSize: "0.65rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              color: "var(--text-muted)",
              padding: "0.3rem 0.5rem 0.2rem",
            }}
          >
            {menuTab.label}
          </div>
          <MenuItem
            disabled={effectiveOrder.indexOf(menu.id) === 0}
            onClick={() => {
              move(menu.id, -1);
              setMenu(null);
            }}
          >
            Move Left
          </MenuItem>
          <MenuItem
            disabled={
              effectiveOrder.indexOf(menu.id) === effectiveOrder.length - 1
            }
            onClick={() => {
              move(menu.id, 1);
              setMenu(null);
            }}
          >
            Move Right
          </MenuItem>
          <MenuItem
            danger
            onClick={() => {
              onHide?.(menu.id);
              setMenu(null);
            }}
          >
            Hide Tab
          </MenuItem>
        </div>
      )}
    </div>
  );
};

const MenuItem: React.FC<{
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
  children: React.ReactNode;
}> = ({ onClick, disabled, danger, children }) => (
  <button
    type="button"
    role="menuitem"
    disabled={disabled}
    onClick={onClick}
    style={{
      textAlign: "left",
      background: "none",
      border: "none",
      borderRadius: "6px",
      padding: "0.4rem 0.5rem",
      fontSize: "0.8rem",
      cursor: disabled ? "not-allowed" : "pointer",
      color: disabled
        ? "var(--text-muted)"
        : danger
          ? "var(--color-danger, #ef4444)"
          : "var(--text-primary)",
      opacity: disabled ? 0.5 : 1,
    }}
    onMouseEnter={(e) => {
      if (!disabled) e.currentTarget.style.background = "var(--input-bg)";
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = "none";
    }}
  >
    {children}
  </button>
);
