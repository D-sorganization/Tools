import React from "react";
import { TABS, type TabId } from "../lib/tabs";

/**
 * Dashboard tab bar (#3543 / #3546).
 *
 * Replaces the eight near-identical inline-styled `<button>` blocks in App.tsx
 * with a single map over the centralized {@link TABS} array. The accent color
 * is the only per-tab variation, applied via the active state.
 */
export const TabBar: React.FC<{
  activeTab: TabId;
  visibleTabs: Record<TabId, boolean>;
  onSelect: (id: TabId) => void;
}> = ({ activeTab, visibleTabs, onSelect }) => {
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
      {TABS.filter((tab) => visibleTabs[tab.id]).map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`tab-btn ${isActive ? "active" : ""}`}
            onClick={() => onSelect(tab.id)}
            style={{
              color: isActive ? tab.accentVar : "var(--text-secondary)",
              borderBottomColor: isActive ? tab.accentVar : "transparent",
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
};
