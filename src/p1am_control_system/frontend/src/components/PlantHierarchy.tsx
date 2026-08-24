import React, { useState, useEffect, useCallback } from "react";
import { Network, HardDrive, Tag, ChevronDown, ChevronRight, Cpu } from "lucide-react";
import { getPlant } from "../api/endpoints";
import type {
  HierarchicalTag,
  HierarchicalEquipment,
  HierarchicalUnit,
  HierarchicalArea,
} from "../api/schemas";

export type {
  HierarchicalTag,
  HierarchicalEquipment,
  HierarchicalUnit,
  HierarchicalArea,
};

interface PlantHierarchyProps {
  onSelectTag: (name: string) => void;
  triggerNotification: (msg: string, type: "success" | "error" | "info") => void;
}

export const PlantHierarchy: React.FC<PlantHierarchyProps> = ({
  onSelectTag,
  triggerNotification,
}) => {
  const [hierarchy, setHierarchy] = useState<HierarchicalArea[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState<boolean>(true);

  const fetchHierarchy = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getPlant();
      setHierarchy(data);
      // Expand areas by default
      const initialExpanded: Record<string, boolean> = {};
      data.forEach((area: HierarchicalArea) => {
        initialExpanded[`area_${area.name}`] = true;
        area.units.forEach((unit) => {
          initialExpanded[`unit_${area.name}_${unit.name}`] = true;
        });
      });
      setExpandedNodes(initialExpanded);
    } catch {
      triggerNotification("Failed to fetch plant hierarchy structure.", "error");
    } finally {
      setLoading(false);
    }
  }, [triggerNotification]);

  useEffect(() => {
    fetchHierarchy();
  }, [fetchHierarchy]);

  const toggleNode = (nodeId: string) => {
    setExpandedNodes((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  };

  return (
    <div className="glass-panel" style={{ padding: "1.5rem" }}>
      <div className="panel-header" style={{ marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Network size={18} color="var(--accent-cyan)" />
          <span style={{ fontWeight: 800 }}>Plant Equipment Hierarchy Tree</span>
        </div>
        <button
          type="button"
          className="btn"
          onClick={fetchHierarchy}
          style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
        >
          Refresh Structure
        </button>
      </div>

      {loading ? (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
          Building plant physical tree...
        </div>
      ) : hierarchy.length === 0 ? (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
          No areas or equipment registered. Ingest a project zip to build the hierarchy.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {hierarchy.map((area) => {
            const areaId = `area_${area.name}`;
            const isAreaExpanded = expandedNodes[areaId];

            return (
              <div
                key={area.name}
                style={{
                  border: "1px solid var(--panel-border)",
                  borderRadius: "6px",
                  overflow: "hidden",
                  background: "rgba(255, 255, 255, 0.005)",
                }}
              >
                {/* Area Node Header */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    padding: "0.6rem 0.85rem",
                    backgroundColor: "rgba(255, 255, 255, 0.02)",
                    cursor: "pointer",
                    gap: "0.5rem",
                    userSelect: "none",
                  }}
                  onClick={() => toggleNode(areaId)}
                >
                  {isAreaExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  <Network size={14} color="var(--accent-cyan)" />
                  <span style={{ fontWeight: 800, fontSize: "0.85rem", color: "var(--text-primary)" }}>
                    {area.name}
                  </span>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      color: "var(--text-muted)",
                      marginLeft: "auto",
                      backgroundColor: "rgba(255,255,255,0.03)",
                      padding: "0.1rem 0.4rem",
                      borderRadius: "10px",
                    }}
                  >
                    {area.units.length} Units
                  </span>
                </div>

                {/* Units List */}
                {isAreaExpanded && (
                  <div style={{ padding: "0.5rem 0.5rem 0.5rem 1.25rem" }}>
                    {area.units.map((unit) => {
                      const unitId = `unit_${area.name}_${unit.name}`;
                      const isUnitExpanded = expandedNodes[unitId];

                      return (
                        <div key={unit.name} style={{ marginBottom: "0.4rem" }}>
                          {/* Unit Node Header */}
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              padding: "0.45rem 0.6rem",
                              cursor: "pointer",
                              gap: "0.5rem",
                              borderRadius: "4px",
                              userSelect: "none",
                              backgroundColor: "rgba(255, 255, 255, 0.01)",
                            }}
                            onClick={() => toggleNode(unitId)}
                          >
                            {isUnitExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                            <Cpu size={14} color="var(--accent-purple)" />
                            <span style={{ fontWeight: 700, fontSize: "0.8rem", color: "var(--text-primary)" }}>
                              {unit.name}
                            </span>
                          </div>

                          {/* Equipment List */}
                          {isUnitExpanded && (
                            <div style={{ padding: "0.25rem 0.25rem 0.25rem 1rem" }}>
                              {unit.equipment.map((equip) => {
                                const equipId = `equip_${area.name}_${unit.name}_${equip.name}`;
                                const isEquipExpanded = expandedNodes[equipId];

                                return (
                                  <div key={equip.name} style={{ marginBottom: "0.3rem" }}>
                                    {/* Equipment Node Header */}
                                    <div
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        padding: "0.35rem 0.5rem",
                                        cursor: "pointer",
                                        gap: "0.5rem",
                                        borderRadius: "4px",
                                        userSelect: "none",
                                      }}
                                      onClick={() => toggleNode(equipId)}
                                    >
                                      {isEquipExpanded ? (
                                        <ChevronDown size={10} />
                                      ) : (
                                        <ChevronRight size={10} />
                                      )}
                                      <HardDrive size={13} color="var(--accent-magenta)" />
                                      <span
                                        style={{
                                          fontWeight: 600,
                                          fontSize: "0.75rem",
                                          color: "var(--text-secondary)",
                                        }}
                                      >
                                        {equip.name}
                                      </span>
                                    </div>

                                    {/* Tags List */}
                                    {isEquipExpanded && (
                                      <div
                                        style={{
                                          display: "grid",
                                          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                                          gap: "0.5rem",
                                          padding: "0.4rem 0.4rem 0.4rem 1.25rem",
                                        }}
                                      >
                                        {equip.tags.map((tag) => (
                                          <div
                                            key={tag.name}
                                            style={{
                                              display: "flex",
                                              alignItems: "center",
                                              justifyContent: "space-between",
                                              padding: "0.4rem 0.6rem",
                                              background: "rgba(255, 255, 255, 0.02)",
                                              border: "1px solid var(--panel-border)",
                                              borderRadius: "4px",
                                              cursor: "pointer",
                                              transition: "all var(--transition-fast)",
                                            }}
                                            className="tag-item-node hover-row"
                                            onClick={() => onSelectTag(tag.name)}
                                          >
                                            <div
                                              style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: "0.4rem",
                                                minWidth: 0,
                                              }}
                                            >
                                              <Tag size={12} color="var(--accent-cyan)" />
                                              <span
                                                style={{
                                                  fontWeight: 700,
                                                  fontSize: "0.75rem",
                                                  color: "var(--text-primary)",
                                                  overflow: "hidden",
                                                  textOverflow: "ellipsis",
                                                  whiteSpace: "nowrap",
                                                }}
                                                title={tag.name}
                                              >
                                                {tag.name.split("_").pop()}
                                              </span>
                                            </div>
                                            {tag.register_type && (
                                              <span
                                                className="mono-text"
                                                style={{
                                                  fontSize: "0.65rem",
                                                  color: "var(--text-muted)",
                                                  background: "rgba(255,255,255,0.03)",
                                                  padding: "0.1rem 0.3rem",
                                                  borderRadius: "2px",
                                                }}
                                              >
                                                {tag.register_type}:{tag.register_num}
                                              </span>
                                            )}
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
