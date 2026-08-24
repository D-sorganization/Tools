import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Search, Cpu, Eye } from "lucide-react";
import { getLadderExplorer } from "../api/endpoints";
import type { LadderTagInfo } from "../api/schemas";

export type { LadderTagInfo };

interface LadderExplorerProps {
  onSelectTag: (name: string) => void;
  triggerNotification: (msg: string, type: "success" | "error" | "info") => void;
}

export const LadderExplorer: React.FC<LadderExplorerProps> = ({
  onSelectTag,
  triggerNotification,
}) => {
  const [tags, setTags] = useState<LadderTagInfo[]>([]);
  const [search, setSearch] = useState<string>("");
  const [selectedArea, setSelectedArea] = useState<string>("All");
  const [selectedRegType, setSelectedRegType] = useState<string>("All");
  const [loading, setLoading] = useState<boolean>(true);

  const fetchTags = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getLadderExplorer();
      setTags(data);
    } catch {
      triggerNotification("Failed to fetch ladder logic registry.", "error");
    } finally {
      setLoading(false);
    }
  }, [triggerNotification]);

  useEffect(() => {
    fetchTags();
  }, [fetchTags]);

  // ⚡ Bolt Optimization: Memoize and pre-compute dropdown options using a single-pass loop instead of chained .map().filter()
  const { areas, regTypes } = useMemo(() => {
    const areaSet = new Set<string>();
    const regTypeSet = new Set<string>();
    for (let i = 0; i < tags.length; i++) {
      const t = tags[i];
      if (t.area) areaSet.add(t.area);
      if (t.register_type) regTypeSet.add(t.register_type);
    }
    return {
      areas: ["All", ...Array.from(areaSet)],
      regTypes: ["All", ...Array.from(regTypeSet)],
    };
  }, [tags]);

  // ⚡ Bolt Optimization: Memoize filtered tags and pull out .toLowerCase() to avoid redundant string allocations on every item
  const filteredTags = useMemo(() => {
    const lowerSearch = search.toLowerCase();
    return tags.filter((t) => {
      const matchesArea = selectedArea === "All" || t.area === selectedArea;
      if (!matchesArea) return false;

      const matchesRegType = selectedRegType === "All" || t.register_type === selectedRegType;
      if (!matchesRegType) return false;

      if (!lowerSearch) return true;

      return (
        t.name.toLowerCase().includes(lowerSearch) ||
        (t.description && t.description.toLowerCase().includes(lowerSearch)) ||
        (t.register_num !== null && String(t.register_num).includes(lowerSearch))
      );
    });
  }, [tags, search, selectedArea, selectedRegType]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Search and Filters Controls */}
      <section className="glass-panel" style={{ padding: "1.25rem" }}>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "1rem",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Search box */}
          <div style={{ position: "relative", flex: "1", minWidth: "250px" }}>
            <span
              style={{
                position: "absolute",
                left: "0.75rem",
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--text-muted)",
                pointerEvents: "none",
              }}
            >
              <Search size={16} />
            </span>
            <input
              type="text"
              placeholder="Search Tag name, register number or description..."
              className="text-input"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                width: "100%",
                paddingLeft: "2.25rem",
                fontSize: "0.85rem",
              }}
            />
          </div>

          {/* Area Filter */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)" }}>
              Area:
            </span>
            <select
              className="select-input"
              value={selectedArea}
              onChange={(e) => setSelectedArea(e.target.value)}
              style={{ fontSize: "0.8rem", padding: "0.4rem 1.5rem 0.4rem 0.75rem" }}
            >
              {areas.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>

          {/* Register Type Filter */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)" }}>
              Reg Type:
            </span>
            <select
              className="select-input"
              value={selectedRegType}
              onChange={(e) => setSelectedRegType(e.target.value)}
              style={{ fontSize: "0.8rem", padding: "0.4rem 1.5rem 0.4rem 0.75rem" }}
            >
              {regTypes.map((rt) => (
                <option key={rt ?? ""} value={rt ?? ""}>
                  {rt}
                </option>
              ))}
            </select>
          </div>

          {/* Refresh Button */}
          <button
            type="button"
            className="btn"
            onClick={fetchTags}
            style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem" }}
          >
            Refresh List
          </button>
        </div>
      </section>

      {/* Grid of PLC Register Mappings */}
      <section className="glass-panel" style={{ padding: "0" }}>
        <div
          className="panel-header"
          style={{ padding: "1.25rem 1.5rem 0.75rem 1.5rem", borderBottom: "none" }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Cpu size={18} color="var(--accent-magenta)" />
            <span style={{ fontWeight: 800 }}>PLC Memory & Ladder Cross-Reference</span>
          </div>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Showing {filteredTags.length} of {tags.length} registered tags
          </span>
        </div>

        {loading ? (
          <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
            Loading database registry...
          </div>
        ) : filteredTags.length === 0 ? (
          <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
            No tags matching current search filters.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="routing-table" style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--panel-border)" }}>
                  <th style={{ textAlign: "left", padding: "0.75rem 1.5rem" }}>Tag Name</th>
                  <th style={{ textAlign: "left", padding: "0.75rem 1rem" }}>Description</th>
                  <th style={{ textAlign: "center", padding: "0.75rem 1rem" }}>Type</th>
                  <th style={{ textAlign: "center", padding: "0.75rem 1rem" }}>Register</th>
                  <th style={{ textAlign: "center", padding: "0.75rem 1rem" }}>Mode</th>
                  <th style={{ textAlign: "right", padding: "0.75rem 1rem" }}>Scale</th>
                  <th style={{ textAlign: "left", padding: "0.75rem 1.5rem" }}>Equipment Path</th>
                  <th style={{ textAlign: "center", padding: "0.75rem 1.5rem" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredTags.map((t, idx) => {
                  const registerRepr =
                    t.register_type && t.register_num !== null
                      ? `${t.register_type}:${t.register_num}${
                          t.data_format ? `:${t.data_format}` : ""
                        }`
                      : "Unmapped";

                  return (
                    <tr
                      key={t.name}
                      style={{
                        borderBottom: "1px solid var(--panel-border)",
                        backgroundColor: idx % 2 === 0 ? "rgba(255,255,255,0.01)" : "none",
                        transition: "background var(--transition-fast)",
                      }}
                      className="hover-row"
                    >
                      <td
                        style={{
                          padding: "0.75rem 1.5rem",
                          fontWeight: 700,
                          fontSize: "0.85rem",
                          color: "var(--accent-cyan)",
                        }}
                      >
                        {t.name}
                      </td>
                      <td
                        style={{
                          padding: "0.75rem 1rem",
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          maxWidth: "200px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                        title={t.description}
                      >
                        {t.description || "-"}
                      </td>
                      <td style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                        <span
                          className={`badge ${
                            t.tag_type === "Boolean"
                              ? "badge-success"
                              : t.tag_type === "Real"
                              ? "badge-info"
                              : "badge-primary"
                          }`}
                          style={{ fontSize: "0.7rem", fontWeight: 700 }}
                        >
                          {t.tag_type}
                        </span>
                      </td>
                      <td
                        style={{
                          padding: "0.75rem 1rem",
                          textAlign: "center",
                          fontWeight: 700,
                          fontSize: "0.8rem",
                        }}
                        className="mono-text"
                      >
                        {registerRepr}
                      </td>
                      <td style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            fontWeight: 600,
                            padding: "0.2rem 0.5rem",
                            borderRadius: "4px",
                            backgroundColor:
                              t.rw_mode === "Read/Write"
                                ? "rgba(56, 189, 248, 0.08)"
                                : "rgba(255, 255, 255, 0.03)",
                            border: `1px solid ${
                              t.rw_mode === "Read/Write"
                                ? "rgba(56, 189, 248, 0.15)"
                                : "var(--panel-border)"
                            }`,
                            color:
                              t.rw_mode === "Read/Write"
                                ? "var(--accent-cyan)"
                                : "var(--text-muted)",
                          }}
                        >
                          {t.rw_mode}
                        </span>
                      </td>
                      <td
                        style={{
                          padding: "0.75rem 1rem",
                          textAlign: "right",
                          fontSize: "0.8rem",
                        }}
                        className="mono-text"
                      >
                        {t.scale_factor !== null ? t.scale_factor.toFixed(2) : "-"}
                      </td>
                      <td
                        style={{
                          padding: "0.75rem 1.5rem",
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {t.area ? `${t.area} ➔ ${t.unit} ➔ ${t.equipment}` : "Default Layout"}
                      </td>
                      <td style={{ padding: "0.75rem 1.5rem", textAlign: "center" }}>
                        <button
                          type="button"
                          className="btn"
                          style={{
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.7rem",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.25rem",
                          }}
                          onClick={() => onSelectTag(t.name)}
                        >
                          <Eye size={12} />
                          Inspect
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
