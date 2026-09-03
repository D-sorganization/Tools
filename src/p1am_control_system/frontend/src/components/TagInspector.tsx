import React from "react";
import { ShieldAlert, Sliders } from "lucide-react";
import type { LadderTagInfo } from "../api/schemas";
import type { InspectorState } from "../lib/inspector";
import { tagName } from "../lib/tags";
import type { InterlockConfig, RoutingConfig } from "../types";
import { limitInputValue, parseLimitInput } from "../lib/limits";

type TagInspectorView = Extract<
  InspectorState,
  { type: "tag" } | { type: "custom_tag" }
>;

export const TagInspector: React.FC<{
  view: TagInspectorView;
  allTags: LadderTagInfo[];
  tagsDict: Record<string, number>;
  tagValues: number[];
  config: RoutingConfig;
  overrideVal: string;
  showOverrideConfirm: boolean;
  deploying: boolean;
  onOverrideValChange: (value: string) => void;
  onShowOverrideConfirmChange: (show: boolean) => void;
  onExecuteOverride: (tagId: number | string) => void;
  onInterlockChange: (
    tagId: number,
    field: keyof InterlockConfig,
    value: number | null,
  ) => void;
  onDeploy: () => void;
}> = ({
  view,
  allTags,
  tagsDict,
  tagValues,
  config,
  overrideVal,
  showOverrideConfirm,
  deploying,
  onOverrideValChange,
  onShowOverrideConfirmChange,
  onExecuteOverride,
  onInterlockChange,
  onDeploy,
}) => {
  const isCustom = view.type === "custom_tag";
  const tagLabel = isCustom ? view.tagName : tagName(view.tagId);
  const tagInfo = allTags.find((t) => t.name === tagLabel);
  const currentVal = isCustom
    ? (tagsDict[tagLabel] ?? 0.0)
    : (tagValues[view.tagId] ?? 0.0);
  const rwMode = tagInfo ? tagInfo.rw_mode : "Read/Write";
  const isWritable = rwMode === "Read/Write";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>
        <h3
          style={{
            fontSize: "1rem",
            fontWeight: 700,
            color: "var(--accent-cyan)",
            textTransform: "uppercase",
          }}
        >
          Tag Inspector
        </h3>
        <div
          style={{
            fontSize: "0.8rem",
            color: "var(--text-primary)",
            fontWeight: 700,
            marginTop: "0.25rem",
          }}
          className="mono-text"
        >
          {tagLabel}
        </div>
        {tagInfo && tagInfo.description && (
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-secondary)",
              marginTop: "0.25rem",
              lineHeight: 1.4,
            }}
          >
            {tagInfo.description}
          </p>
        )}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            background: "var(--input-bg)",
            padding: "0.5rem",
            borderRadius: "4px",
            marginTop: "0.5rem",
            border: "1px solid var(--panel-border)",
          }}
        >
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Current Value:
          </span>
          <span
            className="mono-text"
            style={{
              fontSize: "0.9rem",
              fontWeight: 700,
              color: "var(--accent-cyan)",
            }}
          >
            {currentVal.toFixed(2)}
          </span>
        </div>
      </div>

      {tagInfo && (
        <div
          style={{
            fontSize: "0.75rem",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.5rem",
            background: "rgba(255,255,255,0.01)",
            padding: "0.5rem",
            borderRadius: "4px",
            border: "1px solid var(--panel-border)",
          }}
        >
          <div>
            <span style={{ color: "var(--text-muted)" }}>Reg Type:</span>{" "}
            <strong style={{ color: "var(--text-secondary)" }}>
              {tagInfo.register_type || "None"}
            </strong>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>Reg Num:</span>{" "}
            <strong style={{ color: "var(--text-secondary)" }}>
              {tagInfo.register_num ?? "None"}
            </strong>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>Format:</span>{" "}
            <strong style={{ color: "var(--text-secondary)" }}>
              {tagInfo.data_format || "None"}
            </strong>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>RW Mode:</span>{" "}
            <strong style={{ color: "var(--text-secondary)" }}>
              {tagInfo.rw_mode}
            </strong>
          </div>
        </div>
      )}

      {isWritable && (
        <div
          style={{
            borderTop: "1px solid var(--panel-border)",
            paddingTop: "0.75rem",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.3rem",
              marginBottom: "0.5rem",
            }}
          >
            <ShieldAlert size={14} color="var(--color-warning)" />
            <span
              style={{
                fontSize: "0.8rem",
                fontWeight: 700,
                textTransform: "uppercase",
              }}
            >
              Manual Force Override
            </span>
          </div>
          <div className="input-group">
            <label className="input-label">Override Force Value</label>
            <input
              type="number"
              step="0.1"
              className="form-input"
              value={overrideVal}
              onChange={(e) => onOverrideValChange(e.target.value)}
            />
          </div>
          <button
            type="button"
            onClick={() => onShowOverrideConfirmChange(true)}
            className="btn"
            style={{
              width: "100%",
              color: "var(--color-warning)",
              borderColor: "var(--color-warning)",
              padding: "0.45rem",
              fontSize: "0.8rem",
            }}
          >
            Force Register Write
          </button>

          {showOverrideConfirm && (
            <div
              style={{
                background: "rgba(239, 68, 68, 0.1)",
                border: "1px solid var(--color-error)",
                borderRadius: "4px",
                padding: "0.6rem",
                marginTop: "0.5rem",
              }}
            >
              <div
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  color: "var(--color-error)",
                  marginBottom: "0.25rem",
                }}
              >
                Confirm Direct Write
              </div>
              <p
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-secondary)",
                  marginBottom: "0.6rem",
                }}
              >
                Forcing Tag {tagLabel} to {overrideVal} will overwrite normal
                logic. Continue?
              </p>
              <div style={{ display: "flex", gap: "0.4rem" }}>
                <button
                  type="button"
                  onClick={() =>
                    onExecuteOverride(isCustom ? tagLabel : view.tagId)
                  }
                  className="btn"
                  style={{
                    background: "var(--color-error)",
                    border: "none",
                    color: "#ffffff",
                    padding: "0.2rem 0.5rem",
                    fontSize: "0.75rem",
                  }}
                >
                  Confirm
                </button>
                <button
                  type="button"
                  onClick={() => onShowOverrideConfirmChange(false)}
                  className="btn"
                  style={{ padding: "0.2rem 0.5rem", fontSize: "0.75rem" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {!isCustom && (
        <div
          style={{
            borderTop: "1px solid var(--panel-border)",
            paddingTop: "0.75rem",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.3rem",
              marginBottom: "0.5rem",
            }}
          >
            <Sliders size={14} color="var(--accent-purple)" />
            <span
              style={{
                fontSize: "0.8rem",
                fontWeight: 700,
                textTransform: "uppercase",
              }}
            >
              Safety Limits Interlocks
            </span>
          </div>
          <div className="input-group">
            <label className="input-label">High Trip Limit (Alarm High)</label>
            <input
              type="number"
              step="0.5"
              className="form-input"
              value={limitInputValue(config.interlocks[view.tagId]?.high_limit)}
              placeholder="disabled"
              onChange={(e) =>
                onInterlockChange(
                  view.tagId,
                  "high_limit",
                  parseLimitInput(e.target.value),
                )
              }
            />
          </div>
          <div className="input-group">
            <label className="input-label">Low Trip Limit (Alarm Low)</label>
            <input
              type="number"
              step="0.5"
              className="form-input"
              value={limitInputValue(config.interlocks[view.tagId]?.low_limit)}
              placeholder="disabled"
              onChange={(e) =>
                onInterlockChange(
                  view.tagId,
                  "low_limit",
                  parseLimitInput(e.target.value),
                )
              }
            />
          </div>
          <button
            type="button"
            onClick={onDeploy}
            disabled={deploying}
            className="btn btn-primary"
            style={{
              width: "100%",
              padding: "0.5rem",
              fontSize: "0.85rem",
              marginTop: "0.5rem",
            }}
          >
            {deploying ? "Deploying Configuration..." : "Commit Safety Limits"}
          </button>
        </div>
      )}
    </div>
  );
};
