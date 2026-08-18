import React from "react";
import { CollapsibleSection } from "./CollapsibleSection";
import { formatTcReadout } from "../lib/temperatureTrend";
import type { TcType, TemperatureConfig, TemperatureStatus } from "../types";

/**
 * Thermocouple source selector (Type K / Type R) plus the P1-04THM open-circuit
 * (burnout) fail direction.
 *
 * Split out of TemperatureControl.tsx unchanged. Both controls change what the
 * heater believes the temperature is, so they stay together — but the POSTs that
 * apply them remain in the parent, which owns `busy` and the toast channel.
 */
interface Props {
  config: TemperatureConfig;
  /** Live status; undefined while waiting for the first frame. */
  status: TemperatureStatus | undefined;
  activeTcType: TcType;
  busy: boolean;
  onSelectTcType: (tcType: TcType) => void;
  onSetBurnoutMode: (highSide: boolean) => void;
}

export const ThermocoupleSelector: React.FC<Props> = ({
  config,
  status: s,
  activeTcType,
  busy,
  onSelectTcType,
  onSetBurnoutMode,
}) => (
  <CollapsibleSection
    className="tc-card"
    title="Thermocouple — heater temperature source"
  >
    <div
      role="group"
      aria-label="Thermocouple type"
      style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
    >
      {(["K", "R"] as const).map((tc) => {
        const ch = tc === "K" ? config.type_k : config.type_r;
        const isActive = activeTcType === tc;
        const reading = tc === "K" ? s?.type_k_temp_c : s?.type_r_temp_c;
        return (
          <button
            key={tc}
            type="button"
            onClick={() => !isActive && onSelectTcType(tc)}
            disabled={busy || isActive}
            aria-pressed={isActive}
            title={`${ch.label} on ${ch.tag} — live ${formatTcReadout(reading)}`}
            style={{
              flex: "1 1 0",
              minWidth: "8rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              gap: "0.15rem",
              padding: "0.5rem 0.75rem",
              borderRadius: "6px",
              cursor: isActive ? "default" : "pointer",
              border: "2px solid",
              borderColor: isActive ? "var(--accent-cyan)" : "var(--panel-border)",
              background: isActive ? "var(--cell-hover-bg)" : "var(--input-bg)",
              color: isActive ? "var(--accent-cyan)" : "var(--text-secondary)",
            }}
          >
            <span
              style={{
                display: "flex",
                alignItems: "baseline",
                justifyContent: "space-between",
                gap: "0.5rem",
                width: "100%",
              }}
            >
              <strong style={{ fontSize: "0.95rem" }}>
                Type {tc}
                {isActive ? " ✓" : ""}
              </strong>
              <span
                className="tc-tc-reading"
                aria-label={`Type ${tc} live reading`}
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "1.05rem",
                  fontWeight: 700,
                  // Emphasize the active channel; dim the other so a dead/stuck
                  // sensor is still visible but reads as secondary.
                  color: isActive ? "var(--accent-cyan)" : "var(--text-primary)",
                  opacity: isActive ? 1 : 0.75,
                }}
              >
                {formatTcReadout(reading)}
              </span>
            </span>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
              {ch.label} · {ch.tag}
              {isActive ? " · controlling" : ""}
            </span>
          </button>
        );
      })}
    </div>
    <p
      style={{
        fontSize: "0.72rem",
        color: "var(--text-secondary)",
        margin: "0.5rem 0 0",
        lineHeight: 1.5,
      }}
    >
      The selected thermocouple drives <strong>all</strong> heater control —
      setpoint band, HH cutoff and trends. Switching re-clamps the limits to the
      chosen channel's range. (Type R is wired to a separate THM channel.)
    </p>

    {/* ---- Open-circuit (burnout) fail direction ---- */}
    <div
      style={{
        marginTop: "0.85rem",
        paddingTop: "0.75rem",
        borderTop: "1px solid var(--panel-border)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: "0.78rem", fontWeight: 600 }}>
          On a broken thermocouple, read:
        </span>
        <div
          style={{
            display: "inline-flex",
            borderRadius: "6px",
            overflow: "hidden",
            border: "1px solid var(--panel-border)",
          }}
        >
          {(
            [
              { high: true, label: "Hot (fail-safe)" },
              { high: false, label: "Cold" },
            ] as const
          ).map(({ high, label }) => {
            const active = (s?.burnout_high_side ?? true) === high;
            return (
              <button
                key={label}
                type="button"
                disabled={busy || active}
                onClick={() => onSetBurnoutMode(high)}
                aria-pressed={active}
                style={{
                  border: "none",
                  padding: "0.3rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: active ? "default" : "pointer",
                  background: active
                    ? high
                      ? "var(--accent-cyan)"
                      : "var(--accent-amber, #f59e0b)"
                    : "var(--input-bg)",
                  color: active ? "#0f172a" : "var(--text-secondary)",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>
      <p
        style={{
          fontSize: "0.7rem",
          color: "var(--text-muted)",
          margin: "0.45rem 0 0",
          lineHeight: 1.5,
        }}
      >
        Sets the P1-04THM open-circuit (burnout) direction. <strong>Hot</strong>{" "}
        makes an open sensor read full-scale so the heater shuts off
        (recommended). <strong>Cold</strong> makes it read 0&nbsp;°C. The deglitch
        filter rides out brief dropouts either way and trips on a sustained fault.
      </p>
    </div>
  </CollapsibleSection>
);
