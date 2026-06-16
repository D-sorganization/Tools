import React from "react";
import { ShieldAlert, RefreshCw } from "lucide-react";

interface EStopButtonProps {
  eStopActive: boolean;
  onTriggerEStop: () => void;
  onClearEStop: () => void;
}

/**
 * Emergency-stop control. Always red and always reachable (the parent keeps it
 * in a sticky header). When latched it flips to an amber "CLEAR E-STOP".
 *
 * Styled with explicit inline styles — this project has no Tailwind, so the
 * utility classes this component used to carry never rendered.
 */
export const EStopButton: React.FC<EStopButtonProps> = ({
  eStopActive,
  onTriggerEStop,
  onClearEStop,
}) => {
  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.55rem 1.1rem",
    borderRadius: "8px",
    fontWeight: 800,
    fontSize: "0.85rem",
    letterSpacing: "0.5px",
    border: "none",
    cursor: "pointer",
    color: "#fff",
    transition: "filter 0.12s ease, transform 0.12s ease",
  };

  if (eStopActive) {
    return (
      <button
        type="button"
        onClick={onClearEStop}
        title="Clear the latched E-stop (requires re-arm afterward)"
        style={{
          ...base,
          background: "#d97706",
          boxShadow: "0 0 18px rgba(217,119,6,0.5)",
          animation: "pulse 1.4s ease-in-out infinite",
        }}
      >
        <RefreshCw size={16} />
        CLEAR E-STOP
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onTriggerEStop}
      title="Emergency stop — immediately forces all outputs to zero"
      style={{
        ...base,
        background: "#dc2626",
        boxShadow: "0 0 16px rgba(220,38,38,0.45)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.filter = "brightness(1.1)";
        e.currentTarget.style.transform = "scale(1.03)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.filter = "none";
        e.currentTarget.style.transform = "none";
      }}
    >
      <ShieldAlert size={16} />
      EMERGENCY STOP
    </button>
  );
};
