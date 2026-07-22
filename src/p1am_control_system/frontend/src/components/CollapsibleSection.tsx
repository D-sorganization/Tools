import React, { useState } from "react";
import { ChevronDown } from "lucide-react";

/**
 * A panel section whose body can be collapsed to a single header row, so the
 * operator can minimize the parts of a screen they aren't using. Reusable across
 * panels (pass the card class via `className`); the header stays visible when
 * collapsed and a chevron shows the state.
 */
export const CollapsibleSection: React.FC<{
  title: React.ReactNode;
  defaultOpen?: boolean;
  /** Wrapper class — typically the panel's card class (e.g. "tc-card"). */
  className?: string;
  /** Optional controls rendered at the right of the header (kept clickable). */
  headerExtra?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, defaultOpen = true, className, headerExtra, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={className}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            gap: "0.45rem",
            background: "none",
            border: "none",
            padding: 0,
            margin: 0,
            cursor: "pointer",
            textAlign: "left",
            fontSize: "0.95rem",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
        >
          <ChevronDown
            size={15}
            aria-hidden
            style={{
              flexShrink: 0,
              transition: "transform 0.15s ease",
              transform: open ? "none" : "rotate(-90deg)",
              opacity: 0.7,
            }}
          />
          {title}
        </button>
        {headerExtra}
      </div>
      {open && <div style={{ marginTop: "0.6rem" }}>{children}</div>}
    </section>
  );
};
