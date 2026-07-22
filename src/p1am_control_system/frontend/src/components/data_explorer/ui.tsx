import React from "react";

/**
 * Small theme-aware UI primitives shared by the Data Explorer panels.
 *
 * The rest of the HMI styles controls with inline styles + CSS variables; these
 * wrap that pattern once so the nine explorer panels stay terse and consistent
 * (DRY) instead of repeating the same label/input/button markup everywhere.
 */

export const ExpCard: React.FC<{
  title?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  accent?: string;
}> = ({ title, right, children, accent = "var(--accent-cyan)" }) => (
  <section
    style={{
      background: "var(--panel-bg)",
      border: "1px solid var(--panel-border)",
      borderRadius: "8px",
      padding: "0.85rem 1rem",
      boxShadow: "var(--card-shadow)",
    }}
  >
    {(title || right) && (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
          marginBottom: "0.6rem",
        }}
      >
        <h3
          style={{
            margin: 0,
            fontSize: "0.82rem",
            fontWeight: 700,
            letterSpacing: "0.02em",
            textTransform: "uppercase",
            color: accent,
          }}
        >
          {title}
        </h3>
        {right}
      </div>
    )}
    {children}
  </section>
);

export const Field: React.FC<{
  label: React.ReactNode;
  children: React.ReactNode;
  hint?: React.ReactNode;
}> = ({ label, children, hint }) => (
  <label style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
    <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>
      {label}
    </span>
    {children}
    {hint && (
      <span style={{ fontSize: "0.62rem", color: "var(--text-muted)" }}>
        {hint}
      </span>
    )}
  </label>
);

export const Row: React.FC<{
  children: React.ReactNode;
  gap?: string;
  wrap?: boolean;
}> = ({ children, gap = "0.6rem", wrap = true }) => (
  <div
    style={{
      display: "flex",
      gap,
      flexWrap: wrap ? "wrap" : "nowrap",
      alignItems: "flex-end",
    }}
  >
    {children}
  </div>
);

const baseControl: React.CSSProperties = {
  background: "var(--input-bg)",
  color: "var(--text-primary)",
  border: "1px solid var(--panel-border)",
  borderRadius: "5px",
  padding: "0.32rem 0.45rem",
  fontSize: "0.78rem",
  fontFamily: "var(--font-sans)",
};

export const Btn: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: "primary" | "ghost" | "danger";
  }
> = ({ variant = "ghost", style, children, ...rest }) => {
  const palette: Record<string, React.CSSProperties> = {
    primary: {
      background: "var(--accent-cyan)",
      color: "#04141b",
      borderColor: "var(--accent-cyan)",
      fontWeight: 700,
    },
    ghost: { background: "var(--input-bg)", color: "var(--text-primary)" },
    danger: { background: "transparent", color: "var(--color-error)" },
  };
  return (
    <button
      type="button"
      style={{
        ...baseControl,
        cursor: rest.disabled ? "not-allowed" : "pointer",
        opacity: rest.disabled ? 0.5 : 1,
        whiteSpace: "nowrap",
        ...palette[variant],
        ...style,
      }}
      {...rest}
    >
      {children}
    </button>
  );
};

export const Select: React.FC<
  React.SelectHTMLAttributes<HTMLSelectElement>
> = ({ style, children, ...rest }) => (
  <select style={{ ...baseControl, ...style }} {...rest}>
    {children}
  </select>
);

export const NumInput: React.FC<
  React.InputHTMLAttributes<HTMLInputElement>
> = ({ style, ...rest }) => (
  <input
    type="number"
    style={{ ...baseControl, width: "6.5rem", ...style }}
    {...rest}
  />
);

export const TextInput: React.FC<
  React.InputHTMLAttributes<HTMLInputElement>
> = ({ style, ...rest }) => (
  <input type="text" style={{ ...baseControl, ...style }} {...rest} />
);

export const Check: React.FC<{
  label: React.ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
}> = ({ label, checked, onChange }) => (
  <label
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "0.3rem",
      fontSize: "0.74rem",
      color: "var(--text-secondary)",
      cursor: "pointer",
    }}
  >
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
    />
    {label}
  </label>
);

export const ErrorText: React.FC<{ children: React.ReactNode }> = ({
  children,
}) =>
  children ? (
    <p
      style={{
        margin: "0.4rem 0 0",
        fontSize: "0.72rem",
        color: "var(--color-error)",
      }}
    >
      {children}
    </p>
  ) : null;
