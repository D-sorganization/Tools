import React from "react";
import type { NotificationState } from "../types";

/**
 * Fixed top-right toast banner (#3543).
 *
 * Extracted verbatim from App.tsx's inline notification block so the host no
 * longer carries ~30 lines of inline-styled markup.
 */
export const NotificationBanner: React.FC<{
  notification: NotificationState | null;
}> = ({ notification }) => {
  if (!notification) return null;

  const accent =
    notification.type === "success"
      ? "var(--color-success)"
      : notification.type === "error"
        ? "var(--color-error)"
        : "var(--accent-cyan)";

  const bg =
    notification.type === "success"
      ? "rgba(16, 185, 129, 0.2)"
      : notification.type === "error"
        ? "rgba(239, 68, 68, 0.25)"
        : "rgba(56, 189, 248, 0.15)";

  return (
    <div
      role="status"
      style={{
        position: "fixed",
        top: "1.25rem",
        right: "1.25rem",
        zIndex: 9999,
        padding: "0.75rem 1.25rem",
        borderRadius: "4px",
        fontSize: "0.85rem",
        fontWeight: 500,
        borderLeft: `4px solid ${accent}`,
        color: "#ffffff",
        backgroundColor: bg,
      }}
    >
      {notification.message}
    </div>
  );
};
