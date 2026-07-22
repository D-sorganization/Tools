import React, { useEffect, useRef } from "react";
import type { TabId } from "../lib/tabs";
import { HELP, SYSTEM_OVERVIEW } from "../help/helpContent";
import { renderMarkdown } from "../lib/markdownLite";
import "./HelpModal.css";

/**
 * A modal that shows the help for one HMI tab, plus the system overview.
 *
 * Content comes from {@link HELP} (per tab) rendered by the lightweight Markdown
 * renderer. Accessible: labelled dialog, Escape / backdrop / ✕ to close, and
 * focus moves to the close button on open. Renders nothing when closed.
 */
export interface HelpModalProps {
  tabId: TabId;
  open: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ tabId, open, onClose }) => {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const doc = HELP[tabId];

  return (
    <div
      className="help-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="help-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Help — ${doc.title}`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="help-modal__head">
          <div>
            <span className="help-modal__eyebrow">Help</span>
            <h2 className="help-modal__title">{doc.title}</h2>
          </div>
          <button
            ref={closeRef}
            type="button"
            className="help-modal__close"
            onClick={onClose}
            aria-label="Close help"
          >
            ✕
          </button>
        </header>

        <div className="help-modal__body">
          {renderMarkdown(doc.body)}
          <hr />
          <div className="help-modal__overview">{renderMarkdown(SYSTEM_OVERVIEW)}</div>
        </div>

        <footer className="help-modal__foot">
          Full reference: <code>USER_MANUAL.md</code> · press{" "}
          <kbd>Esc</kbd> to close
        </footer>
      </div>
    </div>
  );
};
