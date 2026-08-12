import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MAX_MORRIS_WORKSPACE_BYTES } from "../model/morrisWorkspaceDocument";
import { MorrisWorkspaceActions } from "./MorrisWorkspaceActions";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("MorrisWorkspaceActions", () => {
  it("uses a keyboard-focusable import button with visible focus styling", async () => {
    const user = userEvent.setup();
    render(<MorrisWorkspaceActions workspace={null} busy={false}
      onImportText={vi.fn()} onImportError={vi.fn()} />);
    const button = screen.getByRole("button", { name: "Import Workspace JSON" });
    const input = screen.getByLabelText("Import Morris workspace JSON");
    const click = vi.spyOn(input, "click");
    expect(button).toHaveClass("focus-visible:outline");
    await user.tab();
    expect(button).toHaveFocus();
    await user.keyboard("{Enter}");
    await user.keyboard(" ");
    expect(click).toHaveBeenCalledTimes(2);
  });

  it("rejects an oversized file before constructing or invoking FileReader", async () => {
    const user = userEvent.setup();
    const onImportError = vi.fn();
    const readAsText = vi.spyOn(FileReader.prototype, "readAsText");
    render(<MorrisWorkspaceActions workspace={null} busy={false}
      onImportText={vi.fn()} onImportError={onImportError} />);
    await user.upload(
      screen.getByLabelText("Import Morris workspace JSON"),
      new File(["x".repeat(MAX_MORRIS_WORKSPACE_BYTES + 1)], "oversized.json", {
        type: "application/json",
      }),
    );
    expect(onImportError).toHaveBeenCalledWith(expect.stringMatching(/2000000-byte file limit/));
    expect(readAsText).not.toHaveBeenCalled();
  });
});
