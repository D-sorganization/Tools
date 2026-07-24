import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { EditableValue } from "./EditableValue";

describe("EditableValue", () => {
  it("renders the formatted reading with its unit", () => {
    render(
      <EditableValue
        value={1400}
        onCommit={() => {}}
        label="HH cutoff"
        unit="°C"
        format={(v) => v.toFixed(0)}
      />,
    );
    expect(screen.getByRole("button", { name: /HH cutoff: 1400 °C/i })).toBeTruthy();
  });

  it("commits a changed value on Enter", async () => {
    const onCommit = vi.fn();
    render(<EditableValue value={100} onCommit={onCommit} label="Setpoint" unit="°C" />);
    fireEvent.click(screen.getByRole("button", { name: /Setpoint/i }));
    const input = screen.getByLabelText("Setpoint") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "250" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(onCommit).toHaveBeenCalledWith(250));
  });

  it("does not commit when the value is unchanged", () => {
    const onCommit = vi.fn();
    render(<EditableValue value={100} onCommit={onCommit} label="Setpoint" />);
    fireEvent.click(screen.getByRole("button"));
    const input = screen.getByLabelText("Setpoint") as HTMLInputElement;
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("clamps to max before committing", async () => {
    const onCommit = vi.fn();
    render(
      <EditableValue value={100} onCommit={onCommit} label="Cutoff" max={1400} />,
    );
    fireEvent.click(screen.getByRole("button"));
    const input = screen.getByLabelText("Cutoff") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "9999" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(onCommit).toHaveBeenCalledWith(1400));
  });

  it("cancels on Escape without committing", () => {
    const onCommit = vi.fn();
    render(<EditableValue value={100} onCommit={onCommit} label="Setpoint" />);
    fireEvent.click(screen.getByRole("button"));
    const input = screen.getByLabelText("Setpoint") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "250" } });
    fireEvent.keyDown(input, { key: "Escape" });
    expect(onCommit).not.toHaveBeenCalled();
    // Back to reading mode showing the original value.
    expect(screen.getByRole("button", { name: /100/ })).toBeTruthy();
  });

  it("marks the input invalid and does not commit garbage on Enter", () => {
    const onCommit = vi.fn();
    render(<EditableValue value={100} onCommit={onCommit} label="Setpoint" />);
    fireEvent.click(screen.getByRole("button"));
    const input = screen.getByLabelText("Setpoint") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "abc" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onCommit).not.toHaveBeenCalled();
    expect(input.getAttribute("aria-invalid")).toBe("true");
  });

  it("renders plain, non-interactive text when disabled", () => {
    render(<EditableValue value={100} onCommit={() => {}} label="Setpoint" disabled />);
    expect(screen.queryByRole("button")).toBeNull();
  });
});
