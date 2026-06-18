import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { EStopButton } from "./EStopButton";

describe("EStopButton", () => {
  it("renders the trigger affordance when not latched and fires onTriggerEStop", () => {
    const onTrigger = vi.fn();
    const onClear = vi.fn();
    render(
      <EStopButton
        eStopActive={false}
        onTriggerEStop={onTrigger}
        onClearEStop={onClear}
      />,
    );

    const btn = screen.getByRole("button", { name: /emergency stop/i });
    fireEvent.click(btn);

    expect(onTrigger).toHaveBeenCalledTimes(1);
    expect(onClear).not.toHaveBeenCalled();
  });

  it("renders the clear affordance when latched and fires onClearEStop", () => {
    const onTrigger = vi.fn();
    const onClear = vi.fn();
    render(
      <EStopButton
        eStopActive={true}
        onTriggerEStop={onTrigger}
        onClearEStop={onClear}
      />,
    );

    const btn = screen.getByRole("button", { name: /clear e-stop/i });
    fireEvent.click(btn);

    expect(onClear).toHaveBeenCalledTimes(1);
    expect(onTrigger).not.toHaveBeenCalled();
  });
});
