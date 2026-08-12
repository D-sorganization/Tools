import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { DecimalInput } from "./DecimalInput";

function Harness({ initial = 12.5 }: { initial?: number }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <DecimalInput aria-label="Engineering value" value={value} onCommit={setValue} />
      <output>{value}</output>
    </>
  );
}

describe("DecimalInput", () => {
  it("allows a temporary minus sign and commits a negative value", () => {
    render(<Harness />);
    const input = screen.getByLabelText("Engineering value") as HTMLInputElement;

    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "-" } });
    expect(input.value).toBe("-");
    fireEvent.change(input, { target: { value: "-17.5" } });
    fireEvent.blur(input);

    expect(screen.getByText("-17.5")).toBeInTheDocument();
    expect(input.value).toBe("-17.5");
  });

  it("selects the complete value on focus for one-action replacement", () => {
    render(<Harness initial={2686} />);
    const input = screen.getByLabelText("Engineering value") as HTMLInputElement;

    fireEvent.focus(input);
    expect(input.selectionStart).toBe(0);
    expect(input.selectionEnd).toBe(4);
  });

  it("restores the committed value when an intermediate draft loses focus", () => {
    render(<Harness initial={10.9} />);
    const input = screen.getByLabelText("Engineering value") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "-" } });
    fireEvent.blur(input);

    expect(input.value).toBe("10.9");
    expect(screen.getByText("10.9")).toBeInTheDocument();
  });

  it("clamps only when the edit is committed", () => {
    const Wrapper = () => {
      const [value, setValue] = useState(0);
      return (
        <DecimalInput
          aria-label="Bounded value"
          value={value}
          min={-20}
          max={20}
          onCommit={setValue}
        />
      );
    };
    render(<Wrapper />);
    const input = screen.getByLabelText("Bounded value") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "-25" } });
    expect(input.value).toBe("-25");
    fireEvent.blur(input);
    expect(input.value).toBe("-20");
  });
});
