import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RotationConverter } from "./RotationConverter";

describe("RotationConverter", () => {
  it("gives every rotation input a descriptive accessible name", () => {
    render(<RotationConverter />);

    expect(
      screen.getByRole("spinbutton", { name: "Quaternion component W" }),
    ).toBeInTheDocument();

    const inputType = screen.getByRole("combobox", { name: "Input Type" });
    fireEvent.change(inputType, { target: { value: "euler" } });
    expect(screen.getByRole("textbox", { name: "Convention" })).toBeInTheDocument();
    expect(
      screen.getByRole("spinbutton", { name: "Euler angle axis 3" }),
    ).toBeInTheDocument();

    fireEvent.change(inputType, { target: { value: "axis_angle" } });
    expect(
      screen.getByRole("spinbutton", {
        name: "Axis-angle component angle (rad)",
      }),
    ).toBeInTheDocument();

    fireEvent.change(inputType, { target: { value: "rodrigues" } });
    expect(
      screen.getByRole("spinbutton", {
        name: "Rodrigues vector component rz",
      }),
    ).toBeInTheDocument();
  });
});
