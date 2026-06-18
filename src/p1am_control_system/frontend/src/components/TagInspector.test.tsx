import { fireEvent, render, screen } from "@testing-library/react";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import { TagInspector } from "./TagInspector";
import type { LadderTagInfo } from "../api/schemas";
import type { RoutingConfig } from "../types";

const routingConfig: RoutingConfig = {
  input_routing: [0, 1, 2, 3, 4, 5],
  output_routing: [10, 11],
  pids: [],
  interlocks: Array.from({ length: 32 }, () => ({
    lolo_limit: 0,
    low_limit: 5,
    high_limit: 95,
    hihi_limit: 100,
  })),
};

const tagInfo = (
  name: string,
  rw_mode: LadderTagInfo["rw_mode"],
): LadderTagInfo => ({
  name,
  rw_mode,
  tag_type: "Real",
  description: `${name} description`,
  register_type: "HR",
  register_num: 402,
  data_format: "float",
  scale_factor: 1,
  area: "process",
  unit: "unit",
  equipment: "bench",
});

const renderInspector = (
  overrides: Partial<React.ComponentProps<typeof TagInspector>> = {},
) => {
  const props: React.ComponentProps<typeof TagInspector> = {
    view: { type: "tag", tagId: 2 },
    allTags: [tagInfo("TAG_2", "Read/Write")],
    tagsDict: {},
    tagValues: [0, 0, 42.25],
    config: routingConfig,
    overrideVal: "7.5",
    showOverrideConfirm: false,
    deploying: false,
    onOverrideValChange: vi.fn(),
    onShowOverrideConfirmChange: vi.fn(),
    onExecuteOverride: vi.fn(),
    onInterlockChange: vi.fn(),
    onDeploy: vi.fn(),
    ...overrides,
  };
  render(<TagInspector {...props} />);
  return props;
};

describe("TagInspector", () => {
  it("renders tag metadata and asks for confirmation before executing an override", () => {
    const props = renderInspector({ showOverrideConfirm: true });

    expect(screen.getByText("TAG_2")).toBeInTheDocument();
    expect(screen.getByText("42.25")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));

    expect(props.onExecuteOverride).toHaveBeenCalledWith(2);
  });

  it("hides manual write controls for read-only custom tags", () => {
    renderInspector({
      view: { type: "custom_tag", tagName: "READ_ONLY_TEMP" },
      allTags: [tagInfo("READ_ONLY_TEMP", "Read-only")],
      tagsDict: { READ_ONLY_TEMP: 18.5 },
    });

    expect(screen.getByText("READ_ONLY_TEMP")).toBeInTheDocument();
    expect(screen.queryByText("Force Register Write")).toBeNull();
    expect(screen.queryByText("Safety Limits Interlocks")).toBeNull();
  });
});
