import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AlarmsHeader } from "./AlarmsHeader";
import type { ActiveAlarm } from "../types";

/**
 * Degraded-alarm-data banner (#4011).
 *
 * When an entry of the `active_alarms` map fails validation it is dropped. The
 * list on screen is then INCOMPLETE, but nothing said so — and the summary line
 * happily reported "All normal — no active alarms" while alarms fired on the
 * PLC. Dropping data silently is the defect; saying so is the fix.
 */

const alarm = (tagId: string, severity = 2): ActiveAlarm => ({
  tag_id: tagId,
  state: "HiHi",
  severity,
  acknowledged: false,
  timestamp: "2026-07-31T10:00:00Z",
});

describe("AlarmsHeader", () => {
  it("summarises a normal system when nothing is dropped", () => {
    render(
      <AlarmsHeader activeAlarms={[]} droppedAlarmCount={0} onAcknowledgeAll={vi.fn()} />,
    );
    expect(screen.getByText(/All normal — no active alarms/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("warns that the alarm list is INCOMPLETE when entries were dropped", () => {
    render(
      <AlarmsHeader activeAlarms={[]} droppedAlarmCount={2} onAcknowledgeAll={vi.fn()} />,
    );
    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent(/2/);
    expect(banner).toHaveTextContent(/incomplete/i);
    // The reassuring summary must NOT be shown while data is known-missing.
    expect(screen.queryByText(/All normal — no active alarms/)).not.toBeInTheDocument();
  });

  it("keeps listing the alarms that DID parse alongside the warning", () => {
    render(
      <AlarmsHeader
        activeAlarms={[alarm("TAG_0"), alarm("TAG_9")]}
        droppedAlarmCount={1}
        onAcknowledgeAll={vi.fn()}
      />,
    );
    expect(screen.getByText(/2 active alarms/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});
