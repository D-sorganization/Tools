import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { DiagnosticsFeed } from "./DiagnosticsFeed";
import type { TemperatureStatus } from "./TemperatureControl";

/** Recent frames ending ~now, so the stream reads as fresh/LIVE. */
function makeData(n: number): { history: number[][]; historyTimes: number[] } {
  const t0 = Date.now() - n * 100;
  const historyTimes = Array.from({ length: n }, (_, i) => t0 + i * 100);
  const history = Array.from({ length: n }, () =>
    Array.from({ length: 32 }, (_, t) => (t === 0 ? 35 : t === 1 ? 47 : t)),
  );
  return { history, historyTimes };
}

const temp = {
  type_k_temp_c: 35.9,
  type_r_temp_c: 475.4,
  relay_on: false,
} as TemperatureStatus;

describe("DiagnosticsFeed", () => {
  it("shows a LIVE bar with rate + K/R summary when connected and fresh", () => {
    const { history, historyTimes } = makeData(20);
    render(
      <DiagnosticsFeed
        history={history}
        historyTimes={historyTimes}
        isConnected
        temperature={temp}
      />,
    );
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByText(/Hz/)).toBeInTheDocument();
    expect(screen.getByText(/K 35\.9°C/)).toBeInTheDocument();
    expect(screen.getByText(/R 475\.4°C/)).toBeInTheDocument();
  });

  it("reads OFFLINE when disconnected", () => {
    const { history, historyTimes } = makeData(5);
    render(
      <DiagnosticsFeed
        history={history}
        historyTimes={historyTimes}
        isConnected={false}
        temperature={temp}
      />,
    );
    expect(screen.getByText("OFFLINE")).toBeInTheDocument();
  });

  it("expands to a raw-value feed carrying the recent tag values", () => {
    const { history, historyTimes } = makeData(10);
    render(
      <DiagnosticsFeed
        history={history}
        historyTimes={historyTimes}
        isConnected
        temperature={temp}
      />,
    );
    expect(screen.queryByLabelText("Recent raw tag values")).toBeNull();
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    const feed = screen.getByLabelText("Recent raw tag values");
    expect(feed.textContent).toContain("T0=35.0");
    expect(feed.textContent).toContain("T1=47.0");
  });
});
