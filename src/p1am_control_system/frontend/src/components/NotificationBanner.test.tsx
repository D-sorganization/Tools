import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { NotificationBanner } from "./NotificationBanner";

describe("NotificationBanner", () => {
  it("renders nothing when there is no notification", () => {
    const { container } = render(<NotificationBanner notification={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the message with a status role", () => {
    render(
      <NotificationBanner
        notification={{ message: "Saved!", type: "success" }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Saved!");
  });
});
