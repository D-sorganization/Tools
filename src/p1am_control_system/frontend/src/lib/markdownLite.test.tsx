import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { renderInline, renderMarkdown } from "./markdownLite";

describe("renderInline", () => {
  it("renders **bold** and `code` spans, leaving plain text intact", () => {
    render(<p>{renderInline("start **bold** mid `code` end")}</p>);
    expect(screen.getByText("bold").tagName).toBe("STRONG");
    expect(screen.getByText("code").tagName).toBe("CODE");
    expect(screen.getByText(/start/)).toBeTruthy();
  });

  it("returns a single plain string when there is no markup", () => {
    const nodes = renderInline("just text");
    expect(nodes).toEqual(["just text"]);
  });
});

describe("renderMarkdown", () => {
  it("renders headings, a paragraph, and both list kinds", () => {
    const md = "## Section\n\nIntro paragraph.\n\n- one\n- two\n\n1. first\n2. second";
    render(<div>{renderMarkdown(md)}</div>);
    // '##' demotes to h3 so it sits under the modal's h2 title.
    expect(screen.getByRole("heading", { level: 3, name: "Section" })).toBeTruthy();
    expect(screen.getByText("Intro paragraph.")).toBeTruthy();
    expect(screen.getAllByRole("listitem")).toHaveLength(4);
  });

  it("renders a horizontal rule and inline markup inside list items", () => {
    const { container } = render(
      <div>{renderMarkdown("- **bold** item\n\n---")}</div>,
    );
    expect(container.querySelector("hr")).toBeTruthy();
    expect(container.querySelector("li strong")?.textContent).toBe("bold");
  });

  it("joins wrapped paragraph lines into one paragraph", () => {
    const { container } = render(
      <div>{renderMarkdown("line one\nline two")}</div>,
    );
    const paras = container.querySelectorAll("p");
    expect(paras).toHaveLength(1);
    expect(paras[0].textContent).toBe("line one line two");
  });

  it("throws on a non-string body (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => renderMarkdown(42)).toThrow(TypeError);
  });
});
