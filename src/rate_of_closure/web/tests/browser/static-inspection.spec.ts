import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { auditSameOriginNetwork, summarizeRuntimeError } from "./support/networkAudit";
import { startStaticReleaseServer } from "./support/staticReleaseServer";

test("runtime diagnostics redact origins and token-like values", () => {
  expect(summarizeRuntimeError("console error",
    "failed https://127.0.0.1:5193/private abcdefghijklmnopqrstuvwxyz123456"))
    .toBe("console error: failed [url] [redacted]");
});

test("static inspection loads exact release assets from a nested subpath", async ({ page }) => {
  const server = await startStaticReleaseServer();
  const audit = auditSameOriginNetwork(page, server.origin, { forbidApi: true });
  try {
    await page.goto(`${server.mountUrl}index.html#impact`);
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toBeVisible();
    expect(new URL(page.url()).pathname).toBe("/release/candidate/index.html");
    expect(new URL(page.url()).hash).toBe("#impact");
    await expect(page.locator("script[src], link[rel=stylesheet]")).not.toHaveCount(0);
    const violations = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(violations.violations.filter((item) =>
      item.impact === "critical" || item.impact === "serious")).toEqual([]);
    audit.assertClean();
  } finally {
    await server.close();
  }
});

test("static fixture preserves directory-index and fragment semantics", async ({ page }) => {
  const server = await startStaticReleaseServer();
  const audit = auditSameOriginNetwork(page, server.origin, { forbidApi: true });
  try {
    const response = await page.goto(`${server.mountUrl}#flight`);
    expect(response?.status()).toBe(200);
    expect(new URL(page.url()).pathname).toBe("/release/candidate/");
    expect(new URL(page.url()).hash).toBe("#flight");
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toBeVisible();
    audit.assertClean();
  } finally {
    await server.close();
  }
});

test("missing declared release script fails closed without leaving loopback", async ({ page }) => {
  const server = await startStaticReleaseServer({ fault: "missing-script" });
  const audit = auditSameOriginNetwork(page, server.origin, { forbidApi: true });
  try {
    const rejectedScript = page.waitForResponse((candidate) =>
      candidate.status() === 404 &&
      /\/assets\/index-[A-Za-z0-9_-]+\.js$/.test(new URL(candidate.url()).pathname));
    const response = await page.goto(server.mountUrl, { waitUntil: "commit" });
    expect(response?.status()).toBe(200);
    await rejectedScript;
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toHaveCount(0);
    audit.assertBoundaryClean();
  } finally {
    await server.close();
  }
});

test("corrupt persisted workspace state restores safe defaults", async ({ page }) => {
  const server = await startStaticReleaseServer();
  const audit = auditSameOriginNetwork(page, server.origin, { forbidApi: true });
  try {
    await page.addInitScript(() => {
      localStorage.setItem("rate-of-closure.web.workspace-modules.v2", "{not-json");
      localStorage.setItem("rate-of-closure.impact-scene-layers.v1", "{not-json");
    });
    await page.goto(server.mountUrl);
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toBeVisible();
    await expect(page.getByRole("tab", { name: "Explorer", exact: true })).toHaveAttribute(
      "aria-selected", "true",
    );
    audit.assertClean();
  } finally {
    await server.close();
  }
});
