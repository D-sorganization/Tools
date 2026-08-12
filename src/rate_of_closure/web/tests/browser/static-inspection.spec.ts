import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { auditSameOriginNetwork } from "./support/networkAudit";
import { startStaticReleaseServer } from "./support/staticReleaseServer";

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
