import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { startCompanionHarness } from "./support/companionHarness";
import { auditSameOriginNetwork } from "./support/networkAudit";

test("production companion serves the application and its qualified capability", async ({ page }) => {
  const companion = await startCompanionHarness();
  const audit = auditSameOriginNetwork(page, companion.origin);
  try {
    await page.goto(`${companion.origin}/`);
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toBeVisible();
    const runtime = await page.locator("#rate-of-closure-web-runtime").textContent();
    expect(JSON.parse(runtime ?? "{}")).toMatchObject({
      schema_version: "rate-of-closure/web-runtime/v1",
      mode: "local_companion",
      release_revision: expect.stringMatching(/^[0-9a-f]{40}$/),
      authority_path: "/api/rate-of-closure/v1",
    });
    const capability = await page.evaluate(async () => {
      const response = await fetch("/api/rate-of-closure/v1/capabilities");
      return { status: response.status, body: await response.json() as unknown };
    });
    expect(capability.status).toBe(200);
    expect(capability.body).toMatchObject({ regional_ground_execution: true });
    const accessibility = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    expect(accessibility.violations.filter((item) =>
      item.impact === "critical" || item.impact === "serious")).toEqual([]);
    audit.assertClean();
  } finally {
    await companion.close();
  }
});
