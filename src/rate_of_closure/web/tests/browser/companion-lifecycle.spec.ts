import { expect, test } from "@playwright/test";

import { startCompanionHarness } from "./support/companionHarness";
import { auditSameOriginNetwork } from "./support/networkAudit";

test("authority hard loss replaces private identity behind the stable gateway", async ({ page }) => {
  const companion = await startCompanionHarness("fast");
  const audit = auditSameOriginNetwork(page, companion.origin);
  try {
    const initialCapability = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/rate-of-closure/v1/capabilities"
      && response.status() === 200);
    await page.goto(`${companion.origin}/`);
    await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
      .toBeVisible();
    await initialCapability;
    const stopped = await companion.command("authority_hard_loss");
    expect(stopped).toMatchObject({ event: "authority_stopped", authority_stopped: true });
    const replaced = await companion.command("observe_replacement");
    expect(replaced).toMatchObject({
      event: "authority_replaced", authority_replaced: true,
      authority_running: true, token_changed: true, port_changed: true,
    });
    const capability = await page.evaluate(async () => {
      const response = await fetch("/api/rate-of-closure/v1/capabilities");
      return { status: response.status, body: await response.json() as unknown };
    });
    expect(capability).toMatchObject({ status: 200,
      body: { regional_ground_execution: true } });
    expect(new URL(page.url()).origin).toBe(companion.origin);
    audit.assertClean();
  } finally {
    await companion.close();
  }
});
