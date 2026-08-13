import { expect, type Page } from "@playwright/test";

export async function openVariation(page: Page): Promise<void> {
  await page.goto("./");
  await page.getByRole("tab", { name: "Variation" }).click();
  await expect(page.getByRole("region", { name: "Variation setup" })).toBeVisible();
}

export async function setNumericField(
  page: Page,
  label: string,
  value: string,
): Promise<void> {
  const field = page.getByRole("textbox", { name: label });
  await field.fill(value);
  await field.blur();
  await expect(field).toHaveValue(value);
}

export function capturePageErrors(page: Page): Error[] {
  const errors: Error[] = [];
  page.on("pageerror", (error) => errors.push(error));
  return errors;
}
