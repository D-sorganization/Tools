import process from "node:process";

import { defineConfig, devices } from "@playwright/test";

// Separate from playwright.config.ts on purpose: that one is testDir ./e2e
// with the project matrix landed by #4473, and repointing its testDir at the
// browser qualification suite would silently disable every e2e spec.

const qualificationOutput = process.env.PLAYWRIGHT_JSON_OUTPUT_NAME
  ?? "test-results/browser-qualification.json";

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 8_000 },
  globalSetup: "./tests/browser/support/exactReleaseBuild.ts",
  outputDir: "test-results/artifacts",
  reporter: [["json", { outputFile: qualificationOutput }]],
  use: {
    actionTimeout: 8_000,
    navigationTimeout: 15_000,
    screenshot: "off",
    trace: "off",
    video: "off",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
