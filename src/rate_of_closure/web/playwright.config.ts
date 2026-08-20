import { defineConfig, devices } from "@playwright/test";

const previewPortText = process.env.RATE_E2E_PORT ?? "4173";
const previewPort = Number(previewPortText);
if (!/^\d+$/.test(previewPortText) || previewPort < 1 || previewPort > 65_535) {
  throw new Error("RATE_E2E_PORT must be an integer from 1 through 65535");
}
const previewUrl = `http://127.0.0.1:${previewPort}`;
const previewCommand = process.env.RATE_E2E_PREBUILT === "1"
  ? `npm run preview -- --host 127.0.0.1 --port ${previewPort} --strictPort`
  : `npm run build && npm run preview -- --host 127.0.0.1 --port ${previewPort} --strictPort`;
const evidencePhase = process.env.RATE_E2E_EVIDENCE_PHASE;
if (evidencePhase !== undefined && !/^[a-z0-9-]+$/.test(evidencePhase)) {
  throw new Error("RATE_E2E_EVIDENCE_PHASE must use lowercase letters, digits, or hyphens");
}
const evidencePath = (root: string): string => evidencePhase === undefined
  ? root
  : `${root}/${evidencePhase}`;
const chromiumArgs = [
  "--disable-background-networking",
  "--disable-component-update",
  "--disable-default-apps",
  "--disable-features=MediaRouter,Translate",
  "--force-color-profile=srgb",
];

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  outputDir: evidencePath("test-results"),
  preserveOutput: "always",
  reporter: process.env.CI
    ? [["line"], ["html", { open: "never", outputFolder: evidencePath("playwright-report") }]]
    : [["list"], ["html", { open: "never", outputFolder: evidencePath("playwright-report") }]],
  use: {
    baseURL: previewUrl,
    colorScheme: "dark",
    deviceScaleFactor: 1,
    headless: true,
    locale: "en-US",
    reducedMotion: "reduce",
    serviceWorkers: "block",
    timezoneId: "UTC",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 1000 },
        launchOptions: { args: chromiumArgs },
      },
    },
    {
      name: "chromium-narrow",
      testMatch: /variation-(layout|visual-state)\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
        launchOptions: { args: chromiumArgs },
      },
    },
    {
      name: "firefox-desktop",
      testMatch: /variation-crossbrowser\.spec\.ts/,
      use: { ...devices["Desktop Firefox"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "webkit-desktop",
      testMatch: /variation-crossbrowser\.spec\.ts/,
      use: { ...devices["Desktop Safari"], viewport: { width: 1440, height: 1000 } },
    },
  ],
  webServer: {
    command: previewCommand,
    url: previewUrl,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
