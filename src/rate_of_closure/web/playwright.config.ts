import { defineConfig, devices } from "@playwright/test";

const previewUrl = "http://127.0.0.1:4173";
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
  outputDir: "test-results",
  preserveOutput: "always",
  reporter: process.env.CI
    ? [["line"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
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
    command: "npm run build && npm run preview -- --host 127.0.0.1 --port 4173 --strictPort",
    url: previewUrl,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
