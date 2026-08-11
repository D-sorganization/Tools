import { defineConfig, devices } from "@playwright/test";

const APP_PORT = 5194;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "test-results",
  timeout: 60_000,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["line"], ["html", { outputFolder: "playwright-report", open: "never" }]]
    : "line",
  use: {
    baseURL: `http://127.0.0.1:${APP_PORT}`,
    viewport: { width: 1600, height: 1200 },
    colorScheme: "dark",
    locale: "en-US",
    timezoneId: "UTC",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1600, height: 1200 },
      },
    },
  ],
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${APP_PORT}`,
    url: `http://127.0.0.1:${APP_PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
