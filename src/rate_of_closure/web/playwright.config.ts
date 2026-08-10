import { defineConfig, devices } from "@playwright/test";

const PORT = 54193;
const browserChannel = process.env.PLAYWRIGHT_CHANNEL;

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "**/*.pw.ts",
  outputDir: "./node_modules/.cache/playwright-results",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(browserChannel ? { channel: browserChannel } : {}),
      },
    },
    {
      name: "chromium-constrained-hidpi",
      use: {
        browserName: "chromium",
        ...(browserChannel ? { channel: browserChannel } : {}),
        viewport: { width: 520, height: 900 },
        deviceScaleFactor: 2,
      },
    },
  ],
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
