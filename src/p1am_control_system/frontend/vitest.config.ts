import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Vitest configuration for the P1AM control-system frontend.
// jsdom gives React Testing Library a DOM; setup file wires
// @testing-library/jest-dom matchers and cleanup.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/test/**", "src/**/*.{test,spec}.{ts,tsx}", "src/main.tsx"],
    },
  },
});
