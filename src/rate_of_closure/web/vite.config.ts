import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { configDefaults } from "vitest/config";

import { morrisAuthorityProxy } from "./morrisAuthorityProxy";

export default defineConfig(({ mode }) => {
  const proxy = morrisAuthorityProxy(loadEnv(mode, process.cwd(), ""));
  return {
  // Relative base so the built bundle works from any static-host subpath
  // (GitHub Pages project sites included), not just a domain root.
  base: "./",
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
    exclude: [...configDefaults.exclude, "e2e/**", "tests/**"],
    // Physics optimization and Monte Carlo cases contend under Vitest's
    // parallel pool; retain a bounded but CI-realistic per-test ceiling.
    testTimeout: 15_000,
  },
  server: {
    port: 5193,
    strictPort: true,
    open: false,
    proxy,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/katex")) return "katex";
          if (id.includes("node_modules/react")) return "react-vendor";
        },
      },
    },
  },
  };
});
