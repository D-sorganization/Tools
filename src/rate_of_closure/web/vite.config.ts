import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { buildAuthorityProxyConfig } from "./authorityProxyConfig";

const authorityProxy = buildAuthorityProxyConfig({
  ROC_AUTHORITY_URL: process.env.ROC_AUTHORITY_URL,
  ROC_AUTHORITY_TOKEN: process.env.ROC_AUTHORITY_TOKEN,
});

export default defineConfig({
  // Relative base so the built bundle works from any static-host subpath
  // (GitHub Pages project sites included), not just a domain root.
  base: "./",
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
    // Physics optimization and Monte Carlo cases contend under Vitest's
    // parallel pool; retain a bounded but CI-realistic per-test ceiling.
    testTimeout: 15_000,
  },
  server: {
    port: 5193,
    strictPort: true,
    open: false,
    proxy: authorityProxy === undefined
      ? undefined
      : { "/api/rate-of-closure": authorityProxy },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/katex")) return "katex";
          if (id.includes("node_modules/react")) return "react-vendor";
        },
      },
    },
  }
});
