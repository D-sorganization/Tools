import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Relative base so the built bundle works from any static-host subpath
  // (GitHub Pages project sites included), not just a domain root.
  base: "./",
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
  server: {
    port: 5193,
    strictPort: true,
    open: false
  },
  build: {
    outDir: "dist",
    sourcemap: true
  }
});
