import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
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
