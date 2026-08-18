import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api (and the /api/stream WebSocket) to the FastAPI backend on :8000.
// Shared by the dev server and `vite preview` so the production build served on
// the Pi routes exactly like development.
const apiProxy = {
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
    ws: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    host: true,
    proxy: apiProxy,
  },
  // `vite preview` serves the minified production build (dist/) — far lighter on
  // the Raspberry Pi than the dev server (no HMR/transpile) and this is what the
  // p1am-frontend systemd service runs in production.
  //
  // SECURITY (#4007): host MUST stay loopback. The backend is deliberately bound
  // to 127.0.0.1, but this preview server also proxies /api and the WebSocket to
  // it — so `host: true` here bound every interface and handed the whole plant
  // VLAN an unauthenticated path to E-stop clear, tag writes and the DB wipe
  // (`curl -X POST http://<pi-ip>:3002/api/estop/clear`). The kiosk Chromium runs
  // on the Pi itself, so loopback is all it needs. To reach the HMI from another
  // machine, forward a port over SSH rather than widening this bind.
  preview: {
    port: 3002,
    host: "127.0.0.1",
    proxy: apiProxy,
  },
  build: {
    // Split rarely-changing vendor code into its own cached chunks so a code
    // change doesn't force the Pi's browser to re-download React/zod/icons, and
    // so the initial parse cost is spread across cacheable files.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          zod: ["zod"],
          icons: ["lucide-react"],
        },
      },
    },
    // A slightly higher warning threshold: the app is a single-page HMI, not a
    // latency-critical public site, and the vendor split keeps chunks cacheable.
    chunkSizeWarningLimit: 700,
  },
});
