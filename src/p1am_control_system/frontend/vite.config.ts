import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Configure Vite to proxy /api backend calls to FastAPI on port 8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    host: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
