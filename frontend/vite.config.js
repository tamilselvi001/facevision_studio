import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app talks to the FastAPI server through this proxy, so the
// frontend can always call relative "/api/..." paths and never needs to know a
// host. In production the same relative paths are served from one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
});
