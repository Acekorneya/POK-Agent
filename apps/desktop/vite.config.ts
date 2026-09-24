import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const port = process.env.VITE_PORT ? parseInt(process.env.VITE_PORT, 10) : 1430;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port, strictPort: true },
});

