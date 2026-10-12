import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const port = process.env.VITE_PORT ? parseInt(process.env.VITE_PORT, 10) : 1430;

/** Group node_modules modules into stable vendor chunks so no single built
 * file is oversized and the app code stays cacheable across UI-only changes.
 * The desktop app loads from local disk (Tauri/WebView2), so this is about
 * parse shape, not download size. */
function manualChunks(id: string): string | undefined {
  if (!id.includes("node_modules")) return undefined;
  if (id.includes("/react/") || id.includes("react-dom") || id.includes("scheduler")) {
    return "react";
  }
  if (
    id.includes("react-markdown") ||
    id.includes("remark") ||
    id.includes("micromark") ||
    id.includes("mdast") ||
    id.includes("hast-") ||
    id.includes("unist") ||
    id.includes("vfile") ||
    id.includes("unified") ||
    id.includes("property-information") ||
    id.includes("space-separated-tokens") ||
    id.includes("comma-separated-tokens") ||
    id.includes("decode-named-character-reference") ||
    id.includes("character-entities")
  ) {
    return "markdown";
  }
  if (id.includes("@tauri-apps")) return "tauri";
  return "vendor";
}

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port, strictPort: true },
  build: {
    rollupOptions: {
      output: { manualChunks },
    },
  },
});
