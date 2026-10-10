import { defineConfig } from "vitest/config";

// served from https://<user>.github.io/filing-signals/ on GitHub Pages
export default defineConfig({
  base: process.env.GITHUB_PAGES ? "/filing-signals/" : "/",
  test: { environment: "node" },
});
