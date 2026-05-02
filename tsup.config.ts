import { defineConfig } from "tsup";

export default defineConfig({
    entry: ["cli.ts"],
    format: ["esm"],
    target: "node20",
    bundle: false,
    clean: true,
    outDir: "dist",
});
