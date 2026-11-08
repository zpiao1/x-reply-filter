import { copyFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distDir = path.join(rootDir, "dist");
const publicDir = path.join(rootDir, "public");
const transformersDistDir = path.join(
  rootDir,
  "node_modules",
  "@huggingface",
  "transformers",
  "dist",
);

await rm(distDir, { recursive: true, force: true });
await mkdir(distDir, { recursive: true });

await Promise.all([
  build({
    entryPoints: [path.join(rootDir, "src/content/index.ts")],
    outfile: path.join(distDir, "content.js"),
    bundle: true,
    format: "iife",
    target: "es2022",
    platform: "browser",
  }),
  build({
    entryPoints: [path.join(rootDir, "src/background/index.ts")],
    outfile: path.join(distDir, "background.js"),
    bundle: true,
    format: "esm",
    target: "es2022",
    platform: "browser",
  }),
  build({
    entryPoints: [path.join(rootDir, "src/offscreen/index.ts")],
    outfile: path.join(distDir, "offscreen.js"),
    bundle: true,
    format: "esm",
    target: "es2022",
    platform: "browser",
  }),
  copyFile(path.join(publicDir, "manifest.json"), path.join(distDir, "manifest.json")),
  copyFile(path.join(publicDir, "offscreen.html"), path.join(distDir, "offscreen.html")),
  copyFile(
    path.join(transformersDistDir, "ort-wasm-simd-threaded.jsep.mjs"),
    path.join(distDir, "ort-wasm-simd-threaded.jsep.mjs"),
  ),
  copyFile(
    path.join(transformersDistDir, "ort-wasm-simd-threaded.jsep.wasm"),
    path.join(distDir, "ort-wasm-simd-threaded.jsep.wasm"),
  ),
]);
