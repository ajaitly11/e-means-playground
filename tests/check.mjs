// Runs both engines on the digits data in Node and prints time and final error.
// Usage: node tests/check.mjs [n] [k] [epochs]
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createRequire } from "node:module";
const Engine = createRequire(import.meta.url)("../engine.js");

const n = +(process.argv[2] || 60000), k = +(process.argv[3] || 100), T = +(process.argv[4] || 3), d = 196;
const u8 = gunzipSync(readFileSync(new URL("../data/mnist14.bin.gz", import.meta.url)));
const X = new Float32Array(n * d);
for (let i = 0; i < n * d; i++) X[i] = u8[i];

for (const seed of [1, 2, 3]) {
  const L = Engine.lloyd(X, n, d, k, { seed, iters: 25 });
  const E = Engine.emeans(X, n, d, k, { seed, nepochs: T });
  const ml = Engine.mse(X, n, d, L.centroids, k), me = Engine.mse(X, n, d, E.centroids, k);
  console.log(`seed ${seed} n=${n} k=${k} | k-means ${(L.ms / 1000).toFixed(2)}s mse ${ml.toFixed(1)} | e-means(${T}) ${(E.ms / 1000).toFixed(2)}s mse ${me.toFixed(1)} B=${E.recipe.batchSize} a0=${E.recipe.alpha0.toFixed(3)} splits=${E.splits} | ${(L.ms / E.ms).toFixed(1)}x, ${(100 * (me / ml - 1)).toFixed(2)}%`);
}
