// Runs one algorithm off the main thread and streams progress back.
importScripts("engine.js");

self.onmessage = function (event) {
  const m = event.data;
  const X = m.data instanceof Float32Array ? m.data : Float32Array.from(m.data);
  const opts = {
    seed: m.seed,
    iters: 25,
    nepochs: m.epochs,
    onProgress: function (point, C) {
      self.postMessage({ type: "progress", point: point, centroids: m.sendCentroids ? C.slice() : null });
    },
  };
  const run = m.algo === "kmeans" ? Engine.lloyd : Engine.emeans;
  const result = run(X, m.n, m.d, m.k, opts);
  // Scored after the clock stops, on every point, the same way for both.
  const score = Engine.evaluate(X, m.n, m.d, result.centroids, m.k);
  self.postMessage({
    type: "done",
    ms: result.ms,
    passes: result.passes,
    mse: score.mse,
    centroids: result.centroids,
    labels: m.wantLabels ? score.labels : null,
    recipe: result.recipe || null,
  });
};
