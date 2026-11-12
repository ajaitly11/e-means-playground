// k-means (Lloyd) and e-means in plain JavaScript.
// e-means follows Algorithm 1 of Bessa et al., "e-means: outperforming k-means
// (in one-tenth the epochs)", arXiv:2611.09501, and the constants of the
// reference code in faiss/contrib/e_means (faiss 1.16.0).
// Both algorithms share the same nearest-centroid kernel, so any speed gap
// comes from the algorithm, not from the implementation.

(function (root) {
  "use strict";

  // Reference constants (faiss/contrib/e_means/base.py).
  const ALPHA_MIN_FACTOR = 0.1;     // cosine floor, epsilon in Eq. 5
  const CHECK_UNUSED_EVERY = 5;     // revival check every 5 updates
  const INIT_USAGE = 2 ** -16;      // initial EMA mass N_j
  const SPLIT_EPSILON = 1e-4;       // relative perturbation when splitting
  const UNUSED_THRESHOLD = 0.1;     // revive N_j < 0.1 * mean(N)
  const NO_REVIVAL_MAX_D = 2;
  const AUTO_QUALITY_MAX_D = 16;
  const BATCH_LAWS = {
    throughput: [0.035134, 0.134749, 0.876473, 0.813344],
    quality: [0.00439175, 0.134749, 0.876473, 0.813344],
  };
  const ALPHA_LAW = [0.8244, -0.4617, 0.9003, -0.4498, -0.1057];

  // Lloyd empty-cluster split, as in faiss Clustering (EPS = 1 / 1024).
  const LLOYD_SPLIT_EPS = 1 / 1024;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function permutation(n, rng) {
    const p = new Int32Array(n);
    for (let i = 0; i < n; i++) p[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = p[i]; p[i] = p[j]; p[j] = t;
    }
    return p;
  }

  // k distinct rows of X, chosen at random. Both algorithms start here.
  function sampleCentroids(X, n, d, k, seed) {
    const rng = mulberry32(seed);
    const pick = new Int32Array(n);
    for (let i = 0; i < n; i++) pick[i] = i;
    const C = new Float32Array(k * d);
    for (let j = 0; j < k; j++) {
      const r = j + Math.floor(rng() * (n - j));
      const t = pick[j]; pick[j] = pick[r]; pick[r] = t;
      C.set(X.subarray(pick[j] * d, pick[j] * d + d), j * d);
    }
    return C;
  }

  function centroidNorms(C, k, d, out) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      const o = j * d;
      for (let t = 0; t < d; t++) s += C[o + t] * C[o + t];
      out[j] = s;
    }
    return out;
  }

  // Nearest centroid for rows idx[from..to) (or rows from..to when idx is null).
  // Writes labels/dists at position (i - from) and returns the summed distance.
  function assign(X, d, idx, from, to, C, cn, k, labels, dists) {
    let total = 0;
    const k4 = k - (k % 4);
    for (let i = from; i < to; i++) {
      const row = idx ? idx[i] : i;
      const xo = row * d;
      let xn = 0;
      for (let t = 0; t < d; t++) xn += X[xo + t] * X[xo + t];
      let best = Infinity, bj = 0;
      let j = 0;
      for (; j < k4; j += 4) {
        const o0 = j * d, o1 = o0 + d, o2 = o1 + d, o3 = o2 + d;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let t = 0; t < d; t++) {
          const v = X[xo + t];
          s0 += v * C[o0 + t]; s1 += v * C[o1 + t];
          s2 += v * C[o2 + t]; s3 += v * C[o3 + t];
        }
        const d0 = cn[j] - 2 * s0, d1 = cn[j + 1] - 2 * s1;
        const d2 = cn[j + 2] - 2 * s2, d3 = cn[j + 3] - 2 * s3;
        if (d0 < best) { best = d0; bj = j; }
        if (d1 < best) { best = d1; bj = j + 1; }
        if (d2 < best) { best = d2; bj = j + 2; }
        if (d3 < best) { best = d3; bj = j + 3; }
      }
      for (; j < k; j++) {
        const o = j * d;
        let s = 0;
        for (let t = 0; t < d; t++) s += X[xo + t] * C[o + t];
        const dd = cn[j] - 2 * s;
        if (dd < best) { best = dd; bj = j; }
      }
      const dist = Math.max(0, best + xn);
      labels[i - from] = bj;
      if (dists) dists[i - from] = dist;
      total += dist;
    }
    return total;
  }

  // Nearest centroid of every row, and the mean squared distance.
  function evaluate(X, n, d, C, k) {
    const cn = centroidNorms(C, k, d, new Float64Array(k));
    const labels = new Int32Array(n);
    const total = assign(X, d, null, 0, n, C, cn, k, labels, null);
    return { labels: labels, mse: total / n };
  }

  function mse(X, n, d, C, k) {
    return evaluate(X, n, d, C, k).mse;
  }

  function now() {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  // ---------------------------------------------------------------- Lloyd

  function lloyd(X, n, d, k, opts) {
    const iters = opts.iters || 25;
    const report = opts.onProgress || function () {};
    const t0 = now();
    const C = sampleCentroids(X, n, d, k, opts.seed);
    const rng = mulberry32(opts.seed ^ 0x9e3779b9);
    const cn = new Float64Array(k);
    const labels = new Int32Array(n);
    const sums = new Float64Array(k * d);
    const counts = new Float64Array(k);
    const history = [];
    for (let it = 0; it < iters; it++) {
      centroidNorms(C, k, d, cn);
      const obj = assign(X, d, null, 0, n, C, cn, k, labels, null);
      sums.fill(0); counts.fill(0);
      for (let i = 0; i < n; i++) {
        const j = labels[i];
        counts[j]++;
        const xo = i * d, so = j * d;
        for (let t = 0; t < d; t++) sums[so + t] += X[xo + t];
      }
      for (let j = 0; j < k; j++) {
        if (counts[j] === 0) continue;
        const so = j * d;
        for (let t = 0; t < d; t++) C[so + t] = sums[so + t] / counts[j];
      }
      splitEmpty(C, counts, n, d, k, rng);
      const point = { pass: it + 1, passes: iters, loss: obj / n, ms: now() - t0 };
      history.push(point);
      report(point, C);
    }
    return { centroids: C, ms: now() - t0, history: history, passes: iters };
  }

  // Give each empty cluster half of a non-empty one, picked in proportion to size.
  function splitEmpty(C, counts, n, d, k, rng) {
    for (let ci = 0; ci < k; ci++) {
      if (counts[ci] !== 0) continue;
      let cj = 0;
      for (;;) {
        const p = (counts[cj] - 1) / (n - k);
        if (rng() < p) break;
        cj = (cj + 1) % k;
      }
      const oi = ci * d, oj = cj * d;
      for (let t = 0; t < d; t++) {
        const v = C[oj + t];
        if (t % 2 === 0) { C[oi + t] = v * (1 + LLOYD_SPLIT_EPS); C[oj + t] = v * (1 - LLOYD_SPLIT_EPS); }
        else { C[oi + t] = v * (1 - LLOYD_SPLIT_EPS); C[oj + t] = v * (1 + LLOYD_SPLIT_EPS); }
      }
      counts[ci] = counts[cj] / 2;
      counts[cj] -= counts[ci];
    }
  }

  // -------------------------------------------------------------- e-means

  function roundPowerOfTwo(v) {
    if (v <= 1) return 1;
    const low = 2 ** Math.floor(Math.log2(v));
    const high = 2 * low;
    return v / low < high / v ? low : high;
  }

  // Batch size, starting step and pass plan from the fitted scaling laws.
  function recipe(n, k, d, nepochs) {
    const law = d <= AUTO_QUALITY_MAX_D ? "quality" : "throughput";
    const [cb, bk, bn, bt] = BATCH_LAWS[law];
    let B = roundPowerOfTwo(cb * k ** bk * n ** bn * nepochs ** bt);
    B = law === "throughput" ? Math.min(B, roundPowerOfTwo(n / 2)) : Math.min(B, Math.floor(n / 2));
    B = Math.max(1, Math.min(B, n));
    const [ca, ak, aB, an, aT] = ALPHA_LAW;
    const alpha0 = Math.min(Math.max(ca * k ** ak * B ** aB * n ** an * nepochs ** aT, 1e-4), 1);
    const whole = Math.floor(nepochs);
    const rem = Math.round((nepochs - whole) * n);
    const epochRows = whole === 0 ? [rem]
      : new Array(whole - 1).fill(n).concat(rem ? [rem] : []).concat([n]);
    let totalSteps = 0;
    for (const r of epochRows) totalSteps += Math.ceil(r / B);
    return {
      batchLaw: law, batchSize: B, alpha0: alpha0, epochRows: epochRows, totalSteps: totalSteps,
      unusedThreshold: d <= NO_REVIVAL_MAX_D ? 0 : UNUSED_THRESHOLD,
    };
  }

  function scheduledAlpha(alpha0, step, total) {
    const f = Math.min(step, total) / Math.max(1, total);
    const cos = 0.5 * (1 + Math.cos(Math.PI * f));
    return alpha0 * (ALPHA_MIN_FACTOR + (1 - ALPHA_MIN_FACTOR) * cos);
  }

  function emeans(X, n, d, k, opts) {
    const nepochs = opts.nepochs || 3;
    const report = opts.onProgress || function () {};
    const every = opts.reportEveryMs || 60;
    const t0 = now();
    const r = recipe(n, k, d, nepochs);
    const C = sampleCentroids(X, n, d, k, opts.seed);
    const usage = new Float64Array(k).fill(INIT_USAGE);
    const cn = new Float64Array(k);
    const B = r.batchSize;
    const labels = new Int32Array(B);
    const bsum = new Float64Array(k * d);
    const bcnt = new Float64Array(k);
    const passSums = new Float64Array(k * d);
    const passCounts = new Float64Array(k);
    const splitRng = mulberry32(opts.seed ^ 0x85ebca6b);
    const history = [];
    let step = 0, nSplits = 0, lastReport = -Infinity;
    const lastEpoch = r.epochRows.length - 1;
    let passesDone = 0;

    for (let e = 0; e <= lastEpoch; e++) {
      const rows = r.epochRows[e];
      const order = permutation(n, mulberry32((opts.seed * 1000003 + e) >>> 0));
      const isLast = e === lastEpoch;
      if (isLast) { passSums.fill(0); passCounts.fill(0); }
      let epochObj = 0, epochSeen = 0;
      for (let from = 0; from < rows; from += B) {
        const to = Math.min(from + B, rows);
        centroidNorms(C, k, d, cn);
        const obj = assign(X, d, order, from, to, C, cn, k, labels, null);
        bsum.fill(0); bcnt.fill(0);
        for (let i = from; i < to; i++) {
          const j = labels[i - from];
          bcnt[j]++;
          const xo = order[i] * d, so = j * d;
          for (let t = 0; t < d; t++) bsum[so + t] += X[xo + t];
        }
        if (isLast) {
          for (let q = 0; q < k * d; q++) passSums[q] += bsum[q];
          for (let j = 0; j < k; j++) passCounts[j] += bcnt[j];
        }
        // Equation 2, written on the centroid: N <- (1-a)N + a n,
        // c <- c + (a n / N_new)(s/n - c). Empty clusters keep their place.
        const alpha = scheduledAlpha(r.alpha0, step, r.totalSteps);
        const decay = 1 - alpha;
        const replace = alpha === 1;
        for (let j = 0; j < k; j++) {
          if (!replace) usage[j] *= decay;
          const cnt = bcnt[j];
          if (cnt === 0) continue;
          const mass = (replace ? 0 : usage[j]) + alpha * cnt;
          const w = (alpha * cnt) / mass;
          const so = j * d;
          for (let t = 0; t < d; t++) {
            const mean = bsum[so + t] / cnt;
            C[so + t] = w === 1 ? mean : C[so + t] + w * (mean - C[so + t]);
          }
          usage[j] = mass;
        }
        if ((step + 1) % CHECK_UNUSED_EVERY === 0 && r.unusedThreshold > 0) {
          nSplits += revive(C, usage, k, d, r.unusedThreshold, splitRng);
        }
        step++;
        epochObj += obj;
        epochSeen += to - from;
        const t = now();
        if (t - lastReport >= every) {
          lastReport = t;
          const point = { pass: passesDone + epochSeen / n, passes: nepochs, loss: epochObj / epochSeen, ms: t - t0 };
          history.push(point);
          report(point, C);
        }
      }
      passesDone += rows / n;
    }
    // Last-epoch-mean finalization (Algorithm 1, line 11).
    for (let j = 0; j < k; j++) {
      if (passCounts[j] <= 0) continue;
      const so = j * d;
      for (let t = 0; t < d; t++) C[so + t] = passSums[so + t] / passCounts[j];
    }
    const ms = now() - t0;
    const point = { pass: nepochs, passes: nepochs, loss: history.length ? history[history.length - 1].loss : NaN, ms: ms };
    history.push(point);
    report(point, C);
    return { centroids: C, ms: ms, history: history, passes: nepochs, recipe: r, splits: nSplits };
  }

  // Revive clusters whose EMA mass fell below tau * mean mass by splitting donors
  // drawn without replacement in proportion to their mass (Gumbel top-k).
  function revive(C, usage, k, d, tau, rng) {
    let total = 0;
    for (let j = 0; j < k; j++) total += usage[j];
    const threshold = (total * tau) / k;
    const unused = [], keys = [];
    for (let j = 0; j < k; j++) {
      if (usage[j] < threshold) unused.push(j);
      else if (usage[j] > 0) keys.push([Math.log(usage[j]) - Math.log(-Math.log(Math.max(rng(), Number.MIN_VALUE))), j]);
    }
    const nsplit = Math.min(unused.length, keys.length);
    if (nsplit === 0) return 0;
    keys.sort((a, b) => b[0] - a[0]);
    for (let s = 0; s < nsplit; s++) {
      const src = keys[s][1], dst = unused[s];
      const os = src * d, od = dst * d;
      for (let t = 0; t < d; t++) {
        const off = SPLIT_EPSILON * (t % 2 === 0 ? 1 : -1);
        const v = C[os + t];
        C[os + t] = v * (1 - off);
        C[od + t] = v * (1 + off);
      }
      if (C[os] === C[od]) C[od] = C[od] + Math.max(Math.abs(C[od]) * 1e-7, 1e-30);
      const half = usage[src] / 2;
      usage[src] = half;
      usage[dst] = half;
    }
    return nsplit;
  }

  const api = { lloyd, emeans, mse, evaluate, recipe, sampleCentroids };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof self !== "undefined" ? self : this);
