"use strict";

const DIGIT_D = 196;   // 14 x 14 pixels
const EPOCHS = 3;      // e-means reads of the data (the paper reports 1.4 to 2.6 to match k-means)
const LLOYD_ITERS = 25;
const INK = "#3f3f46", CORAL = "#e94547", FAINT = "#a1a1aa", LINE = "#e4e4e7";
const $ = (id) => document.getElementById(id);

let busy = null; // { owner, stop() } while an experiment runs

// ------------------------------------------------------------ helpers

const fmt = (n) => n.toLocaleString("en-US");
const secs = (ms) => (ms < 10000 ? (ms / 1000).toFixed(2) : (ms / 1000).toFixed(1)) + "s";
const newSeed = () => Math.floor(Math.random() * 2 ** 31);
const easeOut = (p) => 1 - Math.pow(1 - p, 3);

function countUp(el, to, decimals, ms) {
  const start = performance.now();
  (function tick(now) {
    const p = Math.min(1, (now - start) / (ms || 1200));
    el.textContent = (to * easeOut(p)).toFixed(decimals);
    if (p < 1) requestAnimationFrame(tick);
  })(start);
}

function setupSeg(name, initial, onChange) {
  const seg = document.querySelector(`.seg[data-name="${name}"]`);
  const state = { value: String(initial) };
  const mark = (v) => seg.querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o.dataset.v === v)));
  mark(state.value);
  seg.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      if (busy) return;
      if (onChange && onChange(b.dataset.v) === false) return;
      state.value = b.dataset.v;
      mark(state.value);
    };
  });
  state.set = (v) => { state.value = String(v); mark(state.value); };
  return state;
}

function setRunning(btn, running, idle) {
  btn.firstElementChild.textContent = running ? "Stop" : idle;
  btn.classList.toggle("stop", running);
}

function showResult(id, on) {
  $(id).classList.toggle("show", on);
  if (!on) return;
  // Once the card has opened, glide down just enough to show it, so the
  // run and its result fit in one screen recording.
  const card = $(id);
  let done = false;
  const reveal = (e) => {
    if (done || (e && e.propertyName !== "max-height")) return;
    done = true;
    card.removeEventListener("transitionend", reveal);
    const over = card.getBoundingClientRect().bottom - innerHeight + 20;
    if (over > 0) scrollBy({ top: over, behavior: "smooth" });
  };
  card.addEventListener("transitionend", reveal);
  setTimeout(reveal, 900); // in case the transition never reports back
}

// The digits ship as 4-bit pixels, two per byte (2.3 MB gzipped). The download
// starts when the page opens, so it is usually done before anyone clicks run.
let digitsPromise = null;
const digitsLoad = { got: 0, total: 2321489, listeners: new Set() };
function loadDigits() {
  if (!digitsPromise) {
    digitsPromise = fetch("data/mnist14-4bit.bin.gz").then(async (r) => {
      if (!r.ok) throw new Error(`digits download failed (${r.status})`);
      const total = +r.headers.get("content-length");
      if (total) digitsLoad.total = total;
      const reader = r.body.getReader();
      const parts = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        digitsLoad.got += value.length;
        digitsLoad.listeners.forEach((f) => f());
      }
      let buf = new Uint8Array(await new Blob(parts).arrayBuffer());
      if (buf[0] === 0x1f && buf[1] === 0x8b) {
        const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
        buf = new Uint8Array(await new Response(stream).arrayBuffer());
      } // otherwise the server already decompressed it
      const out = new Uint8Array(buf.length * 2);
      for (let i = 0; i < buf.length; i++) {
        out[2 * i] = (buf[i] >> 4) * 17;
        out[2 * i + 1] = (buf[i] & 15) * 17;
      }
      return out;
    }).catch((err) => { digitsPromise = null; digitsLoad.got = 0; throw err; });
  }
  return digitsPromise;
}

// Shows download progress on a run button until the digits are in.
async function waitForDigits(btn) {
  const label = btn.firstElementChild;
  const show = () => {
    const pct = Math.min(99, Math.floor((100 * digitsLoad.got) / digitsLoad.total));
    label.textContent = `Downloading digits ${pct}%`;
  };
  show();
  digitsLoad.listeners.add(show);
  try {
    return await loadDigits();
  } catch (err) {
    label.textContent = "Download failed, try again";
    throw err;
  } finally {
    digitsLoad.listeners.delete(show);
  }
}

// Runs k-means and e-means at the same time, one worker each.
function runPair(job, onMessage) {
  const workers = [];
  const done = {};
  let fail;
  const promise = new Promise((resolve, reject) => {
    fail = reject;
    for (const algo of ["kmeans", "emeans"]) {
      const w = new Worker("worker.js");
      workers.push(w);
      w.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "done") { done[algo] = m; w.terminate(); }
        onMessage(algo, m);
        if (done.kmeans && done.emeans) resolve(done);
      };
      w.onerror = (e) => reject(e);
      w.postMessage(Object.assign({ algo }, job));
    }
  });
  promise.stop = () => {
    workers.forEach((w) => w.terminate());
    fail(new Error("stopped"));
  };
  return promise;
}

function sizeCanvas(c, h) {
  const r = devicePixelRatio || 1;
  const w = c.clientWidth;
  h = h || c.clientHeight;
  if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) {
    c.width = Math.round(w * r);
    c.height = Math.round(h * r);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return { ctx, w, h };
}

// ------------------------------------------------------- page motion

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    e.target.querySelectorAll(".ticks").forEach((t) => t.classList.add("shown"));
    io.unobserve(e.target);
  }
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el, i) => {
  el.style.transitionDelay = el.closest("header") ? `${i * 0.08}s` : "0s";
  io.observe(el);
});
document.querySelectorAll(".ticks").forEach((t) => {
  const n = +t.dataset.count;
  t.innerHTML = "<i></i>".repeat(n);
  t.querySelectorAll("i").forEach((i, j) => (i.style.transitionDelay = `${0.35 + j * 0.03}s`));
});

// ------------------------------------------------- 01: digits race

const raceN = setupSeg("n", 60000, (v) => { $("race-n-title").textContent = fmt(+v); });
const raceK = setupSeg("k", 100);
$("race-n-title").textContent = fmt(+raceN.value);

function buildReads(id, count) {
  $(id).innerHTML = "<i><b></b></i>".repeat(count);
}
function fillReads(id, pass) {
  const bars = $(id).querySelectorAll("b");
  bars.forEach((b, i) => (b.style.width = Math.max(0, Math.min(1, pass - i)) * 100 + "%"));
  $(id + "-l").textContent = `${Math.min(Math.floor(pass + 1e-9), bars.length)} of ${bars.length} reads done`;
}

function drawDigits(canvas, C, k) {
  const cols = Math.ceil(Math.sqrt(k));
  const side = cols * 14 + (cols - 1) * 2;
  const off = drawDigits.off || (drawDigits.off = document.createElement("canvas"));
  off.width = off.height = side;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(side, side);
  const paper = [250, 250, 250], ink = [10, 10, 11];
  for (let p = 0; p < side * side; p++) {
    img.data[p * 4] = paper[0]; img.data[p * 4 + 1] = paper[1]; img.data[p * 4 + 2] = paper[2]; img.data[p * 4 + 3] = 255;
  }
  for (let j = 0; j < k; j++) {
    const ox = (j % cols) * 16, oy = Math.floor(j / cols) * 16;
    for (let y = 0; y < 14; y++) {
      for (let x = 0; x < 14; x++) {
        const v = Math.max(0, Math.min(255, C[j * DIGIT_D + y * 14 + x])) / 255;
        const p = ((oy + y) * side + ox + x) * 4;
        for (let ch = 0; ch < 3; ch++) img.data[p + ch] = paper[ch] + (ink[ch] - paper[ch]) * v;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  const { ctx: out, w } = sizeCanvas(canvas, canvas.clientWidth);
  out.imageSmoothingEnabled = false;
  out.clearRect(0, 0, w, w);
  out.drawImage(off, 0, 0, w, w);
}

const race = { hist: { kmeans: [], emeans: [] }, done: {}, elapsed: 0 };

function drawRaceChart() {
  const canvas = $("race-chart");
  const { ctx, w, h } = sizeCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const all = race.hist.kmeans.concat(race.hist.emeans);
  const pad = { l: 6, r: 70, t: 26, b: 22 };
  ctx.strokeStyle = LINE; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r + 60, h - pad.b + 0.5); ctx.stroke();
  ctx.font = "500 11px 'DM Mono', monospace";
  ctx.fillStyle = FAINT; ctx.textAlign = "left";
  ctx.fillText("0s", pad.l, h - 6);
  if (all.length < 2) return;
  const xMax = Math.max(race.elapsed, ...all.map((p) => p.ms), 1);
  const yMin = Math.min(...all.map((p) => p.loss));
  const later = race.hist.kmeans.slice(1).concat(race.hist.emeans.slice(2));
  const yMax = Math.max(yMin * 1.03, ...later.map((p) => p.loss));
  const X = (ms) => pad.l + (ms / xMax) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (Math.min(v, yMax) - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);
  if (!race.done.kmeans) { ctx.textAlign = "right"; ctx.fillText(secs(xMax), w - pad.r, h - 6); }

  // Where e-means crossed the line: the moment to point at in the video.
  const eDone = race.done.emeans;
  if (eDone) {
    const x = X(eDone.ms);
    ctx.save();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = CORAL; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(x, pad.t - 14); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = CORAL; ctx.textAlign = x > w / 2 ? "right" : "left";
    ctx.font = "800 13px Archivo, sans-serif";
    ctx.fillText(`e-means done · ${secs(eDone.ms)}`, x + (x > w / 2 ? -6 : 6), pad.t - 10);
  }
  for (const [algo, color] of [["kmeans", INK], ["emeans", CORAL]]) {
    const pts = race.hist[algo];
    if (!pts.length) continue;
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.ms), Y(p.loss)) : ctx.moveTo(X(p.ms), Y(p.loss))));
    ctx.stroke();
    const last = pts[pts.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(X(last.ms), Y(last.loss), 4, 0, 7); ctx.fill();
    if (race.done[algo]) {
      ctx.font = "800 15px Archivo, sans-serif"; ctx.textAlign = "left";
      ctx.fillText(secs(race.done[algo].ms), Math.min(X(last.ms) + 8, w - 64), Y(last.loss) + (algo === "kmeans" ? -8 : 18));
    }
  }
}

function resetRace(k) {
  race.hist = { kmeans: [], emeans: [] };
  race.done = {};
  race.elapsed = 0;
  for (const a of ["k", "e"]) {
    $(`r${a}-clock`).innerHTML = "0.0<small>s</small>";
    $(`r${a}-live`).className = "live";
    const g = $(`r${a}-grid`);
    const { ctx, w } = sizeCanvas(g, g.clientWidth);
    ctx.clearRect(0, 0, w, w);
    g.closest(".lane").classList.remove("won");
  }
  buildReads("rk-reads", LLOYD_ITERS);
  buildReads("re-reads", EPOCHS);
  fillReads("rk-reads", 0);
  fillReads("re-reads", 0);
  showResult("race-result", false);
  drawRaceChart();
}

function setClock(id, ms) {
  $(id).innerHTML = (ms < 10000 ? (ms / 1000).toFixed(1) : Math.floor(ms / 1000)) + "<small>s</small>";
}

function raceNote(dk, de) {
  const x = (dk.ms / de.ms).toFixed(1);
  const diff = (de.mse / dk.mse - 1) * 100;
  if (dk.ms <= de.ms) return `<b>This time k-means finished first.</b> That can happen on very small runs or slow devices. Try more digits.`;
  const reads = `after ${EPOCHS} reads of the data instead of ${LLOYD_ITERS}`;
  if (diff > 0.1) {
    return `<b>e-means finished ${x}× sooner,</b> ${reads}. Its groups were ${diff.toFixed(1)}% looser this run, close to but not quite Meta's claim of equal quality. Run it again: each run starts from new random centres.`;
  }
  const groups = diff < -0.1 ? `groups that are ${Math.abs(diff).toFixed(1)}% tighter` : "groups that are just as good";
  return `<b>Meta's claim holds on your device.</b> e-means made ${groups} ${reads}, so it finished ${x}× sooner.`;
}

function qualityText(dk, de) {
  const diff = (de.mse / dk.mse - 1) * 100;
  if (Math.abs(diff) < 0.1) return "Same";
  return diff < 0 ? `${Math.abs(diff).toFixed(1)}% better` : `${diff.toFixed(1)}% worse`;
}

async function runRace() {
  const btn = $("race-run");
  if (busy && busy.owner === "race") { busy.stop(); return; }
  if (busy) return;
  const n = +raceN.value, k = +raceK.value;
  busy = { owner: "race", stop() {} };
  let all;
  try { all = await waitForDigits(btn); } catch { busy = null; return; }
  resetRace(k);
  const started = performance.now();
  let raf = 0;
  const tick = () => {
    race.elapsed = performance.now() - started;
    if (!race.done.kmeans) setClock("rk-clock", race.elapsed);
    if (!race.done.emeans) setClock("re-clock", race.elapsed);
    drawRaceChart();
    raf = requestAnimationFrame(tick);
  };
  $("rk-live").className = $("re-live").className = "live on";
  const pair = runPair({ data: all.subarray(0, n * DIGIT_D), n, d: DIGIT_D, k, epochs: EPOCHS, seed: newSeed(), sendCentroids: true }, (algo, m) => {
    const a = algo === "kmeans" ? "k" : "e";
    if (m.type === "progress") {
      race.hist[algo].push(m.point);
      fillReads(`r${a}-reads`, m.point.pass);
      if (m.centroids) drawDigits($(`r${a}-grid`), m.centroids, k);
    } else {
      race.done[algo] = m;
      setClock(`r${a}-clock`, m.ms);
      $(`r${a}-clock`).innerHTML = secs(m.ms).replace("s", "<small>s</small>");
      $(`r${a}-live`).className = "live done";
      $(`r${a}-live`).textContent = "Done";
      fillReads(`r${a}-reads`, m.passes);
      drawDigits($(`r${a}-grid`), m.centroids, k);
    }
  });
  $("rk-live").textContent = $("re-live").textContent = "Running";
  busy = { owner: "race", stop: () => pair.stop() };
  setRunning(btn, true);
  raf = requestAnimationFrame(tick);
  try {
    const res = await pair;
    cancelAnimationFrame(raf);
    race.elapsed = Math.max(res.kmeans.ms, res.emeans.ms);
    drawRaceChart();
    const faster = res.emeans.ms < res.kmeans.ms ? "e" : "k";
    $(`r${faster}-grid`).closest(".lane").classList.add("won");
    $("race-tk").textContent = secs(res.kmeans.ms);
    $("race-te").textContent = secs(res.emeans.ms);
    $("race-q").textContent = qualityText(res.kmeans, res.emeans);
    $("race-note").innerHTML = raceNote(res.kmeans, res.emeans);
    showResult("race-result", true);
    countUp($("race-x"), res.kmeans.ms / res.emeans.ms, 1);
  } catch (e) {
    cancelAnimationFrame(raf);
    $("rk-live").className = $("re-live").className = "live";
  } finally {
    busy = null;
    setRunning(btn, false, "Run again");
  }
}
$("race-run").onclick = runRace;
buildReads("rk-reads", LLOYD_ITERS);
buildReads("re-reads", EPOCHS);
fillReads("rk-reads", 0);
fillReads("re-reads", 0);
addEventListener("resize", drawRaceChart);

// ---------------------------------------------- 02: photo colours

const photoK = setupSeg("c", 16, (v) => { $("photo-k-title").textContent = v; });
let uploadedUrl = null;
const photoImg = setupSeg("img", "data/china.jpg", (v) => {
  if (v === "upload" && !uploadedUrl) { $("upload").click(); return false; }
  loadPhoto(v === "upload" ? uploadedUrl : v);
});
$("upload-btn").onclick = () => { if (!busy) $("upload").click(); };
let pixels = null; // { data: Float32Array, w, h }

function loadPhoto(src) {
  const img = new Image();
  img.onload = () => {
    const scale = Math.min(1, 800 / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
    for (const id of ["ph-orig", "ph-k", "ph-e"]) {
      const c = $(id);
      c.width = w; c.height = h;
      c.getContext("2d").drawImage(img, 0, 0, w, h);
      if (id !== "ph-orig") c.classList.add("waiting");
    }
    const rgba = $("ph-orig").getContext("2d").getImageData(0, 0, w, h).data;
    const data = new Float32Array(w * h * 3);
    for (let i = 0, p = 0; i < w * h; i++, p += 4) {
      data[i * 3] = rgba[p]; data[i * 3 + 1] = rgba[p + 1]; data[i * 3 + 2] = rgba[p + 2];
    }
    pixels = { data, w, h };
    $("ph-count").textContent = `Original · ${fmt(w * h)} px`;
    $("ph-kt").textContent = $("ph-et").textContent = "";
    $("ph-ks").innerHTML = $("ph-es").innerHTML = "";
    showResult("photo-result", false);
  };
  img.src = src;
}

function paint(canvas, labels, C, w, h) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const j = labels[i] * 3, p = i * 4;
    img.data[p] = C[j]; img.data[p + 1] = C[j + 1]; img.data[p + 2] = C[j + 2]; img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  canvas.classList.remove("waiting");
}

function swatches(id, C, k) {
  const cols = [];
  for (let j = 0; j < k; j++) cols.push([C[j * 3], C[j * 3 + 1], C[j * 3 + 2]]);
  cols.sort((a, b) => (0.3 * a[0] + 0.59 * a[1] + 0.11 * a[2]) - (0.3 * b[0] + 0.59 * b[1] + 0.11 * b[2]));
  const el = $(id);
  el.classList.remove("shown");
  el.innerHTML = cols.map((c, i) => `<i style="background:rgb(${c.map(Math.round)});transition-delay:${(i * 0.4 / k).toFixed(3)}s"></i>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("shown")));
}

$("upload").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
  uploadedUrl = URL.createObjectURL(f);
  e.target.value = ""; // lets the same file be picked again
  photoImg.set("upload");
  loadPhoto(uploadedUrl);
};

$("photo-run").onclick = async () => {
  if (busy || !pixels) return;
  const btn = $("photo-run");
  const k = +photoK.value;
  const { data, w, h } = pixels;
  $("ph-k").classList.add("waiting"); $("ph-e").classList.add("waiting");
  $("ph-kt").textContent = $("ph-et").textContent = "…";
  showResult("photo-result", false);
  btn.firstElementChild.textContent = "Running…";
  const pair = runPair({ data, n: w * h, d: 3, k, epochs: EPOCHS, seed: newSeed(), wantLabels: true }, (algo, m) => {
    if (m.type !== "done") return;
    const a = algo === "kmeans" ? "k" : "e";
    paint($(`ph-${a}`), m.labels, m.centroids, w, h);
    swatches(`ph-${a}s`, m.centroids, k);
    $(`ph-${a}t`).textContent = secs(m.ms);
  });
  busy = { owner: "photo", stop: () => pair.stop() };
  try {
    const res = await pair;
    const diff = (res.emeans.mse / res.kmeans.mse - 1) * 100;
    $("photo-tk").textContent = secs(res.kmeans.ms);
    $("photo-te").textContent = secs(res.emeans.ms);
    $("photo-q").textContent = Math.abs(diff) < 1 ? "Same" : `${Math.abs(diff).toFixed(1)}% ${diff > 0 ? "higher" : "lower"}`;
    const quick = Math.max(res.kmeans.ms, res.emeans.ms) < 1000;
    const colour = Math.abs(diff) < 1 ? "Both picked almost the same colours."
      : diff > 0 ? `e-means' colours were ${diff.toFixed(1)}% further from the original. That fits Meta's own note that e-means is tuned for large, complex data, not three-number pixels.`
      : `e-means' colours were even ${Math.abs(diff).toFixed(1)}% closer to the original.`;
    $("photo-note").innerHTML = quick
      ? `<b>Both finished in under a second,</b> so on a small, simple job like this you would never notice the speed-up. ${colour}`
      : `<b>e-means finished ${(res.kmeans.ms / res.emeans.ms).toFixed(1)}× sooner.</b> ${colour}`;
    showResult("photo-result", true);
    countUp($("photo-x"), res.kmeans.ms / res.emeans.ms, 1);
  } finally {
    busy = null;
    btn.firstElementChild.textContent = "Run again";
  }
};
loadPhoto("data/china.jpg");
loadDigits().catch(() => {}); // start early; a failure retries on the next click

// --------------------------------------------- 03: does size matter

const SIZES = [1000, 10000, 60000]; // the same for every visitor
const SCALE_K = 49;

function buildSizes() {
  $("scale-rows").innerHTML = SIZES.map((n, i) => `
    <div class="size">
      <div class="n">${n >= 1000 ? n / 1000 + "k" : n}<small>digits</small></div>
      <div class="bar k"><b id="s${i}k"></b><span id="s${i}kt"></span></div>
      <div class="x" id="s${i}x"></div>
      <div class="bar e"><b id="s${i}e"></b><span id="s${i}et"></span></div>
    </div>`).join("");
}
buildSizes();

$("scale-run").onclick = async () => {
  const btn = $("scale-run");
  if (busy && busy.owner === "scale") { busy.stop(); return; }
  if (busy) return;
  busy = { owner: "scale", stop() {} };
  buildSizes();
  showResult("scale-result", false);
  let all;
  try { all = await waitForDigits(btn); } catch { busy = null; return; }
  const times = [];
  let stopped = false, current = null, raf = 0;
  busy = { owner: "scale", stop: () => { stopped = true; if (current) current.stop(); } };
  setRunning(btn, true);
  const redraw = () => {
    const live = (t) => (t.started != null ? performance.now() - t.started : 0);
    const max = Math.max(1, ...times.flatMap((t) => [t.k != null ? t.k : live(t), t.e != null ? t.e : live(t)]));
    times.forEach((t, i) => {
      for (const a of ["k", "e"]) {
        const v = t[a] != null ? t[a] : live(t);
        const pct = (100 * v) / max;
        $(`s${i}${a}`).style.width = pct + "%";
        const label = $(`s${i}${a}t`);
        label.textContent = t[a] != null ? secs(t[a]) : (v / 1000).toFixed(1) + "s";
        // Sit just past the end of the bar, or inside it once the bar is long.
        const inside = pct > 82;
        label.classList.toggle("in", inside);
        label.style.left = inside ? "auto" : `calc(${pct}% + 8px)`;
        label.style.right = inside ? "10px" : "auto";
      }
      if (t.k != null && t.e != null) {
        $(`s${i}x`).textContent = `${(t.k / t.e).toFixed(1)}×`;
        $(`s${i}x`).classList.add("show");
      }
    });
  };
  try {
    for (let i = 0; i < SIZES.length && !stopped; i++) {
      const n = SIZES[i];
      const t = { started: performance.now() };
      times.push(t);
      const loop = () => { redraw(); raf = requestAnimationFrame(loop); };
      raf = requestAnimationFrame(loop);
      current = runPair({ data: all.subarray(0, n * DIGIT_D), n, d: DIGIT_D, k: SCALE_K, epochs: EPOCHS, seed: newSeed() }, (algo, m) => {
        if (m.type === "done") t[algo === "kmeans" ? "k" : "e"] = m.ms;
      });
      const res = await current;
      cancelAnimationFrame(raf);
      t.k = res.kmeans.ms; t.e = res.emeans.ms;
      redraw();
    }
    const first = times[0], last = times[times.length - 1];
    $("scale-small").textContent = secs(first.k - first.e);
    $("scale-small-l").textContent = `saved on ${fmt(SIZES[0])} digits`;
    $("scale-unit").textContent = `seconds saved on ${fmt(SIZES[SIZES.length - 1])} digits`;
    const xs = times.map((t) => t.k / t.e);
    $("scale-note").innerHTML = `<b>e-means was ${Math.min(...xs).toFixed(1)}× to ${Math.max(...xs).toFixed(1)}× faster at every size,</b> but the seconds saved grew with the data: ${secs(first.k - first.e)} on ${fmt(SIZES[0])} digits, ${secs(last.k - last.e)} on ${fmt(SIZES[SIZES.length - 1])}. On small data you won't notice. On large data, like the billions of points Meta works with, the same speed-up is a big saving in computing time and cost.`;
    showResult("scale-result", true);
    countUp($("scale-saved"), (last.k - last.e) / 1000, 1);
  } catch (e) {
    /* stopped */
  } finally {
    cancelAnimationFrame(raf);
    busy = null;
    setRunning(btn, false, "Run again");
  }
};
