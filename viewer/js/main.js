// Main orchestrator. Commit 3: roadnet + streaming replay index + sync timeline.
// Animation polish + metrics + compare layer land in later commits.

import { MapRenderer } from "./renderer.js";
import { ReplayIndex } from "./replay-loader.js";
import { LiveMetricsTracker, loadGroundTruth, countLanes } from "./metrics.js";
import { CompareOverlay } from "./compare.js";

const MANIFEST_URL = "../results/replays/manifest.json";

const state = {
    pairs: [],
    decisionCycle: 35,  // default; overwritten from manifest.decisionCycle
    panels: {
        A: { pair: null, renderer: null, index: null, loaded: false,
             tracker: new LiveMetricsTracker(), groundTruth: null, laneCount: 0,
             liveSeries: { awt: [], aql: [], throughput: [] } },
        B: { pair: null, renderer: null, index: null, loaded: false,
             tracker: new LiveMetricsTracker(), groundTruth: null, laneCount: 0,
             liveSeries: { awt: [], aql: [], throughput: [] } },
    },
    playing: false,
    step: 0,
    maxStep: 0,
    speed: 5,
    rafHandle: null,
    lastTickAt: 0,
    charts: null,
    compare: new CompareOverlay(),
    lastCounts: { A: null, B: null },
};

async function loadManifest() {
    try {
        const resp = await fetch(MANIFEST_URL, { cache: "no-store" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.warn("Manifest fetch failed:", err);
        return null;
    }
}

function populateDropdowns(pairs) {
    state.pairs = pairs;
    for (const panel of ["A", "B"]) {
        const sel = document.querySelector(`.replay-select[data-panel="${panel}"]`);
        sel.innerHTML = '<option value="">— Select replay —</option>';
        pairs.forEach((p, i) => {
            const opt = document.createElement("option");
            opt.value = String(i);
            opt.textContent = p.label;
            sel.appendChild(opt);
        });
        sel.addEventListener("change", () => {
            const btn = document.querySelector(`.load-btn[data-panel="${panel}"]`);
            btn.disabled = sel.value === "";
        });
    }
}

function setStatus(panel, text) {
    document.querySelector(`.status[data-panel="${panel}"]`).textContent = text;
}

async function onLoadClick(panel) {
    const sel = document.querySelector(`.replay-select[data-panel="${panel}"]`);
    const idx = parseInt(sel.value, 10);
    if (Number.isNaN(idx)) return;
    const pair = state.pairs[idx];
    const slot = state.panels[panel];
    slot.pair = pair;
    slot.loaded = false;
    slot.index = null;
    maybeEnablePlayback();

    setStatus(panel, "fetching roadnet…");
    try {
        const resp = await fetch(pair.roadnet, { cache: "no-store" });
        if (!resp.ok) throw new Error(`roadnet HTTP ${resp.status}`);
        const roadnet = await resp.json();

        if (!slot.renderer) {
            const canvas = document.querySelector(`.map-canvas[data-panel="${panel}"]`);
            slot.renderer = new MapRenderer(canvas);
        }
        slot.renderer.drawRoadnet(roadnet);
        slot.laneCount = countLanes(roadnet);
        slot.totalLanes = slot.laneCount;
        // Per-panel compare zones so dots stay aligned with whichever roadnet
        // is loaded in that panel (Hangzhou ≠ Jinan layout).
        state.compare.attach(panel, slot.renderer, roadnet);
        const scaleEl = document.getElementById("vehicle-scale");
        slot.renderer.setVehicleScale(parseFloat(scaleEl.value));

        // Ground-truth final metrics from runner JSON.
        slot.groundTruth = await loadGroundTruth(pair.metrics);
        renderGroundTruth(panel, slot.groundTruth);

        slot.tracker = new LiveMetricsTracker();
        slot.tracker.setDecisionCycle(state.decisionCycle);
        slot.liveSeries = { awt: [], aql: [], throughput: [] };

        setStatus(panel, "indexing replay (streaming)…");
        const t0 = performance.now();
        const index = new ReplayIndex(pair.replay, (prog) => {
            const pct = prog.total ? Math.round((prog.bytes / prog.total) * 100) : 0;
            setStatus(panel, `indexing ${pct}% · ${prog.steps} steps`);
        });
        const steps = await index.build();
        const ms = Math.round(performance.now() - t0);
        slot.index = index;
        slot.loaded = true;
        setStatus(panel, `ready · ${steps} steps · ${ms}ms · ${pair.label}`);

        // Render step 0 immediately so user sees the starting frame.
        renderPanelAtStep(panel, 0);

        maybeEnablePlayback();
        recomputeMaxStep();
    } catch (err) {
        console.error(err);
        setStatus(panel, `error: ${err.message}`);
    }
}

function recomputeMaxStep() {
    const counts = [];
    for (const p of ["A", "B"]) {
        if (state.panels[p].index) counts.push(state.panels[p].index.stepCount);
    }
    // Use min so both panels stay in sync without overrun.
    state.maxStep = counts.length > 0 ? Math.min(...counts) - 1 : 0;
    const slider = document.getElementById("timeline-slider");
    slider.max = String(state.maxStep);
    if (state.step > state.maxStep) state.step = state.maxStep;
    slider.value = String(state.step);
    updateStepDisplay();
}

function renderPanelAtStep(panel, step) {
    const slot = state.panels[panel];
    if (!slot.renderer || !slot.index) return;

    // Advance the tracker through every intermediate step we skipped so the
    // running averages stay correct. Backwards jumps are ignored — we keep
    // the accumulated series unchanged (recomputing 40k steps would freeze
    // the UI), so the metrics shown reflect "max forward step reached".
    slot.tracker.setLaneCount(slot.laneCount || 1);
    const advanceFrom = Math.max(0, slot.tracker.lastStep + 1);
    if (step >= advanceFrom) {
        for (let s = advanceFrom; s <= step; s++) {
            const { cars } = slot.index.parseStep(s);
            const m = slot.tracker.step(s, cars);
            slot.liveSeries.awt.push(m.awt);
            slot.liveSeries.aql.push(m.aql);
            slot.liveSeries.throughput.push(m.throughput);
        }
    }

    // Now render the actual displayed step.
    const { cars, lights } = slot.index.parseStep(step);
    slot.renderer.renderStep(cars, lights);
    state.lastCounts[panel] = state.compare.countQueues(panel, cars);
    maybeApplyIntersectionDots();
    renderLiveMetrics(panel);
}

function renderGroundTruth(panel, gt) {
    const box = document.querySelector(`.metrics-final[data-panel="${panel}"]`)
        || document.querySelector(`.metrics-box[data-panel="${panel}"] .metrics-final`);
    const cells = document.querySelectorAll(
        `.metrics-box[data-panel="${panel}"] .metrics-final td`,
    );
    if (!gt) {
        cells.forEach((c) => (c.textContent = "—"));
        return;
    }
    const fmt = (v) => (Number.isFinite(v) ? v.toFixed(2) : "—");
    const m = gt.metrics || {};
    cells.forEach((c) => {
        const which = c.getAttribute("data-metric");
        if (which === "att") c.textContent = fmt(m.att);
        else if (which === "aql") c.textContent = fmt(m.aql);
        else if (which === "awt") c.textContent = fmt(m.awt);
        else if (which === "backend") {
            const tu = gt.token_usage || {};
            c.textContent = tu.backend || "—";
        }
    });
}

function renderLiveMetrics(panel) {
    const slot = state.panels[panel];
    const s = slot.liveSeries;
    if (!s || s.awt.length === 0) return;
    const last = s.awt.length - 1;
    const cells = document.querySelectorAll(
        `.metrics-box[data-panel="${panel}"] .metrics-live td`,
    );
    const fmt = (v) => (Number.isFinite(v) ? v.toFixed(2) : "—");
    cells.forEach((c) => {
        const w = c.getAttribute("data-metric");
        if (w === "awt") c.textContent = fmt(s.awt[last]);
        else if (w === "aql") c.textContent = fmt(s.aql[last]);
        else if (w === "throughput") c.textContent = String(s.throughput[last]);
    });
    updateCharts();
    applyWinnerHighlight();
}

function setStep(step) {
    state.step = Math.max(0, Math.min(state.maxStep, step | 0));
    renderPanelAtStep("A", state.step);
    renderPanelAtStep("B", state.step);
    document.getElementById("timeline-slider").value = String(state.step);
    updateStepDisplay();
}

function updateStepDisplay() {
    document.getElementById("step-display").textContent =
        `step ${state.step} / ${state.maxStep}`;
}

function tick(now) {
    if (!state.playing) return;
    // Advance based on real time × speed (assume 1 sim step = 1s of game time).
    const dt = (now - state.lastTickAt) / 1000;
    state.lastTickAt = now;
    const advance = Math.max(1, Math.floor(dt * state.speed * 10));
    const next = state.step + advance;
    if (next >= state.maxStep) {
        setStep(state.maxStep);
        stopPlayback();
        return;
    }
    setStep(next);
    state.rafHandle = requestAnimationFrame(tick);
}

function startPlayback() {
    if (state.maxStep <= 0) return;
    state.playing = true;
    state.lastTickAt = performance.now();
    document.getElementById("play-btn").textContent = "⏸ Pause";
    state.rafHandle = requestAnimationFrame(tick);
}

function stopPlayback() {
    state.playing = false;
    if (state.rafHandle) cancelAnimationFrame(state.rafHandle);
    state.rafHandle = null;
    document.getElementById("play-btn").textContent = "▶ Play";
}

function maybeEnablePlayback() {
    const bothLoaded = state.panels.A.loaded && state.panels.B.loaded;
    document.getElementById("play-btn").disabled = !bothLoaded;
    document.getElementById("step-fwd-btn").disabled = !bothLoaded;
    document.getElementById("step-back-btn").disabled = !bothLoaded;
    document.getElementById("timeline-slider").disabled = !bothLoaded;
    if (bothLoaded) {
        document.getElementById("winner-banner").textContent =
            "Both replays selected — playback wiring lands in commit 3.";
    }
}

function wireGlobalControls() {
    document.getElementById("high-contrast").addEventListener("change", (e) => {
        document.body.classList.toggle("high-contrast", e.target.checked);
        // Redraw roadnet so new CSS colors take effect.
        for (const panel of ["A", "B"]) {
            const slot = state.panels[panel];
            if (slot.renderer && slot.pair) {
                fetch(slot.pair.roadnet)
                    .then((r) => r.json())
                    .then((j) => slot.renderer.drawRoadnet(j));
            }
        }
    });

    const scaleSlider = document.getElementById("vehicle-scale");
    const scaleVal = document.getElementById("vehicle-scale-val");
    scaleSlider.addEventListener("input", () => {
        scaleVal.textContent = `${scaleSlider.value}x`;
        const v = parseFloat(scaleSlider.value);
        for (const panel of ["A", "B"]) {
            const r = state.panels[panel].renderer;
            if (r) r.setVehicleScale(v);
        }
    });

    document.getElementById("speed-select").addEventListener("change", (e) => {
        state.speed = parseInt(e.target.value, 10);
    });

    for (const panel of ["A", "B"]) {
        document
            .querySelector(`.load-btn[data-panel="${panel}"]`)
            .addEventListener("click", () => onLoadClick(panel));
    }

    document.getElementById("play-btn").addEventListener("click", () => {
        if (state.playing) stopPlayback();
        else startPlayback();
    });
    document.getElementById("step-fwd-btn").addEventListener("click", () => {
        stopPlayback();
        setStep(state.step + 1);
    });
    document.getElementById("step-back-btn").addEventListener("click", () => {
        stopPlayback();
        setStep(state.step - 1);
    });
    document.getElementById("timeline-slider").addEventListener("input", (e) => {
        stopPlayback();
        setStep(parseInt(e.target.value, 10));
    });

    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        if (e.key === "p" || e.key === " ") {
            e.preventDefault();
            if (state.playing) stopPlayback(); else startPlayback();
        } else if (e.key === "]") {
            stopPlayback();
            setStep(state.step + 1);
        } else if (e.key === "[") {
            stopPlayback();
            setStep(state.step - 1);
        }
    });
}

function initCharts() {
    const mk = (id, label) =>
        new Chart(document.getElementById(id).getContext("2d"), {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Method A",
                        data: [],
                        borderColor: "#1d4ed8",
                        backgroundColor: "transparent",
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                    },
                    {
                        label: "Method B",
                        data: [],
                        borderColor: "#dc2626",
                        backgroundColor: "transparent",
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: { display: true, text: label, font: { size: 12 } },
                    legend: { display: true, labels: { font: { size: 10 }, boxWidth: 16 } },
                    tooltip: { enabled: true },
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: "step", font: { size: 10 } },
                        ticks: { font: { size: 9 }, maxTicksLimit: 6 },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { font: { size: 10 } },
                    },
                },
            },
        });
    state.charts = {
        awt: mk("chart-awt", "AWT — mean cumulative wait (s)"),
        aql: mk("chart-aql", "AQL — mean cars per lane"),
        throughput: mk("chart-throughput", "Throughput — completed vehicles"),
    };
}

let _chartUpdateScheduled = false;
function updateCharts() {
    if (!state.charts || _chartUpdateScheduled) return;
    _chartUpdateScheduled = true;
    // Coalesce rapid updates into one rAF — playback at 20x calls renderLive
    // dozens of times per frame; we only need one chart update per frame.
    requestAnimationFrame(() => {
        _chartUpdateScheduled = false;
        _doUpdateCharts();
    });
}

function _doUpdateCharts() {
    const seriesA = state.panels.A.liveSeries;
    const seriesB = state.panels.B.liveSeries;
    const n = Math.max(seriesA.awt.length, seriesB.awt.length);
    if (n === 0) return;
    // Show at most ~300 points across the full series.
    const stride = Math.max(1, Math.ceil(n / 300));
    const labels = [];
    const dsA = { awt: [], aql: [], throughput: [] };
    const dsB = { awt: [], aql: [], throughput: [] };
    for (let i = 0; i < n; i += stride) {
        labels.push(i);
        dsA.awt.push(seriesA.awt[i] ?? null);
        dsA.aql.push(seriesA.aql[i] ?? null);
        dsA.throughput.push(seriesA.throughput[i] ?? null);
        dsB.awt.push(seriesB.awt[i] ?? null);
        dsB.aql.push(seriesB.aql[i] ?? null);
        dsB.throughput.push(seriesB.throughput[i] ?? null);
    }
    for (const k of ["awt", "aql", "throughput"]) {
        const ch = state.charts[k];
        ch.data.labels = labels;
        ch.data.datasets[0].data = dsA[k];
        ch.data.datasets[1].data = dsB[k];
        ch.update("none");
    }
}

function maybeApplyIntersectionDots() {
    const a = state.lastCounts.A;
    const b = state.lastCounts.B;
    if (!a || !b) return;
    const worldScale = state.panels.A.renderer
        ? state.panels.A.renderer.world.scale.x
        : 1;
    state.compare.apply(a, b, worldScale);
}

function applyWinnerHighlight() {
    const A = state.panels.A.liveSeries;
    const B = state.panels.B.liveSeries;
    if (A.awt.length === 0 || B.awt.length === 0) return;
    const iA = A.awt.length - 1;
    const iB = B.awt.length - 1;
    const compare = (a, b, lowerIsBetter = true) => {
        if (!Number.isFinite(a) || !Number.isFinite(b)) return "tie";
        if (Math.abs(a - b) < 1e-6) return "tie";
        const aBetter = lowerIsBetter ? a < b : a > b;
        return aBetter ? "A" : "B";
    };
    const winners = {
        awt: compare(A.awt[iA], B.awt[iB], true),
        aql: compare(A.aql[iA], B.aql[iB], true),
        throughput: compare(A.throughput[iA], B.throughput[iB], false),
    };
    for (const m of ["awt", "aql", "throughput"]) {
        const tdA = document.querySelector(
            `.metrics-box[data-panel="A"] .metrics-live td[data-metric="${m}"]`,
        );
        const tdB = document.querySelector(
            `.metrics-box[data-panel="B"] .metrics-live td[data-metric="${m}"]`,
        );
        tdA.classList.remove("winner", "loser");
        tdB.classList.remove("winner", "loser");
        if (winners[m] === "A") {
            tdA.classList.add("winner");
            tdB.classList.add("loser");
        } else if (winners[m] === "B") {
            tdB.classList.add("winner");
            tdA.classList.add("loser");
        }
    }
    const wins = { A: 0, B: 0 };
    for (const w of Object.values(winners)) if (w !== "tie") wins[w]++;
    const banner = document.getElementById("winner-banner");
    banner.classList.remove("A-leads", "B-leads", "tied");
    if (wins.A > wins.B) {
        banner.textContent = `Method A leads ${wins.A}/3 metrics at step ${state.step}.`;
        banner.classList.add("A-leads");
    } else if (wins.B > wins.A) {
        banner.textContent = `Method B leads ${wins.B}/3 metrics at step ${state.step}.`;
        banner.classList.add("B-leads");
    } else {
        banner.textContent = `Tied (${wins.A}/3 each) at step ${state.step}.`;
        banner.classList.add("tied");
    }
}

async function init() {
    wireGlobalControls();
    initCharts();
    const manifest = await loadManifest();
    if (!manifest) {
        document.getElementById("winner-banner").innerHTML =
            'Could not load <code>results/replays/manifest.json</code>. ' +
            'Run <code>python scripts/build_viewer_manifest.py</code> first, ' +
            'then serve from project root: <code>python -m http.server 8000</code> ' +
            'and open <code>http://localhost:8000/viewer/</code>.';
        return;
    }
    populateDropdowns(manifest.pairs || []);
    if (manifest.decisionCycle) {
        state.decisionCycle = manifest.decisionCycle;
    }
    document.getElementById("winner-banner").textContent =
        `Discovered ${manifest.pairs.length} replay(s). Select one for each panel.`;
}

init();
