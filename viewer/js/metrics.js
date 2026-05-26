// Live metric computation from replay snapshots + ground-truth loader.
//
// ──────────────────────────────────────────────────────────────────────
// CONVERGENCE DESIGN
// ──────────────────────────────────────────────────────────────────────
//
// The runner (run_gpt4o.py) collects metrics once per DECISION CYCLE
// (= green + yellow + all_red steps, typically 35).  The replay file
// has one line per CityFlow step (interval = 1.0s).  To make the live
// metrics converge to the ground-truth JSON at the FINAL step, the
// tracker must sample at the same decision-cycle boundaries the runner
// used.  The decision cycle length is read from the manifest
// (`manifest.decisionCycle`) and passed into `setDecisionCycle(n)`.
//
// ──────────────────────────────────────────────────────────────────────
//
// AQL (Average Queue Length, "veh/lane"):
//   Ground-truth: runner appends engine.get_lane_vehicle_count() once
//   per cycle → MetricsEvaluator.compute_aql (= mean of per-lane counts
//   over all sampled cycles).
//   Live: we mirror this by accumulating cars.length / totalLanes ONLY
//   at decision-cycle boundaries (step % decisionCycle == 0).
//
// AWT (Average Waiting Time, "s"):
//   Ground-truth: runner accumulates `decision_cycle_change` seconds
//   per cycle for each vehicle whose speed < SPEED_THRESHOLD at that
//   cycle boundary → MetricsEvaluator.compute_awt (= mean over all
//   vehicles ever seen).
//   Live: at every decision-cycle boundary we derive speed from
//   position delta over the preceding frame.  For each vehicle below
//   threshold we add `decisionCycle` seconds (matching runner).  The
//   live value = Σ wait_seconds / |seenIds|.
//
// Throughput ("completed", count):
//   |seen_ids| - |active_ids|.  Strictly non-decreasing.  No
//   ground-truth counterpart (not in MetricsEvaluator).
//
// Speed derivation: replay format has position only; we compute
// speed = ||pos_t - pos_{t-1}|| (CityFlow interval = 1.0s default).

const SPEED_THRESHOLD = 0.1;

export class LiveMetricsTracker {
    constructor() {
        this.reset();
    }

    reset() {
        this.seenIds = new Set();
        this.activeIds = new Set();
        this.waitSeconds = new Map();   // vehicle_id → cumulative wait (seconds)
        this._prevPos = new Map();
        this.lastStep = -1;
        // Running totals for AQL — sampled at decision-cycle boundaries.
        this._aqlSum = 0;       // Σ (cars_at_boundary / lanes)
        this._aqlSamples = 0;   // number of decision-cycle samples
        this._laneCount = 1;
        this._decisionCycle = 1;  // default: every step is a boundary
        // Cached last snapshot to avoid recomputing on every UI read.
        this._last = { aql: 0, awt: 0, throughput: 0 };
    }

    setLaneCount(n) {
        this._laneCount = Math.max(1, n | 0);
    }

    /** Set the decision cycle length (steps).  Must match the runner's
     *  green_duration + yellow_duration + all_red_duration.  */
    setDecisionCycle(n) {
        this._decisionCycle = Math.max(1, n | 0);
    }

    /** Advance by one step, accumulating metrics.  Cheap: O(cars). */
    step(stepIdx, cars) {
        const prev = this._prevPos;
        const next = new Map();

        // Track which vehicles are present and derive per-vehicle speed.
        for (let i = 0; i < cars.length; i++) {
            const c = cars[i];
            this.seenIds.add(c.id);
            next.set(c.id, [c.x, c.y]);
        }

        // Decision-cycle boundary: sample AQL and AWT exactly like the
        // runner does (once per cycle, at step 0, decisionCycle, 2×…).
        const dc = this._decisionCycle;
        if (stepIdx % dc === 0) {
            // AQL: total cars in network / total lanes (matches runner's
            // engine.get_lane_vehicle_count() which returns ALL vehicles).
            this._aqlSum += cars.length / this._laneCount;
            this._aqlSamples += 1;

            // AWT: for each vehicle currently in the network, check speed
            // derived from the previous frame.  If below threshold, add
            // `decisionCycle` seconds of wait time (matching runner's
            // `vehicle_wait_seconds[vid] += decision_cycle_change`).
            for (let i = 0; i < cars.length; i++) {
                const c = cars[i];
                const p = prev.get(c.id);
                if (p !== undefined) {
                    const dx = c.x - p[0];
                    const dy = c.y - p[1];
                    if (dx * dx + dy * dy < SPEED_THRESHOLD * SPEED_THRESHOLD) {
                        this.waitSeconds.set(
                            c.id,
                            (this.waitSeconds.get(c.id) || 0) + dc,
                        );
                    } else {
                        // Ensure vehicle has an entry (like runner's setdefault)
                        if (!this.waitSeconds.has(c.id)) {
                            this.waitSeconds.set(c.id, 0);
                        }
                    }
                } else {
                    // First appearance → register with 0 wait
                    if (!this.waitSeconds.has(c.id)) {
                        this.waitSeconds.set(c.id, 0);
                    }
                }
            }
        }

        this._prevPos = next;
        this.activeIds = new Set(next.keys());
        this.lastStep = stepIdx;

        // Compute current values.
        const aqlAvg = this._aqlSamples > 0
            ? this._aqlSum / this._aqlSamples
            : 0;

        // AWT: mean cumulative wait seconds across ALL vehicles ever seen.
        let waitSum = 0;
        for (const v of this.waitSeconds.values()) waitSum += v;
        const awt = this.seenIds.size > 0
            ? waitSum / this.seenIds.size
            : 0;

        const throughput = this.seenIds.size - this.activeIds.size;

        this._last = { aql: aqlAvg, awt, throughput };
        return this._last;
    }

    snapshot() {
        return this._last;
    }
}

export async function loadGroundTruth(url) {
    if (!url) return null;
    try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) return null;
        return await r.json();
    } catch {
        return null;
    }
}

export function countLanes(roadnet) {
    let total = 0;
    for (const e of roadnet.static.edges) total += (e.nLane || 0);
    return total;
}
