// Per-intersection comparison overlay.
//
// For each non-virtual intersection in EACH panel's roadnet we draw a dot at
// the intersection center, colored by how that panel's queue compares to the
// OTHER panel's overall situation. We do NOT try to match intersection ids
// across panels — that only works when both panels use the same dataset.
// Instead, we compare the panel-wide average queue length per intersection.
//
// Color rule (per intersection):
//   green: this panel's queue here is BELOW its own panel-wide average
//   red:   this panel's queue here is ABOVE its own panel-wide average
//   gray:  within 10% of average
//
// This makes the overlay informative even when the two panels show different
// datasets (it answers "which intersections are the hot-spots for each
// method") while remaining correct when they show the same dataset.

const ZONE_RADIUS = 60; // meters; cars within this radius of a center bin in

export class CompareOverlay {
    constructor() {
        // Per-panel state: { zones: {id: {cx,cy}}, dots: {id: Graphics} }.
        this._panels = { A: null, B: null };
    }

    /** Build zones + attach dots for ONE panel. Must be called once per
     *  panel after that panel's renderer has drawn its roadnet. */
    attach(panel, renderer, roadnet) {
        renderer.compareLayer.removeChildren();
        const zones = {};
        const dots = {};
        for (const n of roadnet.static.nodes) {
            if (n.virtual) continue;
            zones[n.id] = { cx: n.point[0], cy: n.point[1] };
            const g = new PIXI.Graphics();
            g.position.set(n.point[0], -n.point[1]);
            renderer.compareLayer.addChild(g);
            dots[n.id] = g;
        }
        this._panels[panel] = { zones, dots };
    }

    /** Count cars binned per intersection for ONE panel. */
    countQueues(panel, cars) {
        const slot = this._panels[panel];
        if (!slot) return {};
        const counts = {};
        for (const id of Object.keys(slot.zones)) counts[id] = 0;
        const ZR2 = ZONE_RADIUS * ZONE_RADIUS;
        for (const c of cars) {
            let bestId = null;
            let bestD = ZR2;
            for (const id of Object.keys(slot.zones)) {
                const z = slot.zones[id];
                const dx = c.x - z.cx;
                const dy = c.y - z.cy;
                const d2 = dx * dx + dy * dy;
                if (d2 < bestD) { bestD = d2; bestId = id; }
            }
            if (bestId) counts[bestId]++;
        }
        return counts;
    }

    /** Apply per-panel dots based on each panel's own queue distribution. */
    apply(countsA, countsB, worldScale) {
        const radius = Math.max(2.5, 4 / Math.max(0.5, worldScale));
        this._paint("A", countsA, radius);
        this._paint("B", countsB, radius);
    }

    _paint(panel, counts, radius) {
        const slot = this._panels[panel];
        if (!slot) return;
        const ids = Object.keys(counts);
        if (ids.length === 0) return;
        let sum = 0;
        for (const id of ids) sum += counts[id];
        const mean = sum / ids.length;
        const tieBand = Math.max(0.5, mean * 0.1);
        for (const id of ids) {
            const g = slot.dots[id];
            if (!g) continue;
            g.clear();
            const v = counts[id];
            const diff = v - mean;
            let color;
            if (Math.abs(diff) <= tieBand) color = 0x9ca3af;
            else if (diff < 0) color = 0x16a34a; // below avg = better
            else color = 0xdc2626;                // above avg = worse
            g.beginFill(color, 0.95);
            g.lineStyle(0.4, 0xffffff, 0.9);
            g.drawCircle(0, 0, radius);
            g.endFill();
        }
    }

    clear() {
        for (const slot of Object.values(this._panels)) {
            if (!slot) continue;
            for (const g of Object.values(slot.dots)) g.clear();
        }
    }
}
