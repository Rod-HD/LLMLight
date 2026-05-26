// PixiJS renderer for one CityFlow roadnet + replay step.
// Coordinate system: CityFlow stores world coords with +Y pointing up; PIXI
// canvas has +Y pointing down, so every Y gets flipped via transY().

export class MapRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.app = new PIXI.Application({
            view: canvas,
            backgroundAlpha: 0,
            antialias: true,
            resolution: window.devicePixelRatio || 1,
            autoDensity: true,
            resizeTo: canvas.parentElement,
        });

        // ResizeObserver guarantees we refit when the panel layout reflows
        // (e.g. metrics box growing). `resizeTo` only triggers on window
        // resize events, which is not enough here.
        this._ro = new ResizeObserver(() => {
            this.app.resize();
            this.fitToView();
        });
        this._ro.observe(canvas.parentElement);

        // Layers (back → front).
        this.roadLayer = new PIXI.Container();
        this.intersectionLayer = new PIXI.Container();
        this.signalLayer = new PIXI.Container();
        this.vehicleLayer = new PIXI.Container();
        this.compareLayer = new PIXI.Container();

        // World container: receives pan/zoom transform.
        this.world = new PIXI.Container();
        this.world.addChild(
            this.roadLayer,
            this.intersectionLayer,
            this.signalLayer,
            this.vehicleLayer,
            this.compareLayer,
        );
        this.app.stage.addChild(this.world);

        this.bounds = null;       // {minX, minY, maxX, maxY} in CityFlow coords
        this.vehicleScale = 2;    // multiplier from header slider
        this._signalSprites = {}; // roadId → array of PIXI.Graphics (one per lane)
        this._intersectionSprites = {}; // intersectionId → PIXI.Graphics for compare dots

        this._installPan();
        this.app.renderer.on("resize", () => this.fitToView());
    }

    /** Map CityFlow Y → screen Y (flip). */
    static transY(y) { return -y; }

    /** Compute bounding box of all real (non-virtual) nodes. */
    _computeBounds(roadnet) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        for (const node of roadnet.static.nodes) {
            const [x, y] = node.point;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            const ty = MapRenderer.transY(y);
            if (ty < minY) minY = ty;
            if (ty > maxY) maxY = ty;
        }
        return { minX, minY, maxX, maxY };
    }

    fitToView() {
        if (!this.bounds) return;
        const { minX, minY, maxX, maxY } = this.bounds;
        const w = maxX - minX;
        const h = maxY - minY;
        if (w <= 0 || h <= 0) return;
        const cw = this.app.renderer.width / this.app.renderer.resolution;
        const ch = this.app.renderer.height / this.app.renderer.resolution;
        // 5% margin
        const scale = Math.min(cw / w, ch / h) * 0.92;
        this.world.scale.set(scale, scale);
        this.world.position.set(
            cw / 2 - ((minX + maxX) / 2) * scale,
            ch / 2 - ((minY + maxY) / 2) * scale,
        );
    }

    _installPan() {
        // Simple click-drag pan + wheel zoom (no pixi-viewport dep, keeps things lean).
        let dragging = false;
        let lastX = 0, lastY = 0;
        this.canvas.addEventListener("mousedown", (e) => {
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        window.addEventListener("mouseup", () => { dragging = false; });
        window.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            this.world.position.x += e.clientX - lastX;
            this.world.position.y += e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        this.canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            // Zoom around cursor.
            const wx = (mx - this.world.position.x) / this.world.scale.x;
            const wy = (my - this.world.position.y) / this.world.scale.y;
            this.world.scale.x *= factor;
            this.world.scale.y *= factor;
            this.world.position.x = mx - wx * this.world.scale.x;
            this.world.position.y = my - wy * this.world.scale.y;
        }, { passive: false });
    }

    /** Read CSS variable to honour high-contrast mode. */
    _color(varName, fallback) {
        const v = getComputedStyle(document.body).getPropertyValue(varName).trim();
        if (!v) return fallback;
        // Strip leading #.
        return parseInt(v.replace("#", ""), 16);
    }

    drawRoadnet(roadnet) {
        this.roadnet = roadnet;
        this.bounds = this._computeBounds(roadnet);

        const roadColor = this._color("--road", 0xc8cfdb);
        const interColor = this._color("--intersection", 0xaab2c0);
        const laneDividerColor = document.body.classList.contains("high-contrast")
            ? 0x6b7280 : 0xffffff;

        const nodeById = {};
        for (const n of roadnet.static.nodes) nodeById[n.id] = n;
        this._nodeById = nodeById;

        // CityFlow uses right-hand traffic: lanes for a directed edge lie
        // entirely on the RIGHT side of the direction vector (not centered on
        // the polyline). So "centerline" of the lanes is offset from the JSON
        // points by totalWidth/2 to the right. For an east-bound road
        // (direction +X) this means lanes sit at y = -2, -6, -10 (lane 0
        // closest to the centerline, lane N-1 farthest right).
        //
        // The replay's status array for each road is in the same lane order
        // (lane 0 → lane N-1), so signal i lines up with lane i.

        // Compute per-edge geometry once so road + signal + lane-divider draws
        // all use the same vectors.
        const edgeGeom = [];
        for (const edge of roadnet.static.edges) {
            const pts = edge.points;
            const a = pts[0];
            const b = pts[pts.length - 1];
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const len = Math.hypot(dx, dy) || 1;
            const ux = dx / len, uy = dy / len;
            // Right-perpendicular in CityFlow world coords (CW 90°).
            const rx = uy, ry = -ux;
            edgeGeom.push({ edge, a, b, ux, uy, rx, ry, len });
        }

        // ---- Roads (filled quads, right of centerline) ---------------------
        this.roadLayer.removeChildren();
        for (const G of edgeGeom) {
            const totalWidth = (G.edge.laneWidths || []).reduce((s, w) => s + w, 0)
                || (G.edge.nLane || 3) * 4;
            // Quad: from a → b, offset 0 to rightWidth on the right side.
            const a = G.a, b = G.b, rx = G.rx, ry = G.ry, W = totalWidth;
            const q = new PIXI.Graphics();
            q.beginFill(roadColor, 1);
            q.moveTo(a[0],         MapRenderer.transY(a[1]));
            q.lineTo(b[0],         MapRenderer.transY(b[1]));
            q.lineTo(b[0] + rx*W,  MapRenderer.transY(b[1] + ry*W));
            q.lineTo(a[0] + rx*W,  MapRenderer.transY(a[1] + ry*W));
            q.closePath();
            q.endFill();
            this.roadLayer.addChild(q);
        }

        // ---- Lane dividers (white dashed lines between lanes) --------------
        // Drawn as a separate sub-layer so high-contrast theme can keep them
        // bright. Skip the outermost edges (those are the road borders).
        const dividerLayer = new PIXI.Graphics();
        dividerLayer.lineStyle(0.3, laneDividerColor, 0.9);
        const DASH = 4, GAP = 4;
        for (const G of edgeGeom) {
            const widths = G.edge.laneWidths || new Array(G.edge.nLane || 3).fill(4);
            const a = G.a, b = G.b, rx = G.rx, ry = G.ry, ux = G.ux, uy = G.uy;
            let cum = 0;
            // Draw dividers between lane i and lane i+1, NOT at 0 or totalW.
            for (let i = 0; i < widths.length - 1; i++) {
                cum += widths[i];
                const ax = a[0] + rx * cum, ay = a[1] + ry * cum;
                const bx = b[0] + rx * cum, by = b[1] + ry * cum;
                // Dashed line manually (Pixi v7 has no dashed lineStyle).
                const totalLen = Math.hypot(bx - ax, by - ay);
                const cycle = DASH + GAP;
                let t = 0;
                while (t < totalLen) {
                    const t1 = Math.min(t + DASH, totalLen);
                    const sx = ax + ux * t,  sy = ay + uy * t;
                    const ex = ax + ux * t1, ey = ay + uy * t1;
                    dividerLayer.moveTo(sx, MapRenderer.transY(sy));
                    dividerLayer.lineTo(ex, MapRenderer.transY(ey));
                    t += cycle;
                }
            }
        }
        this.roadLayer.addChild(dividerLayer);

        // ---- Intersections -------------------------------------------------
        this.intersectionLayer.removeChildren();
        this._intersectionSprites = {};
        for (const node of roadnet.static.nodes) {
            if (!node.outline || node.outline.length < 6) continue;
            const g = new PIXI.Graphics();
            g.beginFill(interColor, node.virtual ? 0.0 : 1.0);
            const o = node.outline;
            g.moveTo(o[0], MapRenderer.transY(o[1]));
            for (let i = 2; i < o.length; i += 2) {
                g.lineTo(o[i], MapRenderer.transY(o[i + 1]));
            }
            g.closePath();
            g.endFill();
            this.intersectionLayer.addChild(g);
            this._intersectionSprites[node.id] = g;
        }

        // ---- Signal slots: one circle per lane, at stop bar ----------------
        this.signalLayer.removeChildren();
        this._signalSprites = {};
        for (const G of edgeGeom) {
            const edge = G.edge;
            const toNode = nodeById[edge.to];
            if (!toNode || toNode.virtual) continue;
            const widths = edge.laneWidths || new Array(edge.nLane || 3).fill(4);
            const setback = (toNode.width || 15) + 1;
            const baseX = G.b[0] - G.ux * setback;
            const baseY = G.b[1] - G.uy * setback;
            const slots = [];
            let cum = 0;
            for (let li = 0; li < widths.length; li++) {
                const center = cum + widths[li] / 2;
                const cx = baseX + G.rx * center;
                const cy = baseY + G.ry * center;
                const dot = new PIXI.Graphics();
                dot.position.set(cx, MapRenderer.transY(cy));
                this.signalLayer.addChild(dot);
                slots.push(dot);
                cum += widths[li];
            }
            this._signalSprites[edge.id] = slots;
        }

        this.fitToView();
    }

    static SIGNAL_COLORS = {
        r: 0xff3030,
        g: 0x30c850,
        y: 0xffcc00,
        i: 0x000000, // 'i' = off, drawn with alpha 0
    };

    _drawSignalDot(g, status, radius) {
        g.clear();
        if (status === "i") return; // hidden
        const color = MapRenderer.SIGNAL_COLORS[status] ?? 0x888888;
        g.beginFill(color, 1.0);
        g.lineStyle(0.6, 0x000000, 0.6);
        g.drawCircle(0, 0, radius);
        g.endFill();
    }

    setVehicleScale(scale) {
        this.vehicleScale = scale;
    }

    _getVehicleSprite(idx) {
        if (!this._vehiclePool) this._vehiclePool = [];
        let v = this._vehiclePool[idx];
        if (!v) {
            v = new PIXI.Graphics();
            this._vehiclePool[idx] = v;
        }
        return v;
    }

    _vehicleColor() {
        // Single accent color shared with status bar; tinting per id makes the
        // map noisy at small scale. Pick a saturated color that contrasts with
        // road grey on both themes.
        return getComputedStyle(document.body).classList.contains?.("high-contrast")
            ? 0xffd23f
            : 0x1d4ed8;
    }

    renderStep(cars, lights) {
        // ---- Traffic lights -----------------------------------------------
        // Signal-dot radius: stay roughly the size of a lane (~4m wide) so it
        // never overflows the road, but never smaller than ~3 screen px.
        const worldScale = this.world.scale.x || 1;
        const minWorldRadius = 1.5; // half a lane width
        const minScreenPx = 2.5;
        const dotRadius = Math.max(minWorldRadius, minScreenPx / worldScale);
        // Wipe previously-drawn dots to avoid stale colors for roads not
        // present in this step (rare, but possible at sim boundaries).
        for (const slots of Object.values(this._signalSprites)) {
            for (const s of slots) s.clear();
        }
        for (const light of lights) {
            const slots = this._signalSprites[light.roadId];
            if (!slots) continue;
            // CityFlow writes per-direction statuses (left/straight/right);
            // we map them to lanes left→right.
            const n = Math.min(slots.length, light.statuses.length);
            for (let i = 0; i < n; i++) {
                this._drawSignalDot(slots[i], light.statuses[i], dotRadius);
            }
        }

        // ---- Vehicles (pooled, hide excess instead of destroying) ----------
        const carColor = document.body.classList.contains("high-contrast")
            ? 0xffd23f
            : 0x1d4ed8;
        const outlineColor = document.body.classList.contains("high-contrast")
            ? 0x000000
            : 0x0b1742;

        const pool = this._vehiclePool || (this._vehiclePool = []);
        // Ensure layer has at least cars.length sprites; reuse rest.
        for (let i = 0; i < cars.length; i++) {
            const c = cars[i];
            if (!Number.isFinite(c.x) || !Number.isFinite(c.y)) {
                if (pool[i]) pool[i].visible = false;
                continue;
            }
            let g = pool[i];
            if (!g) {
                g = new PIXI.Graphics();
                pool[i] = g;
                this.vehicleLayer.addChild(g);
            }
            g.visible = true;
            const len = c.length * this.vehicleScale;
            const wid = c.width * this.vehicleScale;
            // Only redraw geometry when length/width changes — most steps the
            // car dimensions are identical, so we can skip the begin/end fill.
            const key = `${len.toFixed(2)}_${wid.toFixed(2)}_${carColor}`;
            if (g.__shapeKey !== key) {
                g.clear();
                g.beginFill(carColor, 1);
                g.lineStyle(0.4, outlineColor, 0.9);
                g.drawRoundedRect(-len / 2, -wid / 2, len, wid, Math.min(len, wid) / 4);
                g.endFill();
                g.__shapeKey = key;
            }
            g.position.set(c.x, MapRenderer.transY(c.y));
            g.rotation = -c.rotation;
        }
        // Hide leftover sprites from previous frames with more cars.
        for (let i = cars.length; i < pool.length; i++) {
            if (pool[i].visible) pool[i].visible = false;
        }
    }

    destroy() {
        this.app.destroy(true, { children: true, texture: true, baseTexture: true });
    }
}
