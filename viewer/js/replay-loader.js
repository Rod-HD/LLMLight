// Streaming replay loader for CityFlow replay.txt format.
//
// Each line in replay.txt = 1 simulation step:
//   <car1>,<car2>,...,;<tl1>,<tl2>,...
// where carN = "x y rotation id laneChange length width"
// and   tlN = "roadId status1 status2 status3"  (status: r/g/y/i)
//
// Strategy for >500MB files: pass-1 streams via fetch().body.getReader(),
// keeps only the byte offset of each newline so step lookup is O(1) seek.
// Pass-2 decodes a single step on demand with a small Range request when
// served over HTTP, or a slice of the cached Uint8Array when loaded once.

const TEXT_DECODER = new TextDecoder("utf-8");

export class ReplayIndex {
    /**
     * @param {string} url - replay.txt URL
     * @param {(progress: {bytes:number,total:number,steps:number}) => void} [onProgress]
     */
    constructor(url, onProgress) {
        this.url = url;
        this.onProgress = onProgress || (() => {});
        this.lineStarts = [0];   // byte offset of each step (line-start). Last entry = file length.
        this.totalSize = 0;
        this.buffer = null;       // Uint8Array of full file once loaded.
        this._loaded = false;
    }

    /** Build the index by streaming the file once. */
    async build() {
        const resp = await fetch(this.url);
        if (!resp.ok) throw new Error(`replay HTTP ${resp.status}`);
        const total = parseInt(resp.headers.get("content-length") || "0", 10);
        this.totalSize = total;

        // We need the full bytes for later random access, so accumulate chunks
        // into one Uint8Array. For 1GB this allocates ~1GB on the JS heap;
        // Chrome 64-bit handles that fine. (We do NOT decode to string — that
        // would double the memory and be the slow part of the legacy viewer.)
        const chunks = [];
        let received = 0;
        let scanCarry = 0;          // bytes already scanned in the prev chunk
        const reader = resp.body.getReader();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            // Scan only the new bytes for '\n' (0x0a).
            const globalBase = received;
            for (let i = 0; i < value.length; i++) {
                if (value[i] === 0x0a) {
                    this.lineStarts.push(globalBase + i + 1);
                }
            }
            received += value.length;
            scanCarry = received;
            this.onProgress({
                bytes: received,
                total: total || received,
                steps: this.lineStarts.length - 1,
            });
        }

        // Concatenate chunks.
        this.buffer = new Uint8Array(received);
        let off = 0;
        for (const c of chunks) {
            this.buffer.set(c, off);
            off += c.length;
        }
        // If file doesn't end with newline, treat EOF as the final boundary.
        const last = this.lineStarts[this.lineStarts.length - 1];
        if (last !== received) this.lineStarts.push(received);

        this._loaded = true;
        return this.stepCount;
    }

    get stepCount() {
        // lineStarts has stepCount + 1 entries (each pair of adjacent entries
        // delimits one step). An empty trailing line is filtered here.
        let n = this.lineStarts.length - 1;
        // Trim trailing empty line if any.
        while (n > 0 && this.lineStarts[n] - this.lineStarts[n - 1] <= 1) n--;
        return n;
    }

    /** Decode step `i` (0-indexed) and return raw line string. */
    rawStep(i) {
        if (!this._loaded) throw new Error("ReplayIndex.rawStep before build()");
        if (i < 0 || i >= this.stepCount) return "";
        const a = this.lineStarts[i];
        const b = this.lineStarts[i + 1];
        // Strip trailing \n (and \r) if present.
        let end = b;
        if (end > a && this.buffer[end - 1] === 0x0a) end--;
        if (end > a && this.buffer[end - 1] === 0x0d) end--;
        return TEXT_DECODER.decode(this.buffer.subarray(a, end));
    }

    /**
     * Parse step `i` into structured form. Returns { cars: [...], lights: [...] }.
     * Cars: { x, y, rotation, id, laneChange, length, width }
     * Lights: { roadId, statuses: [s0, s1, s2] }
     */
    parseStep(i) {
        const raw = this.rawStep(i);
        if (!raw) return { cars: [], lights: [] };

        const semi = raw.indexOf(";");
        const carsRaw = semi >= 0 ? raw.slice(0, semi) : raw;
        const lightsRaw = semi >= 0 ? raw.slice(semi + 1) : "";

        const cars = [];
        if (carsRaw.length > 0) {
            // Cars are comma-separated and the source typically ends with a
            // trailing comma — split and skip empties.
            const carTokens = carsRaw.split(",");
            for (let k = 0; k < carTokens.length; k++) {
                const tok = carTokens[k];
                if (!tok) continue;
                const f = tok.split(" ");
                if (f.length < 7) continue;
                cars.push({
                    x: +f[0],
                    y: +f[1],
                    rotation: +f[2],
                    id: f[3],
                    laneChange: +f[4],
                    length: +f[5],
                    width: +f[6],
                });
            }
        }

        const lights = [];
        if (lightsRaw.length > 0) {
            const tlTokens = lightsRaw.split(",");
            for (let k = 0; k < tlTokens.length; k++) {
                const tok = tlTokens[k];
                if (!tok) continue;
                const f = tok.split(" ");
                if (f.length < 2) continue;
                lights.push({
                    roadId: f[0],
                    statuses: f.slice(1),
                });
            }
        }
        return { cars, lights };
    }
}
