import { G9, point, line, np } from "./g9";
import { defaultDevice, init, vmap } from "@jax-js/jax";

// Pre-allocated constant matrices (hoisted out of render closures to avoid
// re-creating identical arrays on every render/trace call).
let FLIP_MATRIX: any;
let REVERSE_MATRIX: any;

function initConstants() {
  FLIP_MATRIX = np.array([[0, 1], [1, 0]], { dtype: np.float64 });
  REVERSE_MATRIX = np.array([[0, 1], [-1, 0]], { dtype: np.float64 });
}

function show(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "";
}

function hideBanner() {
  const el = document.getElementById("loading-banner");
  if (el) el.style.display = "none";
}

function setBanner(msg) {
  const el = document.getElementById("loading-banner");
  if (el) el.innerHTML = msg;
}

async function main() {
  setBanner('<span class="spinner"></span> Initialising jax-js runtime…');

  try {
    const backends = await init();
    console.log("jax-js init complete, backends:", backends);
    defaultDevice("cpu");
    console.log("jax-js default device forced to cpu");
  } catch (e) {
    console.warn("jax-js full init failed, trying cpu-only:", e.message || e);
    try {
      await init("cpu");
      defaultDevice("cpu");
      console.log("jax-js cpu backend ready");
    } catch (e2) {
      console.warn("jax-js cpu init also failed:", e2.message || e2);
    }
  }

  initConstants();

  setBanner('<span class="spinner"></span> Rendering demos…');

  // ---- Demo 1: Basic mirrored points ----
  // xy is used twice: once in np.dot and once in point(). We need .ref on
  // the first usage so xy survives for the second.
  try {
    show("section-points");
    new G9(
      (params) => {
        const xy = params.xy;
        const p2 = np.dot(FLIP_MATRIX.ref, xy.ref);
        return { p1: point(xy), p2: point(p2) };
      },
      { xy: [40, 0] },
    ).align("center", "center").insertInto("#demo-points");
    console.log("Demo 1 (points) OK");
  } catch (e) {
    console.error("Demo 1 (points) failed:", e);
  }

  // ---- Demo 2: Rings ----
  try {
    show("section-rings");
    const SIDES = 8;
    const offsets: number[] = [];
    for (let i = 0; i < SIDES; i++) offsets.push((i / SIDES) * Math.PI * 2);
    const OFFSETS = np.array(offsets, { dtype: np.float64 });

    const outerPoint = (offset: any, angle: any, radius: any) => {
      const a = angle.add(offset);
      const x = np.cos(a.ref).mul(radius.ref);
      const y = np.sin(a).mul(radius);
      return np.concatenate([x, y]);
    };
    const innerPoint = (offset: any, angle: any, radius: any) => {
      const a = angle.add(offset).neg();
      const x = np.cos(a.ref).mul(radius.ref);
      const y = np.sin(a).mul(radius);
      return np.concatenate([x, y]);
    };
    const vmapOuter = vmap(outerPoint, [0, null, null]);
    const vmapInner = vmap(innerPoint, [0, null, null]);

    new G9(
      (params) => {
        const outerFlat = vmapOuter(OFFSETS.ref, params.angle.ref, params.radius.ref).reshape([SIDES * 2]);
        const halfR = params.radius.div(2);
        const innerFlat = vmapInner(OFFSETS.ref, params.angle, halfR).reshape([SIDES * 2]);

        const pts: Record<string, any> = {};
        for (let i = 0; i < SIDES; i++) {
          const lastOuter = i === SIDES - 1;
          pts[`out${i}`] = point((lastOuter ? outerFlat : outerFlat.ref).slice([i * 2, i * 2 + 2]));

          const lastInner = i === SIDES - 1;
          pts[`in${i}`] = point((lastInner ? innerFlat : innerFlat.ref).slice([i * 2, i * 2 + 2]), { fill: "#e11d48" });
        }
        return pts;
      },
      { radius: [120], angle: [0] },
    ).align("center", "center").insertInto("#demo-rings");
    console.log("Demo 2 (rings) OK");
  } catch (e) {
    console.error("Demo 2 (rings) failed:", e);
  }

  // ---- Demo 3: Lines with affects ----
  try {
    show("section-lines");
    new G9(
      (params) => {
        const opts = { "stroke-width": 8 };
        return {
          l1: line(params.line1, { stroke: "black", ...opts }),
          l2: line(params.line2, {
            stroke: "#e11d48",
            affects: { line2: [1, 1, 0, 0] },
            ...opts,
          }),
          l3: line(params.line3, {
            stroke: "#2563eb",
            affects: { line3: [1, 0, 0, 1] },
            ...opts,
          }),
        };
      },
      { line1: [-100, -50, 100, -50], line2: [-100, 0, 100, 0], line3: [-100, 50, 100, 50] },
    ).align("center", "center").insertInto("#demo-lines");
    console.log("Demo 3 (lines) OK");
  } catch (e) {
    console.error("Demo 3 (lines) failed:", e);
  }

  // ---- Demo 4: Dragon curve ----
  try {
    show("section-dragon");
    new G9(
      (params) => {
        const lineOpts = {
          "stroke-width": 3,
          "stroke-linecap": "round",
          affects: { squareness: true },
        };
        const pts = {};

        // Ownership contract: dragon consumes fromPt and toPt exactly once
        // (at leaf via concatenate, at interior via recursive calls).
        function dragon(fromPt, toPt, dir, level, name) {
          if (level === 0) {
            pts["ln" + name] = line(np.concatenate([fromPt, toPt]), lineOpts);
          } else {
            // .ref keeps fromPt/toPt alive for reuse below
            const diff = toPt.ref.sub(fromPt.ref);
            const rotated = np.dot(REVERSE_MATRIX.ref, diff);
            const mid = fromPt.ref
              .add(toPt.ref)
              .add(rotated.mul(params.squareness.ref).mul(dir))
              .div(2.0);
            // Left branch consumes fromPt; mid.ref keeps mid alive for right branch
            dragon(fromPt, mid.ref, -1, level - 1, name + "l");
            // Right branch consumes mid and toPt
            dragon(mid, toPt, 1, level - 1, name + "r");
          }
        }

        dragon(params.fromPt.ref, params.toPt.ref, -1, 5, "");
        pts["from"] = point(params.fromPt.ref, { fill: "#0ea5e9", r: 6 });
        pts["to"] = point(params.toPt.ref, { fill: "#0ea5e9", r: 6 });
        return pts;
      },
      { fromPt: [175, 96], toPt: [-175, 39], squareness: [0.8] },
    ).align("center", "center").insertInto("#demo-dragon");
    console.log("Demo 4 (dragon) OK");
  } catch (e) {
    console.error("Demo 4 (dragon) failed:", e);
  }

  // ---- Demo 5: Fractal tree ----
  try {
    show("section-tree");
    new G9(
      (params) => {
        const pts = {};
        const PI_180 = Math.PI / 180;

        // Ownership: branch consumes base, length, angle exactly once.
        function branch(base, length, angle, depth, name) {
          // angle.ref: keep angle alive for later reuse
          const rad = angle.ref.mul(PI_180);
          const dx = np.cos(rad.ref);
          const dy = np.sin(rad);
          const dir = np.concatenate([dx, dy]);
          // length.ref: keep length alive for later reuse
          // base.ref: keep base alive for line below
          const tip = base.ref.add(dir.mul(length.ref));

          pts["pt" + name] = point(tip.ref, { fill: "#22c55e", r: 3 });
          // base consumed by concatenate (last use of base)
          pts["ln" + name] = line(np.concatenate([base, tip.ref]), {
            stroke: depth > 3 ? "#8B4513" : "#22c55e",
            "stroke-width": Math.max(1, depth * 0.7 + 1),
            "stroke-linecap": "round",
            affects: { deltaAngle: true, attenuation: true },
          });

          if (depth > 0) {
            const nl1 = length.ref.mul(params.attenuation.ref);
            const la = angle.ref.add(params.deltaAngle.ref);
            branch(tip.ref, nl1, la, depth - 1, name + "l");

            const nl2 = length.mul(params.attenuation.ref);
            const ra = angle.sub(params.deltaAngle.ref);
            branch(tip, nl2, ra, depth - 1, name + "r");
          }
        }

        const root = np.array([0, 120], { dtype: np.float64 });
        const startAngle = np.array([-90], { dtype: np.float64 });
        branch(root, params.startLength.ref, startAngle, 4, "");

        pts["root"] = point(np.array([0, 120], { dtype: np.float64 }), {
          fill: "#8B4513",
          r: 5,
          affects: { startLength: true },
        });
        return pts;
      },
      { deltaAngle: [33], startLength: [65], attenuation: [0.7] },
    ).align("center", "center").insertInto("#demo-tree");
    console.log("Demo 5 (tree) OK");
  } catch (e) {
    console.error("Demo 5 (tree) failed:", e);
  }

  hideBanner();
  console.log("All demos rendered");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  setBanner(`Error: ${err.message}. Check the browser console (F12) for details.`);
});
