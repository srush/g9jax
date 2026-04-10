import { G9, point, line, np } from "./g9.js";

function show(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "";
}

function hideBanner() {
  const el = document.getElementById("loading-banner");
  if (el) el.style.display = "none";
}

function main() {
  // ---- Demo 1: Basic mirrored points ----
  show("section-points");
  new G9(
    (params) => {
      const xy = params.xy;
      const flip = np.array([[0, 1], [1, 0]], { dtype: np.float64 });
      const p2 = np.dot(flip, xy);
      return { p1: point(xy.ref), p2: point(p2) };
    },
    { xy: [40, 0] },
  ).align("center", "center").insertInto("#demo-points");

  // ---- Demo 2: Rings ----
  // Use a pure-array approach: precompute the angle offsets as constants
  show("section-rings");
  const SIDES = 8;
  const angleOffsets = [];
  for (let i = 0; i < SIDES; i++) {
    angleOffsets.push((i / SIDES) * Math.PI * 2);
  }

  new G9(
    (params) => {
      const pts = {};
      for (let i = 0; i < SIDES; i++) {
        const offset = np.array([angleOffsets[i]], { dtype: np.float64 });
        const a = params.angle.ref.add(offset);

        // Outer ring
        const ox = np.cos(a.ref).mul(params.radius.ref);
        const oy = np.sin(a.ref).mul(params.radius.ref);
        pts[`out${i}`] = point(np.concatenate([ox, oy]));

        // Inner ring (neg angle, half radius)
        const negA = a.neg();
        const halfR = params.radius.ref.div(2);
        const ix = np.cos(negA.ref).mul(halfR.ref);
        const iy = np.sin(negA).mul(halfR);
        pts[`in${i}`] = point(np.concatenate([ix, iy]), { fill: "#e11d48" });
      }
      return pts;
    },
    { radius: [120], angle: [0] },
  ).align("center", "center").insertInto("#demo-rings");

  // ---- Demo 3: Lines with affects ----
  show("section-lines");
  new G9(
    (params) => {
      const opts = { "stroke-width": 8, "stroke-linecap": "round" };
      return {
        l1: line(params.line1.ref, { stroke: "#1c1917", ...opts }),
        l2: line(params.line2.ref, {
          stroke: "#e11d48",
          affects: { line2: [1, 1, 0, 0] },
          ...opts,
        }),
        l3: line(params.line3.ref, {
          stroke: "#2563eb",
          affects: { line3: [1, 0, 0, 1] },
          ...opts,
        }),
      };
    },
    { line1: [-100, -50, 100, -50], line2: [-100, 0, 100, 0], line3: [-100, 50, 100, 50] },
  ).align("center", "center").insertInto("#demo-lines");

  // ---- Demo 4: Dragon curve ----
  show("section-dragon");
  new G9(
    (params) => {
      const reverseM = np.array([[0, 1], [-1, 0]], { dtype: np.float64 });
      const lineOpts = {
        "stroke-width": 3,
        "stroke-linecap": "round",
        affects: { squareness: true },
      };
      const pts = {};

      function dragon(fromPt, toPt, dir, level, name) {
        if (level === 0) {
          pts["ln" + name] = line(np.concatenate([fromPt, toPt]), lineOpts);
        } else {
          const diff = toPt.ref.sub(fromPt.ref);
          const rotated = np.dot(reverseM.ref, diff);
          const mid = fromPt.ref
            .add(toPt.ref)
            .add(rotated.mul(params.squareness.ref).mul(dir))
            .div(2.0);
          dragon(fromPt, mid.ref, -1, level - 1, name + "l");
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

  // ---- Demo 5: Fractal tree ----
  show("section-tree");
  new G9(
    (params) => {
      const pts = {};
      const PI_180 = Math.PI / 180;

      function branch(base, length, angle, depth, name) {
        const rad = angle.mul(PI_180);
        const dx = np.cos(rad.ref);
        const dy = np.sin(rad);
        const dir = np.concatenate([dx, dy]);
        const tip = base.ref.add(dir.mul(length));

        pts["pt" + name] = point(tip.ref, { fill: "#22c55e", r: 3 });
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
        } else {
          tip?.dispose?.();
          length?.dispose?.();
          angle?.dispose?.();
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

  hideBanner();
}

main();
