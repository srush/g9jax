import { numpy as np, vmap } from "@jax-js/jax";
import {
  G9,
  minimize,
  point,
  line,
  getG9RuntimeStats,
  resetG9RuntimeStats,
  getG9DragDebugEnabled,
  setG9DragDebugEnabled,
  getG9LineSearchEnabled,
  setG9LineSearchEnabled,
  getG9DebugLossStats,
} from "./src/g9";

if (typeof (globalThis as { Float16Array?: typeof Float32Array }).Float16Array === "undefined") {
  (globalThis as { Float16Array: typeof Float32Array }).Float16Array = Float32Array;
}

type ParamState = { name: string; value: any };

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

function run(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`${name}: OK`);
  } catch (error) {
    console.error(`${name}: FAIL`);
    throw error;
  }
}

function toList(x: any): number[] {
  if (typeof x?.dataSync === "function") return Array.from(x.dataSync()) as number[];
  if (typeof x?.js === "function") {
    const value = x.js();
    return Array.isArray(value) ? (value.flat(Infinity) as number[]) : [Number(value)];
  }
  return [Number(x)];
}

class FakeElement {
  style: Record<string, string> = {};
  children: FakeElement[] = [];
  textContent = "";
  attrs: Record<string, string> = {};
  classList = {
    add: () => {},
    remove: () => {},
  };
  appendChild(child: FakeElement): FakeElement { this.children.push(child); return child; }
  removeChild(child: FakeElement): FakeElement { const i = this.children.indexOf(child); if (i >= 0) this.children.splice(i, 1); return child; }
  setAttributeNS(_ns: string | null, key: string, value: string): void { this.attrs[key] = value; }
  setAttribute(key: string, value: string): void { this.attrs[key] = value; }
  addEventListener(): void {}
  removeEventListener(): void {}
  getBoundingClientRect() { return { top: 0, left: 0, width: 800, height: 600 }; }
}

function installFakeDom(): FakeElement {
  const host = new FakeElement();
  (globalThis as any).document = {
    createElementNS: () => new FakeElement(),
    querySelector: () => host,
    addEventListener: () => {},
    removeEventListener: () => {},
    getElementById: () => null,
    documentElement: { classList: { add: () => {}, remove: () => {} } },
  };
  (globalThis as any).window = { addEventListener: () => {}, removeEventListener: () => {} };
  return host;
}

// ---------------------------------------------------------------------------

run("point drag loss minimizes", () => {
  const params: ParamState[] = [
    { name: "xy", value: np.array([40, 0], { dtype: np.float32 }) },
  ];

  const renderFn = (p: any) => {
    const xy = p.xy;
    const flip = np.array([[0, 1], [1, 0]], { dtype: np.float32 });
    const p2 = np.dot(flip, xy.ref);
    return { p1: point(xy), p2: point(p2) };
  };

  const lossFn = (target: any, coords: any) => {
    const d = coords.p1.sub(target);
    return d.ref.mul(d).sum();
  };

  minimize(params, renderFn, lossFn, [50, 10], null, 5);
  const xy = toList(params[0].value);
  assert(Number.isFinite(xy[0]) && Number.isFinite(xy[1]), "point params should remain finite");
});

run("line-search mode toggle keeps optimization stable", () => {
  const prev = getG9LineSearchEnabled();
  setG9LineSearchEnabled(true);
  try {
    const params: ParamState[] = [
      { name: "xy", value: np.array([40, 0], { dtype: np.float32 }) },
    ];
    const renderFn = (p: any) => ({ p1: point(p.xy) });
    const lossFn = (target: any, coords: any) => {
      const d = coords.p1.sub(target);
      return d.ref.mul(d).sum();
    };
    for (const target of [[65, 10], [42, -18], [50, 4]]) {
      minimize(params, renderFn, lossFn, target, null, 8);
    }
    const xy = toList(params[0].value);
    assert(xy.every(Number.isFinite), "line-search mode should keep params finite");
  } finally {
    setG9LineSearchEnabled(prev);
  }
});

run("line-search converges better with more iterations on rings", () => {
  const prev = getG9LineSearchEnabled();
  setG9LineSearchEnabled(true);
  try {
    const SIDES = 8;
    const offsets: number[] = [];
    for (let i = 0; i < SIDES; i++) offsets.push((i / SIDES) * Math.PI * 2);
    const OFFSETS = np.array(offsets, { dtype: np.float32 });
    const renderFn = (p: any) => {
      const outerPt = (offset: any, angle: any, radius: any) => {
        const a = angle.add(offset);
        return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
      };
      const innerPt = (offset: any, angle: any, radius: any) => {
        const a = angle.add(offset).neg();
        return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
      };
      const vmapOuter = vmap(outerPt, [0, null, null]);
      const vmapInner = vmap(innerPt, [0, null, null]);
      const outerFlat = vmapOuter(OFFSETS.ref, p.angle.ref, p.radius.ref).reshape([SIDES * 2]);
      const halfR = p.radius.div(2);
      const innerFlat = vmapInner(OFFSETS.ref, p.angle, halfR).reshape([SIDES * 2]);
      const pts: Record<string, any> = {};
      for (let i = 0; i < SIDES; i++) {
        const lo = i === SIDES - 1;
        pts[`out${i}`] = point((lo ? outerFlat : outerFlat.ref).slice([i * 2, i * 2 + 2]));
        const li = i === SIDES - 1;
        pts[`in${i}`] = point((li ? innerFlat : innerFlat.ref).slice([i * 2, i * 2 + 2]), { fill: "#e11d48" });
      }
      return pts;
    };
    const lossFn = (target: any, coords: any) => {
      const d = coords.out0.sub(target);
      return d.ref.mul(d).sum();
    };
    const runWithIter = (iter: number) => {
      const params: ParamState[] = [
        { name: "radius", value: np.array([120], { dtype: np.float32 }) },
        { name: "angle", value: np.array([0], { dtype: np.float32 }) },
      ];
      let cached: any = undefined;
      for (const target of [[94, 0], [82, 0], [70, 0], [58, 0]]) {
        cached = minimize(params, renderFn as any, lossFn as any, target, { radius: true }, iter, cached);
      }
      return Math.abs(toList(params[0].value)[0] - 58);
    };

    const err10 = runWithIter(10);
    const err14 = runWithIter(14);
    console.log(`line-search residual (10 vs 14 iters): ${err10.toFixed(4)} vs ${err14.toFixed(4)}`);
    assert(err14 <= err10, `expected 14 iters to converge at least as well as 10 (${err14} > ${err10})`);
  } finally {
    setG9LineSearchEnabled(prev);
  }
});

run("line-search converges better with more iterations on hard rings path", () => {
  const prev = getG9LineSearchEnabled();
  setG9LineSearchEnabled(true);
  try {
    const SIDES = 8;
    const offsets: number[] = [];
    for (let i = 0; i < SIDES; i++) offsets.push((i / SIDES) * Math.PI * 2);
    const OFFSETS = np.array(offsets, { dtype: np.float32 });

    const renderFn = (p: any) => {
      const outerPt = (offset: any, angle: any, radius: any) => {
        const a = angle.add(offset);
        return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
      };
      const innerPt = (offset: any, angle: any, radius: any) => {
        const a = angle.add(offset).neg();
        return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
      };
      const vmapOuter = vmap(outerPt, [0, null, null]);
      const vmapInner = vmap(innerPt, [0, null, null]);
      const outerFlat = vmapOuter(OFFSETS.ref, p.angle.ref, p.radius.ref).reshape([SIDES * 2]);
      const halfR = p.radius.div(2);
      const innerFlat = vmapInner(OFFSETS.ref, p.angle, halfR).reshape([SIDES * 2]);
      const pts: Record<string, any> = {};
      for (let i = 0; i < SIDES; i++) {
        const lo = i === SIDES - 1;
        pts[`out${i}`] = point((lo ? outerFlat : outerFlat.ref).slice([i * 2, i * 2 + 2]));
        const li = i === SIDES - 1;
        pts[`in${i}`] = point((li ? innerFlat : innerFlat.ref).slice([i * 2, i * 2 + 2]));
      }
      return pts;
    };
    const lossFn = (target: any, coords: any) => {
      const d = coords.out0.sub(target);
      return d.ref.mul(d).sum();
    };
    const runWithIter = (iter: number) => {
      const params: ParamState[] = [
        { name: "radius", value: np.array([120], { dtype: np.float32 }) },
        { name: "angle", value: np.array([0], { dtype: np.float32 }) },
      ];
      let cached: any = undefined;
      const targets = [[120, 50], [70, -95], [10, 110], [95, -25], [58, 0]];
      for (const target of targets) {
        cached = minimize(params, renderFn as any, lossFn as any, target, null, iter, cached);
      }
      const out0 = toList(renderFn({ radius: params[0].value.ref, angle: params[1].value.ref }).out0.c);
      return Math.hypot(out0[0] - 58, out0[1] - 0);
    };

    const err10 = runWithIter(10);
    const err14 = runWithIter(14);
    console.log(`hard line-search residual (10 vs 14 iters): ${err10.toFixed(4)} vs ${err14.toFixed(4)}`);
    assert(err14 < err10, `expected 14 iterations to beat 10 on hard path (${err14} !< ${err10})`);
  } finally {
    setG9LineSearchEnabled(prev);
  }
});

run("debug mode tracks average optimization loss", () => {
  const prevDebug = getG9DragDebugEnabled();
  setG9DragDebugEnabled(true);
  try {
    const params: ParamState[] = [
      { name: "xy", value: np.array([35, -10], { dtype: np.float32 }) },
    ];
    const renderFn = (p: any) => ({ p1: point(p.xy) });
    const lossFn = (target: any, coords: any) => {
      const d = coords.p1.sub(target);
      return d.ref.mul(d).sum();
    };
    minimize(params, renderFn, lossFn, [42, 8], null, 4);
    const stats = getG9DebugLossStats();
    assert(stats.count > 0, "debug loss stats should record samples");
    assert(Number.isFinite(stats.average), "debug loss average should be finite");
  } finally {
    setG9DragDebugEnabled(prevDebug);
  }
});

run("basic example render path works exactly as in main.ts", () => {
  const xy = np.array([40, 0], { dtype: np.float32 });
  const render = (params: { xy: any }) => {
    const value = params.xy;
    const flip = np.array([[0, 1], [1, 0]], { dtype: np.float32 });
    const p2 = np.dot(flip, value.ref);
    return { p1: point(value), p2: point(p2) };
  };

  const shapes = render({ xy: xy.ref });
  assert(toList(shapes.p1.c).length === 2, "basic example p1 should be 2D");
  assert(toList(shapes.p2.c).length === 2, "basic example p2 should be 2D");
});

run("basic example survives repeated G9 minimize/render cycles", () => {
  const host = installFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => {
      const xy = params.xy;
      const flip = np.array([[0, 1], [1, 0]], { dtype: np.float32 });
      const p2 = np.dot(flip, xy.ref);
      return { p1: point(xy), p2: point(p2) };
    },
    { xy: [40, 0] },
  );

  g9.align("center", "center").insertInto(host as any);

  const lossFn = (target: any, coords: any) => {
    const delta = coords.p1.sub(target);
    return delta.ref.mul(delta).sum();
  };

  for (let i = 0; i < 4; i++) {
    minimize((g9 as any).params, (g9 as any).renderFn, lossFn, [50, 10], null, 3);
    g9.render();
  }

  const xy = toList((g9 as any).params[0].value);
  assert(Number.isFinite(xy[0]) && Number.isFinite(xy[1]), "basic G9 params should remain finite");
});

run("rings example render path works exactly as in main.ts", () => {
  const SIDES = 8;
  const angleOffsets: number[] = [];
  for (let i = 0; i < SIDES; i++) angleOffsets.push((i / SIDES) * Math.PI * 2);

  const radius = np.array([120], { dtype: np.float32 });
  const angle = np.array([0], { dtype: np.float32 });
  const render = (params: { radius: any; angle: any }) => {
    const pts: Record<string, any> = {};
    for (let i = 0; i < SIDES; i++) {
      const offset = np.array([angleOffsets[i]], { dtype: np.float32 });
      const a = params.angle.ref.add(offset);
      const ox = np.cos(a.ref).mul(params.radius.ref);
      const oy = np.sin(a.ref).mul(params.radius.ref);
      pts[`out${i}`] = point(np.concatenate([ox, oy]));
      const negA = a.neg();
      const halfR = params.radius.ref.div(2);
      const ix = np.cos(negA.ref).mul(halfR.ref);
      const iy = np.sin(negA).mul(halfR);
      pts[`in${i}`] = point(np.concatenate([ix, iy]), { fill: "#e11d48" });
    }
    return pts;
  };

  const shapes = render({ radius: radius.ref, angle: angle.ref });
  assert(Object.keys(shapes).length === 16, "rings example should render 16 points");
  assert(toList(shapes.out0.c).length === 2, "rings outer point should be 2D");
  assert(toList(shapes.in0.c).length === 2, "rings inner point should be 2D");
});

run("line drag loss minimizes", () => {
  const params: ParamState[] = [
    { name: "line1", value: np.array([-100, -50, 100, -50], { dtype: np.float32 }) },
  ];

  const renderFn = (p: any) => ({ l1: line(p.line1) });

  const lossFn = (target: any, coords: any) => {
    const cv = coords.l1;
    const fromPt = cv.ref.slice([0, 2]);
    const toPt = cv.slice([2, 4]);
    const dir = toPt.sub(fromPt.ref);
    const predicted = fromPt.add(dir.mul(0.5));
    const delta = predicted.sub(target);
    return delta.ref.mul(delta).sum();
  };

  minimize(params, renderFn, lossFn, [0, -40], null, 5);
  const coords = toList(params[0].value);
  assert(coords.length === 4, "line coords should stay 4D");
  assert(coords.every(Number.isFinite), "line coords should remain finite");
});

run("constrained line supports negative projection drag", () => {
  const params: ParamState[] = [
    { name: "line1", value: np.array([-100, -50, 100, -50], { dtype: np.float32 }) },
    { name: "line2", value: np.array([-100, 0, 100, 0], { dtype: np.float32 }) },
    { name: "line3", value: np.array([-100, 50, 100, 50], { dtype: np.float32 }) },
  ];

  const id = "line3";
  const renderFn = (p: any) => ({
    line1: line(p.line1),
    line2: line(p.line2),
    line3: line(p.line3),
  });
  const lossFn = (target: any, coords: any) => {
    const cv = coords[id];
    const fromPt = cv.ref.slice([0, 2]);
    const toPt = cv.slice([2, 4]);
    const dir = toPt.sub(fromPt.ref);
    const r = target.ref.slice([2, 3]);
    const predicted = fromPt.add(dir.mul(r));
    const t = target.slice([0, 2]);
    const d = predicted.sub(t);
    return d.ref.mul(d).sum();
  };

  minimize(params, renderFn, lossFn, [-220, 50, -0.5], { line3: [1, 0, 0, 1] }, 8);
  const line3 = toList(params[2].value);
  assert(line3[0] < -100, "line3 x1 should move left for negative projection drag");
});

run("rings radius can repeatedly shrink", () => {
  const SIDES = 8;
  const offsets: number[] = [];
  for (let i = 0; i < SIDES; i++) {
    offsets.push((i / SIDES) * Math.PI * 2);
  }
  const OFFSETS = np.array(offsets, { dtype: np.float32 });
  const params: ParamState[] = [
    { name: "radius", value: np.array([120], { dtype: np.float32 }) },
    { name: "angle", value: np.array([0], { dtype: np.float32 }) },
  ];

  const outerPt = (offset: any, angle: any, radius: any) => {
    const a = angle.add(offset);
    return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
  };
  const innerPt = (offset: any, angle: any, radius: any) => {
    const a = angle.add(offset).neg();
    return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
  };
  const vmapOuter = vmap(outerPt, [0, null, null]);
  const vmapInner = vmap(innerPt, [0, null, null]);
  const renderFn = (p: any) => {
    const outerFlat = vmapOuter(OFFSETS.ref, p.angle.ref, p.radius.ref).reshape([SIDES * 2]);
    const halfR = p.radius.div(2);
    const innerFlat = vmapInner(OFFSETS.ref, p.angle, halfR).reshape([SIDES * 2]);
    const pts: Record<string, any> = {};
    for (let i = 0; i < SIDES; i++) {
      const lo = i === SIDES - 1;
      pts[`out${i}`] = point((lo ? outerFlat : outerFlat.ref).slice([i * 2, i * 2 + 2]));
      const li = i === SIDES - 1;
      pts[`in${i}`] = point((li ? innerFlat : innerFlat.ref).slice([i * 2, i * 2 + 2]), { fill: "#e11d48" });
    }
    return pts;
  };

  const lossFn = (target: any, coords: any) => {
    const d = coords.out0.sub(target);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  const targets = [[90, 0], [80, 0], [70, 0], [60, 0]];
  for (const target of targets) {
    cached = minimize(params, renderFn as any, lossFn as any, target, null, 10, cached);
  }
  const radius = toList(params[0].value)[0];
  assert(radius <= 60.5, `radius should shrink to near target, got ${radius}`);
});

run("jit cache is reused across repeated minimizations", () => {
  resetG9RuntimeStats();
  const params: ParamState[] = [
    { name: "xy", value: np.array([40, 0], { dtype: np.float32 }) },
  ];
  const renderFn = (p: any) => {
    const xy = p.xy;
    const flip = np.array([[0, 1], [1, 0]], { dtype: np.float32 });
    const p2 = np.dot(flip, xy.ref);
    return { p1: point(xy), p2: point(p2) };
  };
  const lossFn = (target: any, coords: any) => {
    const d = coords.p1.sub(target);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  for (let i = 0; i < 12; i++) {
    cached = minimize(params, renderFn, lossFn, [60 + i, 10], null, 8, cached);
  }

  const stats = getG9RuntimeStats();
  assert(stats.minimizeCalls >= 12, "expected repeated minimize calls");
  assert(stats.jitBuilds === 1, `expected one jit build, got ${stats.jitBuilds}`);
  assert(stats.jitCacheHits >= 10, `expected cache hits, got ${stats.jitCacheHits}`);
});

run("affects mask is reused from cached state", () => {
  resetG9RuntimeStats();
  const params: ParamState[] = [
    { name: "line1", value: np.array([-100, -50, 100, -50], { dtype: np.float32 }) },
    { name: "line2", value: np.array([-100, 0, 100, 0], { dtype: np.float32 }) },
    { name: "line3", value: np.array([-100, 50, 100, 50], { dtype: np.float32 }) },
  ];
  const renderFn = (p: any) => ({
    line1: line(p.line1),
    line2: line(p.line2),
    line3: line(p.line3),
  });
  const lossFn = (target: any, coords: any) => {
    const cv = coords.line3;
    const fromPt = cv.ref.slice([0, 2]);
    const toPt = cv.slice([2, 4]);
    const dir = toPt.sub(fromPt.ref);
    const r = target.ref.slice([2, 3]);
    const predicted = fromPt.add(dir.mul(r));
    const t = target.slice([0, 2]);
    const d = predicted.sub(t);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  const affects = { line3: [1, 0, 0, 1] };
  for (let i = 0; i < 10; i++) {
    cached = minimize(params, renderFn, lossFn, [-140 - i * 4, 50, -0.2], affects, 8, cached);
  }

  assert(cached.affectsMask, "expected cached affects mask");
  assert(cached.affectsRef === affects, "expected affects reference to be reused");
});

run("dragon render survives optimization path", () => {
  const params: ParamState[] = [
    { name: "fromPt", value: np.array([175, 96], { dtype: np.float32 }) },
    { name: "toPt", value: np.array([-175, 39], { dtype: np.float32 }) },
    { name: "squareness", value: np.array([0.8], { dtype: np.float32 }) },
  ];

  const renderFn = (p: any) => {
    const reverseM = np.array([[0, 1], [-1, 0]], { dtype: np.float32 });
    const shapes: Record<string, any> = {};
    function dragon(a: any, b: any, dir: number, level: number, name: string): void {
      if (level === 0) {
        shapes[`ln${name}`] = line(np.concatenate([a, b]));
      } else {
        const diff = b.ref.sub(a.ref);
        const rotated = np.dot(reverseM.ref, diff);
        const mid = a.ref.add(b.ref).add(rotated.mul(p.squareness.ref).mul(dir)).div(2);
        dragon(a, mid.ref, -1, level - 1, `${name}l`);
        dragon(mid, b, 1, level - 1, `${name}r`);
      }
    }
    dragon(p.fromPt.ref, p.toPt.ref, -1, 4, "");
    shapes.from = point(p.fromPt.ref);
    shapes.to = point(p.toPt.ref);
    return shapes;
  };

  const lossFn = (target: any, coords: any) => {
    const delta = coords.from.sub(target);
    return delta.ref.mul(delta).sum();
  };

  minimize(params, renderFn, lossFn, [100, 50], null, 3);
  assert(toList(params[0].value).every(Number.isFinite), "dragon fromPt should remain finite");
  assert(toList(params[1].value).every(Number.isFinite), "dragon toPt should remain finite");
  assert(toList(params[2].value).every(Number.isFinite), "dragon squareness should remain finite");
});

run("tree render survives optimization path", () => {
  const params: ParamState[] = [
    { name: "deltaAngle", value: np.array([33], { dtype: np.float32 }) },
    { name: "startLength", value: np.array([65], { dtype: np.float32 }) },
    { name: "attenuation", value: np.array([0.7], { dtype: np.float32 }) },
  ];

  const renderFn = (p: any) => {
    const shapes: Record<string, any> = {};
    const PI_180 = Math.PI / 180;
    function branch(base: any, length: any, angle: any, depth: number, name: string): void {
      const rad = angle.ref.mul(PI_180);
      const dx = np.cos(rad.ref);
      const dy = np.sin(rad);
      const dir = np.concatenate([dx, dy]);
      const tip = base.ref.add(dir.mul(length.ref));
      shapes[`pt${name}`] = point(tip.ref);
      shapes[`ln${name}`] = line(np.concatenate([base, tip.ref]));
      if (depth > 0) {
        branch(tip.ref, length.ref.mul(p.attenuation.ref), angle.ref.add(p.deltaAngle.ref), depth - 1, `${name}l`);
        branch(tip, length.mul(p.attenuation.ref), angle.sub(p.deltaAngle.ref), depth - 1, `${name}r`);
      }
    }
    branch(np.array([0, 120], { dtype: np.float32 }), p.startLength.ref, np.array([-90], { dtype: np.float32 }), 3, "");
    shapes.root = point(np.array([0, 120], { dtype: np.float32 }));
    return shapes;
  };

  const lossFn = (target: any, coords: any) => {
    const delta = coords.root.sub(target);
    return delta.ref.mul(delta).sum();
  };

  minimize(params, renderFn, lossFn, [0, 120], null, 3);
  assert(toList(params[0].value).every(Number.isFinite), "tree deltaAngle should remain finite");
  assert(toList(params[1].value).every(Number.isFinite), "tree startLength should remain finite");
  assert(toList(params[2].value).every(Number.isFinite), "tree attenuation should remain finite");
});

run("snake demo runs offline and supports repeated minimization", () => {
  const host = installFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => {
      const pts: Record<string, any> = {};
      const turns = [params.r1, params.r2, params.r3, params.r4];
      const segmentLength = 36;
      let angle = np.array([0], { dtype: np.float32 });
      let head = np.concatenate([params.x.ref, params.y.ref]);
      pts.p0 = point(head.ref, { fill: "#ef4444", r: 7, affects: { x: true, y: true } });
      for (let i = 0; i < turns.length; i++) {
        angle = angle.ref.add(turns[i].ref);
        const step = np.concatenate([
          np.cos(angle.ref).mul(segmentLength),
          np.sin(angle.ref).mul(segmentLength),
        ]);
        const next = head.ref.add(step.ref);
        pts[`s${i}`] = line(np.concatenate([head.ref, next.ref]), {
          stroke: "#1f2937",
          "stroke-width": 5 - i,
          "stroke-linecap": "round",
          affects: { r1: true, r2: true, r3: true, r4: true },
        });
        pts[`p${i + 1}`] = point(next.ref, { fill: "#0ea5e9", r: 5 });
        head = next;
      }
      return pts;
    },
    {
      r1: [6.2594],
      r2: [12.5397],
      r3: [12.708],
      r4: [6.0184],
      x: [84],
      y: [12],
    },
  );
  g9.align("center", "center").insertInto(host as any);

  const lossFn = (target: any, coords: any) => {
    const d = coords.p0.sub(target);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  for (const target of [[88, 20], [70, 28], [96, -6]]) {
    cached = minimize((g9 as any).params, (g9 as any).renderFn, lossFn, target, { x: true, y: true }, 8, cached);
    g9.render();
  }

  const x = toList((g9 as any).params[4].value)[0];
  const y = toList((g9 as any).params[5].value)[0];
  assert(Number.isFinite(x) && Number.isFinite(y), "snake demo params should remain finite");
});

run("tongs demo runs offline and supports repeated minimization", () => {
  const host = installFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => {
      const rotate = (xy: any, a: any) => {
        const c = np.cos(a.ref);
        const s = np.sin(a.ref);
        const x = xy.ref.slice([0, 1]);
        const y = xy.ref.slice([1, 2]);
        const rx = c.ref.mul(x.ref).sub(s.ref.mul(y.ref));
        const ry = s.ref.mul(x.ref).add(c.ref.mul(y.ref));
        return np.concatenate([rx, ry]);
      };

      const pts: Record<string, any> = {};
      const SEGMENT = 68;
      let x = np.array([0], { dtype: np.float32 });
      let yTop = np.array([0], { dtype: np.float32 });
      let yBottom = np.sin(params.b.ref).mul(-SEGMENT);

      for (let i = 0; i < 4; i++) {
        const dir = i % 2 === 0 ? -1 : 1;
        const nx = x.ref.add(np.cos(params.b.ref).mul(SEGMENT));
        const nyTop = yTop.ref.add(np.sin(params.b.ref).mul(SEGMENT * dir));
        const nyBottom = yBottom.ref.sub(np.sin(params.b.ref).mul(SEGMENT * dir));

        const a0 = rotate(np.concatenate([x.ref, yTop.ref]), params.a.ref);
        const a1 = rotate(np.concatenate([nx.ref, nyTop.ref]), params.a.ref);
        const b0 = rotate(np.concatenate([x.ref, yBottom.ref]), params.a.ref);
        const b1 = rotate(np.concatenate([nx.ref, nyBottom.ref]), params.a.ref);

        pts[`u${i}`] = line(np.concatenate([a0.ref, a1.ref]), {
          stroke: "#111827",
          "stroke-width": 8,
          "stroke-linecap": "round",
        });
        pts[`l${i}`] = line(np.concatenate([b0.ref, b1.ref]), {
          stroke: "#111827",
          "stroke-width": 8,
          "stroke-linecap": "round",
        });
        pts[`up${i}`] = point(a1.ref, { fill: "#0ea5e9", r: 4, affects: { a: true, b: true } });
        pts[`lp${i}`] = point(b1.ref, { fill: "#0ea5e9", r: 4, affects: { a: true, b: true } });

        x = nx;
        yTop = nyTop;
        yBottom = nyBottom;
      }
      return pts;
    },
    { a: [0.3], b: [0.85] },
  );
  g9.align("center", "center").insertInto(host as any);

  const lossFn = (target: any, coords: any) => {
    const d = coords.up3.sub(target);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  for (const target of [[120, -20], [110, 0], [95, 15]]) {
    cached = minimize((g9 as any).params, (g9 as any).renderFn, lossFn, target, { a: true, b: true }, 8, cached);
    g9.render();
  }

  const a = toList((g9 as any).params[0].value)[0];
  const b = toList((g9 as any).params[1].value)[0];
  assert(Number.isFinite(a) && Number.isFinite(b), "tongs demo params should remain finite");
});

run("bezier demo runs offline and supports repeated minimization", () => {
  const host = installFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => {
      const pts: Record<string, any> = {};
      const start = params.start;
      const middle = params.middle;
      const end = params.end;
      const steps = 30;
      const t = params.t;

      pts.ctrl1 = line(np.concatenate([start.ref, middle.ref]), { stroke: "rgba(0,0,0,0.25)" });
      pts.ctrl2 = line(np.concatenate([middle.ref, end.ref]), { stroke: "rgba(0,0,0,0.25)" });
      pts.pStart = point(start.ref, { fill: "#0ea5e9", r: 6 });
      pts.pMiddle = point(middle.ref, { fill: "#f97316", r: 6 });
      pts.pEnd = point(end.ref, { fill: "#0ea5e9", r: 6 });

      const curve = [];
      for (let i = 0; i < steps; i++) {
        const r = t.ref.mul(i / steps);
        const oneMinus = np.array([1], { dtype: np.float32 }).sub(r.ref);
        const a = start.ref.mul(oneMinus.ref).add(middle.ref.mul(r.ref));
        const b = middle.ref.mul(oneMinus.ref).add(end.ref.mul(r.ref));
        const c = a.ref.mul(oneMinus.ref).add(b.ref.mul(r.ref));
        if (i % 4 === 0) {
          pts[`step${i}`] = line(np.concatenate([a.ref, b.ref]), {
            stroke: "rgba(0,0,0,0.12)",
            affects: { t: true },
          });
        }
        curve.push(c);
      }

      for (let i = 1; i < curve.length; i++) {
        pts[`curve${i}`] = line(np.concatenate([curve[i - 1].ref, curve[i].ref]), {
          stroke: "#111827",
          "stroke-width": 4,
          affects: { t: true },
        });
      }

      const tY = np.array([140], { dtype: np.float32 });
      const tX = t.ref.mul(240).sub(120);
      pts.tAxis = line(np.array([-120, 140, 120, 140], { dtype: np.float32 }), {
        stroke: "#94a3b8",
        "stroke-width": 2,
      });
      pts.tKnob = point(np.concatenate([tX, tY]), {
        fill: "#16a34a",
        r: 7,
        affects: { t: true },
      });
      return pts;
    },
    {
      start: [-110, 62],
      middle: [0, -150],
      end: [130, 58],
      t: [0.5],
    },
  );
  g9.align("center", "center").insertInto(host as any);

  const lossFn = (target: any, coords: any) => {
    const d = coords.tKnob.sub(target);
    return d.ref.mul(d).sum();
  };

  let cached: any = undefined;
  for (const target of [[-96, 140], [96, 140], [24, 140]]) {
    cached = minimize((g9 as any).params, (g9 as any).renderFn, lossFn, target, { t: true }, 8, cached);
    g9.render();
  }

  const t = toList((g9 as any).params[3].value)[0];
  assert(Number.isFinite(t), "bezier demo t parameter should remain finite");
});

run("adaptive-step optimizer converges better with 8 iterations than 6", () => {
  const SIDES = 8;
  const offsets: number[] = [];
  for (let i = 0; i < SIDES; i++) {
    offsets.push((i / SIDES) * Math.PI * 2);
  }
  const OFFSETS = np.array(offsets, { dtype: np.float32 });

  const renderFn = (p: any) => {
    const outerPt = (offset: any, angle: any, radius: any) => {
      const a = angle.add(offset);
      return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
    };
    const innerPt = (offset: any, angle: any, radius: any) => {
      const a = angle.add(offset).neg();
      return np.concatenate([np.cos(a.ref).mul(radius.ref), np.sin(a).mul(radius)]);
    };
    const vmapOuter = vmap(outerPt, [0, null, null]);
    const vmapInner = vmap(innerPt, [0, null, null]);
    const outerFlat = vmapOuter(OFFSETS.ref, p.angle.ref, p.radius.ref).reshape([SIDES * 2]);
    const halfR = p.radius.div(2);
    const innerFlat = vmapInner(OFFSETS.ref, p.angle, halfR).reshape([SIDES * 2]);
    const pts: Record<string, any> = {};
    for (let i = 0; i < SIDES; i++) {
      const lo = i === SIDES - 1;
      pts[`out${i}`] = point((lo ? outerFlat : outerFlat.ref).slice([i * 2, i * 2 + 2]));
      const li = i === SIDES - 1;
      pts[`in${i}`] = point((li ? innerFlat : innerFlat.ref).slice([i * 2, i * 2 + 2]), { fill: "#e11d48" });
    }
    return pts;
  };
  const lossFn = (target: any, coords: any) => {
    const d = coords.out0.sub(target);
    return d.ref.mul(d).sum();
  };
  const runWithIter = (iter: number) => {
    const params: ParamState[] = [
      { name: "radius", value: np.array([120], { dtype: np.float32 }) },
      { name: "angle", value: np.array([0], { dtype: np.float32 }) },
    ];
    let cached: any = undefined;
    for (const target of [[94, 0], [82, 0], [70, 0], [58, 0]]) {
      cached = minimize(params, renderFn as any, lossFn as any, target, { radius: true }, iter, cached);
    }
    return Math.abs(toList(params[0].value)[0] - 58);
  };

  const err6 = runWithIter(6);
  const err8 = runWithIter(8);
  console.log(`convergence residual (6 vs 8 iters): ${err6.toFixed(4)} vs ${err8.toFixed(4)}`);
  assert(err8 < err6, `expected 8 iterations to converge better than 6 (${err8} !< ${err6})`);
});

console.log("All offline regression tests passed.");
