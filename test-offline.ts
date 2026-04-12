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
  listeners: Record<string, Array<(event: any) => void>> = {};
  classList = {
    add: () => {},
    remove: () => {},
    contains: () => false,
  };
  appendChild(child: FakeElement): FakeElement { this.children.push(child); return child; }
  removeChild(child: FakeElement): FakeElement { const i = this.children.indexOf(child); if (i >= 0) this.children.splice(i, 1); return child; }
  setAttributeNS(_ns: string | null, key: string, value: string): void { this.attrs[key] = value; }
  setAttribute(key: string, value: string): void { this.attrs[key] = value; }
  addEventListener(type: string, handler: (event: any) => void): void {
    (this.listeners[type] ??= []).push(handler);
  }
  removeEventListener(type: string, handler: (event: any) => void): void {
    const list = this.listeners[type];
    if (!list) return;
    const idx = list.indexOf(handler);
    if (idx >= 0) list.splice(idx, 1);
  }
  dispatch(type: string, event: any): void {
    for (const handler of this.listeners[type] ?? []) handler(event);
  }
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

function installInteractiveFakeDom(): {
  host: FakeElement;
  dispatchDocumentEvent: (type: string, event: any) => void;
} {
  const host = new FakeElement();
  const documentListeners = new Map<string, Array<(event: any) => void>>();
  const activeClasses = new Set<string>();
  const addDocumentListener = (type: string, handler: (event: any) => void) => {
    const list = documentListeners.get(type) ?? [];
    list.push(handler);
    documentListeners.set(type, list);
  };
  const removeDocumentListener = (type: string, handler: (event: any) => void) => {
    const list = documentListeners.get(type);
    if (!list) return;
    const idx = list.indexOf(handler);
    if (idx >= 0) list.splice(idx, 1);
  };
  const dispatchDocumentEvent = (type: string, event: any) => {
    for (const handler of documentListeners.get(type) ?? []) handler(event);
  };

  (globalThis as any).document = {
    createElementNS: () => new FakeElement(),
    querySelector: () => host,
    addEventListener: addDocumentListener,
    removeEventListener: removeDocumentListener,
    getElementById: () => null,
    documentElement: {
      classList: {
        add: (cls: string) => { activeClasses.add(cls); },
        remove: (cls: string) => { activeClasses.delete(cls); },
        contains: (cls: string) => activeClasses.has(cls),
      },
    },
  };
  (globalThis as any).window = {
    addEventListener: () => {},
    removeEventListener: () => {},
    requestAnimationFrame: () => 1,
    cancelAnimationFrame: () => {},
    setInterval: () => 0,
    clearInterval: () => {},
    matchMedia: () => ({ matches: false }),
  };
  return { host, dispatchDocumentEvent };
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

run("line-search mode keeps optimization stable", () => {
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

run("optimize loss event includes container id and finite loss", () => {
  const events: Array<{ containerId: string; loss: number }> = [];
  const listeners = new Map<string, (event: any) => void>();
  const host = installFakeDom();
  (globalThis as any).document.addEventListener = (name: string, cb: (event: any) => void) => {
    listeners.set(name, cb);
  };
  (globalThis as any).document.dispatchEvent = (event: { type: string; detail: any }) => {
    if (event.type === "g9:opt-loss") events.push(event.detail);
    const listener = listeners.get(event.type);
    if (listener) listener(event);
    return true;
  };

  const g9 = new G9(
    (params: Record<string, any>) => ({ p1: point(params.xy) }),
    { xy: [40, 0] },
  );
  g9.align("center", "center").insertInto("#demo-points");
  const lossFn = (target: any, coords: any) => {
    const d = coords.p1.sub(target);
    return d.ref.mul(d).sum();
  };

  (g9 as any)._minimize("p1", lossFn, [52, 14], null, true, undefined);
  assert(events.length > 0, "expected optimize loss events");
  assert(events.some((e) => e.containerId === "demo-points"), "expected container id in optimize loss event");
  assert(events.every((e) => Number.isFinite(e.loss)), "expected finite optimize loss values in events");
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
  const LEVELS = 5;
  const PER_LEVEL = 20;
  const TOTAL = LEVELS * PER_LEVEL;
  const OFFSET_VALUES = Array.from({ length: TOTAL }, (_unused, i) => (i / PER_LEVEL) * Math.PI * 2);
  const LEVEL_IDS = Array.from({ length: TOTAL }, (_unused, i) => Math.floor(i / PER_LEVEL));
  const BASE_RADII = [1.0, 0.84, 0.68, 0.52, 0.36];
  const DIR = [1, -1, 1, -1, 1];

  const radius = np.array([120], { dtype: np.float32 });
  const angle = np.array([0], { dtype: np.float32 });
  const render = (params: { radius: any; angle: any }) => {
    const pts: Record<string, any> = {};
    for (let i = 0; i < TOTAL; i++) {
      const offset = np.array([OFFSET_VALUES[i]], { dtype: np.float32 });
      const dir = np.array([DIR[LEVEL_IDS[i]]], { dtype: np.float32 });
      const a = params.angle.ref.mul(dir.ref).add(offset);
      const baseScale = np.array([BASE_RADII[LEVEL_IDS[i]]], { dtype: np.float32 });
      const levelRadius = params.radius.ref.mul(baseScale.ref);
      const px = np.cos(a.ref).mul(levelRadius.ref);
      const py = np.sin(a).mul(levelRadius);
      const color = LEVEL_IDS[i] === 0
        ? "#111827"
        : LEVEL_IDS[i] === 1
          ? "#2563eb"
          : LEVEL_IDS[i] === 2
            ? "#10b981"
            : LEVEL_IDS[i] === 3
              ? "#f59e0b"
              : "#ef4444";
      pts[`p${i}`] = point(np.concatenate([px, py]), { fill: color });
    }
    return pts;
  };

  const shapes = render({ radius, angle });
  assert(Object.keys(shapes).length === TOTAL, `rings example should render ${TOTAL} points`);
  const level0 = toList(shapes.p0.c);
  const level99 = toList(shapes.p99.c);
  const level1 = toList(shapes.p20.c);
  assert(level0.length === 2, "rings point should be 2D");
  assert(level99.length === 2, "rings final point should be 2D");
  assert(Math.sign(level0[1]) !== Math.sign(level1[1]), "adjacent levels should rotate in opposite directions");
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

run("line drag end does not snap back to drag start", () => {
  const { host, dispatchDocumentEvent } = installInteractiveFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => ({
      l1: line(params.line1),
      l2: line(params.line2, { affects: { line2: [1, 1, 0, 0] } }),
      l3: line(params.line3, { affects: { line3: [1, 0, 0, 1] } }),
    }),
    {
      line1: [-100, -50, 100, -50],
      line2: [-100, 0, 100, 0],
      line3: [-100, 50, 100, 50],
    },
  );
  g9.align("center", "center").insertInto(host as any);

  const lineEl = ((g9 as any).elements.l2 as any).el as FakeElement;
  const eventAt = (clientX: number, clientY: number) => ({
    clientX,
    clientY,
    cancelable: true,
    stopPropagation: () => {},
    preventDefault: () => {},
  });

  // Start drag at center and pull diagonally.
  lineEl.dispatch("mousedown", eventAt(400, 300));
  dispatchDocumentEvent("mousemove", eventAt(440, 320));
  dispatchDocumentEvent("mouseup", eventAt(440, 320));

  const line2 = toList((g9 as any).params[1].value);
  assert(Math.abs(line2[1]) > 10, `line2 y1 should stay displaced after mouseup, got ${line2[1]}`);
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

run("particles demo keeps params finite under minimization", () => {
  const COUNT = 100;
  const GRID = 10;
  const STEP = 28;
  const initialPos: number[] = [];
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      initialPos.push((gx - (GRID - 1) / 2) * STEP);
      initialPos.push((gy - (GRID - 1) / 2) * STEP);
    }
  }

  const renderFn = (p: any) => {
    const flat = p.pos;
    const pts: Record<string, any> = {};
    for (let i = 0; i < COUNT; i++) {
      const start = i * 2;
      const c = i === COUNT - 1
        ? flat.slice([start, start + 2])
        : flat.ref.slice([start, start + 2]);
      pts[`p${i}`] = point(c, { affects: { pos: true } });
    }
    return pts;
  };

  const lossFn = (target: any, coords: any) => {
    const d = coords.p0.sub(target);
    return d.ref.mul(d).sum();
  };

  const params: ParamState[] = [
    { name: "pos", value: np.array(initialPos, { dtype: np.float32 }) },
  ];

  let cached: any = undefined;
  for (const target of [[0, 0], [24, -30], [-18, 26]]) {
    cached = minimize(params, renderFn as any, lossFn as any, target, { pos: true }, 6, cached);
  }

  const posVals = toList(params[0].value);
  assert(posVals.length === COUNT * 2, `expected ${COUNT * 2} particle coordinates`);
  assert(posVals.every(Number.isFinite), "particle coordinates should remain finite");
});

run("secondary objectives are summed with main objective", () => {
  const baseRender = (p: any) => {
    const xy = np.concatenate([p.x.ref, p.y.ref]);
    return { p0: point(xy, { affects: { x: true, y: true } }) };
  };
  const lossFn = (target: any, coords: any) => {
    const d = coords.p0.sub(target);
    return d.ref.mul(d).sum();
  };

  const paramsNoSecondary: ParamState[] = [
    { name: "x", value: np.array([0], { dtype: np.float32 }) },
    { name: "y", value: np.array([0], { dtype: np.float32 }) },
  ];
  minimize(paramsNoSecondary, baseRender as any, lossFn as any, [30, 0], { x: true, y: true }, 6);
  const xNoSecondary = Math.abs(toList(paramsNoSecondary[0].value)[0]);

  const penaltyWeight = np.array([4], { dtype: np.float32 });
  const renderWithSecondary = (p: any) => {
    const xy = np.concatenate([p.x.ref, p.y.ref]);
    const springPenalty = p.x.ref.mul(p.x).sum().mul(penaltyWeight.ref);
    return {
      shapes: { p0: point(xy, { affects: { x: true, y: true } }) },
      secondary: { springPenalty },
    };
  };
  const paramsSecondary: ParamState[] = [
    { name: "x", value: np.array([0], { dtype: np.float32 }) },
    { name: "y", value: np.array([0], { dtype: np.float32 }) },
  ];
  minimize(paramsSecondary, renderWithSecondary as any, lossFn as any, [30, 0], { x: true, y: true }, 6);
  const xSecondary = Math.abs(toList(paramsSecondary[0].value)[0]);
  assert(xSecondary < xNoSecondary, `secondary penalty should reduce x magnitude (${xSecondary} < ${xNoSecondary})`);
});

run("blocker wall demo applies strong negative secondary inside wall", () => {
  const ONE = np.array([1], { dtype: np.float32 });
  const SHARP = np.array([0.35], { dtype: np.float32 });
  const WALL_SECONDARY = np.array([-10000], { dtype: np.float32 });
  const WALL_X0 = np.array([-20], { dtype: np.float32 });
  const WALL_X1 = np.array([20], { dtype: np.float32 });
  const WALL_Y0 = np.array([-130], { dtype: np.float32 });
  const WALL_Y1 = np.array([130], { dtype: np.float32 });
  const WALL_LEFT_SEG = np.array([-20, -130, -20, 130], { dtype: np.float32 });
  const WALL_RIGHT_SEG = np.array([20, -130, 20, 130], { dtype: np.float32 });

  const renderFn = (params: any) => {
    const sigmoid = (x: any) => ONE.ref.div(ONE.ref.add(np.exp(x.neg())));
    const insideBand = (value: any, minVal: any, maxVal: any) => {
      const geMin = sigmoid(value.ref.sub(minVal.ref).mul(SHARP.ref));
      const leMax = sigmoid(maxVal.ref.sub(value.ref).mul(SHARP.ref));
      return geMin.ref.mul(leMax.ref);
    };

    const inWallX = insideBand(params.bx, WALL_X0, WALL_X1);
    const inWallY = insideBand(params.by, WALL_Y0, WALL_Y1);
    const inWall = inWallX.ref.mul(inWallY.ref);
    const shapes: Record<string, any> = {
      wallLeft: line(WALL_LEFT_SEG, {
        stroke: "#111827",
        "stroke-width": 16,
        "stroke-linecap": "round",
      }),
      wallRight: line(WALL_RIGHT_SEG, {
        stroke: "#111827",
        "stroke-width": 16,
        "stroke-linecap": "round",
      }),
      ball: point(np.concatenate([params.bx.ref, params.by.ref]), {
        fill: "#f97316",
        r: 8,
        affects: { bx: true, by: true },
      }),
    };
    return {
      shapes,
      secondary: {
        wallBlock: inWall.ref.mul(WALL_SECONDARY.ref),
      },
    };
  };

  const outside = renderFn({
    bx: np.array([-170], { dtype: np.float32 }),
    by: np.array([0], { dtype: np.float32 }),
  });
  const inside = renderFn({
    bx: np.array([0], { dtype: np.float32 }),
    by: np.array([0], { dtype: np.float32 }),
  });
  const outsidePenalty = toList((outside as any).secondary.wallBlock)[0];
  const insidePenalty = toList((inside as any).secondary.wallBlock)[0];
  assert(insidePenalty < outsidePenalty - 9000, `inside wall penalty should be near -10000 (${insidePenalty} vs ${outsidePenalty})`);

  const lossFn = (target: any, coords: any) => {
    const d = coords.ball.sub(target);
    return d.ref.mul(d).sum();
  };
  const params: ParamState[] = [
    { name: "bx", value: np.array([-170], { dtype: np.float32 }) },
    { name: "by", value: np.array([0], { dtype: np.float32 }) },
  ];
  let cached: any = undefined;
  for (const target of [[-120, 0], [-80, 0], [-40, 0], [0, 0]]) {
    cached = minimize(params, renderFn as any, lossFn as any, target, { bx: true, by: true }, 6, cached);
  }
  const bx = toList(params[0].value)[0];
  const by = toList(params[1].value)[0];
  assert(Number.isFinite(bx) && Number.isFinite(by), "blocker wall params should remain finite");
});

run("snake demo runs offline and supports repeated minimization", () => {
  const host = installFakeDom();
  const g9 = new G9(
    (params: Record<string, any>) => {
      const pts: Record<string, any> = {};
      const turns = [params.r1, params.r2, params.r3, params.r4];
      const jointAffects = [{ r1: true }, { r2: true }, { r3: true }, { r4: true }];
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
          affects: jointAffects[i],
        });
        pts[`p${i + 1}`] = point(next.ref, {
          fill: "#0ea5e9",
          r: 5,
          affects: jointAffects[i],
        });
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

run("snake tiny line drag does not overshoot turn parameters", () => {
  const params: ParamState[] = [
    { name: "r1", value: np.array([6.2594], { dtype: np.float32 }) },
    { name: "r2", value: np.array([12.5397], { dtype: np.float32 }) },
    { name: "r3", value: np.array([12.708], { dtype: np.float32 }) },
    { name: "r4", value: np.array([6.0184], { dtype: np.float32 }) },
    { name: "x", value: np.array([84], { dtype: np.float32 }) },
    { name: "y", value: np.array([12], { dtype: np.float32 }) },
  ];

  const renderFn = (p: any) => {
    const pts: Record<string, any> = {};
    const turns = [p.r1, p.r2, p.r3, p.r4];
    const segmentLength = 36;
    const lineAffects = [{ r1: true }, { r2: true }, { r3: true }, { r4: true }];
    const pointAffects = [{ r1: true }, { r2: true }, { r3: true }, { r4: true }];
    let angle = np.array([0], { dtype: np.float32 });
    let head = np.concatenate([p.x.ref, p.y.ref]);
    pts.p0 = point(head.ref, { affects: { x: true, y: true } });
    for (let i = 0; i < turns.length; i++) {
      angle = angle.ref.add(turns[i].ref);
      const step = np.concatenate([
        np.cos(angle.ref).mul(segmentLength),
        np.sin(angle.ref).mul(segmentLength),
      ]);
      const next = head.ref.add(step.ref);
      pts[`s${i}`] = line(np.concatenate([head.ref, next.ref]), { affects: lineAffects[i] });
      pts[`p${i + 1}`] = point(next.ref, { affects: pointAffects[i] });
      head = next;
    }
    return pts;
  };

  const p: Record<string, any> = {};
  for (const ps of params) p[ps.name] = ps.value.ref;
  const shapes = renderFn(p);
  const s1 = toList(shapes.s1.c);
  const cx = (s1[0] + s1[2]) / 2;
  const cy = (s1[1] + s1[3]) / 2;
  const ldx = s1[2] - s1[0];
  const ldy = s1[3] - s1[1];
  const ll2 = ldx * ldx + ldy * ldy;
  const r = ll2 > 0 ? ((cx - s1[0]) * ldx + (cy - s1[1]) * ldy) / ll2 : 0;
  const before = params.map((ps) => toList(ps.value)[0]);

  const lineLoss = (target: any, coords: any) => {
    const cv = coords.s1;
    const fromPt = cv.ref.slice([0, 2]);
    const toPt = cv.slice([2, 4]);
    const dir = toPt.sub(fromPt.ref);
    const rr = target.ref.slice([2, 3]);
    const predicted = fromPt.add(dir.mul(rr));
    const t = target.slice([0, 2]);
    const d = predicted.sub(t);
    return d.ref.mul(d).sum();
  };

  minimize(params, renderFn as any, lineLoss as any, [cx + 1, cy, r], { r2: true }, 5);
  const after = params.map((ps) => toList(ps.value)[0]);
  const deltas = after.map((v, i) => Math.abs(v - before[i]));
  assert(deltas[1] < 0.02, `expected small r2 change for tiny drag, got ${deltas[1]}`);
  assert(deltas[0] < 1e-6 && deltas[2] < 1e-6 && deltas[3] < 1e-6, "other turn params should not move");
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
        const localAffects = i < 3
          ? { b: true, dragIter: [1] }
          : { a: true, dragIter: [1] };
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
          affects: localAffects,
        });
        pts[`l${i}`] = line(np.concatenate([b0.ref, b1.ref]), {
          stroke: "#111827",
          "stroke-width": 8,
          "stroke-linecap": "round",
          affects: localAffects,
        });
        pts[`up${i}`] = point(a1.ref, { fill: "#0ea5e9", r: 4, affects: localAffects });
        pts[`lp${i}`] = point(b1.ref, { fill: "#0ea5e9", r: 4, affects: localAffects });

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

run("tongs tiny line drag does not overshoot parameters", () => {
  const rotate = (xy: any, a: any) => {
    const c = np.cos(a.ref);
    const s = np.sin(a.ref);
    const x = xy.ref.slice([0, 1]);
    const y = xy.ref.slice([1, 2]);
    const rx = c.ref.mul(x.ref).sub(s.ref.mul(y.ref));
    const ry = s.ref.mul(x.ref).add(c.ref.mul(y.ref));
    return np.concatenate([rx, ry]);
  };

  const renderFn = (params: any) => {
    const pts: Record<string, any> = {};
    const SEGMENT = 68;
    let x = np.array([0], { dtype: np.float32 });
    let yTop = np.array([0], { dtype: np.float32 });
    let yBottom = np.sin(params.b.ref).mul(-SEGMENT);

    for (let i = 0; i < 4; i++) {
      const dir = i % 2 === 0 ? -1 : 1;
      const localAffects = i < 3
        ? { b: true, dragIter: [1] }
        : { a: true, dragIter: [1] };
      const nx = x.ref.add(np.cos(params.b.ref).mul(SEGMENT));
      const nyTop = yTop.ref.add(np.sin(params.b.ref).mul(SEGMENT * dir));
      const nyBottom = yBottom.ref.sub(np.sin(params.b.ref).mul(SEGMENT * dir));
      const a0 = rotate(np.concatenate([x.ref, yTop.ref]), params.a.ref);
      const a1 = rotate(np.concatenate([nx.ref, nyTop.ref]), params.a.ref);
      const b0 = rotate(np.concatenate([x.ref, yBottom.ref]), params.a.ref);
      const b1 = rotate(np.concatenate([nx.ref, nyBottom.ref]), params.a.ref);
      pts[`u${i}`] = line(np.concatenate([a0.ref, a1.ref]), { affects: localAffects });
      pts[`l${i}`] = line(np.concatenate([b0.ref, b1.ref]), { affects: localAffects });
      pts[`up${i}`] = point(a1.ref, { affects: localAffects });
      pts[`lp${i}`] = point(b1.ref, { affects: localAffects });
      x = nx;
      yTop = nyTop;
      yBottom = nyBottom;
    }
    return pts;
  };

  const params: ParamState[] = [
    { name: "a", value: np.array([0.3], { dtype: np.float32 }) },
    { name: "b", value: np.array([0.85], { dtype: np.float32 }) },
  ];
  const pObj: Record<string, any> = {};
  for (const p of params) pObj[p.name] = p.value.ref;
  const shapes = renderFn(pObj);
  const u2 = toList(shapes.u2.c);
  const cx = (u2[0] + u2[2]) / 2;
  const cy = (u2[1] + u2[3]) / 2;
  const ldx = u2[2] - u2[0];
  const ldy = u2[3] - u2[1];
  const ll2 = ldx * ldx + ldy * ldy;
  const r = ll2 > 0 ? ((cx - u2[0]) * ldx + (cy - u2[1]) * ldy) / ll2 : 0;
  const before = params.map((p) => toList(p.value)[0]);

  const lossFn = (target: any, coords: any) => {
    const cv = coords.u2;
    const fromPt = cv.ref.slice([0, 2]);
    const toPt = cv.slice([2, 4]);
    const dir = toPt.sub(fromPt.ref);
    const rr = target.ref.slice([2, 3]);
    const predicted = fromPt.add(dir.mul(rr));
    const t = target.slice([0, 2]);
    const d = predicted.sub(t);
    return d.ref.mul(d).sum();
  };

  minimize(params, renderFn as any, lossFn as any, [cx - 1, cy, r], { b: true, dragIter: [1] }, 1);
  const after = params.map((p) => toList(p.value)[0]);
  const deltaA = Math.abs(after[0] - before[0]);
  const deltaB = Math.abs(after[1] - before[1]);
  assert(deltaA < 1e-6, `a should stay fixed on u2 tiny drag, got ${deltaA}`);
  assert(deltaB < 0.05, `b should move only slightly on tiny drag, got ${deltaB}`);
});

run("adaptive-step optimizer converges better with 8 iterations than 6", () => {
  const prev = getG9LineSearchEnabled();
  setG9LineSearchEnabled(false);
  try {
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
  } finally {
    setG9LineSearchEnabled(prev);
  }
});

console.log("All offline regression tests passed.");
