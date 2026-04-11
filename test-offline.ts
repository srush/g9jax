import { numpy as np } from "@jax-js/jax";
import { G9, minimize, point, line } from "./src/g9";

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

console.log("All offline regression tests passed.");
