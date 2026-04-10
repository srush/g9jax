import { numpy as np, jvp } from "@jax-js/jax";

type ShapeArgs = { type: "point" | "line"; c: any } & Record<string, any>;
type ParamState = { name: string; value: any };
type LossFn = (...values: any[]) => any;
type RenderFn = (
  params: Record<string, any>,
  targetId: string | null,
) => Record<string, ShapeArgs>;

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

export function point(
  loc: any,
  opts: Record<string, any> = {},
): ShapeArgs {
  return { type: "point", c: loc, ...opts };
}

export function line(
  loc: any,
  opts: Record<string, any> = {},
): ShapeArgs {
  return { type: "line", c: loc, ...opts };
}

// ---------------------------------------------------------------------------
// SVG utilities
// ---------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function setAttrs(el: Element, attrs: Record<string, any>): void {
  for (const [k, v] of Object.entries(attrs)) {
    if (v != null) el.setAttributeNS(null, k, String(v));
  }
}

function toJS(x: any): any {
  if (typeof x === "number") return x;
  if (typeof x?.dataSync === "function") return x.dataSync()[0];
  if (typeof x?.js === "function") return x.js();
  return Number(x);
}

function toJSArr(arr: any): number[] {
  if (typeof arr?.dataSync === "function") return Array.from(arr.dataSync());
  if (typeof arr?.js === "function") {
    const v = arr.js();
    return Array.isArray(v) ? v.flat(Infinity).map(Number) : [Number(v)];
  }
  if (Array.isArray(arr)) return arr.map(Number);
  return [Number(arr)];
}

// ---------------------------------------------------------------------------
// Forward-mode gradient computation
//
// Given f(x1,...,xN) -> scalar, compute df/dxi for each component by calling
// jvp once per component with a one-hot tangent vector (basis direction).
// ---------------------------------------------------------------------------

function forwardGrad(
  scalarFn: LossFn,
  paramArrays: any[],
): Float64Array {
  const sizes = paramArrays.map((p) => p.shape[0]);
  const total = sizes.reduce((a, b) => a + b, 0);
  const grad = new Float64Array(total);

  let col = 0;
  for (let pi = 0; pi < paramArrays.length; pi++) {
    for (let j = 0; j < sizes[pi]; j++) {
      const tangents = paramArrays.map((p, idx) => {
        const buf = Array(sizes[idx]).fill(0);
        if (idx === pi) buf[j] = 1.0;
        return np.array(buf, { dtype: np.float64 });
      });
      const primals = paramArrays.map((p) => p.ref);

      try {
        const [, tOut] = jvp(scalarFn, primals, tangents);
        grad[col] = toJS(tOut);
      } catch (e) {
        console.warn("jvp error for col", col, e);
        grad[col] = 0;
      }
      col++;
    }
  }
  return grad;
}

// ---------------------------------------------------------------------------
// Gradient descent with backtracking line search
// ---------------------------------------------------------------------------

function readVec(params: ParamState[]): number[] {
  const vals = [];
  for (const p of params) for (const v of toJSArr(p.value)) vals.push(v);
  return vals;
}

function writeVec(params: ParamState[], x: number[]): void {
  let off = 0;
  for (const p of params) {
    const n = p.value.shape[0];
    p.value = np.array(x.slice(off, off + n), { dtype: np.float64 });
    off += n;
  }
}

function buildEvalArrays(params: ParamState[], x: number[]): any[] {
  let off = 0;
  return params.map((p) => {
    const n = p.value.shape[0];
    const arr = np.array(x.slice(off, off + n), { dtype: np.float64 });
    off += n;
    return arr;
  });
}

function buildAffectsMask(
  params: ParamState[],
  affects: Record<string, any> | null | undefined,
): Float64Array {
  const total = params.reduce((s, p) => s + p.value.shape[0], 0);
  const mask = new Float64Array(total).fill(1);
  if (!affects) return mask;
  let idx = 0;
  for (const p of params) {
    const n = p.value.shape[0];
    for (let j = 0; j < n; j++) {
      if (!(p.name in affects)) {
        mask[idx] = 0;
      } else {
        const a = affects[p.name];
        if (a !== true) {
          const av = Array.isArray(a) ? a : toJSArr(a);
          if (av[j] === 0) mask[idx] = 0;
        }
      }
      idx++;
    }
  }
  return mask;
}

export function minimize(
  params: ParamState[],
  lossFn: LossFn,
  affects: Record<string, any> | null | undefined,
  maxIter = 30,
): void {
  const dim = params.reduce((s, p) => s + p.value.shape[0], 0);
  if (dim === 0) return;
  const mask = buildAffectsMask(params, affects);
  let x = readVec(params);

  function evalLossAt(values: number[]) {
    const arrays = buildEvalArrays(params, values);
    const loss = lossFn(...arrays.map((arr) => arr.ref));
    const out = toJS(loss);
    arrays.forEach((arr) => arr.dispose?.());
    return out;
  }

  function evalGradAt(values: number[]) {
    const arrays = buildEvalArrays(params, values);
    const g = forwardGrad(lossFn, arrays);
    arrays.forEach((arr) => arr.dispose?.());
    for (let i = 0; i < dim; i++) g[i] *= mask[i];
    return g;
  }

  for (let it = 0; it < maxIter; it++) {
    const f0 = evalLossAt(x);
    const g = evalGradAt(x);

    let gnorm2 = 0;
    for (let i = 0; i < dim; i++) gnorm2 += g[i] * g[i];
    if (gnorm2 < 1e-12) break;

    const x0 = x.slice();
    let lr = 1.0;
    let improved = false;

    for (let ls = 0; ls < 10; ls++) {
      const xn = x0.map((v, i) => v - lr * g[i]);
      const f1 = evalLossAt(xn);
      if (f1 < f0 - 1e-4 * lr * gnorm2) {
        x = xn;
        improved = true;
        break;
      }
      lr *= 0.5;
    }

    if (!improved) {
      break;
    }
  }

  writeVec(params, x);
}

// ---------------------------------------------------------------------------
// Shape SVG elements
// ---------------------------------------------------------------------------

class PointEl {
  container!: SVGSVGElement;
  g9!: G9;
  el!: SVGCircleElement;
  args!: ShapeArgs;

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      affects: Record<string, any> | null | undefined,
    ) => void,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "circle");
    setAttrs(this.el, { id, r: 5, fill: "#333", cursor: "grab" });
    container.appendChild(this.el);

    addDrag(this.el, (_evt) => {
      const c0 = toJSArr(this.args.c);
      return (dx, dy) => {
        const tx = c0[0] + dx;
        const ty = c0[1] + dy;
        doMinimize(
          id,
          (...pv) => {
            const shapes = g9._callRender(pv, id);
            if (!shapes?.[id]) return np.array([0], { dtype: np.float64 });
            const c = shapes[id].c;
            const target = np.array([tx, ty], { dtype: np.float64 });
            const d = c.sub(target);
            return d.ref.mul(d).sum();
          },
          this.args.affects,
        );
      };
    });
  }
  unmount(): void {
    this.container.removeChild(this.el);
  }

  update(args: ShapeArgs): void {
    this.args = args;
    const c = toJSArr(args.c);
    const a: Record<string, any> = { cx: c[0], cy: c[1] };
    if (args.fill) a.fill = args.fill;
    if (args.r) a.r = toJS(args.r);
    if (args.stroke) a.stroke = args.stroke;
    if (args["stroke-width"]) a["stroke-width"] = args["stroke-width"];
    setAttrs(this.el, a);
  }
}

class LineEl {
  container!: SVGSVGElement;
  g9!: G9;
  el!: SVGLineElement;
  args!: ShapeArgs;

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      affects: Record<string, any> | null | undefined,
    ) => void,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "line");
    setAttrs(this.el, { id, stroke: "#000", "stroke-width": 2, cursor: "grab" });
    container.appendChild(this.el);

    addDrag(this.el, (evt) => {
      const c = toJSArr(this.args.c);
      const off = g9.getOffset();
      const cx = evt.clientX - off.left;
      const cy = evt.clientY - off.top;
      const ldx = c[2] - c[0], ldy = c[3] - c[1];
      const pdx = cx - c[0], pdy = cy - c[1];
      const ll = Math.sqrt(ldx * ldx + ldy * ldy) || 1;
      const r = Math.sqrt(pdx * pdx + pdy * pdy) / ll;

      return (dx, dy) => {
        const tx = cx + dx, ty = cy + dy;
        doMinimize(
          id,
          (...pv) => {
            const shapes = g9._callRender(pv, id);
            if (!shapes?.[id]) return np.array([0], { dtype: np.float64 });
            const cv = shapes[id].c;
            const fromPt = cv.ref.slice([0, 2]);
            const toPt = cv.slice([2, 4]);
            const dir = toPt.sub(fromPt.ref);
            const predicted = fromPt.add(dir.mul(r));
            const target = np.array([tx, ty], { dtype: np.float64 });
            const d = predicted.sub(target);
            return d.ref.mul(d).sum();
          },
          this.args.affects,
        );
      };
    });
  }
  unmount(): void {
    this.container.removeChild(this.el);
  }

  update(args: ShapeArgs): void {
    this.args = args;
    const c = toJSArr(args.c);
    const a: Record<string, any> = {
      x1: c[0],
      y1: c[1],
      x2: c[2],
      y2: c[3],
    };
    if (args.stroke) a.stroke = args.stroke;
    if (args["stroke-width"]) a["stroke-width"] = args["stroke-width"];
    if (args["stroke-linecap"]) a["stroke-linecap"] = args["stroke-linecap"];
    setAttrs(this.el, a);
  }
}

// ---------------------------------------------------------------------------
// Mouse / touch drag
// ---------------------------------------------------------------------------

function addDrag(
  el: SVGElement,
  onStartCb: (event: MouseEvent | Touch) => (dx: number, dy: number) => void,
): void {
  function firstPointer(event: MouseEvent | TouchEvent): MouseEvent | Touch {
    return "touches" in event ? event.touches[0] : event;
  }

  function start(e: MouseEvent | TouchEvent) {
    e.stopPropagation();
    e.preventDefault();
    const f = firstPointer(e);
    const onDrag = onStartCb(f);
    const sx = f.clientX, sy = f.clientY;

    function move(ev: MouseEvent | TouchEvent) {
      ev.preventDefault();
      const m = firstPointer(ev);
      onDrag(m.clientX - sx, m.clientY - sy);
    }
    function end(ev: MouseEvent | TouchEvent) {
      ev.preventDefault();
      document.removeEventListener("mousemove", move);
      document.removeEventListener("touchmove", move);
      document.removeEventListener("mouseup", end);
      document.removeEventListener("touchend", end);
      document.removeEventListener("touchcancel", end);
    }
    document.addEventListener("mousemove", move);
    document.addEventListener("touchmove", move, { passive: false });
    document.addEventListener("mouseup", end);
    document.addEventListener("touchend", end);
    document.addEventListener("touchcancel", end);
  }
  el.addEventListener("mousedown", start);
  el.addEventListener("touchstart", start, { passive: false });
}

// ---------------------------------------------------------------------------
// G9 controller
// ---------------------------------------------------------------------------

export class G9 {
  renderFn: RenderFn;
  params: ParamState[];
  elements: Record<string, PointEl | LineEl>;
  node: SVGSVGElement;
  parent: Element | null;
  xAlign: string;
  yAlign: string;
  xOff: number;
  yOff: number;
  _rect: DOMRect | null;

  constructor(
    renderFn: RenderFn,
    initialParams: Record<string, number | number[]>,
  ) {
    this.renderFn = renderFn;
    this.params = [];
    for (const [name, value] of Object.entries(initialParams)) {
      const arr = Array.isArray(value) ? value : [value];
      this.params.push({ name, value: np.array(arr, { dtype: np.float64 }) });
    }
    this.elements = {};
    this.node = document.createElementNS(SVG_NS, "svg");
    this.node.style.width = "100%";
    this.node.style.height = "100%";
    this.node.style.overflow = "visible";
    this.parent = null;
    this.xAlign = "center";
    this.yAlign = "center";
    this.xOff = 0;
    this.yOff = 0;
    this._rect = null;
  }

  getOffset(): { top: number; left: number } {
    const r = this._rect || { top: 0, left: 0 };
    return { top: r.top + this.yOff, left: r.left + this.xOff };
  }

  align(x = "center", y = "center"): this {
    this.xAlign = x;
    this.yAlign = y;
    this.resize();
    return this;
  }

  resize(rerender = true): void {
    if (!this.parent) return;
    const r = this.parent.getBoundingClientRect();
    this._rect = r;
    this.xOff = this.xAlign === "left" ? 0 : this.xAlign === "center" ? r.width / 2 : r.width;
    this.yOff = this.yAlign === "top" ? 0 : this.yAlign === "center" ? r.height / 2 : r.height;
    this.node.setAttribute("viewBox", `${-this.xOff} ${-this.yOff} ${r.width} ${r.height}`);
    if (rerender) this.render();
  }

  insertInto(sel: string | Element): this {
    this.parent = typeof sel === "string" ? document.querySelector(sel) : sel;
    this.parent.textContent = "";
    this.parent.appendChild(this.node);
    const h = () => this.resize();
    window.addEventListener("resize", h);
    this.resize();
    return this;
  }

  _callRender(
    paramValues: any[],
    targetId: string | null,
  ): Record<string, ShapeArgs> {
    const obj: Record<string, any> = {};
    for (let i = 0; i < this.params.length; i++) {
      obj[this.params[i].name] = paramValues[i];
    }
    return this.renderFn(obj, targetId);
  }

  _minimize(
    id: string,
    lossFn: LossFn,
    affects: Record<string, any> | null | undefined,
  ): void {
    minimize(this.params, lossFn, affects, 30);
    this.render();
  }

  render(): void {
    const vals = this.params.map((p) => p.value.ref);
    const renderables = this._callRender(vals, null);
    if (!renderables) return;

    const ids = new Set(Object.keys(renderables));
    for (const k of Object.keys(this.elements)) {
      if (!ids.has(k)) {
        this.elements[k].unmount();
        delete this.elements[k];
      }
    }
    for (const [id, shape] of Object.entries(renderables)) {
      if (!this.elements[id]) {
        const elem = shape.type === "line" ? new LineEl() : new PointEl();
        elem.mount(id, this.node, this._minimize.bind(this), this);
        this.elements[id] = elem;
      }
      this.elements[id].update(shape);
    }
  }
}

export { np };
