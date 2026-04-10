import { numpy as np, jit, jacfwd } from "@jax-js/jax";

type ShapeArgs = { type: "point" | "line"; c: any } & Record<string, any>;
type ParamState = { name: string; value: any };
type LossFn = (target: any, coords: Record<string, any>) => any;
type RenderFn = (
  params: Record<string, any>,
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
// Coordinate-only render: returns Record<string, Array> (a JsTree).
// The render function returns ShapeArgs with { type, c, ...opts }.
// We strip everything except c, giving jax a pure JsTree to trace through.
// ---------------------------------------------------------------------------

function renderCoords(
  renderFn: RenderFn,
  paramNames: string[],
  paramValues: any[],
): Record<string, any> {
  const obj: Record<string, any> = {};
  for (let i = 0; i < paramNames.length; i++) {
    obj[paramNames[i]] = paramValues[i];
  }
  const shapes = renderFn(obj);
  const coords: Record<string, any> = {};
  for (const [id, shape] of Object.entries(shapes)) {
    coords[id] = shape.c;
  }
  return coords;
}

// ---------------------------------------------------------------------------
// Gradient descent with backtracking line search
// ---------------------------------------------------------------------------

function readVec(params: ParamState[]): number[] {
  const vals: number[] = [];
  for (const p of params) for (const v of toJSArr(p.value)) vals.push(v);
  return vals;
}

function readVecNonDestructive(params: ParamState[]): number[] {
  const vals: number[] = [];
  for (const p of params) {
    const arr = p.value.ref;
    for (const v of toJSArr(arr)) vals.push(v);
  }
  return vals;
}

function writeVec(params: ParamState[], sizes: number[], x: number[]): void {
  let off = 0;
  for (let i = 0; i < params.length; i++) {
    const n = sizes[i];
    params[i].value = np.array(x.slice(off, off + n), { dtype: np.float64 });
    off += n;
  }
}

function buildAffectsMask(
  params: ParamState[],
  sizes: number[],
  affects: Record<string, any> | null | undefined,
): Float64Array {
  const total = sizes.reduce((a, b) => a + b, 0);
  const mask = new Float64Array(total).fill(1);
  if (!affects) return mask;
  let idx = 0;
  for (let pi = 0; pi < params.length; pi++) {
    const n = sizes[pi];
    for (let j = 0; j < n; j++) {
      if (!(params[pi].name in affects)) {
        mask[idx] = 0;
      } else {
        const a = affects[params[pi].name];
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

type CachedJit = { jitLoss: any; jitGrad: any; jitRender: any; targetLen: number };

export function minimize(
  params: ParamState[],
  renderFn: RenderFn,
  lossFn: LossFn,
  target: number[],
  affects: Record<string, any> | null | undefined,
  maxIter = 30,
  cached?: CachedJit,
): CachedJit {
  const sizes = params.map((p) => p.value.shape[0]);
  const paramNames = params.map((p) => p.name);
  const dim = sizes.reduce((a, b) => a + b, 0);
  if (dim === 0) return cached ?? { jitLoss: null, jitGrad: null, jitRender: null, targetLen: 0 };
  const mask = buildAffectsMask(params, sizes, affects);
  let x = readVec(params);

  const tLen = target.length;

  let jitLoss: any, jitGrad: any, jitRender: any;
  if (cached && cached.targetLen === tLen) {
    jitLoss = cached.jitLoss;
    jitGrad = cached.jitGrad;
    jitRender = cached.jitRender;
  } else {
    const splitParams = (combined: any) => {
      const pv: any[] = [];
      let off = tLen;
      for (let i = 0; i < sizes.length; i++) {
        const n = sizes[i];
        const isLast = i === sizes.length - 1;
        pv.push((isLast ? combined : combined.ref).slice([off, off + n]));
        off += n;
      }
      return pv;
    };
    const combinedFn = (combined: any) => {
      const t = combined.ref.slice([0, tLen]);
      const pv = splitParams(combined);
      const coords = renderCoords(renderFn, paramNames, pv);
      return lossFn(t, coords);
    };
    const renderOnlyFn = (flat: any) => {
      const pv: any[] = [];
      let off = 0;
      for (let i = 0; i < sizes.length; i++) {
        const n = sizes[i];
        const isLast = i === sizes.length - 1;
        pv.push((isLast ? flat : flat.ref).slice([off, off + n]));
        off += n;
      }
      return renderCoords(renderFn, paramNames, pv);
    };
    jitLoss = jit(combinedFn);
    jitGrad = jit(jacfwd(combinedFn));
    jitRender = jit(renderOnlyFn);
  }

  for (let it = 0; it < maxIter; it++) {
    const combined = np.array([...target, ...x], { dtype: np.float64 });
    const f0 = toJS(jitLoss(combined.ref));
    const fullG = toJSArr(jitGrad(combined));
    const g = fullG.slice(tLen);
    for (let i = 0; i < dim; i++) g[i] *= mask[i];

    let gnorm2 = 0;
    for (let i = 0; i < dim; i++) gnorm2 += g[i] * g[i];
    if (gnorm2 < 1e-12) break;

    let gmax = 0;
    for (let i = 0; i < dim; i++) gmax = Math.max(gmax, Math.abs(g[i]));

    const x0 = x.slice();
    let lr = Math.min(1.0, 10.0 / gmax);
    let improved = false;

    for (let ls = 0; ls < 20; ls++) {
      const xn = x0.map((v, i) => v - lr * g[i]);
      const lsArr = np.array([...target, ...xn], { dtype: np.float64 });
      const f1 = toJS(jitLoss(lsArr));
      if (f1 < f0 - 1e-4 * lr * gnorm2) {
        x = xn;
        improved = true;
        break;
      }
      lr *= 0.5;
    }

    if (!improved) break;
  }

  writeVec(params, sizes, x);
  return { jitLoss, jitGrad, jitRender, targetLen: tLen };
}

// ---------------------------------------------------------------------------
// Shape SVG elements
// ---------------------------------------------------------------------------

class PointEl {
  container!: SVGSVGElement;
  g9!: G9;
  el!: SVGCircleElement;
  args!: ShapeArgs;
  _cachedCoords!: number[];

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      target: number[],
      affects: Record<string, any> | null | undefined,
      cached?: CachedJit,
    ) => CachedJit,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "circle");
    setAttrs(this.el, { id, r: 5, fill: "#333", cursor: "grab" });
    container.appendChild(this.el);

    addDrag(this.el, (_evt) => {
      const c0 = this._cachedCoords;
      const lossFn: LossFn = (target, coords) => {
        if (!coords[id]) return np.array([0], { dtype: np.float64 });
        const d = coords[id].sub(target);
        return d.ref.mul(d).sum();
      };
      let cached: CachedJit | undefined;
      return (dx, dy) => {
        cached = doMinimize(id, lossFn, [c0[0] + dx, c0[1] + dy], this.args.affects, cached);
      };
    });
  }
  unmount(): void {
    this.container.removeChild(this.el);
  }

  updateCoords(c: number[]): void {
    this._cachedCoords = c;
    this.el.setAttributeNS(null, "cx", String(c[0]));
    this.el.setAttributeNS(null, "cy", String(c[1]));
  }

  updateMeta(args: ShapeArgs): void {
    this.args = args;
    const a: Record<string, any> = {};
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
  _cachedCoords!: number[];

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      target: number[],
      affects: Record<string, any> | null | undefined,
      cached?: CachedJit,
    ) => CachedJit,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "line");
    setAttrs(this.el, { id, stroke: "#000", "stroke-width": 2, cursor: "grab" });
    container.appendChild(this.el);

    addDrag(this.el, (evt) => {
      const c = this._cachedCoords;
      const off = g9.getOffset();
      const cx = evt.clientX - off.left;
      const cy = evt.clientY - off.top;
      const ldx = c[2] - c[0], ldy = c[3] - c[1];
      const pdx = cx - c[0], pdy = cy - c[1];
      const ll = Math.sqrt(ldx * ldx + ldy * ldy) || 1;
      const r = Math.sqrt(pdx * pdx + pdy * pdy) / ll;

      const lossFn: LossFn = (target, coords) => {
        if (!coords[id]) return np.array([0], { dtype: np.float64 });
        const cv = coords[id];
        const fromPt = cv.ref.slice([0, 2]);
        const toPt = cv.slice([2, 4]);
        const dir = toPt.sub(fromPt.ref);
        const predicted = fromPt.add(dir.mul(r));
        const d = predicted.sub(target);
        return d.ref.mul(d).sum();
      };
      let cached: CachedJit | undefined;
      return (dx, dy) => {
        cached = doMinimize(id, lossFn, [cx + dx, cy + dy], this.args.affects, cached);
      };
    });
  }
  unmount(): void {
    this.container.removeChild(this.el);
  }

  updateCoords(c: number[]): void {
    this._cachedCoords = c;
    this.el.setAttributeNS(null, "x1", String(c[0]));
    this.el.setAttributeNS(null, "y1", String(c[1]));
    this.el.setAttributeNS(null, "x2", String(c[2]));
    this.el.setAttributeNS(null, "y2", String(c[3]));
  }

  updateMeta(args: ShapeArgs): void {
    this.args = args;
    const a: Record<string, any> = {};
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

  _minimize(
    id: string,
    lossFn: LossFn,
    target: number[],
    affects: Record<string, any> | null | undefined,
    cached?: CachedJit,
  ): CachedJit {
    const c = minimize(this.params, this.renderFn, lossFn, target, affects, 10, cached);
    this._renderFast(c.jitRender);
    return c;
  }

  _renderFast(jitRender: any): void {
    const x = readVecNonDestructive(this.params);
    const flat = np.array(x, { dtype: np.float64 });
    const coords: Record<string, any> = jitRender(flat);

    const allArrays: any[] = [];
    const ids: string[] = [];
    for (const [id, c] of Object.entries(coords)) {
      ids.push(id);
      allArrays.push(c);
    }
    if (allArrays.length === 0) return;

    const concatenated = np.concatenate(allArrays);
    const allCoords: number[] = typeof concatenated?.dataSync === "function"
      ? Array.from(concatenated.dataSync())
      : toJSArr(concatenated);

    let off = 0;
    for (let i = 0; i < ids.length; i++) {
      const elem = this.elements[ids[i]];
      if (!elem) continue;
      const n = elem instanceof LineEl ? 4 : 2;
      elem.updateCoords(allCoords.slice(off, off + n));
      off += n;
    }
  }

  render(metaOnly = false): void {
    const obj: Record<string, any> = {};
    for (const p of this.params) obj[p.name] = p.value.ref;
    const renderables = this.renderFn(obj);
    if (!renderables) return;

    const ids = new Set(Object.keys(renderables));
    for (const k of Object.keys(this.elements)) {
      if (!ids.has(k)) {
        this.elements[k].unmount();
        delete this.elements[k];
      }
    }

    const entries = Object.entries(renderables);
    let needMount = false;
    for (const [id, shape] of entries) {
      if (!this.elements[id]) {
        needMount = true;
        const elem = shape.type === "line" ? new LineEl() : new PointEl();
        elem.mount(id, this.node, this._minimize.bind(this), this);
        this.elements[id] = elem;
      }
    }

    if (needMount || metaOnly) {
      for (const [id, shape] of entries) {
        this.elements[id].updateMeta(shape);
      }
    }

    const coordArrays: any[] = [];
    const coordIds: string[] = [];
    for (const [id, shape] of entries) {
      coordArrays.push(shape.c);
      coordIds.push(id);
    }

    if (coordArrays.length > 0) {
      const flat = np.concatenate(coordArrays);
      const allCoords: number[] = typeof flat?.dataSync === "function"
        ? Array.from(flat.dataSync())
        : toJSArr(flat);
      let off = 0;
      for (let i = 0; i < coordIds.length; i++) {
        const elem = this.elements[coordIds[i]];
        const n = elem instanceof LineEl ? 4 : 2;
        elem.updateCoords(allCoords.slice(off, off + n));
        off += n;
      }
    }
  }
}

export { np, jit };
