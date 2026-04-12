import { numpy as jaxNp, jit, grad, jacfwd, vmap } from "@jax-js/jax";

function isTracerLike(value: any): boolean {
  return !!value
    && typeof value === "object"
    && "_trace" in value
    && typeof value.dataSync !== "function";
}

function concatWithGradCompat(values: any[]): any {
  if (!Array.isArray(values) || values.length === 0) {
    return jaxNp.concatenate(values as any);
  }
  const anchor = values.find(isTracerLike);
  if (!anchor) return jaxNp.concatenate(values);
  const zero = anchor.ref.sum().mul(0);
  const zeroRef = () => (zero && typeof zero === "object" && "ref" in zero ? (zero as any).ref : zero);
  const lifted = values.map((value) => {
    if (isTracerLike(value)) return value;
    if (value && typeof value === "object" && "ref" in value) {
      // `zero` is reused across elements, so take `.ref` each time.
      return (value as any).ref.add(zeroRef());
    }
    return value;
  });
  return jaxNp.concatenate(lifted);
}

function isReverseModeUnsupported(error: any): boolean {
  if (error?.name === "NonlinearError") return true;
  const message = String(error?.message ?? error);
  return message.includes("Nonlinear operation in backward pass");
}

const np: typeof jaxNp = new Proxy(jaxNp as any, {
  get(target, prop, receiver) {
    if (prop === "concatenate") return concatWithGradCompat;
    const value = Reflect.get(target, prop, receiver);
    return typeof value === "function" ? value.bind(target) : value;
  },
}) as typeof jaxNp;

type ShapeArgs = { type: "point" | "line"; c: any } & Record<string, any>;
type RenderShapeMap = Record<string, ShapeArgs>;
type SecondaryScores = Record<string, any>;
type RenderOutput = RenderShapeMap | {
  shapes: RenderShapeMap;
  secondary?: SecondaryScores;
};
type ParamState = { name: string; value: any };
type LossFn = (target: any, coords: Record<string, any>) => any;
type RenderFn = (
  params: Record<string, any>,
) => RenderOutput;

type RuntimeStats = {
  minimizeCalls: number;
  jitBuilds: number;
  jitCacheHits: number;
  warmupBuilds: number;
};

const runtimeStats: RuntimeStats = {
  minimizeCalls: 0,
  jitBuilds: 0,
  jitCacheHits: 0,
  warmupBuilds: 0,
};

export function getG9RuntimeStats(): RuntimeStats {
  return { ...runtimeStats };
}

export function resetG9RuntimeStats(): void {
  runtimeStats.minimizeCalls = 0;
  runtimeStats.jitBuilds = 0;
  runtimeStats.jitCacheHits = 0;
  runtimeStats.warmupBuilds = 0;
}

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
const MOBILE_POINT_RADIUS_SCALE = 1.35;
const DRAG_ITER_ADAPTIVE = 10;
const DRAG_ITER_LINE_SEARCH = 5;
const LINE_SEARCH_TRIALS = 12;
const DRAG_RENDER_EVERY = 2;
let activeDragCount = 0;
let dragDebugEnabled = false;
let lineSearchEnabled = true;
const liveG9Instances = new Set<G9>();
const debugLossStats = {
  sum: 0,
  count: 0,
  last: 0,
};

function setAttrs(el: Element, attrs: Record<string, any>): void {
  for (const [k, v] of Object.entries(attrs)) {
    if (v != null) el.setAttributeNS(null, k, String(v));
  }
}

function hasCoarsePointer(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(pointer: coarse)").matches;
}

function scalePointRadius(radius: number): number {
  return hasCoarsePointer() ? radius * MOBILE_POINT_RADIUS_SCALE : radius;
}

function beginGlobalDrag(): void {
  if (typeof document === "undefined") return;
  activeDragCount += 1;
  if (activeDragCount === 1) {
    const root = (document as any).documentElement;
    root?.classList?.add("g9-dragging");
  }
}

function endGlobalDrag(): void {
  if (typeof document === "undefined") return;
  activeDragCount = Math.max(0, activeDragCount - 1);
  if (activeDragCount === 0) {
    const root = (document as any).documentElement;
    root?.classList?.remove("g9-dragging");
  }
}

export function setG9DragDebugEnabled(enabled: boolean): void {
  dragDebugEnabled = enabled;
  if (enabled) {
    debugLossStats.sum = 0;
    debugLossStats.count = 0;
    debugLossStats.last = 0;
  }
  for (const g9 of liveG9Instances) g9.syncDebugVisibility();
}

export function getG9DragDebugEnabled(): boolean {
  return dragDebugEnabled;
}

export function setG9LineSearchEnabled(enabled: boolean): void {
  lineSearchEnabled = enabled;
}

export function getG9LineSearchEnabled(): boolean {
  return lineSearchEnabled;
}

export function getG9DebugLossStats(): { average: number; count: number; last: number } {
  const average = debugLossStats.count > 0 ? debugLossStats.sum / debugLossStats.count : 0;
  return {
    average,
    count: debugLossStats.count,
    last: debugLossStats.last,
  };
}

type LossListener = (loss: number | null) => void;
const g9LossListeners = new Set<LossListener>();

export function onG9LossUpdate(listener: LossListener): () => void {
  g9LossListeners.add(listener);
  return () => g9LossListeners.delete(listener);
}

function emitG9Loss(loss: number | null): void {
  for (const listener of g9LossListeners) listener(loss);
}

function markDraggable(el: SVGElement): void {
  el.style.touchAction = "none";
  el.style.userSelect = "none";
  (el.style as any).webkitUserSelect = "none";
  (el.style as any).webkitTouchCallout = "none";
  (el.style as any).webkitTapHighlightColor = "transparent";
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

function evalLoss(jitLoss: any, targetLen: number, x: number[], combinedBuffer: Float32Array): number {
  for (let i = 0; i < x.length; i++) combinedBuffer[targetLen + i] = x[i];
  return Number(toJS(jitLoss(np.array(combinedBuffer, { dtype: np.float32 }))));
}

function emitOptimizeLoss(containerId: string | null, loss: number): void {
  if (!containerId || typeof document === "undefined" || !Number.isFinite(loss)) return;
  document.dispatchEvent(new CustomEvent("g9:opt-loss", {
    detail: { containerId, loss },
  }));
}

// ---------------------------------------------------------------------------
// Coordinate-only render: returns Record<string, Array> (a JsTree).
// The render function returns ShapeArgs with { type, c, ...opts }.
// We strip everything except c, giving jax a pure JsTree to trace through.
// ---------------------------------------------------------------------------

function isShapeArgs(value: any): value is ShapeArgs {
  return !!value
    && typeof value === "object"
    && "c" in value
    && (value.type === "point" || value.type === "line");
}

function parseRenderOutput(output: RenderOutput): {
  shapes: RenderShapeMap;
  secondary: SecondaryScores | null;
} {
  if (!output || typeof output !== "object") {
    throw new TypeError("Render function must return an object.");
  }
  if ("shapes" in output && !isShapeArgs((output as any).shapes)) {
    const rawShapes = (output as any).shapes;
    if (!rawShapes || typeof rawShapes !== "object") {
      throw new TypeError("Render output 'shapes' must be an object.");
    }
    const shapes: RenderShapeMap = {};
    for (const [id, shape] of Object.entries(rawShapes)) {
      if (!isShapeArgs(shape)) throw new TypeError(`Render entry '${id}' is not a valid shape.`);
      shapes[id] = shape;
    }
    const rawSecondary = (output as any).secondary;
    const secondary = rawSecondary && typeof rawSecondary === "object"
      ? rawSecondary as SecondaryScores
      : null;
    return { shapes, secondary };
  }

  const shapes: RenderShapeMap = {};
  let secondary: SecondaryScores | null = null;
  for (const [id, value] of Object.entries(output as Record<string, any>)) {
    if (id === "secondary" && value && typeof value === "object" && !isShapeArgs(value)) {
      secondary = value as SecondaryScores;
      continue;
    }
    if (!isShapeArgs(value)) {
      throw new TypeError(`Render entry '${id}' is not a valid shape.`);
    }
    shapes[id] = value;
  }
  return { shapes, secondary };
}

function sumSecondaryScores(loss: any, secondary: SecondaryScores | null): any {
  if (!secondary) return loss;
  let total = loss;
  for (const value of Object.values(secondary)) {
    if (value == null) continue;
    let scalar: any;
    if (typeof value === "number") {
      scalar = np.array([value], { dtype: np.float32 }).sum();
    } else if (value && typeof value === "object" && typeof (value as any).sum === "function") {
      scalar = (value as any).sum();
    } else if (value && typeof value === "object" && "ref" in (value as any) && typeof (value as any).ref?.sum === "function") {
      scalar = (value as any).ref.sum();
    } else {
      scalar = np.array([Number(value)], { dtype: np.float32 }).sum();
    }
    total = total.ref.add(scalar);
  }
  return total;
}

function renderEval(
  renderFn: RenderFn,
  paramNames: string[],
  paramValues: any[],
): { coords: Record<string, any>; secondary: SecondaryScores | null } {
  const obj: Record<string, any> = {};
  for (let i = 0; i < paramNames.length; i++) {
    obj[paramNames[i]] = paramValues[i];
  }
  const { shapes, secondary } = parseRenderOutput(renderFn(obj));
  const coords: Record<string, any> = {};
  for (const [id, shape] of Object.entries(shapes)) {
    coords[id] = shape.c;
  }
  return { coords, secondary };
}

// ---------------------------------------------------------------------------
// Gradient descent with backtracking line search
// ---------------------------------------------------------------------------

function readVec(params: ParamState[]): number[] {
  const vals: number[] = [];
  for (const p of params) for (const v of toJSArr(p.value.ref)) vals.push(v);
  return vals;
}


function writeVec(params: ParamState[], sizes: number[], x: number[]): void {
  let off = 0;
  for (let i = 0; i < params.length; i++) {
    const n = sizes[i];
    params[i].value = np.array(x.slice(off, off + n), { dtype: np.float32 });
    off += n;
  }
}

function buildAffectsMask(
  params: ParamState[],
  sizes: number[],
  affects: Record<string, any> | null | undefined,
): Float64Array | null {
  if (!affects) return null;
  const total = sizes.reduce((a, b) => a + b, 0);
  const mask = new Float64Array(total).fill(1);
  let hasRestrictions = false;
  let idx = 0;
  for (let pi = 0; pi < params.length; pi++) {
    const n = sizes[pi];
    for (let j = 0; j < n; j++) {
      if (!(params[pi].name in affects)) {
        mask[idx] = 0;
        hasRestrictions = true;
      } else {
        const a = affects[params[pi].name];
        if (a !== true) {
          const av = Array.isArray(a) ? a : toJSArr(a);
          if (av[j] === 0) {
            mask[idx] = 0;
            hasRestrictions = true;
          }
        }
      }
      idx++;
    }
  }
  return hasRestrictions ? mask : null;
}

type CachedJit = {
  jitLoss: any;
  jitGrad: any;
  jitGradFallback: any;
  useGradFallback: boolean;
  jitRender: any;
  jitBatchLoss: any;
  renderIds: string[];
  targetLen: number;
  lastX: number[];
  lastLoss: number;
  lastConverged: boolean;
  lastHitLimit: boolean;
  affectsRef: Record<string, any> | null | undefined;
  affectsMask: Float64Array | null;
  combinedBuffer: Float32Array;
  trialBuffer: Float32Array;
  trialBatchBuffer: Float32Array;
  x0Buffer: Float64Array;
  gBuffer: Float64Array;
  renderBuffer: Float32Array;
  velocityBuffer: Float64Array;
  bfgsH: Float64Array;
  bfgsStep: Float64Array;
  bfgsGradNext: Float64Array;
  bfgsY: Float64Array;
  bfgsHy: Float64Array;
};

export function minimize(
  params: ParamState[],
  renderFn: RenderFn,
  lossFn: LossFn,
  target: number[],
  affects: Record<string, any> | null | undefined,
  maxIter = 30,
  cached?: CachedJit,
): CachedJit {
  runtimeStats.minimizeCalls += 1;
  const sizes = params.map((p) => p.value.shape[0]);
  const paramNames = params.map((p) => p.name);
  const dim = sizes.reduce((a, b) => a + b, 0);
  if (dim === 0) {
    if (cached) return cached;
    return {
      jitLoss: null,
      jitGrad: null,
      jitGradFallback: null,
      useGradFallback: false,
      jitRender: null,
      jitBatchLoss: null,
      renderIds: [],
      targetLen: 0,
      lastLoss: 0,
      lastConverged: false,
      lastHitLimit: false,
      lastX: [],
      affectsRef: null,
      affectsMask: null,
      combinedBuffer: null,
      trialBuffer: null,
      trialBatchBuffer: null,
      x0Buffer: null,
      gBuffer: null,
      renderBuffer: null,
      velocityBuffer: null,
      bfgsH: null,
      bfgsStep: null,
      bfgsGradNext: null,
      bfgsY: null,
      bfgsHy: null,
    };
  }

  const tLen = target.length;
  const totalLen = tLen + dim;
  const trialBatchLen = totalLen * LINE_SEARCH_TRIALS;
  const combinedBuffer = cached?.combinedBuffer && cached.combinedBuffer.length === totalLen
    ? cached.combinedBuffer
    : new Float32Array(totalLen);
  const trialBuffer = cached?.trialBuffer && cached.trialBuffer.length === totalLen
    ? cached.trialBuffer
    : new Float32Array(totalLen);
  const trialBatchBuffer = cached?.trialBatchBuffer && cached.trialBatchBuffer.length === trialBatchLen
    ? cached.trialBatchBuffer
    : new Float32Array(trialBatchLen);
  const x0Buffer = cached?.x0Buffer && cached.x0Buffer.length === dim
    ? cached.x0Buffer
    : new Float64Array(dim);
  const gBuffer = cached?.gBuffer && cached.gBuffer.length === dim
    ? cached.gBuffer
    : new Float64Array(dim);
  const velocityBuffer = cached?.velocityBuffer && cached.velocityBuffer.length === dim
    ? cached.velocityBuffer
    : new Float64Array(dim);
  const renderBuffer = cached?.renderBuffer && cached.renderBuffer.length === dim
    ? cached.renderBuffer
    : new Float32Array(dim);
  const bfgsH = cached?.bfgsH && cached.bfgsH.length === dim * dim
    ? cached.bfgsH
    : new Float64Array(dim * dim);
  const bfgsStep = cached?.bfgsStep && cached.bfgsStep.length === dim
    ? cached.bfgsStep
    : new Float64Array(dim);
  const bfgsGradNext = cached?.bfgsGradNext && cached.bfgsGradNext.length === dim
    ? cached.bfgsGradNext
    : new Float64Array(dim);
  const bfgsY = cached?.bfgsY && cached.bfgsY.length === dim
    ? cached.bfgsY
    : new Float64Array(dim);
  const bfgsHy = cached?.bfgsHy && cached.bfgsHy.length === dim
    ? cached.bfgsHy
    : new Float64Array(dim);

  let jitLoss: any, jitGrad: any, jitGradFallback: any, jitRender: any, jitBatchLoss: any;
  let useGradFallback = cached?.useGradFallback ?? false;
  let renderIds: string[] = cached?.renderIds ?? [];
  if (cached && cached.targetLen === tLen) {
    runtimeStats.jitCacheHits += 1;
    jitLoss = cached.jitLoss;
    jitGrad = cached.jitGrad;
    jitGradFallback = cached.jitGradFallback ?? cached.jitGrad;
    useGradFallback = cached.useGradFallback ?? false;
    jitRender = cached.jitRender;
    jitBatchLoss = cached.jitBatchLoss;
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
      const { coords, secondary } = renderEval(renderFn, paramNames, pv);
      return sumSecondaryScores(lossFn(t, coords), secondary);
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
      const { coords } = renderEval(renderFn, paramNames, pv);
      renderIds = Object.keys(coords);
      const arrays: any[] = [];
      for (const c of Object.values(coords)) arrays.push(c);
      return np.concatenate(arrays);
    };
    jitLoss = jit(combinedFn);
    jitGrad = jit(grad(combinedFn));
    jitGradFallback = jit(jacfwd(combinedFn));
    useGradFallback = false;
    jitRender = jit(renderOnlyFn);
    jitBatchLoss = jit(vmap(combinedFn, [0]));
    runtimeStats.jitBuilds += 1;
    const probeX = readVec(params);
    jitRender(np.array(probeX, { dtype: np.float32 }));
  }

  if (maxIter === 0) {
    return {
      jitLoss,
      jitGrad,
      jitGradFallback,
      useGradFallback,
      jitRender,
      jitBatchLoss,
      renderIds,
      targetLen: tLen,
      lastLoss: cached?.lastLoss ?? 0,
      lastConverged: cached?.lastConverged ?? false,
      lastHitLimit: cached?.lastHitLimit ?? false,
      lastX: [],
      affectsRef: null,
      affectsMask: null,
      combinedBuffer,
      trialBuffer,
      trialBatchBuffer,
      x0Buffer,
      gBuffer,
      renderBuffer,
      velocityBuffer,
      bfgsH,
      bfgsStep,
      bfgsGradNext,
      bfgsY,
      bfgsHy,
    };
  }

  const affectsMask = cached && cached.affectsRef === affects
    ? cached.affectsMask
    : buildAffectsMask(params, sizes, affects);
  let x = readVec(params);
  for (let i = 0; i < tLen; i++) {
    const tv = target[i];
    combinedBuffer[i] = tv;
    trialBuffer[i] = tv;
  }
  for (let ls = 0; ls < LINE_SEARCH_TRIALS; ls++) {
    const row = ls * totalLen;
    for (let i = 0; i < tLen; i++) trialBatchBuffer[row + i] = target[i];
  }

  if (lineSearchEnabled) {
    bfgsH.fill(0);
    for (let i = 0; i < dim; i++) bfgsH[i * dim + i] = 1;
  }

  let converged = false;
  let iterationsUsed = 0;
  for (let it = 0; it < maxIter; it++) {
    iterationsUsed = it + 1;
    for (let i = 0; i < dim; i++) combinedBuffer[tLen + i] = x[i];
    const combined = np.array(combinedBuffer, { dtype: np.float32 });
    let fullG: number[];
    try {
      fullG = toJSArr((useGradFallback ? jitGradFallback : jitGrad)(combined));
    } catch (error) {
      if (useGradFallback || !isReverseModeUnsupported(error)) throw error;
      useGradFallback = true;
      fullG = toJSArr(jitGradFallback(combined));
    }
    if (affectsMask) {
      for (let i = 0; i < dim; i++) gBuffer[i] = fullG[tLen + i] * affectsMask[i];
    } else {
      for (let i = 0; i < dim; i++) gBuffer[i] = fullG[tLen + i];
    }

    let gnorm2 = 0;
    for (let i = 0; i < dim; i++) gnorm2 += gBuffer[i] * gBuffer[i];
    if (gnorm2 < 1e-12) {
      converged = true;
      break;
    }

    let gmax = 0;
    for (let i = 0; i < dim; i++) gmax = Math.max(gmax, Math.abs(gBuffer[i]));

    if (lineSearchEnabled) {
      for (let i = 0; i < dim; i++) x0Buffer[i] = x[i];
      const f0 = evalLoss(jitLoss, tLen, x, combinedBuffer);
      let descentDot = 0;
      let stepNorm2 = 0;
      const maxBfgsCoordStep = 18.0;
      const maxBfgsStepNorm = 24.0;
      for (let i = 0; i < dim; i++) {
        const row = i * dim;
        let projected = 0;
        for (let j = 0; j < dim; j++) projected += bfgsH[row + j] * gBuffer[j];
        const mask = affectsMask ? affectsMask[i] : 1;
        const rawStep = -projected * mask;
        const step = rawStep > maxBfgsCoordStep
          ? maxBfgsCoordStep
          : rawStep < -maxBfgsCoordStep
            ? -maxBfgsCoordStep
            : rawStep;
        bfgsStep[i] = step;
        descentDot += gBuffer[i] * step;
        stepNorm2 += step * step;
      }
      if (stepNorm2 > maxBfgsStepNorm * maxBfgsStepNorm) {
        const scale = maxBfgsStepNorm / (Math.sqrt(stepNorm2) + 1e-12);
        descentDot = 0;
        stepNorm2 = 0;
        for (let i = 0; i < dim; i++) {
          bfgsStep[i] *= scale;
          descentDot += gBuffer[i] * bfgsStep[i];
          stepNorm2 += bfgsStep[i] * bfgsStep[i];
        }
      }

      if (!(descentDot < 0) || stepNorm2 < 1e-20) {
        bfgsH.fill(0);
        for (let i = 0; i < dim; i++) bfgsH[i * dim + i] = 1;
        descentDot = -gnorm2;
        stepNorm2 = 0;
        for (let i = 0; i < dim; i++) {
          const fallback = -gBuffer[i];
          const clipped = fallback > maxBfgsCoordStep
            ? maxBfgsCoordStep
            : fallback < -maxBfgsCoordStep
              ? -maxBfgsCoordStep
              : fallback;
          bfgsStep[i] = clipped;
          stepNorm2 += clipped * clipped;
        }
      }

      let alpha = Math.min(1.0, 8.0 / (Math.sqrt(stepNorm2) + 1e-6));
      let fillAlpha = alpha;
      for (let ls = 0; ls < LINE_SEARCH_TRIALS; ls++) {
        const row = ls * totalLen + tLen;
        for (let i = 0; i < dim; i++) {
          trialBatchBuffer[row + i] = x0Buffer[i] + fillAlpha * bfgsStep[i];
        }
        fillAlpha *= 0.5;
      }

      const batchedTrialLosses = toJSArr(
        jitBatchLoss(np.array(trialBatchBuffer, { dtype: np.float32 }).reshape([LINE_SEARCH_TRIALS, totalLen])),
      );
      let accepted = false;
      let acceptedAlpha = alpha;
      for (let ls = 0; ls < LINE_SEARCH_TRIALS; ls++) {
        const fTrial = batchedTrialLosses[ls];
        if (Number.isFinite(fTrial) && fTrial <= f0 + 1e-4 * acceptedAlpha * descentDot) {
          for (let i = 0; i < dim; i++) x[i] = x0Buffer[i] + acceptedAlpha * bfgsStep[i];
          accepted = true;
          break;
        }
        acceptedAlpha *= 0.5;
      }

      if (!accepted) {
        bfgsH.fill(0);
        for (let i = 0; i < dim; i++) bfgsH[i * dim + i] = 1;
        const fallbackStep = Math.min(0.12, 2.0 / (Math.sqrt(gnorm2) + 1e-6));
        for (let i = 0; i < dim; i++) x[i] = x0Buffer[i] - fallbackStep * gBuffer[i];
        continue;
      }

      for (let i = 0; i < dim; i++) combinedBuffer[tLen + i] = x[i];
      const nextCombined = np.array(combinedBuffer, { dtype: np.float32 });
      let nextGradFull: number[];
      try {
        nextGradFull = toJSArr((useGradFallback ? jitGradFallback : jitGrad)(nextCombined));
      } catch (error) {
        if (useGradFallback || !isReverseModeUnsupported(error)) throw error;
        useGradFallback = true;
        nextGradFull = toJSArr(jitGradFallback(nextCombined));
      }
      if (affectsMask) {
        for (let i = 0; i < dim; i++) bfgsGradNext[i] = nextGradFull[tLen + i] * affectsMask[i];
      } else {
        for (let i = 0; i < dim; i++) bfgsGradNext[i] = nextGradFull[tLen + i];
      }

      let ys = 0;
      let sNorm2 = 0;
      let yNorm2 = 0;
      for (let i = 0; i < dim; i++) {
        const si = x[i] - x0Buffer[i];
        bfgsY[i] = bfgsGradNext[i] - gBuffer[i];
        ys += bfgsY[i] * si;
        sNorm2 += si * si;
        yNorm2 += bfgsY[i] * bfgsY[i];
      }

      const curvatureFloor = 1e-6 * Math.sqrt(sNorm2 * yNorm2 + 1e-30);
      if (!(ys > Math.max(1e-12, curvatureFloor))) {
        bfgsH.fill(0);
        for (let i = 0; i < dim; i++) bfgsH[i * dim + i] = 1;
        continue;
      }

      for (let i = 0; i < dim; i++) {
        const row = i * dim;
        let hy = 0;
        for (let j = 0; j < dim; j++) hy += bfgsH[row + j] * bfgsY[j];
        bfgsHy[i] = hy;
      }
      let yHy = 0;
      for (let i = 0; i < dim; i++) yHy += bfgsY[i] * bfgsHy[i];
      const coeff = (ys + yHy) / (ys * ys);
      for (let i = 0; i < dim; i++) {
        const row = i * dim;
        const si = x[i] - x0Buffer[i];
        for (let j = 0; j < dim; j++) {
          const sj = x[j] - x0Buffer[j];
          bfgsH[row + j] += coeff * si * sj - (bfgsHy[i] * sj + si * bfgsHy[j]) / ys;
        }
      }
      continue;
    }

    const lr = Math.min(0.35, 8.0 / (gmax + 1e-6));
    const momentum = 0.7;
    const maxStep = 18.0;
    for (let i = 0; i < dim; i++) {
      const v = momentum * velocityBuffer[i] - lr * gBuffer[i];
      velocityBuffer[i] = v;
      const delta = v > maxStep ? maxStep : v < -maxStep ? -maxStep : v;
      x[i] += delta;
    }
  }

  const hitLimit = maxIter > 0 && !converged && iterationsUsed >= maxIter;
  const loss = evalLoss(jitLoss, tLen, x, combinedBuffer);
  if (dragDebugEnabled && Number.isFinite(loss)) {
    debugLossStats.sum += loss;
    debugLossStats.count += 1;
    debugLossStats.last = loss;
  }

  writeVec(params, sizes, x);
  return {
    jitLoss,
    jitGrad,
    jitGradFallback,
    useGradFallback,
    jitRender,
    jitBatchLoss,
    renderIds,
    targetLen: tLen,
    lastLoss: Number.isFinite(loss) ? loss : 0,
    lastConverged: converged,
    lastHitLimit: hitLimit,
    lastX: x,
    affectsRef: affects,
    affectsMask,
    combinedBuffer,
    trialBuffer,
    trialBatchBuffer,
    x0Buffer,
    gBuffer,
    renderBuffer,
    velocityBuffer,
    bfgsH,
    bfgsStep,
    bfgsGradNext,
    bfgsY,
    bfgsHy,
  };
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
  _cached!: CachedJit;

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      target: number[],
      affects: Record<string, any> | null | undefined,
      forceRender: boolean,
      cached?: CachedJit,
    ) => CachedJit,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "circle");
    setAttrs(this.el, { id, r: scalePointRadius(5), fill: "#333", cursor: "grab" });
    markDraggable(this.el);
    container.appendChild(this.el);

    const lossFn: LossFn = (target, coords) => {
      if (!coords[id]) return np.array([0], { dtype: np.float32 });
      const d = coords[id].sub(target);
      return d.ref.mul(d).sum();
    };
    this._cached = g9._warmup(lossFn, [0, 0]);

    addDrag(this.el, (_evt) => {
      this._cached.lastConverged = false;
      this._cached.lastHitLimit = false;
      const c0 = this._cachedCoords.slice();
      return {
        drag: (dx, dy) => {
          const pullX = c0[0] + dx;
          const pullY = c0[1] + dy;
          this._cached = doMinimize(id, lossFn, [pullX, pullY], this.args.affects, false, this._cached);
          const model = this._cachedCoords;
          this.g9.setDragDebug([pullX, pullY], [model[0], model[1]]);
        },
        end: () => {
          const c = this._cachedCoords;
          this._cached = doMinimize(id, lossFn, [c[0], c[1]], this.args.affects, true, this._cached);
          this.g9.clearDragDebug();
        },
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
    if (args.r != null) a.r = scalePointRadius(toJS(args.r));
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
  _cached!: CachedJit;

  mount(
    id: string,
    container: SVGSVGElement,
    doMinimize: (
      id: string,
      lossFn: LossFn,
      target: number[],
      affects: Record<string, any> | null | undefined,
      forceRender: boolean,
      cached?: CachedJit,
    ) => CachedJit,
    g9: G9,
  ): void {
    this.container = container;
    this.g9 = g9;
    this.el = document.createElementNS(SVG_NS, "line");
    setAttrs(this.el, { id, stroke: "#000", "stroke-width": 2, cursor: "grab" });
    markDraggable(this.el);
    container.appendChild(this.el);

    const lossFn: LossFn = (target, coords) => {
      if (!coords[id]) return np.array([0], { dtype: np.float32 });
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
    this._cached = g9._warmup(lossFn, [0, 0, 0]);

    addDrag(this.el, (evt) => {
      this._cached.lastConverged = false;
      this._cached.lastHitLimit = false;
      const c = this._cachedCoords.slice();
      const off = g9.getOffset();
      const cx = evt.clientX - off.left;
      const cy = evt.clientY - off.top;
      const ldx = c[2] - c[0], ldy = c[3] - c[1];
      const pdx = cx - c[0], pdy = cy - c[1];
      const ll2 = ldx * ldx + ldy * ldy;
      const r = ll2 > 0 ? (pdx * ldx + pdy * ldy) / ll2 : 0;
      let latestPullX = cx;
      let latestPullY = cy;

      return {
        drag: (dx, dy) => {
          latestPullX = cx + dx;
          latestPullY = cy + dy;
          this._cached = doMinimize(id, lossFn, [latestPullX, latestPullY, r], this.args.affects, false, this._cached);
          const model = this._cachedCoords;
          const targetX = model[0] + (model[2] - model[0]) * r;
          const targetY = model[1] + (model[3] - model[1]) * r;
          this.g9.setDragDebug([latestPullX, latestPullY], [targetX, targetY]);
        },
        end: () => {
          const c = this._cachedCoords;
          const ldx = c[2] - c[0], ldy = c[3] - c[1];
          const ll2 = ldx * ldx + ldy * ldy;
          const rr = ll2 > 0 ? ((latestPullX - c[0]) * ldx + (latestPullY - c[1]) * ldy) / ll2 : r;
          this._cached = doMinimize(id, lossFn, [latestPullX, latestPullY, rr], this.args.affects, true, this._cached);
          this.g9.clearDragDebug();
        },
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

type DragSession = {
  drag: (dx: number, dy: number) => void;
  end?: () => void;
  shouldContinue?: () => boolean;
};

function addDrag(
  el: SVGElement,
  onStartCb: (event: MouseEvent | Touch) => DragSession,
): void {
  const scheduleFrame = (cb: () => void): number => {
    if (typeof requestAnimationFrame === "function") {
      return requestAnimationFrame(() => cb());
    }
    return setTimeout(cb, 0) as unknown as number;
  };
  const cancelFrame = (id: number): void => {
    if (typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(id);
    } else {
      clearTimeout(id);
    }
  };

  function firstPointer(event: MouseEvent | TouchEvent): MouseEvent | Touch {
    return "touches" in event ? event.touches[0] : event;
  }

  function start(e: MouseEvent | TouchEvent) {
    e.stopPropagation();
    if (e.cancelable) e.preventDefault();
    beginGlobalDrag();
    el.classList.add("g9-active-drag");
    const f = firstPointer(e);
    let session: DragSession;
    try {
      session = onStartCb(f);
    } catch (error) {
      el.classList.remove("g9-active-drag");
      endGlobalDrag();
      throw error;
    }
    const onDrag = session.drag;
    const shouldContinue = session.shouldContinue;
    const sx = f.clientX, sy = f.clientY;
    let latestDx = 0;
    let latestDy = 0;
    let rafId = 0;

    const tick = () => {
      if (shouldContinue && !shouldContinue()) return;
      onDrag(latestDx, latestDy);
      rafId = scheduleFrame(tick);
    };
    // Immediate first solve gives instant visual feedback on touch-down.
    onDrag(0, 0);
    rafId = scheduleFrame(tick);

    function move(ev: MouseEvent | TouchEvent) {
      ev.stopPropagation();
      if (ev.cancelable) ev.preventDefault();
      const m = firstPointer(ev);
      latestDx = m.clientX - sx;
      latestDy = m.clientY - sy;
    }
    function end(ev: MouseEvent | TouchEvent) {
      ev.stopPropagation();
      if (ev.cancelable) ev.preventDefault();
      if (rafId !== 0) {
        cancelFrame(rafId);
        rafId = 0;
      }
      onDrag(latestDx, latestDy);
      document.removeEventListener("mousemove", move);
      document.removeEventListener("touchmove", move);
      document.removeEventListener("mouseup", end);
      document.removeEventListener("touchend", end);
      document.removeEventListener("touchcancel", end);
      el.classList.remove("g9-active-drag");
      session.end?.();
      endGlobalDrag();
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
  containerId: string | null;
  xAlign: string;
  yAlign: string;
  xOff: number;
  yOff: number;
  _rect: DOMRect | null;
  _debugPullEl: SVGCircleElement | null;
  _debugTargetEl: SVGCircleElement | null;
  _dragRenderCounter: number;

  constructor(
    renderFn: RenderFn,
    initialParams: Record<string, number | number[]>,
  ) {
    this.renderFn = renderFn;
    this.params = [];
    for (const [name, value] of Object.entries(initialParams)) {
      const arr = Array.isArray(value) ? value : [value];
      this.params.push({ name, value: np.array(arr, { dtype: np.float32 }) });
    }
    this.elements = {};
    this.node = document.createElementNS(SVG_NS, "svg");
    this.node.style.width = "100%";
    this.node.style.height = "100%";
    this.node.style.overflow = "visible";
    this.parent = null;
    this.containerId = null;
    this.xAlign = "center";
    this.yAlign = "center";
    this.xOff = 0;
    this.yOff = 0;
    this._rect = null;
    this._debugPullEl = null;
    this._debugTargetEl = null;
    this._dragRenderCounter = 0;
    liveG9Instances.add(this);
  }

  _ensureDebugMarkers(): void {
    if (this._debugPullEl && this._debugTargetEl) return;
    const pull = document.createElementNS(SVG_NS, "circle");
    const target = document.createElementNS(SVG_NS, "circle");
    setAttrs(pull, {
      r: scalePointRadius(5),
      fill: "#fb923c",
      stroke: "#ffffff",
      "stroke-width": 1.5,
      "pointer-events": "none",
      opacity: 0.95,
    });
    setAttrs(target, {
      r: scalePointRadius(4),
      fill: "#ec4899",
      stroke: "#ffffff",
      "stroke-width": 1.5,
      "pointer-events": "none",
      opacity: 0.95,
    });
    pull.style.display = "none";
    target.style.display = "none";
    this.node.appendChild(pull);
    this.node.appendChild(target);
    this._debugPullEl = pull;
    this._debugTargetEl = target;
  }

  syncDebugVisibility(): void {
    if (!this._debugPullEl || !this._debugTargetEl) return;
    if (dragDebugEnabled) return;
    this._debugPullEl.style.display = "none";
    this._debugTargetEl.style.display = "none";
  }

  setDragDebug(pull: [number, number], target: [number, number]): void {
    if (!dragDebugEnabled) return;
    this._ensureDebugMarkers();
    if (!this._debugPullEl || !this._debugTargetEl) return;
    this._debugPullEl.style.display = "";
    this._debugTargetEl.style.display = "";
    setAttrs(this._debugPullEl, { cx: pull[0], cy: pull[1] });
    setAttrs(this._debugTargetEl, { cx: target[0], cy: target[1] });
  }

  clearDragDebug(): void {
    if (!this._debugPullEl || !this._debugTargetEl) return;
    this._debugPullEl.style.display = "none";
    this._debugTargetEl.style.display = "none";
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
    this.containerId = typeof sel === "string" && sel.startsWith("#") ? sel.slice(1) : null;
    this.parent.textContent = "";
    this.parent.appendChild(this.node);
    const h = () => this.resize();
    window.addEventListener("resize", h);
    this.resize();
    this.syncDebugVisibility();
    return this;
  }

  destroy(): void {
    liveG9Instances.delete(this);
    this.clearDragDebug();
  }

  _warmup(lossFn: LossFn, target: number[]): CachedJit {
    const sizes = this.params.map((p) => p.value.shape[0]);
    const paramNames = this.params.map((p) => p.name);
    const dim = sizes.reduce((a, b) => a + b, 0);
    const tLen = target.length;
    const x: number[] = [];
    for (const p of this.params) {
      const v = p.value.ref;
      for (const n of toJSArr(v)) x.push(n);
    }

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
      const { coords, secondary } = renderEval(this.renderFn, paramNames, pv);
      return sumSecondaryScores(lossFn(t, coords), secondary);
    };
    let renderIds: string[] = [];
    const renderOnlyFn = (flat: any) => {
      const pv: any[] = [];
      let off = 0;
      for (let i = 0; i < sizes.length; i++) {
        const n = sizes[i];
        const isLast = i === sizes.length - 1;
        pv.push((isLast ? flat : flat.ref).slice([off, off + n]));
        off += n;
      }
      const { coords } = renderEval(this.renderFn, paramNames, pv);
      renderIds = Object.keys(coords);
      const arrays: any[] = [];
      for (const c of Object.values(coords)) arrays.push(c);
      return np.concatenate(arrays);
    };
    const jitLoss = jit(combinedFn);
    const jitGrad = jit(grad(combinedFn));
    const jitGradFallback = jit(jacfwd(combinedFn));
    let useGradFallback = false;
    const jitRender = jit(renderOnlyFn);
    const jitBatchLoss = jit(vmap(combinedFn, [0]));
    runtimeStats.warmupBuilds += 1;
    jitRender(np.array(x, { dtype: np.float32 }));
    const totalLen = tLen + dim;
    const warmupCombined = new Float32Array(totalLen);
    for (let i = 0; i < tLen; i++) warmupCombined[i] = target[i] ?? 0;
    for (let i = 0; i < dim; i++) warmupCombined[tLen + i] = x[i];
    jitLoss(np.array(warmupCombined, { dtype: np.float32 }));
    try {
      jitGrad(np.array(warmupCombined, { dtype: np.float32 }));
    } catch (error) {
      if (!isReverseModeUnsupported(error)) throw error;
      useGradFallback = true;
      jitGradFallback(np.array(warmupCombined, { dtype: np.float32 }));
    }
    const warmupBatch = new Float32Array(totalLen * LINE_SEARCH_TRIALS);
    for (let ls = 0; ls < LINE_SEARCH_TRIALS; ls++) {
      const row = ls * totalLen;
      for (let i = 0; i < totalLen; i++) warmupBatch[row + i] = warmupCombined[i];
    }
    jitBatchLoss(np.array(warmupBatch, { dtype: np.float32 }).reshape([LINE_SEARCH_TRIALS, totalLen]));
    return {
      jitLoss,
      jitGrad,
      jitGradFallback,
      useGradFallback,
      jitRender,
      jitBatchLoss,
      renderIds,
      targetLen: tLen,
      lastX: x,
      lastLoss: 0,
      lastConverged: false,
      lastHitLimit: false,
      affectsRef: null,
      affectsMask: null,
      combinedBuffer: new Float32Array(tLen + dim),
      trialBuffer: new Float32Array(tLen + dim),
      trialBatchBuffer: new Float32Array((tLen + dim) * LINE_SEARCH_TRIALS),
      x0Buffer: new Float64Array(dim),
      gBuffer: new Float64Array(dim),
      renderBuffer: new Float32Array(dim),
      velocityBuffer: new Float64Array(dim),
      bfgsH: new Float64Array(dim * dim),
      bfgsStep: new Float64Array(dim),
      bfgsGradNext: new Float64Array(dim),
      bfgsY: new Float64Array(dim),
      bfgsHy: new Float64Array(dim),
    };
  }

  _minimize(
    id: string,
    lossFn: LossFn,
    target: number[],
    affects: Record<string, any> | null | undefined,
    forceRender = false,
    cached?: CachedJit,
  ): CachedJit {
    const affectsObj = affects ?? {};
    const dragIterRaw = "dragIter" in affectsObj ? (affectsObj as any).dragIter : null;
    const dragIterOverride = dragIterRaw == null
      ? NaN
      : Array.isArray(dragIterRaw)
        ? Number(dragIterRaw[0])
        : Number(dragIterRaw);
    const maxIterCap = Number.isFinite(dragIterOverride)
      ? Math.max(0, Math.floor(dragIterOverride))
      : lineSearchEnabled
        ? DRAG_ITER_LINE_SEARCH
        : DRAG_ITER_ADAPTIVE;
    const targetChanged = !cached || cached.targetLen !== target.length || target.some((v, i) => {
      return Math.abs(v - cached.combinedBuffer[i]) > 1e-4;
    });
    const shouldRun = !cached || targetChanged || (!cached.lastConverged && !cached.lastHitLimit);
    const dragIterations = shouldRun ? maxIterCap : 0;
    const c = minimize(this.params, this.renderFn, lossFn, target, affects, dragIterations, cached);
    emitOptimizeLoss(this.containerId, c.lastLoss);
    this._dragRenderCounter += 1;
    // Always render when the drag target moves so a single moved frame cannot get skipped.
    const shouldRender = forceRender || targetChanged || this._dragRenderCounter % DRAG_RENDER_EVERY === 0;
    if (c.lastX.length > 0 && shouldRender) {
      this._renderFast(c);
      this._dragRenderCounter = 0;
    }
    return c;
  }

  _renderFast(cached: CachedJit): void {
    const n = cached.lastX.length;
    const flatBuf = cached.renderBuffer.length === n ? cached.renderBuffer : new Float32Array(n);
    for (let i = 0; i < n; i++) flatBuf[i] = cached.lastX[i];
    cached.renderBuffer = flatBuf;
    const result = cached.jitRender(np.array(flatBuf, { dtype: np.float32 }));
    const allCoords: number[] = typeof result?.dataSync === "function"
      ? Array.from(result.dataSync())
      : toJSArr(result);

    let off = 0;
    for (const id of cached.renderIds) {
      const elem = this.elements[id];
      if (!elem) continue;
      const n = elem instanceof LineEl ? 4 : 2;
      elem.updateCoords(allCoords.slice(off, off + n));
      off += n;
    }
  }

  render(metaOnly = false): void {
    const obj: Record<string, any> = {};
    for (const p of this.params) obj[p.name] = p.value.ref;
    const { shapes: renderables } = parseRenderOutput(this.renderFn(obj));
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
    if (this._debugPullEl && this._debugTargetEl) {
      this.node.appendChild(this._debugPullEl);
      this.node.appendChild(this._debugTargetEl);
    }
  }
}

export { np, jit };
