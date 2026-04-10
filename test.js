import { numpy as np, jvp } from "@jax-js/jax";

// Polyfill Float16Array for Node.js < 23
if (typeof globalThis.Float16Array === "undefined") {
  globalThis.Float16Array = Float32Array;
}

// Test 1: Simple jvp works
console.log("Test 1: basic jvp of x^2");
const f = (x) => x.ref.mul(x);
const x = np.array([3.0], { dtype: np.float64 });
const t = np.array([1.0], { dtype: np.float64 });
const [pOut, tOut] = jvp(f, [x], [t]);
console.log("  f(3) =", pOut.js(), "(expected [9])");
console.log("  f'(3) =", tOut.js(), "(expected [6])");

// Test 2: Multi-param jvp
console.log("\nTest 2: jvp with two params, f(a,b) = (a-b)^2");
const g = (a, b) => {
  const d = a.sub(b);
  return d.ref.mul(d);
};
const a = np.array([5.0], { dtype: np.float64 });
const b = np.array([2.0], { dtype: np.float64 });

const [_p1, t1] = jvp(g, [a.ref, b.ref], [np.array([1.0]), np.array([0.0])]);
console.log("  df/da(5,2) =", t1.js(), "(expected [6])");

const [_p2, t2] = jvp(g, [a.ref, b.ref], [np.array([0.0]), np.array([1.0])]);
console.log("  df/db(5,2) =", t2.js(), "(expected [-6])");

a.dispose();
b.dispose();

// Test 3: Forward gradient with multi-dim param
console.log("\nTest 3: forward gradient of loss(xy) = (xy[0]-10)^2 + (xy[1]-20)^2");
function forwardGrad(fn, paramArrays) {
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
      const [_pO, tO] = jvp(fn, primals, tangents);
      const v = tO.js();
      grad[col] = Array.isArray(v) ? v[0] : v;
      col++;
    }
  }
  return grad;
}

const xy = np.array([3.0, 7.0], { dtype: np.float64 });
const lossFn = (xy) => {
  const target = np.array([10.0, 20.0], { dtype: np.float64 });
  const d = xy.sub(target);
  return d.ref.mul(d).sum();
};
const grad = forwardGrad(lossFn, [xy]);
console.log("  grad =", Array.from(grad), "(expected [-14, -26])");
xy.dispose();

// Test 4: np.dot differentiability
console.log("\nTest 4: jvp through np.dot (matrix-vector)");
const mat = np.array([[0, 1], [1, 0]], { dtype: np.float64 });
const vec = np.array([3.0, 7.0], { dtype: np.float64 });
const dotFn = (v) => {
  const result = np.dot(mat.ref, v);
  return result.ref.mul(result).sum();
};
const dGrad = forwardGrad(dotFn, [vec]);
console.log("  grad =", Array.from(dGrad));
console.log("  (dot([0,1;1,0], [3,7]) = [7,3], loss=49+9=58, grad should be [6,14])");
mat.dispose();
vec.dispose();

// Test 5: np.concatenate differentiability
console.log("\nTest 5: jvp through np.concatenate");
const aa = np.array([1.0, 2.0], { dtype: np.float64 });
const bb = np.array([3.0, 4.0], { dtype: np.float64 });
const concatFn = (a, b) => {
  const c = np.concatenate([a, b]);
  return c.ref.mul(c).sum();
};
const cGrad = forwardGrad(concatFn, [aa, bb]);
console.log("  grad =", Array.from(cGrad), "(expected [2, 4, 6, 8])");
aa.dispose();
bb.dispose();

// Test 6: sin/cos differentiability
console.log("\nTest 6: jvp through sin/cos");
const angle = np.array([1.0], { dtype: np.float64 });
const sinCosFn = (a) => {
  const s = np.sin(a.ref);
  const c = np.cos(a);
  return s.ref.mul(s).add(c.ref.mul(c));
};
const scGrad = forwardGrad(sinCosFn, [angle]);
console.log("  grad =", Array.from(scGrad), "(expected ~[0] since sin^2+cos^2=1)");
angle.dispose();

console.log("\nAll tests completed!");
