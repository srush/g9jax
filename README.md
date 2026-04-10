# g9-jaxjs

**Automatically interactive graphics** using [jax-js](https://github.com/ekzhang/jax-js) forward-mode differentiation.

Inspired by [g9.js](https://omrelli.ug/g9/) and [g9py](https://github.com/srush/g9py). Instead of
numerical finite differences (g9.js) or reverse-mode backpropagation (g9py/PyTorch), this
implementation uses **forward-mode automatic differentiation** via `jvp` (Jacobian-vector products)
from jax-js.

## How it works

1. You write a **render function** that takes parameters (jax-js arrays) and returns a dictionary of
   shapes (points, lines).
2. The library renders these as SVG elements and attaches drag handlers.
3. When the user drags a shape, g9-jaxjs constructs a **loss function** measuring the distance
   between the shape's current position and the drag target.
4. It computes the **gradient of the loss** with respect to all parameters using forward-mode AD
   (`jvp`): for each parameter component, a single `jvp` call with a one-hot tangent vector
   yields that component of the gradient.
5. **Gradient descent with backtracking line search** updates the parameters to minimize the loss.
6. The scene re-renders with the new parameters, so all shapes move consistently.

### Why forward-mode?

Forward-mode AD (`jvp`) computes directional derivatives: given a function `f(x)` and a tangent
vector `v`, it returns `(f(x), Jf · v)` in a single pass. To get the full gradient of a scalar
loss, we call `jvp` once per parameter dimension with basis vectors `e_i`.

This is efficient when the number of parameters is small (typical for interactive graphics), and
it avoids the complexity of building a reverse-mode tape. jax-js's `jvp` traces through all
standard array operations, so arbitrary differentiable render functions work automatically.

## Getting started

```bash
npm install
npm run dev
```

Open the local URL and drag the shapes in each demo.

## Project structure

```
src/g9.js     Core library: shapes, jvp-based gradient, optimizer, SVG rendering, drag handling
src/main.js   Demo wiring: five interactive examples
index.html    Demo page
```

## Demos

| Demo | Description |
|------|-------------|
| **Basic** | Two mirrored points `(x,y)` and `(y,x)` |
| **Rings** | Concentric circles controlled by radius and angle |
| **Lines** | Three lines with different `affects` constraints |
| **Dragon curve** | Recursive fractal with draggable endpoints and squareness |
| **Fractal tree** | Recursive tree with draggable branch angle and attenuation |

## API

```js
import { G9, point, line, np } from "./g9.js";

const g = new G9(
  (params) => ({
    myPoint: point(params.pos),
    myLine: line(params.lineCoords, { stroke: "red", affects: { lineCoords: [1,1,0,0] } }),
  }),
  { pos: [0, 0], lineCoords: [-50, 0, 50, 0] },
);

g.align("center", "center").insertInto("#my-container");
```

- **`point(coords, opts?)`** – a draggable circle at `[x, y]`
- **`line(coords, opts?)`** – a draggable line from `[x1, y1, x2, y2]`
- **`affects`** option on shapes restricts which parameters a drag can modify
- **`np`** is re-exported from `@jax-js/jax` for building differentiable render functions
