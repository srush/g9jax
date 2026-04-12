# g9-jaxjs

Check it out at [g9-jaxjs](https://srush.github.io/g9jax/)

**Automatically interactive graphics** using [jax-js](https://github.com/ekzhang/jax-js) forward-mode differentiation.

Inspired by [g9.js](https://omrelli.ug/g9/). Instead of
numerical finite differences (g9.js), this implementation uses from jax-js enabling higher param spaces.

## How it works

1. You write a **render function** that takes parameters (jax-js arrays) and returns a dictionary of
   shapes (points, lines).
2. The library renders these and attaches drag handlers.
3. When the user drags a shape, g9-jaxjs constructs a **loss function** measuring the distance
   between the shape's current position and the drag target.
4. It computes the **gradient of the loss** with respect and optimizes
5. **Gradient descent with backtracking line search** updates the parameters to minimize the loss.
6. The scene re-renders with the new parameters, so all shapes move consistently.

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

- **`point(coords, opts?)`** – a draggable circle at `[x, y]`
- **`line(coords, opts?)`** – a draggable line from `[x1, y1, x2, y2]`
- **`affects`** option on shapes restricts which parameters a drag can modify
- **`np`** is re-exported from `@jax-js/jax` for building differentiable render functions
