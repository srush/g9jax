import {
  G9,
  point,
  line,
  np,
  jit,
  getG9DragDebugEnabled,
  setG9DragDebugEnabled,
  getG9DebugLossStats,
  getG9GradientArrowsEnabled,
  setG9GradientArrowsEnabled,
} from "./g9";
import { defaultDevice, init, vmap, type Device } from "@jax-js/jax";
import { shouldReuseMountedDemo } from "./demo-run-policy";

function show(id: string) {
  const el = document.getElementById(id);
  if (el) el.style.display = "";
}

function hideBanner() {
  const el = document.getElementById("loading-banner");
  if (el) el.style.display = "none";
}

function setBanner(msg: string, showSpinner = false) {
  const el = document.getElementById("loading-banner");
  if (!(el instanceof HTMLElement)) return;
  el.textContent = "";
  if (showSpinner) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    el.appendChild(spinner);
    el.appendChild(document.createTextNode(" "));
  }
  el.appendChild(document.createTextNode(msg));
}

function updateDemoLoss(card: Element | null, loss: number | null): void {
  const lossEl = card?.querySelector(".loss-value");
  if (!(lossEl instanceof HTMLElement)) return;
  if (loss == null || !Number.isFinite(loss)) {
    lossEl.textContent = "—";
    return;
  }
  lossEl.textContent = loss.toExponential(3);
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

const demoMountState = new Map<string, G9>();
const demoSourceState = new Map<string, string>();
const queuedDemoTargets = new Set<string>();

function runDemoFromTextarea(sectionId: string, canvasSelector: string, forceRemount = false) {
  const section = document.getElementById(sectionId);
  if (!section) return;
  show(sectionId);
  const textarea = section.querySelector("textarea");
  if (!textarea) return;
  const source = textarea.value;
  const card = section.querySelector(`.demo-code[data-target="${canvasSelector}"]`);
  const existing = demoMountState.get(canvasSelector);
  const previousSource = demoSourceState.get(canvasSelector);
  if (shouldReuseMountedDemo(canvasSelector, !!existing, previousSource, source, forceRemount)) {
    existing.render();
    updateDemoLoss(card, null);
    return;
  }

  const fn = new Function("G9", "point", "line", "np", "jit", "vmap", source);
  const g9 = fn(G9, point, line, np, jit, vmap);
  if (g9 instanceof G9) {
    if (existing) {
      existing.destroy();
      demoMountState.delete(canvasSelector);
    }
    g9.align("center", "center").insertInto(canvasSelector);
    demoMountState.set(canvasSelector, g9);
    demoSourceState.set(canvasSelector, source);
    updateDemoLoss(card, null);
  }
}

function setRunError(card: Element | null, message: string) {
  const errEl = card?.querySelector(".error-msg");
  if (errEl instanceof HTMLElement) errEl.textContent = message;
}

function runDemoFromCard(card: Element): void {
  const section = card.closest(".demo-section");
  const sectionId = section?.id ?? "";
  const canvasSelector = card instanceof HTMLElement ? card.dataset.target ?? "" : "";
  if (!sectionId || !canvasSelector) return;
  runDemoFromTextarea(sectionId, canvasSelector, true);
}

function bindRunButtons(): void {
  document.addEventListener("click", (event) => {
    const node = event.target;
    const source = node instanceof Element ? node : node instanceof Node ? node.parentElement : null;
    if (!source) return;
    const button = source.closest(".run-btn");
    if (!button) return;
    const card = button.closest(".demo-code");
    if (!card) return;
    try {
      runDemoFromCard(card);
      setRunError(card, "");
    } catch (error: any) {
      setRunError(card, String(error?.message ?? error));
    }
  });
}

function bindDemoLossRows(): void {
  document.addEventListener("g9:opt-loss", (event) => {
    const custom = event as CustomEvent<{ containerId: string; loss: number }>;
    const containerId = custom.detail?.containerId;
    if (!containerId) return;
    const codeCard = document.querySelector(`.demo-code[data-target="#${containerId}"]`);
    const demoCard = codeCard instanceof Element ? codeCard.closest(".demo-card") : null;
    updateDemoLoss(demoCard, custom.detail?.loss ?? null);
  });
}

(window as any).__runDemoFromTextarea = runDemoFromTextarea;
(window as any).__g9SetDragDebugEnabled = setG9DragDebugEnabled;

function bindDebugControls(): void {
  const debugToggle = document.getElementById("debug-drag-toggle") as HTMLInputElement | null;
  const debugBox = document.getElementById("debug-stats-box");
  const avgLossValue = document.getElementById("avg-opt-loss-value");
  const avgLossCount = document.getElementById("avg-opt-loss-count");
  let statsTimer: number | null = null;

  const renderDebugStats = () => {
    const debugEnabled = debugToggle?.checked ?? false;
    if (!debugEnabled) {
      if (debugBox instanceof HTMLElement) debugBox.style.display = "none";
      return;
    }
    if (debugBox instanceof HTMLElement) debugBox.style.display = "block";
    const stats = getG9DebugLossStats();
    if (avgLossValue) {
      avgLossValue.textContent = stats.count > 0 ? stats.average.toExponential(3) : "—";
    }
    if (avgLossCount) {
      avgLossCount.textContent = String(stats.count);
    }
  };

  if (debugToggle) {
    debugToggle.checked = getG9DragDebugEnabled();
    debugToggle.addEventListener("change", () => {
      setG9DragDebugEnabled(debugToggle.checked);
      renderDebugStats();
      if (debugToggle.checked) {
        if (statsTimer == null) statsTimer = window.setInterval(renderDebugStats, 200);
      } else if (statsTimer != null) {
        window.clearInterval(statsTimer);
        statsTimer = null;
      }
    });
  }

  renderDebugStats();
  if (debugToggle?.checked) statsTimer = window.setInterval(renderDebugStats, 200);

  const gradArrowsToggle = document.getElementById("gradient-arrows-toggle") as HTMLInputElement | null;
  if (gradArrowsToggle) {
    gradArrowsToggle.checked = getG9GradientArrowsEnabled();
    gradArrowsToggle.addEventListener("change", () => {
      setG9GradientArrowsEnabled(gradArrowsToggle.checked);
    });
  }
}
(window as any).__g9SetDragDebugEnabled = setG9DragDebugEnabled;
(window as any).__g9GetDragDebugEnabled = getG9DragDebugEnabled;

function moveDemoSection(sectionId: string, beforeSectionId: string): void {
  const section = document.getElementById(sectionId);
  const before = document.getElementById(beforeSectionId);
  const parent = before?.parentElement;
  if (!section || !before || !parent) return;
  parent.insertBefore(section, before);
}

type DemoSpec = { sectionId: string; canvasSelector: string };

function isDragActive(): boolean {
  return document.documentElement?.classList.contains("g9-dragging") ?? false;
}

function setupDemoLazyMount(demos: readonly DemoSpec[]): void {
  const bySection = new Map<string, DemoSpec>();
  for (const demo of demos) {
    bySection.set(demo.sectionId, demo);
    const section = document.getElementById(demo.sectionId);
    if (section) section.dataset.demoTarget = demo.canvasSelector;
  }

  const queue: DemoSpec[] = [];
  const deferred: DemoSpec[] = [];
  let draining = false;
  let autoMountReady = false;

  const enqueueDemo = (demo: DemoSpec) => {
    if (demoMountState.has(demo.canvasSelector)) return;
    if (queuedDemoTargets.has(demo.canvasSelector)) return;
    queuedDemoTargets.add(demo.canvasSelector);
    queue.push(demo);
    void drainQueue();
  };

  const enableAutoMount = () => {
    if (autoMountReady) return;
    autoMountReady = true;
    for (const demo of deferred.splice(0, deferred.length)) enqueueDemo(demo);
  };

  const installAutoMountGate = () => {
    window.addEventListener("scroll", enableAutoMount, { passive: true, once: true });
    window.addEventListener("wheel", enableAutoMount, { passive: true, once: true });
    window.addEventListener("touchmove", enableAutoMount, { passive: true, once: true });
    window.addEventListener("keydown", enableAutoMount, { once: true });
  };

  const drainQueue = async () => {
    if (draining) return;
    draining = true;
    while (queue.length > 0) {
      if (isDragActive()) {
        await nextFrame();
        continue;
      }
      const demo = queue.shift();
      if (!demo) continue;
      queuedDemoTargets.delete(demo.canvasSelector);
      if (demoMountState.has(demo.canvasSelector)) continue;
      try {
        runDemoFromTextarea(demo.sectionId, demo.canvasSelector);
        console.log(`${demo.sectionId} lazy mounted`);
      } catch (error) {
        console.error(`${demo.sectionId} lazy mount failed:`, error);
      }
      await nextFrame();
    }
    draining = false;
  };

  if (typeof IntersectionObserver === "undefined") return;

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const target = entry.target;
      const sectionId = target instanceof HTMLElement ? target.id : "";
      if (!sectionId) continue;
      const demo = bySection.get(sectionId);
      if (!demo) continue;
      observer.unobserve(target);
      if (autoMountReady) {
        enqueueDemo(demo);
      } else {
        deferred.push(demo);
      }
    }
  }, {
    root: null,
    rootMargin: "100px 0px",
    threshold: 0.01,
  });

  for (const demo of demos) {
    const section = document.getElementById(demo.sectionId);
    if (section) observer.observe(section);
  }

  installAutoMountGate();
}

async function main() {
  setBanner("Initialising jax-js runtime…", true);

  let readyDevices: Device[] = [];
  try {
    readyDevices = await init();
    console.log("jax-js init complete, devices:", readyDevices);
  } catch (e: any) {
    console.warn("jax-js full init failed, trying cpu-only:", e.message || e);
    try {
      readyDevices = await init("cpu");
    } catch (e2: any) {
      console.warn("jax-js cpu init also failed:", e2.message || e2);
    }
  }

  const preferred: Device[] = ["webgpu", "wasm", "webgl", "cpu"];
  const chosen = preferred.find((d) => readyDevices.includes(d)) ?? readyDevices[0];
  defaultDevice(chosen);
  console.log("jax-js default device:", chosen);

  const select = document.getElementById("backend-select") as HTMLSelectElement | null;
  if (select) {
    select.innerHTML = "";
    for (const d of readyDevices) {
      const opt = document.createElement("option");
      opt.value = d;
      opt.textContent = d;
      if (d === chosen) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      defaultDevice(select.value as any);
      console.log("Backend switched to", defaultDevice().toString());
    });
  }

  bindRunButtons();
  bindDemoLossRows();
  setG9DragDebugEnabled(true);
  bindDebugControls();
  moveDemoSection("section-tongs", "section-blocker");

  const demos: DemoSpec[] = [
    ["section-points", "#demo-points"],
    ["section-rings", "#demo-rings"],
    ["section-cube", "#demo-cube"],
    ["section-lines", "#demo-lines"],
    ["section-tongs", "#demo-tongs"],
    ["section-blocker", "#demo-blocker"],
    ["section-particles", "#demo-particles"],
    ["section-dragon", "#demo-dragon"],
    ["section-tree", "#demo-tree"],
    ["section-snake", "#demo-snake"],
  ].map(([sectionId, canvasSelector]) => ({ sectionId, canvasSelector }));

  // Unhide sections first for layout consistency.
  for (const { sectionId } of demos) show(sectionId);

  // Render one demo immediately so first interaction is responsive.
  await nextFrame();
  await nextFrame();

  const firstDemo = demos[0];
  try {
    runDemoFromTextarea(firstDemo.sectionId, firstDemo.canvasSelector);
    console.log(`${firstDemo.sectionId} ready`);
  } catch (error: any) {
    const message = String(error?.message ?? error);
    console.error(`${firstDemo.sectionId} initial mount failed:`, error);
    const card = document.querySelector(`.demo-code[data-target="${firstDemo.canvasSelector}"]`);
    setRunError(card, message);
  }
  hideBanner();

  setupDemoLazyMount(demos.slice(1));
}

main().catch((err) => {
  console.error("Fatal error:", err);
  setBanner(`Error: ${err.message}. Check the browser console (F12) for details.`);
});
