import {
  G9,
  point,
  line,
  np,
  jit,
  getG9DragDebugEnabled,
  setG9DragDebugEnabled,
  getG9LineSearchEnabled,
  setG9LineSearchEnabled,
  getG9DebugLossStats,
} from "./g9";
import { defaultDevice, init, vmap, type Device } from "@jax-js/jax";

function show(id: string) {
  const el = document.getElementById(id);
  if (el) el.style.display = "";
}

function hideBanner() {
  const el = document.getElementById("loading-banner");
  if (el) el.style.display = "none";
}

function setBanner(msg: string) {
  const el = document.getElementById("loading-banner");
  if (el) el.innerHTML = msg;
}

function updateDemoLoss(card: Element | null, loss: number | null): void {
  const lossEl = card?.querySelector(".demo-loss-value");
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

function runDemoFromTextarea(sectionId: string, canvasSelector: string) {
  const section = document.getElementById(sectionId);
  if (!section) return;
  show(sectionId);
  const textarea = section.querySelector("textarea");
  if (!textarea) return;
  const card = section.querySelector(`.demo-code[data-target="${canvasSelector}"]`);
  const existing = demoMountState.get(canvasSelector);
  if (existing) {
    existing.render();
    updateDemoLoss(card, null);
    return;
  }
  const fn = new Function("G9", "point", "line", "np", "jit", "vmap", textarea.value);
  const g9 = fn(G9, point, line, np, jit, vmap);
  if (g9 instanceof G9) {
    g9.align("center", "center").insertInto(canvasSelector);
    demoMountState.set(canvasSelector, g9);
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
  runDemoFromTextarea(sectionId, canvasSelector);
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
    const card = document.querySelector(`.demo-code[data-target="#${containerId}"]`);
    updateDemoLoss(card, custom.detail?.loss ?? null);
  });
}

(window as any).__runDemoFromTextarea = runDemoFromTextarea;
(window as any).__g9SetDragDebugEnabled = setG9DragDebugEnabled;
(window as any).__g9SetLineSearchEnabled = setG9LineSearchEnabled;

function bindDebugControls(): void {
  const debugToggle = document.getElementById("debug-drag-toggle") as HTMLInputElement | null;
  const lineSearchToggle = document.getElementById("line-search-toggle") as HTMLInputElement | null;
  const debugBox = document.getElementById("debug-stats-box");
  const avgLossValue = document.getElementById("avg-opt-loss-value");
  const avgLossCount = document.getElementById("avg-opt-loss-count");

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
    });
  }

  if (lineSearchToggle) {
    lineSearchToggle.checked = getG9LineSearchEnabled();
    lineSearchToggle.addEventListener("change", () => {
      setG9LineSearchEnabled(lineSearchToggle.checked);
    });
  }

  renderDebugStats();
  window.setInterval(renderDebugStats, 200);
}
(window as any).__g9SetDragDebugEnabled = setG9DragDebugEnabled;
(window as any).__g9GetDragDebugEnabled = getG9DragDebugEnabled;

async function main() {
  setBanner('<span class="spinner"></span> Initialising jax-js runtime…');

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

  const preferred: Device[] = ["webgpu", "webgl", "cpu"];
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
  bindDebugControls();

  const demos = [
    ["section-points", "#demo-points"],
    ["section-rings", "#demo-rings"],
    ["section-lines", "#demo-lines"],
    ["section-dragon", "#demo-dragon"],
    ["section-tree", "#demo-tree"],
    ["section-snake", "#demo-snake"],
    ["section-tongs", "#demo-tongs"],
    ["section-bezier", "#demo-bezier"],
  ] as const;

  // Unhide sections first for layout consistency.
  for (const [sectionId] of demos) show(sectionId);

  // Render one demo immediately, then mount the rest progressively.
  await nextFrame();
  await nextFrame();

  const [firstSectionId, firstCanvas] = demos[0];
  runDemoFromTextarea(firstSectionId, firstCanvas);
  hideBanner();
  console.log(`${firstSectionId} OK`);

  const mountRemaining = async () => {
    await nextFrame();
    for (let i = 1; i < demos.length; i++) {
      const [sectionId, canvasSelector] = demos[i];
      try {
        runDemoFromTextarea(sectionId, canvasSelector);
        console.log(`${sectionId} OK`);
      } catch (e) {
        console.error(`${sectionId} failed:`, e);
      }
      await nextFrame();
    }
    console.log("All demos rendered");
  };

  void mountRemaining();

  for (let i = 1; i < demos.length; i++) {
    const [sectionId, canvasSelector] = demos[i];
    try {
      const section = document.getElementById(sectionId);
      if (section) section.dataset.demoTarget = canvasSelector;
    } catch (e) {
      console.error(`${sectionId} failed:`, e);
    }
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  setBanner(`Error: ${err.message}. Check the browser console (F12) for details.`);
});
