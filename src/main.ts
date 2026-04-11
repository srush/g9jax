import { G9, point, line, np, jit, getG9DragDebugEnabled, setG9DragDebugEnabled } from "./g9";
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
  const existing = demoMountState.get(canvasSelector);
  if (existing) {
    existing.render();
    return;
  }
  const fn = new Function("G9", "point", "line", "np", "jit", "vmap", textarea.value);
  const g9 = fn(G9, point, line, np, jit, vmap);
  if (g9 instanceof G9) {
    g9.align("center", "center").insertInto(canvasSelector);
    demoMountState.set(canvasSelector, g9);
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

function bindDragDebugToggle(): void {
  const checkbox = document.getElementById("drag-debug-toggle");
  if (!(checkbox instanceof HTMLInputElement)) return;
  checkbox.checked = getG9DragDebugEnabled();
  checkbox.addEventListener("change", () => {
    setG9DragDebugEnabled(checkbox.checked);
  });
}

(window as any).__runDemoFromTextarea = runDemoFromTextarea;
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
  bindDragDebugToggle();

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
