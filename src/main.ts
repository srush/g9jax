import { G9, point, line, np, jit } from "./g9";
import { defaultDevice, init, vmap, devices } from "@jax-js/jax";

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

function runDemoFromTextarea(sectionId: string, canvasSelector: string) {
  const section = document.getElementById(sectionId);
  if (!section) return;
  show(sectionId);
  const textarea = section.querySelector("textarea");
  if (!textarea) return;
  const fn = new Function("G9", "point", "line", "np", "jit", "vmap", textarea.value);
  const g9 = fn(G9, point, line, np, jit, vmap);
  if (g9 instanceof G9) {
    g9.align("center", "center").insertInto(canvasSelector);
  }
}

async function main() {
  setBanner('<span class="spinner"></span> Initialising jax-js runtime…');

  let backendName = "unknown";
  try {
    const backends = await init();
    console.log("jax-js init complete, backends:", backends);
    defaultDevice("cpu");
    backendName = defaultDevice().toString();
    console.log("jax-js default device forced to cpu");
  } catch (e: any) {
    console.warn("jax-js full init failed, trying cpu-only:", e.message || e);
    try {
      await init("cpu");
      defaultDevice("cpu");
      backendName = defaultDevice().toString();
      console.log("jax-js cpu backend ready");
    } catch (e2: any) {
      console.warn("jax-js cpu init also failed:", e2.message || e2);
    }
  }

  const backendEl = document.getElementById("backend-info");
  if (backendEl) {
    const allDevices = devices.map((d: any) => d.toString()).join(", ");
    backendEl.textContent = `Backend: ${backendName} | Available: ${allDevices}`;
  }

  setBanner('<span class="spinner"></span> Rendering demos…');

  const demos = [
    ["section-points", "#demo-points"],
    ["section-rings", "#demo-rings"],
    ["section-lines", "#demo-lines"],
    ["section-dragon", "#demo-dragon"],
    ["section-tree", "#demo-tree"],
  ];

  for (const [sectionId, canvasSelector] of demos) {
    try {
      runDemoFromTextarea(sectionId, canvasSelector);
      console.log(`${sectionId} OK`);
    } catch (e) {
      console.error(`${sectionId} failed:`, e);
    }
  }

  hideBanner();
  console.log("All demos rendered");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  setBanner(`Error: ${err.message}. Check the browser console (F12) for details.`);
});
