import { chromium } from "playwright";

const BASE_URL = process.env.G9_URL ?? "http://127.0.0.1:5173/g9jax/";
const TINY_DX = Number(process.env.SENS_DX ?? 6);
const TINY_DY = Number(process.env.SENS_DY ?? 3);

function l2(valuesA, valuesB) {
  const n = Math.max(valuesA.length, valuesB.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = (valuesA[i] ?? 0) - (valuesB[i] ?? 0);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function safeRatio(numerator, denominator) {
  return denominator > 1e-12 ? numerator / denominator : 0;
}

async function mountDemo(page, sectionId, canvasSelector) {
  await page.evaluate((id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "center" });
  }, sectionId);
  await page.evaluate(
    ({ id, selector }) => {
      if (typeof window.__runDemoFromTextarea !== "function") {
        throw new Error("__runDemoFromTextarea not available");
      }
      window.__runDemoFromTextarea(id, selector);
    },
    { id: sectionId, selector: canvasSelector },
  );
  await page.waitForSelector(`${canvasSelector} svg`, { timeout: 120000 });
}

async function dragNearCenter(page, selector, dx, dy) {
  const locator = page.locator(selector).first();
  await locator.waitFor({ state: "attached", timeout: 120000 });
  const box = await locator.boundingBox();
  if (!box) throw new Error(`No bounding box for ${selector}`);
  const sx = box.x + box.width / 2;
  const sy = box.y + box.height / 2;
  await page.mouse.move(sx, sy);
  await page.mouse.down();
  await page.mouse.move(sx + dx, sy + dy, { steps: 18 });
  await page.mouse.up();
}

async function readLine4(page, selector) {
  return page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!(el instanceof SVGLineElement)) throw new Error(`line not found: ${sel}`);
    return [
      Number(el.getAttribute("x1") ?? "0"),
      Number(el.getAttribute("y1") ?? "0"),
      Number(el.getAttribute("x2") ?? "0"),
      Number(el.getAttribute("y2") ?? "0"),
    ];
  }, selector);
}

async function runLinesProbe(page) {
  await mountDemo(page, "section-lines", "#demo-lines");
  const line2Before = await readLine4(page, "#demo-lines svg line#l2");
  await dragNearCenter(page, "#demo-lines svg line#l2", TINY_DX, TINY_DY);
  const line2AfterOwnDrag = await readLine4(page, "#demo-lines svg line#l2");
  await dragNearCenter(page, "#demo-lines svg line#l3", -TINY_DX, TINY_DY);
  const line2AfterSwitchDrag = await readLine4(page, "#demo-lines svg line#l2");
  const pointerDistance = Math.hypot(TINY_DX, TINY_DY);
  return {
    line2Move: l2(line2Before, line2AfterOwnDrag),
    line2MovePerPointerPx: safeRatio(l2(line2Before, line2AfterOwnDrag), pointerDistance),
    line2ResetDrift: l2(line2AfterOwnDrag, line2AfterSwitchDrag),
    line2ResetDriftPerPointerPx: safeRatio(l2(line2AfterOwnDrag, line2AfterSwitchDrag), pointerDistance),
  };
}

async function runSnakeProbe(page) {
  await mountDemo(page, "section-snake", "#demo-snake");
  const before = await readLine4(page, "#demo-snake svg line#s1");
  await dragNearCenter(page, "#demo-snake svg line#s1", TINY_DX, TINY_DY);
  const after = await readLine4(page, "#demo-snake svg line#s1");
  const pointerDistance = Math.hypot(TINY_DX, TINY_DY);
  return {
    segmentMove: l2(before, after),
    segmentMovePerPointerPx: safeRatio(l2(before, after), pointerDistance),
  };
}

async function runTongsProbe(page) {
  await mountDemo(page, "section-tongs", "#demo-tongs");
  const before = await readLine4(page, "#demo-tongs svg line#u2");
  await dragNearCenter(page, "#demo-tongs svg line#u2", -TINY_DX, TINY_DY);
  const after = await readLine4(page, "#demo-tongs svg line#u2");
  const pointerDistance = Math.hypot(TINY_DX, TINY_DY);
  return {
    segmentMove: l2(before, after),
    segmentMovePerPointerPx: safeRatio(l2(before, after), pointerDistance),
  };
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 120000 });
  await page.waitForFunction(() => typeof window.__runDemoFromTextarea === "function", { timeout: 120000 });

  const report = {
    url: BASE_URL,
    tinyDrag: { dx: TINY_DX, dy: TINY_DY, pointerDistance: Math.hypot(TINY_DX, TINY_DY) },
    lines: await runLinesProbe(page),
    snake: await runSnakeProbe(page),
    tongs: await runTongsProbe(page),
  };

  await page.screenshot({
    path: "/opt/cursor/artifacts/drag_sensitivity_probe.png",
    fullPage: true,
  });
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
