const materials = ["Plastic", "Glass", "Metal", "Paper", "Cardboard", "Organic", "Background"];

// Fallback-only copy of ROUTES in web/server.py (the API response carries the
// authoritative bin); keep the two in sync.
const routes = {
  Plastic: "Recycling",
  Glass: "Recycling",
  Metal: "Recycling",
  Paper: "Recycling",
  Cardboard: "Recycling",
  Organic: "Compost",
  Background: "Review",
};

const classColors = {
  Plastic: "#11a66a",
  Glass: "#2d72d9",
  Metal: "#d04437",
  Paper: "#d0a31f",
  Cardboard: "#9a6a2f",
  Organic: "#168a45",
  Background: "#6b7280",
};

// Chapter order for arrow keys, the flow rail, and scroll-spy. The whole
// presentation is one continuous scroll; "pages" survive only as anchor
// aliases so old links (#demo, #data) keep working.
const FLOW = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13"];
const PAGE_ANCHORS = { home: "s1", data: "s2", eda: "s4", modeling: "s5", demo: "s13", scan: "scan-live" };
const PAGE_OF = { s1: "home", s2: "data", s3: "data", s4: "eda", s13: "demo" }; // others: modeling
const WORDS = ["Localize.", "Verify.", "Fuse.", "Route it right."];

const HERO_DEMO_BOXES = [
  { x: 55, y: 8, w: 14, h: 22, label: "Plastic", confidence: 94 },
  { x: 34, y: 38, w: 12, h: 24, label: "Plastic", confidence: 91 },
  { x: 63, y: 44, w: 13, h: 26, label: "Plastic", confidence: 93 },
  { x: 18, y: 60, w: 14, h: 26, label: "Plastic", confidence: 88 },
  { x: 46, y: 64, w: 12, h: 25, label: "Plastic", confidence: 90 },
];

// One sample per material class, each verified against the live pipeline (web/server.py) -
// not just plausible-looking stock photos. fallbackResult is only shown if the real API call
// fails; the button always tries the live model first (see performScan/apiBlob below).
const SAMPLES = [
  {
    id: "plastic", src: "assets/sample-plastic.jpg", label: "Floating bottles",
    result: {
      className: "Plastic", confidence: 92.4, context: "Outdoor / water", latency: 412,
      detections: HERO_DEMO_BOXES,
      scores: { Plastic: 92.4, Glass: 2.6, Metal: 1.1, Paper: 0.8, Cardboard: 0.6, Organic: 0.7, Background: 1.8 },
    },
  },
  {
    id: "glass", src: "assets/sample-glass.jpg", label: "Glass jar",
    result: {
      className: "Glass", confidence: 98, context: "Beach", latency: 574,
      detections: [{ x: 8, y: 17, w: 71, h: 69, label: "Glass", confidence: 99 }],
      scores: { Plastic: 1, Glass: 98, Metal: 0, Paper: 0, Cardboard: 0, Organic: 0, Background: 0 },
    },
  },
  {
    id: "can", src: "assets/sample-can.jpg", label: "Crushed can",
    result: {
      className: "Metal", confidence: 95.1, context: "Studio / close-up", latency: 388,
      detections: [{ x: 24, y: 16, w: 52, h: 66, label: "Metal", confidence: 95 }],
      scores: { Plastic: 1.4, Glass: 0.7, Metal: 95.1, Paper: 0.5, Cardboard: 0.6, Organic: 0.4, Background: 1.3 },
    },
  },
  {
    id: "paper", src: "assets/sample-paper.jpg", label: "Paper flyer",
    result: {
      className: "Paper", confidence: 97, context: "Street", latency: 44,
      detections: [{ x: 17, y: 17, w: 83, h: 60, label: "Paper", confidence: 99 }],
      scores: { Plastic: 0, Glass: 0, Metal: 1, Paper: 97, Cardboard: 1, Organic: 0, Background: 1 },
    },
  },
  {
    id: "cardboard", src: "assets/sample-cardboard.jpg", label: "Cardboard box",
    result: {
      className: "Cardboard", confidence: 93, context: "Indoor", latency: 44,
      detections: [{ x: 14, y: 14, w: 69, h: 66, label: "Cardboard", confidence: 86 }],
      scores: { Plastic: 2, Glass: 1, Metal: 1, Paper: 1, Cardboard: 93, Organic: 1, Background: 2 },
    },
  },
  {
    id: "organic", src: "assets/sample-organic.jpg", label: "Banana",
    result: {
      className: "Organic", confidence: 95, context: "Street", latency: 42,
      detections: [{ x: 24, y: 18, w: 54, h: 57, label: "Organic", confidence: 94 }],
      scores: { Plastic: 2, Glass: 0, Metal: 1, Paper: 1, Cardboard: 0, Organic: 95, Background: 2 },
    },
  },
];

const elements = {
  boxLayer: document.querySelector("#boxLayer"),
  confidenceMeter: document.querySelector("#confidenceMeter"),
  contextValue: document.querySelector("#contextValue"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  clearHistory: document.querySelector("#clearHistory"),
  historyCount: document.querySelector("#historyCount"),
  historyList: document.querySelector("#historyList"),
  imageStage: document.querySelector("#imageStage"),
  latencyValue: document.querySelector("#latencyValue"),
  loadingMask: document.querySelector("#loadingMask"),
  confidenceLabel: document.querySelector("#confidenceLabel"),
  possibleMaterialValue: document.querySelector("#possibleMaterialValue"),
  stagePills: Array.from(document.querySelectorAll("#stagePills li")),
  previewImage: document.querySelector("#previewImage"),
  resultColumn: document.querySelector(".result-column"),
  resultClass: document.querySelector("#resultClass"),
  resultConfidence: document.querySelector("#resultConfidence"),
  resultTopLabel: document.querySelector("#resultTopLabel"),
  binRoute: document.querySelector("#binRoute"),
  scoreList: document.querySelector("#scoreList"),
  scoreSummary: document.querySelector("#scoreSummary"),
  topPredictionList: document.querySelector("#topPredictionList"),
  modelStatus: document.querySelector("#modelStatus"),
};

const HISTORY_KEY = "wastewise.scanHistory.v3";
const MAX_HISTORY_ITEMS = 12;
const REVIEW_BOX_DISPLAY_MIN = 50;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let uploadObjectUrl = null;
let scanToken = 0;
let scanTimeouts = [];
let scanning = false;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function routeFor(className) {
  return routes[className] || "Review";
}

function emptyScores() {
  return Object.fromEntries(materials.map((material) => [material, 0]));
}

function withStrongestScore(className, confidence, scores) {
  return { ...scores, [className]: confidence };
}

function makeResult(partial) {
  return {
    bin: routeFor(partial.className),
    context: partial.context || "Unknown",
    detections: partial.detections || [{ x: 24, y: 18, w: 46, h: 58 }],
    latency: partial.latency || 0,
    mode: partial.mode || "Browser",
    possibleMaterial: partial.possibleMaterial || null,
    scores: withStrongestScore(partial.className, partial.confidence, partial.scores || emptyScores()),
    topPredictions: partial.topPredictions || null,
    ...partial,
  };
}

function setModelStatus(kind, detail) {
  elements.modelStatus.dataset.state = kind;
  elements.modelStatus.querySelector("strong").textContent = detail;
}

function setLoading(isLoading) {
  elements.loadingMask.classList.toggle("is-visible", isLoading);
}

function syncImageBounds() {
  if (!elements.imageStage.classList.contains("has-image") || !elements.previewImage.naturalWidth) {
    elements.imageStage.style.removeProperty("--image-left");
    elements.imageStage.style.removeProperty("--image-top");
    elements.imageStage.style.removeProperty("--image-width");
    elements.imageStage.style.removeProperty("--image-height");
    return;
  }

  const stage = elements.imageStage.getBoundingClientRect();
  const imageRatio = elements.previewImage.naturalWidth / elements.previewImage.naturalHeight;
  const stageRatio = stage.width / stage.height;
  let width = stage.width;
  let height = stage.height;
  let left = 0;
  let top = 0;

  if (imageRatio > stageRatio) {
    height = stage.width / imageRatio;
    top = (stage.height - height) / 2;
  } else {
    width = stage.height * imageRatio;
    left = (stage.width - width) / 2;
  }

  elements.imageStage.style.setProperty("--image-left", `${left}px`);
  elements.imageStage.style.setProperty("--image-top", `${top}px`);
  elements.imageStage.style.setProperty("--image-width", `${width}px`);
  elements.imageStage.style.setProperty("--image-height", `${height}px`);
}

function setPreviewImage(imageSrc) {
  if (!imageSrc) {
    elements.previewImage.removeAttribute("src");
    elements.previewImage.hidden = true;
    elements.imageStage.classList.remove("has-image");
    elements.boxLayer.replaceChildren();
    return;
  }

  elements.previewImage.src = imageSrc;
  elements.previewImage.hidden = false;
  elements.imageStage.classList.add("has-image");
  if (elements.previewImage.complete) syncImageBounds();
}

/* ---------------------------------------------------------------- routing */

const navLinks = Array.from(document.querySelectorAll("[data-page-link]"));

let currentFlowIndex = 0;

function updateFlowRail() {
  document.querySelectorAll(".flow-rail a").forEach((dot, i) => {
    const on = i === currentFlowIndex;
    dot.classList.toggle("active", on);
    if (on) dot.setAttribute("aria-current", "true");
    else dot.removeAttribute("aria-current");
  });
}

function setCurrentSection(sectionId) {
  const i = FLOW.indexOf(sectionId);
  if (i === -1) return;
  currentFlowIndex = i;
  updateFlowRail();
  const page = PAGE_OF[sectionId] || "modeling";
  navLinks.forEach((a) => {
    const on = a.getAttribute("href").replace("#", "") === page;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "true");
    else a.removeAttribute("aria-current");
  });
}

// One chapter on screen at a time: each chapter is its own scroll story, and
// the end-of-chapter card / arrow keys / rail dots are the stops between them.
let chapterNavigated = false;

function showChapter(sectionId, scrollTarget) {
  if (!FLOW.includes(sectionId)) sectionId = "s1";
  FLOW.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.hidden = id !== sectionId;
  });
  setCurrentSection(sectionId);
  const target = scrollTarget && document.getElementById(scrollTarget);
  if (target) {
    requestAnimationFrame(() =>
      target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" }),
    );
  } else {
    window.scrollTo({ top: 0 });
  }
  const sec = document.getElementById(sectionId);
  if (sec) {
    if (chapterNavigated) {
      // move keyboard/SR focus with the chapter, but not on initial load
      sec.setAttribute("tabindex", "-1");
      sec.focus({ preventScroll: true });
    }
    if (!reduceMotion) {
      sec.classList.remove("chapter-enter");
      void sec.offsetWidth; // restart the entrance animation
      sec.classList.add("chapter-enter");
    }
  }
  chapterNavigated = true;
}

// Old page-name hashes (#demo, #data, #scan) map to their first chapter.
function handleHashChange() {
  const hash = (window.location.hash || "").replace("#", "");
  if (hash === "scan-live" || hash === "scan") {
    showChapter("s13", "scan-live");
    return;
  }
  showChapter(PAGE_ANCHORS[hash] || hash || "s1");
}

/* ------------------------------------------------- presentation controls */

window.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
  const tag = (event.target && event.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const lightboxEl = document.getElementById("lightbox");
  if (lightboxEl && !lightboxEl.hidden) {
    if (event.key === "Escape" || event.key === "ArrowLeft" || event.key === "ArrowRight") closeLightbox();
    return;
  }
  // buttons and the chart tab explorers own their arrow keys
  if (tag === "BUTTON" || (event.target && event.target.closest && event.target.closest('[role="tablist"]'))) return;
  const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
  if (!step) return;
  const next = currentFlowIndex + step;
  if (next < 0 || next >= FLOW.length) return;
  window.location.hash = FLOW[next];
});

/* ------------------------------------------------------------------ theme */

const themeToggle = document.getElementById("themeToggle");
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function store(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (err) {
    /* private mode: the choice just does not survive a reload */
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  if (themeToggle) {
    themeToggle.setAttribute(
      "aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
    );
  }
}

if (themeToggle) {
  // The inline <head> script already resolved the starting theme; only react to clicks.
  applyTheme(document.documentElement.dataset.theme || "light");
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    store("ww-theme", next);
    showToast(next === "dark" ? "Dark theme" : "Light theme");
  });
}

// Keep following the system until the visitor picks a side.
systemDark.addEventListener("change", (event) => {
  let saved = null;
  try {
    saved = window.localStorage.getItem("ww-theme");
  } catch (err) {
    saved = null;
  }
  if (!saved) applyTheme(event.matches ? "dark" : "light");
});

/* ---------------------------------------------------------------- lightbox */

const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxCaption = document.getElementById("lightboxCaption");

let lightboxReturnFocus = null;

function closeLightbox() {
  lightbox.hidden = true;
  document.body.classList.remove("no-scroll");
  if (lightboxReturnFocus) {
    lightboxReturnFocus.focus({ preventScroll: true });
    lightboxReturnFocus = null;
  }
}

function openLightbox(img) {
  lightboxImg.src = img.src;
  lightboxImg.alt = img.alt;
  const caption = img.closest("figure").querySelector("figcaption");
  lightboxCaption.innerHTML = caption ? caption.innerHTML : "";
  lightbox.hidden = false;
  document.body.classList.add("no-scroll");
  lightboxReturnFocus = img;
  const closeBtn = document.getElementById("lightboxClose");
  if (closeBtn) closeBtn.focus({ preventScroll: true });
}

if (lightbox) {
  document.querySelectorAll("figure.fig img").forEach((img) => {
    img.tabIndex = 0;
    img.setAttribute("role", "button");
    img.addEventListener("click", () => openLightbox(img));
    img.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openLightbox(img);
      }
    });
  });
  // the close button's click bubbles here too
  lightbox.addEventListener("click", closeLightbox);
}

window.addEventListener("hashchange", handleHashChange);

// Alias links (#demo, #data, ...) have no matching element id, so a click that
// does not change the hash fires no hashchange event; re-run the routing by
// hand. Delegated so it also covers the dynamic "Go to scanner" tile.
document.addEventListener("click", (event) => {
  const link = event.target.closest && event.target.closest("[data-page-link]");
  if (link && link.getAttribute("href") === window.location.hash) handleHashChange();
});

/* ---------------------------------------------------------------- history */

function readHistory() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeHistory(items) {
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY_ITEMS)));
}

function formatHistoryTime(isoDate) {
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) return "saved scan";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function addHistoryRecord(result, sourceLabel, imageSrc) {
  const record = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    savedAt: new Date().toISOString(),
    sourceLabel,
    imageSrc: imageSrc && !imageSrc.startsWith("blob:") ? imageSrc : "",
    className: result.className,
    confidence: result.confidence,
    bin: result.bin,
    context: result.context,
    latency: result.latency,
    mode: result.mode,
    possibleMaterial: result.possibleMaterial,
    detections: result.detections,
    scores: result.scores,
    model: result.model,
  };

  const next = [
    record,
    ...readHistory().filter((item) => {
      const sameScan =
        item.className === record.className &&
        item.confidence === record.confidence &&
        item.bin === record.bin &&
        item.imageSrc === record.imageSrc;
      const recent = Date.now() - Date.parse(item.savedAt || 0) < 1200;
      return !(sameScan && recent);
    }),
  ];

  writeHistory(next);
  renderHistory();
}

function restoreHistoryRecord(record) {
  setPreviewImage(record.imageSrc || "");
  renderResult(record, `History: ${record.sourceLabel}`, {
    historyImageSrc: record.imageSrc,
    skipHistory: true,
  });
  updateStagePills(5, false);
  if (window.location.hash === "#scan") handleHashChange();
  else window.location.hash = "scan";
  showToast("History scan restored");
}

function historyTitle(record) {
  return `${record.className} - ${record.bin} - ${record.confidence}%`;
}

function renderHistory() {
  const history = readHistory();
  elements.historyCount.textContent = `${history.length} saved`;

  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.innerHTML = `
      <strong>No saved scans yet</strong>
      <span>Run a scan on the scanner page; every result is saved here automatically.</span>
      <a class="btn primary" href="#demo" data-page-link>Go to scanner</a>
    `;
    elements.historyList.replaceChildren(empty);
    return;
  }

  elements.historyList.replaceChildren(
    ...history.map((record) => {
      // Built with DOM APIs instead of innerHTML so stored strings render as text.
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "history-tile";

      const image = document.createElement("span");
      image.className = "history-tile-image";
      if (record.imageSrc) {
        const img = document.createElement("img");
        img.src = record.imageSrc;
        img.alt = "";
        image.append(img);
      } else {
        const fallback = document.createElement("span");
        fallback.className = "history-tile-fallback";
        fallback.textContent = String(record.className || "?").slice(0, 1);
        image.append(fallback);
      }

      const body = document.createElement("span");
      body.className = "history-tile-body";
      const title = document.createElement("strong");
      title.textContent = historyTitle(record);
      const meta = document.createElement("small");
      meta.textContent = `${record.sourceLabel} · ${formatHistoryTime(record.savedAt)}`;
      const detail = document.createElement("small");
      detail.textContent = `${record.context} context · ${record.latency} ms`;
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = record.bin;
      body.append(title, meta, detail, badge);

      tile.append(image, body);
      tile.addEventListener("click", () => restoreHistoryRecord(record));
      return tile;
    }),
  );
}

/* ---------------------------------------------------------------- scanner rendering */

function renderBoxes(result) {
  syncImageBounds();
  elements.boxLayer.replaceChildren();
  if (!elements.imageStage.classList.contains("has-image")) return;

  result.detections.forEach((box) => {
    const boxConfidence = box.confidence ?? result.confidence;
    const labelClass = box.label || result.className;
    // Drop very-low-confidence boxes so the overlay isn't noisy.
    if (boxConfidence < REVIEW_BOX_DISPLAY_MIN && labelClass === "Background") return;

    const node = document.createElement("div");
    const label = document.createElement("span");
    const color = classColors[labelClass] || classColors.Background;
    node.className = "detection-box";
    node.style.setProperty("--box-color", color);
    node.style.left = `${box.x}%`;
    node.style.top = `${box.y}%`;
    node.style.width = `${box.w}%`;
    node.style.height = `${box.h}%`;
    label.textContent = `${labelClass} ${boxConfidence}%`;
    node.append(label);
    elements.boxLayer.append(node);
  });
}

function scoreValue(scores, material) {
  return clamp(Math.round(Number(scores?.[material] || 0)), 0, 100);
}

function buildTopPredictions(scores, providedPredictions) {
  const normalized = Array.isArray(providedPredictions)
    ? providedPredictions
        .map((item) => {
          const className = item.className || item.label || item.key;
          return {
            className,
            confidence: clamp(
              Math.round(Number(item.confidence ?? scores?.[className] ?? 0)),
              0,
              100,
            ),
          };
        })
        .filter((item) => materials.includes(item.className) && item.className !== "Background")
    : [];

  const seen = new Set(normalized.map((item) => item.className));
  const fromScores = materials
    .filter((material) => material !== "Background" && !seen.has(material))
    .map((material) => ({ className: material, confidence: scoreValue(scores, material) }));

  return [...normalized, ...fromScores]
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);
}

function renderTopPredictions(scores, topPredictions) {
  const rows = buildTopPredictions(scores, topPredictions).map((item, index) => {
    const row = document.createElement("div");
    const rank = document.createElement("span");
    const name = document.createElement("strong");
    const meter = document.createElement("span");
    const fill = document.createElement("span");
    const score = document.createElement("em");

    row.className = "top-prediction-row";
    row.style.setProperty("--score-color", classColors[item.className] || classColors.Background);
    rank.textContent = `${index + 1}`;
    name.textContent = item.className;
    meter.className = "top-prediction-meter";
    fill.style.width = `${item.confidence}%`;
    score.textContent = `${item.confidence}%`;

    meter.append(fill);
    row.append(rank, name, meter, score);
    return row;
  });

  elements.topPredictionList.replaceChildren(...rows);
}

function renderScores(scores, topPredictions = null) {
  renderTopPredictions(scores, topPredictions);

  const rows = materials.map((material) => {
    const value = scoreValue(scores, material);
    const row = document.createElement("div");
    const name = document.createElement("strong");
    const track = document.createElement("span");
    const fill = document.createElement("span");
    const score = document.createElement("em");

    row.className = "score-row";
    row.style.setProperty("--score-color", classColors[material] || classColors.Background);
    track.className = "score-track";
    name.textContent = material;
    fill.style.width = `${value}%`;
    score.textContent = `${value}%`;

    track.append(fill);
    row.append(name, track, score);
    return row;
  });

  elements.scoreList.replaceChildren(...rows);
  elements.scoreSummary.textContent = `Top 3 / ${materials.length} classes`;
}

function updateStagePills(stageIdx, isScanning) {
  elements.stagePills.forEach((li, i) => {
    const done = stageIdx > i || stageIdx === 5;
    const active = stageIdx === i && isScanning;
    li.dataset.state = done ? "done" : active ? "active" : "pending";
  });
}

function renderResult(rawResult, sourceLabel, options = {}) {
  const result = makeResult(rawResult);
  const possibleMaterial = result.possibleMaterial || {
    className: result.className,
    confidence: result.confidence,
    source: "classifier",
  };
  const possibleText = possibleMaterial?.className
    ? `${possibleMaterial.className} ${possibleMaterial.confidence}%`
    : "--";
  const displayConfidence = result.confidence;
  const confidenceTone = displayConfidence >= 85 ? "high" : displayConfidence >= 60 ? "medium" : "low";

  elements.resultColumn.dataset.resultState = "waste";
  elements.resultColumn.dataset.confidenceTone = confidenceTone;
  elements.resultTopLabel.textContent = "Top material";
  elements.confidenceLabel.textContent = "Confidence";
  elements.resultClass.textContent = result.className;
  elements.resultConfidence.textContent = `${displayConfidence}%`;
  elements.confidenceMeter.style.width = `${displayConfidence}%`;
  elements.binRoute.classList.remove("is-idle");
  elements.binRoute.querySelector("span").textContent = "Bin route";
  elements.binRoute.querySelector("strong").textContent = result.bin;
  elements.possibleMaterialValue.textContent = possibleText;
  elements.contextValue.textContent = result.context;
  elements.latencyValue.textContent = `${result.latency} ms`;

  renderBoxes(result);
  renderScores(result.scores, result.topPredictions);
  if (!options.skipHistory) {
    addHistoryRecord(
      result,
      sourceLabel,
      options.historyImageSrc || options.imageSrc || result.image || elements.previewImage.currentSrc,
    );
  }
}

async function predictWithApi(blob, fileName) {
  const form = new FormData();
  form.append("image", blob, fileName);
  const response = await fetch("api/predict", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  const result = await response.json();
  if (result.error) throw new Error(result.error);
  setModelStatus("api", "Real model");
  return result;
}

/* ---------------------------------------------------------------- scan sequencing */

function clearScanTimers() {
  scanTimeouts.forEach(clearTimeout);
  scanTimeouts = [];
}

function scheduleStage(fn, ms) {
  scanTimeouts.push(window.setTimeout(fn, ms));
}

function performScan({ src, sourceLabel, apiBlob, apiFileName, fallbackResult, historyImageSrc }) {
  clearScanTimers();
  const token = (scanToken += 1);
  scanning = true;
  setLoading(true);
  setPreviewImage(src);
  updateStagePills(0, true);
  scheduleStage(() => updateStagePills(1, true), 300);

  const minWait = new Promise((resolve) => scheduleStage(resolve, 1400));
  const apiPromise = apiBlob
    ? predictWithApi(apiBlob, apiFileName).catch((error) => {
        console.warn("Model API unavailable, using fallback:", error);
        setModelStatus("fallback", "Simulated fallback");
        return null;
      })
    : Promise.resolve(null);

  Promise.all([apiPromise, minWait]).then(([apiResult]) => {
    if (token !== scanToken) return;
    const result = apiResult || { ...fallbackResult, mode: fallbackResult.mode || "Simulated fallback" };
    updateStagePills(2, true);
    scheduleStage(() => updateStagePills(3, true), 300);
    scheduleStage(() => updateStagePills(4, true), 600);
    scheduleStage(() => {
      scanning = false;
      updateStagePills(5, false);
      setLoading(false);
      renderResult(result, sourceLabel, { imageSrc: src, historyImageSrc });
    }, 1200);
  });
}

function pickSample(id) {
  const sample = SAMPLES.find((s) => s.id === id);
  if (!sample) return;
  const fallbackResult = { ...sample.result, mode: "Simulated fallback" };
  fetch(sample.src)
    .then((r) => r.blob())
    .then((blob) => {
      performScan({
        src: sample.src,
        sourceLabel: sample.label,
        apiBlob: blob,
        apiFileName: `${sample.id}.jpg`,
        fallbackResult,
        historyImageSrc: sample.src,
      });
    })
    .catch(() => {
      performScan({
        src: sample.src,
        sourceLabel: sample.label,
        apiBlob: null,
        apiFileName: null,
        fallbackResult,
        historyImageSrc: sample.src,
      });
    });
}

function makeHistoryImageSrc(image) {
  const maxSide = 900;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return "";

  const scale = Math.min(1, maxSide / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.74);
}

function makeApiImageBlob(image, fileName) {
  const maxSide = 1280;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return Promise.resolve(null);

  const scale = Math.min(1, maxSide / Math.max(width, height));
  if (scale === 1 && image.src.startsWith("blob:")) {
    return Promise.resolve(null);
  }

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => resolve(blob ? new File([blob], fileName.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg" }) : null),
      "image/jpeg",
      0.82,
    );
  });
}

function buildScores(className, confidence, stats) {
  const base = {
    Plastic: 24 + Math.round(stats.saturation * 18),
    Glass: 18 + Math.round(stats.blueLift * 24),
    Metal: 22 + Math.round((1 - stats.saturation) * 30),
    Paper: 20 + Math.round(stats.brightness * 24),
    Cardboard: 18 + Math.round(stats.redLift * 22),
    Organic: 16 + Math.round(stats.greenRatio * 58),
    Background: stats.brightness < 0.22 ? 44 : 8,
  };

  return withStrongestScore(className, confidence, base);
}

function chooseClass(stats, fileName) {
  const name = fileName.toLowerCase();

  if (name.includes("plastic") || name.includes("bottle")) return "Plastic";
  if (name.includes("glass")) return "Glass";
  if (name.includes("metal") || name.includes("can")) return "Metal";
  if (name.includes("paper")) return "Paper";
  if (name.includes("cardboard")) return "Cardboard";
  if (name.includes("organic") || name.includes("food")) return "Organic";

  if (stats.greenRatio > 0.3) return "Organic";
  if (stats.blueLift > 0.22 && stats.brightness > 0.42) return "Glass";
  if (stats.saturation < 0.2 && stats.brightness < 0.46) return "Metal";
  if (stats.redLift > 0.18 && stats.brightness > 0.48) return "Cardboard";
  if (stats.brightness > 0.62 && stats.saturation < 0.34) return "Paper";
  return "Plastic";
}

function inferContext(stats) {
  if (stats.greenRatio > 0.3) return "Grass";
  if (stats.blueLift > 0.18 && stats.brightness > 0.55) return "Outdoor";
  if (stats.brightness < 0.38) return "Indoor";
  return "Mixed";
}

function readImageStats(image) {
  const canvas = document.createElement("canvas");
  const size = 96;
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);

  const { data } = context.getImageData(0, 0, size, size);
  let red = 0;
  let green = 0;
  let blue = 0;
  let greenPixels = 0;
  let saturationTotal = 0;

  for (let index = 0; index < data.length; index += 4) {
    const r = data[index] / 255;
    const g = data[index + 1] / 255;
    const b = data[index + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    red += r;
    green += g;
    blue += b;
    saturationTotal += max === 0 ? 0 : (max - min) / max;

    if (g > r * 1.08 && g > b * 1.08 && g > 0.22) {
      greenPixels += 1;
    }
  }

  const pixels = data.length / 4;
  const avgRed = red / pixels;
  const avgGreen = green / pixels;
  const avgBlue = blue / pixels;
  const brightness = (avgRed + avgGreen + avgBlue) / 3;

  return {
    brightness,
    greenRatio: greenPixels / pixels,
    saturation: saturationTotal / pixels,
    blueLift: clamp(avgBlue - (avgRed + avgGreen) / 2, 0, 1),
    redLift: clamp(avgRed - (avgGreen + avgBlue) / 2, 0, 1),
  };
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("Unsupported file");
    return;
  }

  if (uploadObjectUrl) {
    URL.revokeObjectURL(uploadObjectUrl);
  }

  uploadObjectUrl = URL.createObjectURL(file);
  const fallbackImage = new Image();
  fallbackImage.onload = async () => {
    const stats = readImageStats(fallbackImage);
    const historyImageSrc = makeHistoryImageSrc(fallbackImage);
    const apiFile = (await makeApiImageBlob(fallbackImage, file.name)) || file;
    const className = chooseClass(stats, file.name);
    const confidence = clamp(
      Math.round(64 + stats.saturation * 18 + stats.brightness * 14),
      61,
      91,
    );
    const fallbackResult = {
      className,
      confidence,
      context: inferContext(stats),
      latency: clamp(Math.round(34 + apiFile.size / 42000), 38, 118),
      mode: "Browser fallback",
      detections: [{ x: 22, y: 18, w: 52, h: 60, label: className, confidence }],
      scores: buildScores(className, confidence, stats),
    };
    performScan({
      src: uploadObjectUrl,
      sourceLabel: "Uploaded image",
      apiBlob: apiFile,
      apiFileName: apiFile.name,
      fallbackResult,
      historyImageSrc,
    });
  };
  fallbackImage.onerror = () => {
    showToast("Image error");
    setLoading(false);
  };
  fallbackImage.src = uploadObjectUrl;
}

elements.fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragging");
  });
});

elements.dropZone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer.files[0]);
});

document.querySelectorAll("#samplePicker button").forEach((button) => {
  button.addEventListener("click", () => pickSample(button.dataset.sample));
});

elements.clearHistory.addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
  showToast("Scan history cleared");
});

const toast = document.querySelector("#toast");

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 1600);
}

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle("is-visible", entry.isIntersecting);
      });
    },
    { threshold: 0.24 },
  );
  document.querySelectorAll(".reveal-on-scroll").forEach((node) => revealObserver.observe(node));
}

/* --------------------------------------------- scroll story: steps + progress */

// Story paragraphs brighten as they cross the middle of the viewport.
// body.story-js gates the dimmed base state, so no JS (or reduced motion)
// means fully readable static text.
function initStorySteps() {
  const steps = document.querySelectorAll(".scene-story p");
  if (!steps.length || !("IntersectionObserver" in window) || reduceMotion) return;
  document.body.classList.add("story-js");
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle("is-active", entry.isIntersecting);
      });
    },
    { rootMargin: "-15% 0px -28% 0px" },
  );
  steps.forEach((p) => io.observe(p));
  // ponytail: paragraphs near a chapter's end may never reach the IO band on
  // tall viewports — brighten the visible chapter's steps at scroll bottom
  const brightenTail = () => {
    if (window.innerHeight + window.scrollY < document.documentElement.scrollHeight - 2) return;
    steps.forEach((p) => {
      const chapter = p.closest(".pres-section");
      if (!chapter || !chapter.hidden) p.classList.add("is-active");
    });
  };
  window.addEventListener("scroll", brightenTail, { passive: true });
}

function initScrollProgress() {
  const bar = document.createElement("div");
  bar.className = "scroll-progress";
  bar.setAttribute("aria-hidden", "true");
  bar.innerHTML = "<span></span>";
  document.body.append(bar);
  const fill = bar.firstElementChild;
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    fill.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
  };
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

window.addEventListener("resize", syncImageBounds);
elements.previewImage.addEventListener("load", syncImageBounds);

/* ---------------------------------------------------------------- hero: typewriter */

function startTypewriter() {
  const typedWord = document.querySelector("#typedWord");
  if (!typedWord) return;
  if (reduceMotion) {
    typedWord.textContent = WORDS[0];
    return;
  }
  let wordIndex = 0;
  let charIndex = 0;
  let direction = 1;
  const tick = () => {
    const word = WORDS[wordIndex];
    charIndex += direction;
    if (charIndex > word.length) {
      direction = 0;
      typedWord.textContent = word;
      window.setTimeout(() => {
        direction = -1;
        tick();
      }, 1700);
      return;
    }
    if (charIndex <= 0) {
      direction = 1;
      wordIndex = (wordIndex + 1) % WORDS.length;
      charIndex = 0;
    }
    typedWord.textContent = WORDS[wordIndex].slice(0, Math.max(0, charIndex));
    window.setTimeout(tick, direction === -1 ? 38 : 75);
  };
  tick();
}

/* ---------------------------------------------------------------- hero: count-up */

function countUp(el, target, decimals, duration, formatter) {
  if (!el) return;
  if (reduceMotion) {
    el.textContent = formatter ? formatter(target) : target.toFixed(decimals);
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const value = target * eased;
    el.textContent = formatter ? formatter(value) : value.toFixed(decimals);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function startHeroCounts() {
  countUp(document.querySelector("#heroAcc"), 92.93, 2, 1100, (v) => `${v.toFixed(2)}%`);
  countUp(document.querySelector("#heroClasses"), 7, 0, 900, (v) => `${Math.round(v)}`);
  countUp(document.querySelector("#heroBoxes"), 81, 0, 1100, (v) => `${Math.round(v)}k`);
}

let evidenceCounted = false;
function startEvidenceCounts() {
  if (evidenceCounted) return;
  evidenceCounted = true;
  countUp(document.querySelector("#correctMetric"), 92.93, 2, 1100, (v) => `${v.toFixed(2)}%`);
  countUp(document.querySelector("#riskMetric"), 74.9, 1, 1100, (v) => `${v.toFixed(1)}%`);
  countUp(document.querySelector("#scanMetric"), 2406, 0, 1200, (v) => Math.round(v).toLocaleString());
  countUp(document.querySelector("#domainMetric"), 39.0, 1, 1100, (v) => `${v.toFixed(1)}%`);
  // YOLO26m clean-val numbers (runs/audits/detector_clean_val_yolo26m_final100.json).
  // Keep in sync with the static values in index.html - these animations overwrite them.
  countUp(document.querySelector("#yoloPrecision"), 83.4, 1, 1100, (v) => `${v.toFixed(1)}%`);
  countUp(document.querySelector("#yoloRecall"), 67.2, 1, 1100, (v) => `${v.toFixed(1)}%`);
  countUp(document.querySelector("#yoloMap5095"), 57.0, 1, 1100, (v) => `${v.toFixed(1)}%`);
}

// fire the metric count-ups when the s7 metric cards scroll into view
function initEvidenceCounts() {
  const host = document.querySelector("#s7 .metrics");
  if (!host) return;
  if (!("IntersectionObserver" in window)) {
    startEvidenceCounts();
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        startEvidenceCounts();
        io.disconnect();
      });
    },
    { threshold: 0.3 },
  );
  io.observe(host);
}

/* ---------------------------------------------------------------- hero: demo loop */

function startHeroDemoLoop() {
  const sweep = document.querySelector("#heroDemoSweep");
  const boxLayer = document.querySelector("#heroDemoBoxes");
  const status = document.querySelector("#heroDemoStatus");
  const resultBox = document.querySelector("#heroDemoResult");
  if (!sweep || !boxLayer || !status || !resultBox) return;

  const renderDemoBoxes = (shown) => {
    boxLayer.replaceChildren(
      ...HERO_DEMO_BOXES.map((b, i) => {
        const node = document.createElement("div");
        node.className = "detection-box demo-box";
        node.dataset.shown = i < shown ? "1" : "0";
        node.style.setProperty("--box-color", classColors[b.label]);
        node.style.left = `${b.x}%`;
        node.style.top = `${b.y}%`;
        node.style.width = `${b.w}%`;
        node.style.height = `${b.h}%`;
        const label = document.createElement("span");
        label.textContent = `${b.label} ${b.confidence}%`;
        node.append(label);
        return node;
      }),
    );
  };

  if (reduceMotion) {
    status.textContent = "RESULT READY";
    sweep.hidden = true;
    renderDemoBoxes(HERO_DEMO_BOXES.length);
    resultBox.dataset.done = "1";
    resultBox.querySelector("strong").textContent = "Plastic → Recycling · 92.4%";
    return;
  }

  let timers = [];
  const seq = (fn, ms) => timers.push(window.setTimeout(fn, ms));
  const run = () => {
    timers.forEach(clearTimeout);
    timers = [];
    status.textContent = "LOCALIZING…";
    sweep.hidden = false;
    renderDemoBoxes(0);
    resultBox.dataset.done = "0";
    resultBox.querySelector("strong").textContent = "Analyzing…";

    seq(() => {
      status.textContent = "VERIFYING CROPS…";
      sweep.hidden = true;
      let shown = 0;
      const reveal = () => {
        shown += 1;
        renderDemoBoxes(shown);
        if (shown < HERO_DEMO_BOXES.length) {
          seq(reveal, 320);
        } else {
          seq(() => {
            status.textContent = "RESULT READY";
            resultBox.dataset.done = "1";
            resultBox.querySelector("strong").textContent = "Plastic → Recycling · 92.4%";
            seq(run, 3200);
          }, 500);
        }
      };
      reveal();
    }, 1600);
  };
  run();
}

/* ---------------------------------------------------- figure type badges */

const FIG_TYPES = [
  [/^cm_/, "\u{1F9E9}", "confusion matrix"],
  [/^train_yolo26m_labels/, "\u{1F5BC}", "label grid"],
  [/^(train_|metric_)/, "\u{1F4C8}", "training curves"],
  [/^heat_/, "\u{1F525}", "heatmap"],
  [/^(cmp_|eda_|ml_)/, "\u{1F4CA}", "chart"],
  [/^(wf_|workflow_)/, "\u{1F5FA}", "diagram"],
  [/^(result_|app_screenshot|audit_)/, "\u{1F4F7}", "result photo"],
];

function injectFigBadges() {
  document.querySelectorAll("figure.fig > img").forEach((img) => {
    const name = (img.getAttribute("src") || "").split("/").pop();
    const hit = FIG_TYPES.find(([re]) => re.test(name));
    if (!hit) return;
    const badge = document.createElement("span");
    badge.className = "fig-type";
    badge.setAttribute("aria-hidden", "true");
    badge.textContent = `${hit[1]} ${hit[2]}`;
    img.closest("figure").prepend(badge);
  });
}

/* --------------------------------------------- new training round (s10) */

const NM_CLASS_SHORT = ["Pla", "Gla", "Met", "Pap", "Car", "Org", "BG"];
const NM_SWEEP_COLORS = {
  yolo26m: "#047857",
  yolov8m: "#d0a31f",
  yolo11m: "#2d72d9",
  rtdetr_l: "#7c3aed",
  fasterrcnn: "#d04437",
  retinanet: "#0e7490",
  fcos: "#6b7280",
};

function nmEl(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function nmEscape(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const nmVizObserver =
  "IntersectionObserver" in window
    ? new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add("go");
            entry.target.querySelectorAll("[data-w]").forEach((bar) => {
              bar.style.width = bar.dataset.w;
            });
            nmVizObserver.unobserve(entry.target);
          });
        },
        { threshold: 0.25 },
      )
    : null;

function nmObserve(card) {
  if (!card) return;
  if (nmVizObserver && !reduceMotion) {
    nmVizObserver.observe(card);
  } else {
    card.classList.add("go");
    card.querySelectorAll("[data-w]").forEach((bar) => {
      bar.style.width = bar.dataset.w;
    });
  }
}

/* horizontal bar boards ---------------------------------------------- */

function nmBarRow({ name, sub, bars, value, chip, deployed }) {
  const row = nmEl("div", `bar-row${deployed ? " is-deployed" : ""}`);
  const label = nmEl("div", "bar-name", `${name}${sub ? `<small>${sub}</small>` : ""}`);
  const track = nmEl("div", "bar-track");
  bars.forEach(({ w, muted }) => {
    const bar = nmEl("span", `bar${muted ? " muted" : ""}`);
    bar.dataset.w = `${Math.max(0, w)}%`;
    track.append(bar);
  });
  const val = nmEl("div", "bar-val", `<span class="val-num">${value}</span>${chip ? `<span class="lat-chip">${chip}</span>` : ""}`);
  row.append(label, track, val);
  return row;
}

function nmRenderSweepBars(data) {
  const host = document.getElementById("nmSweepBars");
  if (!host) return;
  const board = nmEl("div", "bar-board");
  const models = Object.entries(data.detector_sweep.models).sort(
    (a, b) => b[1].test.map50 - a[1].test.map50,
  );
  models.forEach(([id, m]) => {
    board.append(
      nmBarRow({
        name: m.nice,
        sub: `${m.group} · ${m.train_hours.toFixed(1)} h`,
        bars: [
          { w: (m.val.map50 / 0.8) * 100, muted: true },
          { w: (m.test.map50 / 0.8) * 100 },
        ],
        value: `${(m.test.map50 * 100).toFixed(1)}%`,
        chip: `val ${(m.val.map50 * 100).toFixed(1)}%`,
        deployed: id === "yolo26m",
      }),
    );
  });
  host.append(
    board,
    nmEl(
      "div",
      "board-legend",
      `<span><i style="background:var(--accent)"></i>test mAP50 (unseen sessions)</span>
       <span><i style="background:color-mix(in srgb, var(--info) 55%, var(--track-bg))"></i>validation mAP50</span>
       <span>bars scaled to 80%</span>`,
    ),
  );
  nmObserve(host.closest(".viz-card"));
}

function nmRenderBoard(data) {
  const host = document.getElementById("nmBoard");
  if (!host) return;
  const board = nmEl("div", "bar-board");
  const rows = [...data.stage1, { ...data.deployed, deployed: true }].sort(
    (a, b) => b.test_accuracy - a.test_accuracy,
  );
  const scale = (v) => ((v - 0.75) / 0.25) * 100;
  rows.forEach((m) => {
    board.append(
      nmBarRow({
        name: m.nice,
        sub: m.train_seconds ? `head trained in ${m.train_seconds.toFixed(0)} s` : "",
        bars: [
          { w: scale(m.val_accuracy), muted: true },
          { w: scale(m.test_accuracy) },
        ],
        value: `${(m.test_accuracy * 100).toFixed(1)}%`,
        chip: `${m.latency_ms.toFixed(1)} ms/crop`,
        deployed: Boolean(m.deployed),
      }),
    );
  });
  host.append(
    board,
    nmEl(
      "div",
      "board-legend",
      `<span><i style="background:var(--accent)"></i>test accuracy</span>
       <span><i style="background:color-mix(in srgb, var(--info) 55%, var(--track-bg))"></i>validation accuracy</span>
       <span>2,696 quarantined test crops · latency = median CPU ms per crop</span>`,
    ),
  );
  nmObserve(host.closest(".viz-card"));
}

/* per-class F1 heatmap ------------------------------------------------- */

function nmHeatColor(cell, rate, strongAt = 0.88) {
  const p = Math.max(4, Math.min(85, ((rate - 0.7) / 0.3) * 85));
  cell.style.background = `color-mix(in srgb, var(--accent) ${p}%, var(--surface-2))`;
  if (rate >= strongAt) cell.classList.add("strong-cell");
}

function nmRenderHeat(data) {
  const host = document.getElementById("nmHeat");
  if (!host) return;
  const models = [...data.stage1, data.deployed].sort((a, b) => b.test_macro_f1 - a.test_macro_f1);
  const grid = nmEl("div", "hm-grid");
  grid.style.gridTemplateColumns = `minmax(110px, auto) repeat(${data.classes.length}, 1fr)`;
  grid.append(nmEl("span", "hm-lab col"));
  NM_CLASS_SHORT.forEach((c) => grid.append(nmEl("span", "hm-lab col", c)));
  let idx = 0;
  models.forEach((m) => {
    grid.append(nmEl("span", "hm-lab", m.nice));
    data.classes.forEach((cls) => {
      const f1 = m.per_class[cls] ? m.per_class[cls].f1 : 0;
      const cell = nmEl("span", "hm-cell", (f1 * 100).toFixed(1));
      cell.title = `${m.nice} · ${cls}: F1 ${(f1 * 100).toFixed(2)}%`;
      nmHeatColor(cell, f1);
      cell.style.transitionDelay = `${(idx += 1) * 12}ms`;
      grid.append(cell);
    });
  });
  host.append(grid);
  nmObserve(host.closest(".viz-card"));
}

/* confusion matrix explorer -------------------------------------------- */

function nmRenderCm(host, model, classes) {
  host.replaceChildren();
  host.append(
    nmEl(
      "div",
      "cm-meta",
      `<span>test accuracy <strong>${(model.test_accuracy * 100).toFixed(2)}%</strong></span>
       <span>macro F1 <strong>${(model.test_macro_f1 * 100).toFixed(2)}%</strong></span>
       <span>CPU <strong>${model.latency_ms.toFixed(1)} ms/crop</strong></span>`,
    ),
  );
  const grid = nmEl("div", "hm-grid");
  grid.style.gridTemplateColumns = `minmax(46px, auto) repeat(${classes.length}, 1fr)`;
  grid.append(nmEl("span", "hm-lab col", "true ↓"));
  NM_CLASS_SHORT.forEach((c) => grid.append(nmEl("span", "hm-lab col", c)));
  model.cm.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    grid.append(nmEl("span", "hm-lab", NM_CLASS_SHORT[r]));
    row.forEach((count, c) => {
      const rate = count / total;
      const cell = nmEl("span", "hm-cell", String(count));
      cell.title = `${classes[r]} → ${classes[c]}: ${count} (${(rate * 100).toFixed(1)}%)`;
      if (r === c) {
        cell.style.background = `color-mix(in srgb, var(--accent) ${Math.round(rate * 85)}%, var(--surface-2))`;
        if (rate >= 0.8) cell.classList.add("strong-cell");
      } else {
        const p = Math.min(80, Math.round(rate * 260));
        cell.style.background = p < 3
          ? "var(--surface-2)"
          : `color-mix(in srgb, var(--danger) ${p}%, var(--surface-2))`;
        if (p >= 55) cell.classList.add("strong-cell");
      }
      cell.style.transitionDelay = reduceMotion ? "0ms" : `${(r * classes.length + c) * 9}ms`;
      grid.append(cell);
    });
  });
  host.append(grid);
  requestAnimationFrame(() =>
    requestAnimationFrame(() => grid.querySelectorAll(".hm-cell").forEach((c) => c.classList.add("go"))),
  );
}

function nmRenderCmExplorer(data) {
  const tabs = document.getElementById("nmCmTabs");
  const host = document.getElementById("nmCm");
  if (!tabs || !host) return;
  host.setAttribute("role", "tabpanel");
  data.stage1.forEach((model, i) => {
    const tab = nmEl(
      "button",
      "viz-tab",
      `${model.nice} <small>${(model.test_accuracy * 100).toFixed(1)}%</small>`,
    );
    tab.type = "button";
    tab.id = `nmCmTab${i}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", "nmCm");
    tab.setAttribute("aria-selected", String(i === 0));
    tab.addEventListener("click", () => {
      tabs.querySelectorAll(".viz-tab").forEach((t) => t.setAttribute("aria-selected", "false"));
      tab.setAttribute("aria-selected", "true");
      nmRenderCm(host, model, data.classes);
    });
    tabs.append(tab);
  });
  nmRenderCm(host, data.stage1[0], data.classes);
  nmObserve(host.closest(".viz-card"));
}

/* svg line charts ------------------------------------------------------- */

function nmLineChart({ width = 680, height = 330, margin, xDomain, yDomain, yTicks, xTicks, series }) {
  const m = margin || { l: 46, r: 118, t: 12, b: 30 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const sx = (x) => m.l + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * iw;
  const sy = (y) => m.t + ih - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * ih;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart-svg");
  let inner = "";
  yTicks.forEach((t) => {
    inner += `<line class="grid" x1="${m.l}" y1="${sy(t)}" x2="${m.l + iw}" y2="${sy(t)}"></line>`;
    inner += `<text class="tick-lab" x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  });
  xTicks.forEach((t) => {
    inner += `<text class="tick-lab" x="${sx(t)}" y="${height - 8}" text-anchor="middle">${t}</text>`;
  });
  inner += `<line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + ih}"></line>`;
  inner += `<line class="axis" x1="${m.l}" y1="${m.t + ih}" x2="${m.l + iw}" y2="${m.t + ih}"></line>`;
  svg.innerHTML = inner;
  const labelled = [];
  series.forEach(({ xs, ys, color, label, dashed, delay }, i) => {
    const d = xs.map((x, j) => `${j ? "L" : "M"}${sx(x).toFixed(1)} ${sy(ys[j]).toFixed(1)}`).join(" ");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("class", `series${dashed ? " dashed" : " draw"}`);
    path.setAttribute("stroke", color);
    if (!dashed) path.style.setProperty("--delay", `${delay !== undefined ? delay : i * 0.22}s`);
    svg.append(path);
    const lastX = sx(xs[xs.length - 1]);
    const lastY = sy(ys[ys.length - 1]);
    if (label) {
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", lastX);
      dot.setAttribute("cy", lastY);
      dot.setAttribute("r", 4);
      dot.setAttribute("fill", color);
      dot.setAttribute("class", "end-dot");
      dot.style.setProperty("--delay", `${i * 0.22}s`);
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", lastX + 9);
      text.setAttribute("fill", color);
      text.setAttribute("class", "end-lab");
      text.style.setProperty("--delay", `${i * 0.22}s`);
      text.textContent = label;
      svg.append(dot, text);
      labelled.push({ text, y: lastY + 4 });
    }
  });
  // push overlapping end labels apart (min 14px vertical separation)
  labelled.sort((a, b) => a.y - b.y);
  labelled.forEach((entry, i) => {
    if (i > 0 && entry.y < labelled[i - 1].y + 14) entry.y = labelled[i - 1].y + 14;
    entry.text.setAttribute("y", entry.y);
  });
  return svg;
}

function nmFinalizePathLengths(svg) {
  svg.querySelectorAll("path.series.draw").forEach((path) => {
    const len = Math.ceil(path.getTotalLength());
    path.style.setProperty("--len", String(len));
  });
}

function nmRenderSweepCurves(data) {
  const host = document.getElementById("nmSweepCurves");
  if (!host) return;
  const curves = data.detector_sweep.curves;
  const series = Object.entries(curves).map(([id, c]) => ({
    xs: c.epoch,
    ys: c.map50,
    color: NM_SWEEP_COLORS[id] || "#6b7280",
    label: `${data.detector_sweep.models[id].nice} ${(c.map50[c.map50.length - 1] * 100).toFixed(0)}%`,
  }));
  const svg = nmLineChart({
    xDomain: [1, 30],
    yDomain: [0, 0.8],
    yTicks: [0.2, 0.4, 0.6, 0.8],
    xTicks: [5, 10, 15, 20, 25, 30],
    series,
  });
  host.append(svg);
  nmFinalizePathLengths(svg);
  nmObserve(host.closest(".viz-card"));
}

function nmRenderTorchCurves(data) {
  const host = document.getElementById("nmTorchCurves");
  if (!host) return;
  const tc = data.torch_comparison;
  const colors = { resnet50: "#d04437", mobilenetv2: "#2d72d9" };
  const series = [];
  ["resnet50", "mobilenetv2"].forEach((id, i) => {
    const h = tc[id].history;
    const xs = h.val_accuracy.map((_, j) => j + 1);
    series.push({ xs, ys: h.accuracy, color: colors[id], dashed: true });
    series.push({
      xs,
      ys: h.val_accuracy,
      color: colors[id],
      label: `${tc[id].model_name} ${(tc[id].accuracy * 100).toFixed(1)}%`,
      delay: i * 0.25,
    });
  });
  const svg = nmLineChart({
    height: 300,
    xDomain: [1, 15],
    yDomain: [0.7, 1.0],
    yTicks: [0.7, 0.8, 0.9, 1.0],
    xTicks: [1, 3, 6, 9, 12, 15],
    series,
  });
  const chips = nmEl("div", "stat-chips");
  ["resnet50", "mobilenetv2"].forEach((id) => {
    const m = tc[id];
    chips.append(
      nmEl(
        "span",
        "stat-chip",
        `<i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${colors[id]}"></i>
         ${m.model_name} <strong>${(m.parameters / 1e6).toFixed(1)}M params</strong>
         <strong>${m.size_mb.toFixed(1)} MB</strong>
         <strong>${m.avg_latency_ms.toFixed(1)} ms</strong>
         <strong>best val-loss @ ep ${m.best_epoch_val_loss}</strong>`,
      ),
    );
  });
  host.append(chips, svg);
  nmFinalizePathLengths(svg);
  nmObserve(host.closest(".viz-card"));
}

/* raw log viewer --------------------------------------------------------- */

const NM_LOGS = [
  { label: "Stage-2 backbone sweep", url: "assets/data/stage1_train.log" },
  { label: "RT-DETR-l training", url: "assets/data/rtdetr_train.log" },
  { label: "Torch retrain pair", url: "assets/data/torch_cmp.log" },
];
const nmLogCache = {};

function nmHighlightLog(text) {
  return nmEscape(text)
    .split("\n")
    .map((line) => {
      if (/^===|Test accuracy|macro[_ ]?F1|\bDONE\b|\bSTART\b|best model|saved/i.test(line)) {
        return `<span class="log-hl">${line}</span>`;
      }
      if (/^\[INFO\]/.test(line)) return `<span class="log-dim">${line}</span>`;
      return line;
    })
    .join("\n");
}

function nmShowLog(host, url) {
  if (nmLogCache[url]) {
    host.innerHTML = nmLogCache[url];
    return;
  }
  host.textContent = "Loading log…";
  fetch(url)
    .then((res) => (res.ok ? res.text() : Promise.reject(new Error(String(res.status)))))
    .then((text) => {
      nmLogCache[url] = nmHighlightLog(text);
      host.innerHTML = nmLogCache[url];
    })
    .catch(() => {
      host.textContent = `Log unavailable offline — see ${url} in the repository.`;
    });
}

function nmRenderLogs() {
  const tabs = document.getElementById("nmLogTabs");
  const host = document.getElementById("nmLog");
  if (!tabs || !host) return;
  host.setAttribute("role", "tabpanel");
  NM_LOGS.forEach((log, i) => {
    const tab = nmEl("button", "viz-tab", log.label);
    tab.type = "button";
    tab.id = `nmLogTab${i}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", "nmLog");
    tab.setAttribute("aria-selected", String(i === 0));
    tab.addEventListener("click", () => {
      tabs.querySelectorAll(".viz-tab").forEach((t) => t.setAttribute("aria-selected", "false"));
      tab.setAttribute("aria-selected", "true");
      nmShowLog(host, log.url);
    });
    tabs.append(tab);
  });
  nmShowLog(host, NM_LOGS[0].url);
}

function initNewModels() {
  if (!document.getElementById("s10")) return;
  nmRenderLogs(); // independent of the metrics JSON
  fetch("assets/data/new_models.json")
    .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
    .then((data) => {
      nmRenderSweepBars(data);
      nmRenderSweepCurves(data);
      nmRenderBoard(data);
      nmRenderHeat(data);
      nmRenderCmExplorer(data);
      nmRenderTorchCurves(data);
    })
    .catch((err) => {
      document.querySelectorAll("#s10 .viz-body").forEach((body) => {
        body.textContent = "Chart data unavailable (assets/data/new_models.json failed to load).";
      });
      console.error("new_models.json", err);
    });
}

/* ---------------------------------------------------------------- init */

setModelStatus("pending", "Upload image");
renderScores(emptyScores());
renderHistory();
updateStagePills(0, false);
handleHashChange();
startTypewriter();
startHeroCounts();
startHeroDemoLoop();
injectFigBadges();
initNewModels();
initStorySteps();
initScrollProgress();
initEvidenceCounts();
