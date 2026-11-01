/**
 * Iranian License Plate Recognition (YOLOv8 ALPR)
 * Browser Real-time inference using ONNX Runtime Web
 */

// 41 Classes from YOLOv8 trained model
const CLASS_NAMES = [
  "plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "alef", "be", "pe", "te", "se", "jim", "dal", "sin", "shin",
  "sad", "ta", "za", "ein", "fe", "ghaf", "kaf", "gaf", "lam",
  "mim", "noon", "vav", "he", "ye", "ze", "taxi", "disabled",
  "police", "D", "S", "protocol"
];

// Mapping labels to Persian characters & Persian digits for display
const FA_MAP = {
  "0": "۰", "1": "۱", "2": "۲", "3": "۳", "4": "۴",
  "5": "۵", "6": "۶", "7": "۷", "8": "۸", "9": "۹",
  "alef": "الف", "be": "ب", "pe": "پ", "te": "ت", "se": "ث",
  "jim": "ج", "dal": "د", "sin": "س", "shin": "ش", "sad": "ص",
  "ta": "ط", "za": "ظ", "ein": "ع", "fe": "ف", "ghaf": "ق",
  "kaf": "ک", "gaf": "گ", "lam": "ل", "mim": "م", "noon": "ن",
  "vav": "و", "he": "ه", "ye": "ی", "ze": "ز",
  "taxi": "تکسی", "disabled": "♿", "police": "پلیس",
  "D": "D", "S": "S", "protocol": "تشریفات"
};

let session = null;
let isProcessing = false;
let videoStream = null;
let animationFrameId = null;

// FPS tracking
let frameTimes = [];
const FPS_WINDOW = 15;

// DOM Elements
const canvas = document.getElementById("displayCanvas");
const ctx = canvas.getContext("2d");
const video = document.getElementById("webcam");
const modelStatus = document.getElementById("modelStatus");
const loadModelBtn = document.getElementById("loadModelBtn");
const hfModelUrlInput = document.getElementById("hfModelUrl");
const localModelFileInput = document.getElementById("localModelFile");
const cameraBtn = document.getElementById("cameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const mediaFileInput = document.getElementById("mediaFileInput");
const confThresholdInput = document.getElementById("confThreshold");
const iouThresholdInput = document.getElementById("iouThreshold");
const confValDisplay = document.getElementById("confVal");
const iouValDisplay = document.getElementById("iouVal");
const fpsDisplay = document.getElementById("fpsDisplay");
const latencyDisplay = document.getElementById("latencyDisplay");
const detectedCountDisplay = document.getElementById("detectedCount");
const platesPrintBox = document.getElementById("platesPrintBox");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const overlayText = document.getElementById("overlayText");
const backendBadge = document.getElementById("backendBadge");

// Radio toggles
const modelSourceRadios = document.querySelectorAll('input[name="modelSource"]');
const hfInputGroup = document.getElementById("hfInputGroup");
const localInputGroup = document.getElementById("localInputGroup");

modelSourceRadios.forEach(radio => {
  radio.addEventListener("change", (e) => {
    if (e.target.value === "hf") {
      hfInputGroup.classList.remove("hidden");
      localInputGroup.classList.add("hidden");
    } else {
      hfInputGroup.classList.add("hidden");
      localInputGroup.classList.remove("hidden");
    }
  });
});

confThresholdInput.addEventListener("input", (e) => {
  confValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
});

iouThresholdInput.addEventListener("input", (e) => {
  iouValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
});

clearHistoryBtn.addEventListener("click", () => {
  platesPrintBox.innerHTML = '<div class="empty-state">No plates detected yet. Start camera or upload an image/video.</div>';
});

// Configure ONNX Runtime Web
if (typeof ort !== "undefined") {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
}

function showLoading(msg) {
  loadingOverlay.classList.remove("hidden");
  overlayText.textContent = msg;
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function setStatus(msg, type = "info") {
  modelStatus.textContent = msg;
  modelStatus.className = `status-msg status-${type}`;
}

/**
 * Load Model from HF CDN URL or Local ArrayBuffer
 */
async function loadModel(source) {
  showLoading("Loading YOLOv8 ONNX model into WebAssembly engine...");
  setStatus("Loading model...", "info");

  try {
    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };

    if (source instanceof ArrayBuffer) {
      session = await ort.InferenceSession.create(source, sessionOptions);
    } else {
      session = await ort.InferenceSession.create(source, sessionOptions);
    }

    backendBadge.textContent = "Engine: WebAssembly (WASM)";
    setStatus("Model loaded successfully! Ready for inference.", "success");
    console.log("Model loaded. Input names:", session.inputNames, "Output names:", session.outputNames);
  } catch (err) {
    console.error("Failed to load model:", err);
    setStatus(`Failed to load model: ${err.message}`, "error");
    alert(`Could not load model: ${err.message}\nIf using Hugging Face URL, ensure internet connection or select 'Custom / Local .onnx'.`);
  } finally {
    hideLoading();
  }
}

loadModelBtn.addEventListener("click", () => {
  const url = hfModelUrlInput.value.trim();
  if (!url) return alert("Please specify Hugging Face ONNX URL.");
  loadModel(url);
});

localModelFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (event) => {
    await loadModel(event.target.result);
  };
  reader.readAsArrayBuffer(file);
});

/**
 * Image preprocessing for YOLOv8 (640x640 letterbox + RGB normalization)
 */
function preprocess(imageSource) {
  const modelW = 640;
  const modelH = 640;

  // Temporary canvas for letterboxing/resizing
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = modelW;
  tempCanvas.height = modelH;
  const tCtx = tempCanvas.getContext("2d");

  // Determine scaling & padding (Letterbox)
  const srcW = imageSource.videoWidth || imageSource.width;
  const srcH = imageSource.videoHeight || imageSource.height;

  const scale = Math.min(modelW / srcW, modelH / srcH);
  const nw = Math.round(srcW * scale);
  const nh = Math.round(srcH * scale);
  const padX = Math.floor((modelW - nw) / 2);
  const padY = Math.floor((modelH - nh) / 2);

  // Fill with gray (114, 114, 114) typical for YOLO
  tCtx.fillStyle = "#727272";
  tCtx.fillRect(0, 0, modelW, modelH);
  tCtx.drawImage(imageSource, 0, 0, srcW, srcH, padX, padY, nw, nh);

  const imgData = tCtx.getImageData(0, 0, modelW, modelH);
  const { data } = imgData;

  // CHW format Float32Array: shape [1, 3, 640, 640]
  const float32Data = new Float32Array(3 * modelW * modelH);
  const channelSize = modelW * modelH;

  for (let i = 0; i < channelSize; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    float32Data[i] = r;
    float32Data[i + channelSize] = g;
    float32Data[i + channelSize * 2] = b;
  }

  const tensor = new ort.Tensor("float32", float32Data, [1, 3, modelH, modelW]);
  return { tensor, scale, padX, padY, srcW, srcH };
}

/**
 * Non-Maximum Suppression (NMS)
 */
function nms(boxes, scores, iouThreshold) {
  const indices = Array.from(Array(boxes.length).keys())
    .sort((a, b) => scores[b] - scores[a]);

  const selected = [];
  while (indices.length > 0) {
    const current = indices.shift();
    selected.push(current);

    const [x1, y1, x2, y2] = boxes[current];
    const areaA = (x2 - x1) * (y2 - y1);

    for (let i = indices.length - 1; i >= 0; i--) {
      const idx = indices[i];
      const [bx1, by1, bx2, by2] = boxes[idx];
      const areaB = (bx2 - bx1) * (by2 - by1);

      const xx1 = Math.max(x1, bx1);
      const yy1 = Math.max(y1, by1);
      const xx2 = Math.min(x2, bx2);
      const yy2 = Math.min(y2, by2);

      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const union = areaA + areaB - inter;
      const iou = union > 0 ? inter / union : 0;

      if (iou >= iouThreshold) {
        indices.splice(i, 1);
      }
    }
  }
  return selected;
}

/**
 * Format recognized plate characters into standard Iranian license plate syntax:
 * Example: "12 | 345 ب 67" or "۱۲ ب ۳۴۵ ایران ۶۷"
 */
function formatPlate(charsList) {
  if (charsList.length === 0) return "No Chars";

  // Raw English names
  const rawNames = charsList.map(c => c.name);
  const faChars = charsList.map(c => FA_MAP[c.name] || c.name);

  if (charsList.length >= 7) {
    try {
      // Typically: 2 digits + 1 letter + 3 digits + 2 city digits
      const part1 = faChars.slice(0, 2).join("");
      const letter = faChars[2];
      const part2 = faChars.slice(3, 6).join("");
      const part3 = faChars.slice(6, 8).join("");
      return `${part1}  ${letter}  ${part2}  - ایران ${part3}`;
    } catch (e) {
      return faChars.join(" ");
    }
  }
  return faChars.join(" ");
}

/**
 * Postprocess YOLOv8 output tensor: [1, 45, 8400]
 */
function postprocess(outputTensor, scale, padX, padY, srcW, srcH) {
  const confThresh = parseFloat(confThresholdInput.value);
  const iouThresh = parseFloat(iouThresholdInput.value);

  const [_, numFeatures, numPredictions] = outputTensor.dims; // 1, 45, 8400
  const data = outputTensor.data;

  const rawBoxes = [];
  const rawScores = [];
  const rawClassIds = [];

  for (let i = 0; i < numPredictions; i++) {
    // cx, cy, w, h
    const cx = data[0 * numPredictions + i];
    const cy = data[1 * numPredictions + i];
    const w = data[2 * numPredictions + i];
    const h = data[3 * numPredictions + i];

    // Find highest class score among 41 classes (features 4..44)
    let maxScore = -Infinity;
    let maxClass = -1;
    for (let c = 0; c < 41; c++) {
      const score = data[(4 + c) * numPredictions + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    if (maxScore >= confThresh) {
      // Map coordinates back from letterboxed 640x640 to original srcW, srcH
      const x1 = Math.max(0, Math.min(srcW, (cx - w / 2 - padX) / scale));
      const y1 = Math.max(0, Math.min(srcH, (cy - h / 2 - padY) / scale));
      const x2 = Math.max(0, Math.min(srcW, (cx + w / 2 - padX) / scale));
      const y2 = Math.max(0, Math.min(srcH, (cy + h / 2 - padY) / scale));

      rawBoxes.push([x1, y1, x2, y2]);
      rawScores.push(maxScore);
      rawClassIds.push(maxClass);
    }
  }

  // Apply NMS
  const keep = nms(rawBoxes, rawScores, iouThresh);

  const detections = keep.map(idx => ({
    box: rawBoxes[idx],
    score: rawScores[idx],
    classId: rawClassIds[idx],
    className: CLASS_NAMES[rawClassIds[idx]] || `id_${rawClassIds[idx]}`
  }));

  // Separate plates and characters
  const plates = detections.filter(d => d.className === "plate");
  const characters = detections.filter(d => d.className !== "plate");

  const results = plates.map(plate => {
    const [px1, py1, px2, py2] = plate.box;

    // Filter characters inside plate box
    const insideChars = characters.filter(c => {
      const [cx1, cy1, cx2, cy2] = c.box;
      const centerCenterX = (cx1 + cx2) / 2;
      const centerCenterY = (cy1 + cy2) / 2;
      return centerCenterX >= px1 && centerCenterX <= px2 &&
             centerCenterY >= py1 && centerCenterY <= py2;
    });

    // Sort characters left to right
    insideChars.sort((a, b) => a.box[0] - b.box[0]);

    return {
      plateBox: plate.box,
      score: plate.score,
      characters: insideChars.map(c => ({
        name: c.className,
        box: c.box,
        score: c.score
      })),
      text: formatPlate(insideChars.map(c => ({ name: c.className, box: c.box })))
    };
  });

  return { results, detections };
}

/**
 * Render detections on Canvas & update Plate Print Box
 */
function render(imageSource, analysis) {
  const { results, detections } = analysis;
  const srcW = imageSource.videoWidth || imageSource.width;
  const srcH = imageSource.videoHeight || imageSource.height;

  canvas.width = srcW;
  canvas.height = srcH;
  ctx.drawImage(imageSource, 0, 0, srcW, srcH);

  // Draw detected characters first (smaller boxes)
  detections.forEach(d => {
    if (d.className !== "plate") {
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeStyle = "rgba(59, 130, 246, 0.8)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  });

  // Draw plates and decoded text
  results.forEach(res => {
    const [px1, py1, px2, py2] = res.plateBox;
    const pw = px2 - px1;
    const ph = py2 - py1;

    // Plate bounding box (green)
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 4;
    ctx.strokeRect(px1, py1, pw, ph);

    // Text banner background
    const label = `${res.text} (${(res.score * 100).toFixed(0)}%)`;
    ctx.font = "bold 20px -apple-system, sans-serif";
    const textWidth = ctx.measureText(label).width;
    const textH = 28;

    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.fillRect(px1, Math.max(0, py1 - textH - 6), textWidth + 16, textH + 6);
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1;
    ctx.strokeRect(px1, Math.max(0, py1 - textH - 6), textWidth + 16, textH + 6);

    // Text banner string
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, px1 + 8, Math.max(20, py1 - 8));
  });

  detectedCountDisplay.textContent = `Plates Found: ${results.length}`;

  // Append new detected plates to print box
  if (results.length > 0) {
    addPlatesToPrintBox(results);
  }
}

/**
 * Add recognized plate item to the print box
 */
let lastLoggedTime = 0;
function addPlatesToPrintBox(results) {
  const empty = platesPrintBox.querySelector(".empty-state");
  if (empty) empty.remove();

  const now = Date.now();
  // Prevent spamming UI if webcam shows same plate every 20ms
  if (now - lastLoggedTime < 800) return;
  lastLoggedTime = now;

  results.forEach(res => {
    const item = document.createElement("div");
    item.className = "plate-item";
    const timeStr = new Date().toLocaleTimeString();

    item.innerHTML = `
      <div class="plate-badge-container">
        <div class="plate-ir-strip"><span>I.R.</span><span>IRAN</span></div>
        <span>${res.text}</span>
      </div>
      <div class="plate-meta">
        <span>Confidence: ${(res.score * 100).toFixed(1)}%</span>
        <span>${timeStr}</span>
      </div>
    `;

    platesPrintBox.prepend(item);

    // Limit log size to 50
    if (platesPrintBox.children.length > 50) {
      platesPrintBox.removeChild(platesPrintBox.lastChild);
    }
  });
}

/**
 * Single Frame Inference
 */
async function processFrame(sourceElement) {
  if (!session || isProcessing) return;
  isProcessing = true;

  const t0 = performance.now();
  try {
    const pre = preprocess(sourceElement);
    const feeds = {};
    feeds[session.inputNames[0]] = pre.tensor;

    const outputMap = await session.run(feeds);
    const outputTensor = outputMap[session.outputNames[0]];

    const analysis = postprocess(outputTensor, pre.scale, pre.padX, pre.padY, pre.srcW, pre.srcH);
    render(sourceElement, analysis);

    const t1 = performance.now();
    const latency = t1 - t0;
    updateFPS(latency);
  } catch (err) {
    console.error("Inference error:", err);
  } finally {
    isProcessing = false;
  }
}

function updateFPS(latency) {
  frameTimes.push(performance.now());
  if (frameTimes.length > FPS_WINDOW) frameTimes.shift();

  let fps = 0;
  if (frameTimes.length > 1) {
    const elapsed = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
    fps = (frameTimes.length - 1) / elapsed;
  }

  fpsDisplay.textContent = `${fps.toFixed(1)} FPS`;
  latencyDisplay.textContent = `(${latency.toFixed(0)} ms)`;
}

/**
 * Camera Stream loop
 */
async function cameraLoop() {
  if (!videoStream) return;
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    await processFrame(video);
  }
  animationFrameId = requestAnimationFrame(cameraLoop);
}

// Camera Controls
cameraBtn.addEventListener("click", async () => {
  if (!session) {
    alert("Please load a model first!");
    return;
  }
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = videoStream;
    await video.play();

    cameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    cameraLoop();
  } catch (err) {
    console.error("Camera access error:", err);
    alert(`Could not start camera: ${err.message}`);
  }
});

stopCameraBtn.addEventListener("click", () => {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  cameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  fpsDisplay.textContent = "0.0 FPS";
});

/**
 * Media File Upload (Image or Video)
 */
mediaFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (!session) {
    alert("Please load a model first!");
    return;
  }

  // Stop camera if running
  if (videoStream) stopCameraBtn.click();

  const fileType = file.type;
  if (fileType.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        processFrame(img);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  } else if (fileType.startsWith("video/")) {
    const videoUrl = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = videoUrl;
    video.loop = true;
    video.play();

    const videoPlayHandler = () => {
      videoStream = true; // flag to keep loop active
      cameraLoop();
      video.removeEventListener("playing", videoPlayHandler);
    };
    video.addEventListener("playing", videoPlayHandler);
    stopCameraBtn.disabled = false;
  }
});

// Auto-load default HF model on page start
window.addEventListener("DOMContentLoaded", () => {
  const defaultUrl = hfModelUrlInput.value.trim();
  if (defaultUrl) {
    loadModel(defaultUrl);
  }
});
