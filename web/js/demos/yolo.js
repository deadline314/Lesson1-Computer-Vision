window.init_yolo = function() {
  const yoloCanvas = document.getElementById('yoloCanvas');
  if (!yoloCanvas) return;
  const yoloCtx = yoloCanvas.getContext('2d');
  let yoloMode = 'anchor';
  let selectedCell = null;
  let nmsMode = false;

  const gtBoxes = [
    { x: 0.25, y: 0.35, w: 0.18, h: 0.30, label: 'person' },
    { x: 0.65, y: 0.55, w: 0.28, h: 0.22, label: 'car' },
    { x: 0.50, y: 0.78, w: 0.10, h: 0.10, label: 'dog' }
  ];

  function drawYolo() {
    const gridSize = parseInt(document.getElementById('gridSlider').value);
    const numAnchors = parseInt(document.getElementById('anchorSlider').value);
    const W = yoloCanvas.width, H = yoloCanvas.height;
    const cellW = W / gridSize, cellH = H / gridSize;

    const gridValEl = document.getElementById('gridVal');
    if (gridValEl) gridValEl.textContent = `${gridSize} × ${gridSize}`;
    const anchorValEl = document.getElementById('anchorVal');
    if (anchorValEl) anchorValEl.textContent = numAnchors;
    const gridInfoEl = document.getElementById('gridInfo');
    if (gridInfoEl) gridInfoEl.textContent = `${gridSize} × ${gridSize}`;
    const anchorInfoEl = document.getElementById('anchorInfo');
    if (anchorInfoEl) anchorInfoEl.textContent = yoloMode === 'anchor' ? numAnchors : 1;

    const numClasses = 20;
    const perAnchor = 5 + numClasses;
    const totalPerCell = (yoloMode === 'anchor' ? numAnchors : 1) * perAnchor;
    const perAnchorEl = document.getElementById('perAnchor');
    if (perAnchorEl) perAnchorEl.textContent = `5 + ${numClasses} = ${perAnchor}`;
    const outputShapeEl = document.getElementById('outputShape');
    if (outputShapeEl) outputShapeEl.textContent = `(B, ${gridSize}, ${gridSize}, ${totalPerCell})`;

    yoloCtx.clearRect(0, 0, W, H);

    yoloCtx.fillStyle = '#fff8e7';
    yoloCtx.fillRect(0, 0, W, H);

    gtBoxes.forEach(b => {
      yoloCtx.fillStyle = 'rgba(232, 156, 80, 0.2)';
      yoloCtx.strokeStyle = '#e89c50';
      yoloCtx.lineWidth = 2.5;
      const bx = (b.x - b.w/2) * W, by = (b.y - b.h/2) * H;
      yoloCtx.fillRect(bx, by, b.w * W, b.h * H);
      yoloCtx.strokeRect(bx, by, b.w * W, b.h * H);
      yoloCtx.fillStyle = '#e89c50';
      yoloCtx.beginPath();
      yoloCtx.arc(b.x * W, b.y * H, 5, 0, 2 * Math.PI);
      yoloCtx.fill();
      yoloCtx.fillStyle = '#3d2f1f';
      yoloCtx.font = '700 13px Kalam, cursive';
      yoloCtx.fillText(b.label, bx + 4, by - 6);
    });

    yoloCtx.strokeStyle = 'rgba(61, 47, 31, 0.3)';
    yoloCtx.lineWidth = 1;
    for (let i = 1; i < gridSize; i++) {
      yoloCtx.beginPath();
      yoloCtx.moveTo(i * cellW, 0); yoloCtx.lineTo(i * cellW, H); yoloCtx.stroke();
      yoloCtx.beginPath();
      yoloCtx.moveTo(0, i * cellH); yoloCtx.lineTo(W, i * cellH); yoloCtx.stroke();
    }

    gtBoxes.forEach(b => {
      const cellX = Math.floor(b.x * gridSize);
      const cellY = Math.floor(b.y * gridSize);
      yoloCtx.fillStyle = 'rgba(106, 155, 94, 0.25)';
      yoloCtx.fillRect(cellX * cellW, cellY * cellH, cellW, cellH);
      yoloCtx.strokeStyle = 'rgba(106, 155, 94, 0.9)';
      yoloCtx.lineWidth = 2.5;
      yoloCtx.strokeRect(cellX * cellW, cellY * cellH, cellW, cellH);
    });

    if (nmsMode) {
      const b = gtBoxes[1];
      const predictions = [];
      for (let i = 0; i < 8; i++) {
        const jitter = 0.05;
        predictions.push({
          x: b.x + (Math.random() - 0.5) * jitter,
          y: b.y + (Math.random() - 0.5) * jitter,
          w: b.w * (0.85 + Math.random() * 0.3),
          h: b.h * (0.85 + Math.random() * 0.3),
          conf: 0.4 + Math.random() * 0.55
        });
      }
      predictions.sort((a,b) => b.conf - a.conf);

      predictions.forEach((p, i) => {
        const isKept = i === 0;
        yoloCtx.strokeStyle = isKept ? '#c8553d' : 'rgba(139, 109, 181, 0.5)';
        yoloCtx.lineWidth = isKept ? 3 : 1.5;
        yoloCtx.setLineDash(isKept ? [] : [4, 3]);
        const bx = (p.x - p.w/2) * W, by = (p.y - p.h/2) * H;
        yoloCtx.strokeRect(bx, by, p.w * W, p.h * H);
        if (isKept) {
          yoloCtx.fillStyle = '#c8553d';
          yoloCtx.font = '700 12px JetBrains Mono';
          yoloCtx.fillText(`${p.conf.toFixed(2)} ✓`, bx + 4, by + 14);
        }
      });
      yoloCtx.setLineDash([]);
    }

    if (selectedCell) {
      const { cx, cy } = selectedCell;
      if (cx < gridSize && cy < gridSize) {
        const centerX = (cx + 0.5) * cellW;
        const centerY = (cy + 0.5) * cellH;
        yoloCtx.fillStyle = 'rgba(74, 135, 168, 0.25)';
        yoloCtx.fillRect(cx * cellW, cy * cellH, cellW, cellH);
        yoloCtx.strokeStyle = '#4a87a8';
        yoloCtx.lineWidth = 3;
        yoloCtx.strokeRect(cx * cellW, cy * cellH, cellW, cellH);

        if (yoloMode === 'anchor') {
          const anchorShapes = [
            { w: 0.10, h: 0.10 }, { w: 0.20, h: 0.20 },
            { w: 0.30, h: 0.18 }, { w: 0.15, h: 0.30 },
            { w: 0.08, h: 0.08 }
          ];
          const colors = ['#c8553d', '#e89c50', '#6a9b5e', '#4a87a8', '#8b6db5'];
          for (let i = 0; i < numAnchors; i++) {
            const a = anchorShapes[i % anchorShapes.length];
            const ax = centerX - a.w * W / 2, ay = centerY - a.h * H / 2;
            yoloCtx.strokeStyle = colors[i];
            yoloCtx.lineWidth = 2;
            yoloCtx.setLineDash([5, 4]);
            yoloCtx.strokeRect(ax, ay, a.w * W, a.h * H);
          }
          yoloCtx.setLineDash([]);
        } else {
          yoloCtx.fillStyle = '#4a87a8';
          yoloCtx.beginPath();
          yoloCtx.arc(centerX, centerY, 7, 0, 2 * Math.PI);
          yoloCtx.fill();
          yoloCtx.strokeStyle = '#4a87a8';
          yoloCtx.lineWidth = 2;
          yoloCtx.setLineDash([5, 4]);
          const predW = 0.25, predH = 0.20;
          yoloCtx.strokeRect(centerX - predW * W / 2, centerY - predH * H / 2, predW * W, predH * H);
          yoloCtx.setLineDash([]);
        }
      }
    }

    const infoEl = document.getElementById('cellInfo');
    if (infoEl && selectedCell) {
      const { cx, cy } = selectedCell;
      const gridSize2 = parseInt(document.getElementById('gridSlider').value);
      const numAnchors2 = parseInt(document.getElementById('anchorSlider').value);
      const isResponsible = gtBoxes.some(b => Math.floor(b.x * gridSize2) === cx && Math.floor(b.y * gridSize2) === cy);
      if (yoloMode === 'anchor') {
        infoEl.innerHTML = `
          <div><strong>Cell (${cx}, ${cy})</strong></div>
          <div style="margin-top: 8px; color: ${isResponsible ? 'var(--green)' : 'var(--ink-faint)'};">
            ${isResponsible ? '✓ 負責預測一個 GT box' : '○ 不負責(predict background)'}
          </div>
          <div style="margin-top: 10px;">輸出 ${numAnchors2} 個 anchor 的:</div>
          <div style="margin-left: 8px; margin-top: 4px; color: var(--ink-faint); font-family: 'JetBrains Mono'; font-size: 11px;">
            (tx, ty, tw, th, conf, c1...c20)<br>
            × ${numAnchors2} = ${numAnchors2 * 25} numbers
          </div>
        `;
      } else {
        infoEl.innerHTML = `
          <div><strong>Cell (${cx}, ${cy})</strong></div>
          <div style="margin-top: 8px; color: ${isResponsible ? 'var(--green)' : 'var(--ink-faint)'};">
            ${isResponsible ? '✓ 負責預測一個 GT box' : '○ 不負責'}
          </div>
          <div style="margin-top: 10px;">直接輸出:</div>
          <div style="margin-left: 8px; margin-top: 4px; color: var(--ink-faint); font-family: 'JetBrains Mono'; font-size: 11px;">
            (cx, cy, w, h, conf, c1...c20)<br>
            = 25 numbers (no anchor)
          </div>
        `;
      }
    } else if (infoEl) {
      infoEl.textContent = '點擊左邊任一格子來看 detail';
    }
  }

  yoloCanvas.addEventListener('click', (e) => {
    const rect = yoloCanvas.getBoundingClientRect();
    const scale = yoloCanvas.width / rect.width;
    const x = (e.clientX - rect.left) * scale;
    const y = (e.clientY - rect.top) * scale;
    const gridSize = parseInt(document.getElementById('gridSlider').value);
    selectedCell = {
      cx: Math.floor(x / yoloCanvas.width * gridSize),
      cy: Math.floor(y / yoloCanvas.height * gridSize)
    };
    drawYolo();
  });

  const gridSlider = document.getElementById('gridSlider');
  if (gridSlider) gridSlider.addEventListener('input', drawYolo);
  const anchorSlider = document.getElementById('anchorSlider');
  if (anchorSlider) anchorSlider.addEventListener('input', drawYolo);
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      yoloMode = btn.dataset.mode;
      const aSlider = document.getElementById('anchorSlider');
      if (aSlider) aSlider.disabled = (yoloMode === 'free');
      drawYolo();
    });
  });
  const nmsBtn = document.getElementById('nmsBtn');
  if (nmsBtn) {
    nmsBtn.addEventListener('click', () => {
      nmsMode = !nmsMode;
      nmsBtn.textContent = nmsMode ? '隱藏 NMS' : '顯示重複偵測 + NMS';
      drawYolo();
    });
  }
  drawYolo();

  // IoU canvas
  const iouCanvas = document.getElementById('iouCanvas');
  if (!iouCanvas) return;
  const iouCtx = iouCanvas.getContext('2d');
  let predBox = { x: 100, y: 100, w: 120, h: 100 };
  const gtBox = { x: 60, y: 60, w: 140, h: 130 };
  let iouDragging = false;
  let iouDragOffset = { x: 0, y: 0 };

  function computeIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    if (x2 <= x1 || y2 <= y1) return 0;
    const inter = (x2 - x1) * (y2 - y1);
    const union = a.w * a.h + b.w * b.h - inter;
    return inter / union;
  }

  function drawIoU() {
    const W = iouCanvas.width, H = iouCanvas.height;
    iouCtx.clearRect(0, 0, W, H);
    iouCtx.fillStyle = '#fff8e7';
    iouCtx.fillRect(0, 0, W, H);

    const x1 = Math.max(gtBox.x, predBox.x);
    const y1 = Math.max(gtBox.y, predBox.y);
    const x2 = Math.min(gtBox.x + gtBox.w, predBox.x + predBox.w);
    const y2 = Math.min(gtBox.y + gtBox.h, predBox.y + predBox.h);

    if (x2 > x1 && y2 > y1) {
      iouCtx.fillStyle = 'rgba(232, 197, 71, 0.5)';
      iouCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }

    iouCtx.strokeStyle = '#e89c50';
    iouCtx.lineWidth = 3;
    iouCtx.strokeRect(gtBox.x, gtBox.y, gtBox.w, gtBox.h);
    iouCtx.fillStyle = '#e89c50';
    iouCtx.font = '700 12px Kalam, cursive';
    iouCtx.fillText('GT', gtBox.x + 4, gtBox.y - 6);

    iouCtx.strokeStyle = '#4a87a8';
    iouCtx.lineWidth = 3;
    iouCtx.setLineDash([6, 4]);
    iouCtx.strokeRect(predBox.x, predBox.y, predBox.w, predBox.h);
    iouCtx.setLineDash([]);
    iouCtx.fillStyle = '#4a87a8';
    iouCtx.fillText('Pred', predBox.x + 4, predBox.y - 6);
    iouCtx.fillStyle = '#4a87a8';
    iouCtx.beginPath();
    iouCtx.arc(predBox.x + predBox.w/2, predBox.y + predBox.h/2, 6, 0, 2*Math.PI);
    iouCtx.fill();

    const iou = computeIoU(gtBox, predBox);
    const iouValEl = document.getElementById('iouVal');
    if (iouValEl) iouValEl.textContent = iou.toFixed(3);
    const status = iou >= 0.5 ? '✓ Match!' : iou >= 0.3 ? '~ 邊緣' : '✗ Miss';
    const statusEl = document.getElementById('iouStatus');
    if (statusEl) {
      statusEl.textContent = status;
      statusEl.style.color = iou >= 0.5 ? 'var(--green)' : iou >= 0.3 ? 'var(--orange)' : 'var(--red)';
    }
  }

  iouCanvas.addEventListener('mousedown', (e) => {
    const rect = iouCanvas.getBoundingClientRect();
    const scale = iouCanvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    if (mx >= predBox.x && mx <= predBox.x + predBox.w && my >= predBox.y && my <= predBox.y + predBox.h) {
      iouDragging = true;
      iouDragOffset = { x: mx - predBox.x, y: my - predBox.y };
    }
  });
  iouCanvas.addEventListener('mousemove', (e) => {
    if (!iouDragging) return;
    const rect = iouCanvas.getBoundingClientRect();
    const scale = iouCanvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    predBox.x = Math.max(0, Math.min(iouCanvas.width - predBox.w, mx - iouDragOffset.x));
    predBox.y = Math.max(0, Math.min(iouCanvas.height - predBox.h, my - iouDragOffset.y));
    drawIoU();
  });
  window.addEventListener('mouseup', () => { iouDragging = false; });
  iouCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const rect = iouCanvas.getBoundingClientRect();
    const scale = iouCanvas.width / rect.width;
    const t = e.touches[0];
    const mx = (t.clientX - rect.left) * scale;
    const my = (t.clientY - rect.top) * scale;
    if (mx >= predBox.x && mx <= predBox.x + predBox.w && my >= predBox.y && my <= predBox.y + predBox.h) {
      iouDragging = true;
      iouDragOffset = { x: mx - predBox.x, y: my - predBox.y };
    }
  });
  iouCanvas.addEventListener('touchmove', (e) => {
    if (!iouDragging) return;
    e.preventDefault();
    const rect = iouCanvas.getBoundingClientRect();
    const scale = iouCanvas.width / rect.width;
    const t = e.touches[0];
    const mx = (t.clientX - rect.left) * scale;
    const my = (t.clientY - rect.top) * scale;
    predBox.x = Math.max(0, Math.min(iouCanvas.width - predBox.w, mx - iouDragOffset.x));
    predBox.y = Math.max(0, Math.min(iouCanvas.height - predBox.h, my - iouDragOffset.y));
    drawIoU();
  });
  window.addEventListener('touchend', () => { iouDragging = false; });

  drawIoU();

  // NMS Interactive Demo
  (function() {
    const canvas = document.getElementById('nmsCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const threshSlider = document.getElementById('nmsThresh');
    const threshVal = document.getElementById('nmsThreshVal');
    const stepBtn = document.getElementById('nmsStepBtn');
    const autoBtn = document.getElementById('nmsAutoBtn');
    const resetBtn = document.getElementById('nmsResetBtn');
    const status = document.getElementById('nmsStatus');

    const INITIAL_BOXES = [
      { x: 60, y: 50, w: 120, h: 140, conf: 0.92 },
      { x: 80, y: 60, w: 110, h: 130, conf: 0.87 },
      { x: 70, y: 55, w: 115, h: 135, conf: 0.75 },
      { x: 180, y: 160, w: 100, h: 120, conf: 0.88 },
      { x: 190, y: 170, w: 95, h: 110, conf: 0.72 },
      { x: 185, y: 155, w: 105, h: 125, conf: 0.65 },
      { x: 30, y: 200, w: 90, h: 80, conf: 0.80 },
      { x: 40, y: 195, w: 85, h: 85, conf: 0.55 }
    ];

    let boxes, kept, eliminated, examining, pendingIdx, sortedIndices, currentKeptIdx;

    function computeIoUNms(a, b) {
      const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
      const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const union = a.w * a.h + b.w * b.h - inter;
      return union > 0 ? inter / union : 0;
    }

    function resetNMS() {
      boxes = INITIAL_BOXES.map(b => ({...b}));
      kept = new Set();
      eliminated = new Set();
      examining = -1;
      sortedIndices = boxes.map((_, i) => i).sort((a, b) => boxes[b].conf - boxes[a].conf);
      pendingIdx = 0;
      currentKeptIdx = -1;
      status.textContent = '點 Step 或 Auto Play 開始 NMS 流程。';
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, 320, 320);
      ctx.fillStyle = '#f5ecd9';
      ctx.fillRect(0, 0, 320, 320);
      ctx.strokeStyle = '#c4b289';
      ctx.strokeRect(0, 0, 320, 320);

      boxes.forEach((b, i) => {
        let color, lw;
        if (kept.has(i)) { color = '#6a9b5e'; lw = 3; }
        else if (eliminated.has(i)) { color = '#c8553d'; lw = 2; }
        else if (i === examining) { color = '#e8c547'; lw = 3; }
        else { color = 'rgba(74,135,168,0.6)'; lw = 2; }

        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        if (eliminated.has(i)) { ctx.setLineDash([4, 4]); } else { ctx.setLineDash([]); }
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);

        ctx.fillStyle = color;
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillText(b.conf.toFixed(2), b.x + 2, b.y - 4);
      });
    }

    function nmsStep() {
      const thresh = parseFloat(threshSlider.value);
      if (pendingIdx >= sortedIndices.length) {
        status.textContent = '✅ NMS 完成!綠色 = 最終保留的 boxes。';
        examining = -1;
        draw();
        return false;
      }

      const idx = sortedIndices[pendingIdx];
      if (eliminated.has(idx)) { pendingIdx++; return nmsStep(); }

      if (currentKeptIdx === -1) {
        kept.add(idx);
        currentKeptIdx = idx;
        examining = idx;
        status.textContent = `保留 box #${idx} (conf=${boxes[idx].conf.toFixed(2)}),檢查跟它重疊的 boxes...`;
        pendingIdx++;
        draw();
        return true;
      }

      let foundToEliminate = false;
      for (let j = pendingIdx; j < sortedIndices.length; j++) {
        const jIdx = sortedIndices[j];
        if (eliminated.has(jIdx) || kept.has(jIdx)) continue;
        const iou = computeIoUNms(boxes[currentKeptIdx], boxes[jIdx]);
        if (iou > thresh) {
          eliminated.add(jIdx);
          examining = jIdx;
          status.textContent = `Box #${jIdx} (conf=${boxes[jIdx].conf.toFixed(2)}) IoU=${iou.toFixed(2)} > ${thresh} → 淘汰!`;
          foundToEliminate = true;
          draw();
          return true;
        }
      }

      if (!foundToEliminate) {
        currentKeptIdx = -1;
        examining = -1;
        draw();
        return nmsStep();
      }
      return true;
    }

    let autoTimer = null;
    function autoPlay() {
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = '▶ Auto Play'; return; }
      autoBtn.textContent = '⏸ Stop';
      autoTimer = setInterval(() => {
        if (!nmsStep()) { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = '▶ Auto Play'; }
      }, 600);
    }

    threshSlider.addEventListener('input', () => { threshVal.textContent = parseFloat(threshSlider.value).toFixed(2); });
    stepBtn.addEventListener('click', nmsStep);
    autoBtn.addEventListener('click', autoPlay);
    resetBtn.addEventListener('click', () => { if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = '▶ Auto Play'; } resetNMS(); });
    resetNMS();
  })();
};
window.init_yolo();
