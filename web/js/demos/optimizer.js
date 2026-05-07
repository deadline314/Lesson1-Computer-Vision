window.init_optimizer = function() {
  const canvas = document.getElementById('optCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  const COLORS = {
    sgd: '#c8553d',
    momentum: '#6b9b37',
    adam: '#4a87a8'
  };
  const TRAIL_MAX = 500;
  const CONTOUR_LEVELS = 20;

  let landscape = 'valley';
  let running = false;
  let animId = null;
  let iteration = 0;
  let startX = 0.75, startY = 0.25;

  const lrSlider = document.getElementById('optLrSlider');
  const momSlider = document.getElementById('optMomSlider');
  const lrVal = document.getElementById('optLrVal');
  const momVal = document.getElementById('optMomVal');
  const showSGD = document.getElementById('optShowSGD');
  const showMom = document.getElementById('optShowMom');
  const showAdam = document.getElementById('optShowAdam');

  function getLr() { return parseInt(lrSlider.value) / 1000; }
  function getMom() { return parseInt(momSlider.value) / 100; }

  lrSlider.addEventListener('input', () => { lrVal.textContent = getLr().toFixed(3); });
  momSlider.addEventListener('input', () => { momVal.textContent = getMom().toFixed(2); });

  function lossFunc(x, y) {
    if (landscape === 'valley') {
      const a = 10;
      return a * (y - x * x) * (y - x * x) + (1 - x) * (1 - x);
    } else if (landscape === 'saddle') {
      return x * x - y * y + 0.1 * x * x * x * x + 0.1 * y * y * y * y;
    } else {
      const s1 = Math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) * 8);
      const s2 = Math.exp(-((x + 0.5) ** 2 + (y + 0.8) ** 2) * 6) * 0.8;
      const s3 = Math.exp(-((x + 0.8) ** 2 + (y - 0.3) ** 2) * 10) * 0.6;
      return -(s1 + s2 + s3) + x * x * 0.05 + y * y * 0.05 + 1;
    }
  }

  function gradient(x, y) {
    const h = 1e-5;
    const dx = (lossFunc(x + h, y) - lossFunc(x - h, y)) / (2 * h);
    const dy = (lossFunc(x, y + h) - lossFunc(x, y - h)) / (2 * h);
    return [dx, dy];
  }

  const RANGE = { valley: {xmin: -1.5, xmax: 2, ymin: -1, ymax: 3},
                  saddle: {xmin: -2, xmax: 2, ymin: -2, ymax: 2},
                  multi: {xmin: -2, xmax: 2, ymin: -2, ymax: 2} };

  function getRange() { return RANGE[landscape]; }
  function toCanvas(x, y) {
    const r = getRange();
    return [(x - r.xmin) / (r.xmax - r.xmin) * W, (1 - (y - r.ymin) / (r.ymax - r.ymin)) * H];
  }
  function fromCanvas(cx, cy) {
    const r = getRange();
    return [r.xmin + cx / W * (r.xmax - r.xmin), r.ymax - cy / H * (r.ymax - r.ymin)];
  }

  let optimizers = {};

  function resetOptimizers() {
    const r = getRange();
    const sx = r.xmin + startX * (r.xmax - r.xmin);
    const sy = r.ymin + startY * (r.ymax - r.ymin);
    optimizers = {
      sgd: { x: sx, y: sy, trail: [[sx, sy]] },
      momentum: { x: sx, y: sy, vx: 0, vy: 0, trail: [[sx, sy]] },
      adam: { x: sx, y: sy, mx: 0, my: 0, vx: 0, vy: 0, t: 0, trail: [[sx, sy]] }
    };
    iteration = 0;
    updateStats();
  }

  function stepOptimizers() {
    const lr = getLr();
    const mu = getMom();
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    // SGD
    const o1 = optimizers.sgd;
    const [g1x, g1y] = gradient(o1.x, o1.y);
    o1.x -= lr * g1x;
    o1.y -= lr * g1y;
    if (o1.trail.length < TRAIL_MAX) o1.trail.push([o1.x, o1.y]);

    // Momentum
    const o2 = optimizers.momentum;
    const [g2x, g2y] = gradient(o2.x, o2.y);
    o2.vx = mu * o2.vx - lr * g2x;
    o2.vy = mu * o2.vy - lr * g2y;
    o2.x += o2.vx;
    o2.y += o2.vy;
    if (o2.trail.length < TRAIL_MAX) o2.trail.push([o2.x, o2.y]);

    // Adam
    const o3 = optimizers.adam;
    o3.t++;
    const [g3x, g3y] = gradient(o3.x, o3.y);
    o3.mx = beta1 * o3.mx + (1 - beta1) * g3x;
    o3.my = beta1 * o3.my + (1 - beta1) * g3y;
    o3.vx = beta2 * o3.vx + (1 - beta2) * g3x * g3x;
    o3.vy = beta2 * o3.vy + (1 - beta2) * g3y * g3y;
    const mxh = o3.mx / (1 - Math.pow(beta1, o3.t));
    const myh = o3.my / (1 - Math.pow(beta1, o3.t));
    const vxh = o3.vx / (1 - Math.pow(beta2, o3.t));
    const vyh = o3.vy / (1 - Math.pow(beta2, o3.t));
    o3.x -= lr * mxh / (Math.sqrt(vxh) + eps);
    o3.y -= lr * myh / (Math.sqrt(vyh) + eps);
    if (o3.trail.length < TRAIL_MAX) o3.trail.push([o3.x, o3.y]);

    iteration++;
  }

  function drawContours() {
    const r = getRange();
    const res = 4;
    const cols = Math.ceil(W / res);
    const rows = Math.ceil(H / res);
    const values = new Float32Array(cols * rows);
    let minV = Infinity, maxV = -Infinity;

    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const x = r.xmin + (i / cols) * (r.xmax - r.xmin);
        const y = r.ymax - (j / rows) * (r.ymax - r.ymin);
        let v = lossFunc(x, y);
        values[j * cols + i] = v;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }

    maxV = Math.min(maxV, minV + (maxV - minV) * 0.7);

    const imgData = ctx.createImageData(W, H);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        let v = values[j * cols + i];
        let t = Math.max(0, Math.min(1, (v - minV) / (maxV - minV)));
        t = Math.sqrt(t);

        const r1 = Math.round(255 - t * 80);
        const g1 = Math.round(250 - t * 100);
        const b1 = Math.round(240 - t * 60);

        for (let dy = 0; dy < res && j * res + dy < H; dy++) {
          for (let dx = 0; dx < res && i * res + dx < W; dx++) {
            const idx = ((j * res + dy) * W + (i * res + dx)) * 4;
            imgData.data[idx] = r1;
            imgData.data[idx + 1] = g1;
            imgData.data[idx + 2] = b1;
            imgData.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // contour lines
    ctx.strokeStyle = 'rgba(80, 60, 40, 0.15)';
    ctx.lineWidth = 1;
    for (let level = 0; level < CONTOUR_LEVELS; level++) {
      const threshold = minV + (level / CONTOUR_LEVELS) * (maxV - minV);
      for (let j = 0; j < rows - 1; j++) {
        for (let i = 0; i < cols - 1; i++) {
          const v00 = values[j * cols + i];
          const v10 = values[j * cols + i + 1];
          const v01 = values[(j + 1) * cols + i];
          const v11 = values[(j + 1) * cols + i + 1];
          const above = [(v00 >= threshold) ? 1 : 0, (v10 >= threshold) ? 1 : 0,
                         (v01 >= threshold) ? 1 : 0, (v11 >= threshold) ? 1 : 0];
          const sum = above[0] + above[1] + above[2] + above[3];
          if (sum > 0 && sum < 4) {
            ctx.beginPath();
            ctx.arc(i * res + res / 2, j * res + res / 2, 0.5, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }
  }

  function drawTrails() {
    const entries = [
      ['sgd', showSGD],
      ['momentum', showMom],
      ['adam', showAdam]
    ];

    entries.forEach(([key, checkbox]) => {
      if (!checkbox.checked) return;
      const trail = optimizers[key].trail;
      if (trail.length < 2) return;

      ctx.beginPath();
      ctx.strokeStyle = COLORS[key];
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.8;
      const [sx, sy] = toCanvas(trail[0][0], trail[0][1]);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < trail.length; i++) {
        const [px, py] = toCanvas(trail[i][0], trail[i][1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // draw ball
      const last = trail[trail.length - 1];
      const [bx, by] = toCanvas(last[0], last[1]);
      ctx.beginPath();
      ctx.arc(bx, by, 7, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[key];
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // glow
      const grd = ctx.createRadialGradient(bx, by, 0, bx, by, 14);
      grd.addColorStop(0, COLORS[key] + '44');
      grd.addColorStop(1, COLORS[key] + '00');
      ctx.beginPath();
      ctx.arc(bx, by, 14, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
    });
  }

  function drawStartMarker() {
    const r = getRange();
    const sx = r.xmin + startX * (r.xmax - r.xmin);
    const sy = r.ymin + startY * (r.ymax - r.ymin);
    const [cx, cy] = toCanvas(sx, sy);
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "12px 'Patrick Hand', cursive";
    ctx.fillStyle = '#555';
    ctx.fillText('start', cx + 8, cy - 4);
  }

  function drawMinimum() {
    let mx, my;
    if (landscape === 'valley') { mx = 1; my = 1; }
    else if (landscape === 'saddle') { mx = 0; my = 0; }
    else { mx = 0.5; my = 0.5; }
    const [cx, cy] = toCanvas(mx, my);
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#e8c547';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.font = "11px 'Patrick Hand', cursive";
    ctx.fillStyle = '#888';
    if (landscape !== 'saddle') ctx.fillText('min', cx + 7, cy + 3);
    else ctx.fillText('saddle', cx + 7, cy + 3);
  }

  function render() {
    drawContours();
    drawMinimum();
    drawStartMarker();
    drawTrails();
  }

  function updateStats() {
    const iterEl = document.getElementById('optIter');
    const sgdEl = document.getElementById('optLossSGD');
    const momEl = document.getElementById('optLossMom');
    const adamEl = document.getElementById('optLossAdam');
    if (iterEl) iterEl.textContent = iteration;
    if (sgdEl) sgdEl.textContent = lossFunc(optimizers.sgd.x, optimizers.sgd.y).toFixed(4);
    if (momEl) momEl.textContent = lossFunc(optimizers.momentum.x, optimizers.momentum.y).toFixed(4);
    if (adamEl) adamEl.textContent = lossFunc(optimizers.adam.x, optimizers.adam.y).toFixed(4);
  }

  function animLoop() {
    if (!running) return;
    for (let i = 0; i < 2; i++) stepOptimizers();
    render();
    updateStats();
    animId = requestAnimationFrame(animLoop);
  }

  document.getElementById('optStart').addEventListener('click', () => {
    if (running) {
      running = false;
      if (animId) cancelAnimationFrame(animId);
      document.getElementById('optStart').textContent = '▶ Start';
    } else {
      if (iteration === 0) resetOptimizers();
      running = true;
      document.getElementById('optStart').textContent = '⏸ Pause';
      animLoop();
    }
  });

  document.getElementById('optReset').addEventListener('click', () => {
    running = false;
    if (animId) cancelAnimationFrame(animId);
    document.getElementById('optStart').textContent = '▶ Start';
    resetOptimizers();
    render();
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    const r = getRange();
    startX = cx / W;
    startY = 1 - cy / H;
    running = false;
    if (animId) cancelAnimationFrame(animId);
    document.getElementById('optStart').textContent = '▶ Start';
    resetOptimizers();
    render();
  });

  function setLandscape(type) {
    landscape = type;
    running = false;
    if (animId) cancelAnimationFrame(animId);
    document.getElementById('optStart').textContent = '▶ Start';
    if (type === 'valley') { startX = 0.75; startY = 0.25; }
    else if (type === 'saddle') { startX = 0.6; startY = 0.6; }
    else { startX = 0.2; startY = 0.2; }
    resetOptimizers();
    render();
  }

  document.getElementById('optLandValley').addEventListener('click', () => setLandscape('valley'));
  document.getElementById('optLandSaddle').addEventListener('click', () => setLandscape('saddle'));
  document.getElementById('optLandMulti').addEventListener('click', () => setLandscape('multi'));

  // LR Scheduler Visualization
  const schedCanvas = document.getElementById('schedCanvas');
  if (schedCanvas) {
    const sctx = schedCanvas.getContext('2d');
    const SW = schedCanvas.width, SH = schedCanvas.height;
    const schedEpochSlider = document.getElementById('schedEpochSlider');
    const schedWarmupSlider = document.getElementById('schedWarmupSlider');
    const schedEpochVal = document.getElementById('schedEpochVal');
    const schedWarmupVal = document.getElementById('schedWarmupVal');

    function getSchedules(totalEpochs, warmupEpochs) {
      const baseLr = 1.0;
      const stepSize = Math.floor(totalEpochs / 3);
      const schedules = {
        step: [],
        cosine: [],
        onecycle: [],
        warmcos: []
      };

      for (let e = 0; e < totalEpochs; e++) {
        const t = e / (totalEpochs - 1);

        // StepLR: decay by 0.1 every stepSize
        const stepDecay = Math.pow(0.1, Math.floor(e / stepSize));
        schedules.step.push(baseLr * stepDecay);

        // CosineAnnealing
        schedules.cosine.push(baseLr * 0.5 * (1 + Math.cos(Math.PI * t)));

        // OneCycleLR
        const peak = 0.3;
        let onecycleVal;
        if (t < peak) {
          onecycleVal = baseLr * 0.1 + (baseLr - baseLr * 0.1) * (t / peak);
        } else {
          onecycleVal = baseLr * 0.5 * (1 + Math.cos(Math.PI * (t - peak) / (1 - peak)));
        }
        schedules.onecycle.push(onecycleVal);

        // Warmup + Cosine
        let warmcosVal;
        if (e < warmupEpochs) {
          warmcosVal = baseLr * (e + 1) / warmupEpochs;
        } else {
          const remaining = (e - warmupEpochs) / (totalEpochs - warmupEpochs);
          warmcosVal = baseLr * 0.5 * (1 + Math.cos(Math.PI * remaining));
        }
        schedules.warmcos.push(warmcosVal);
      }
      return schedules;
    }

    function drawSchedules() {
      const totalEpochs = parseInt(schedEpochSlider.value);
      const warmupEpochs = parseInt(schedWarmupSlider.value);
      schedEpochVal.textContent = totalEpochs;
      schedWarmupVal.textContent = warmupEpochs;

      const schedules = getSchedules(totalEpochs, warmupEpochs);

      sctx.clearRect(0, 0, SW, SH);

      // background
      sctx.fillStyle = '#fdfbf5';
      sctx.fillRect(0, 0, SW, SH);

      // axes
      const pad = { left: 50, right: 20, top: 20, bottom: 35 };
      const plotW = SW - pad.left - pad.right;
      const plotH = SH - pad.top - pad.bottom;

      sctx.strokeStyle = '#ccc';
      sctx.lineWidth = 1;
      sctx.beginPath();
      sctx.moveTo(pad.left, pad.top);
      sctx.lineTo(pad.left, pad.top + plotH);
      sctx.lineTo(pad.left + plotW, pad.top + plotH);
      sctx.stroke();

      // y-axis labels
      sctx.font = "11px 'JetBrains Mono', monospace";
      sctx.fillStyle = '#888';
      sctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + plotH * (1 - i / 4);
        const val = (i / 4).toFixed(2);
        sctx.fillText(val, pad.left - 6, y + 4);
        sctx.beginPath();
        sctx.strokeStyle = '#eee';
        sctx.moveTo(pad.left, y);
        sctx.lineTo(pad.left + plotW, y);
        sctx.stroke();
      }

      // x-axis label
      sctx.textAlign = 'center';
      sctx.fillText('0', pad.left, pad.top + plotH + 16);
      sctx.fillText(String(totalEpochs), pad.left + plotW, pad.top + plotH + 16);
      sctx.fillText('Epoch', pad.left + plotW / 2, pad.top + plotH + 30);

      // y-axis title
      sctx.save();
      sctx.translate(14, pad.top + plotH / 2);
      sctx.rotate(-Math.PI / 2);
      sctx.textAlign = 'center';
      sctx.fillText('LR', 0, 0);
      sctx.restore();

      const schedColors = { step: '#c8553d', cosine: '#4a87a8', onecycle: '#6b9b37', warmcos: '#7c5cbf' };
      const checkboxes = document.querySelectorAll('.schedCheck');
      const visible = {};
      checkboxes.forEach(cb => { visible[cb.dataset.sched] = cb.checked; });

      Object.entries(schedules).forEach(([key, values]) => {
        if (!visible[key]) return;
        sctx.beginPath();
        sctx.strokeStyle = schedColors[key];
        sctx.lineWidth = 2.5;
        values.forEach((v, i) => {
          const x = pad.left + (i / (values.length - 1)) * plotW;
          const y = pad.top + plotH * (1 - v);
          if (i === 0) sctx.moveTo(x, y);
          else sctx.lineTo(x, y);
        });
        sctx.stroke();
      });

      // legend
      let legendX = pad.left + 10;
      sctx.font = "11px 'Patrick Hand', cursive";
      Object.entries(schedColors).forEach(([key, color]) => {
        if (!visible[key]) return;
        sctx.fillStyle = color;
        sctx.fillRect(legendX, pad.top + 4, 14, 3);
        sctx.fillText(key, legendX + 18, pad.top + 10);
        legendX += sctx.measureText(key).width + 36;
      });
    }

    schedEpochSlider.addEventListener('input', drawSchedules);
    schedWarmupSlider.addEventListener('input', drawSchedules);
    document.querySelectorAll('.schedCheck').forEach(cb => cb.addEventListener('change', drawSchedules));
    drawSchedules();
  }

  // Quiz logic
  document.querySelectorAll('.quiz-option').forEach(opt => {
    opt.addEventListener('click', () => {
      const parent = opt.parentElement;
      const siblings = parent.querySelectorAll('.quiz-option');
      const feedback = opt.nextElementSibling && opt.nextElementSibling.classList.contains('quiz-feedback')
        ? opt.nextElementSibling : null;

      let fbEl = null;
      let sib = opt.nextElementSibling;
      while (sib) {
        if (sib.classList && sib.classList.contains('quiz-feedback')) { fbEl = sib; break; }
        if (sib.classList && sib.classList.contains('quiz-option')) { sib = sib.nextElementSibling; continue; }
        sib = sib.nextElementSibling;
      }

      const isCorrect = opt.dataset.correct === 'true';
      const group = [];
      let node = opt;
      while (node.previousElementSibling && node.previousElementSibling.classList.contains('quiz-option')) {
        node = node.previousElementSibling;
      }
      while (node && node.classList && node.classList.contains('quiz-option')) {
        group.push(node);
        node = node.nextElementSibling;
      }

      group.forEach(o => {
        o.style.opacity = '0.6';
        o.style.pointerEvents = 'none';
        if (o.dataset.correct === 'true') o.style.border = '2px solid var(--green)';
      });
      opt.style.opacity = '1';
      if (!isCorrect) opt.style.border = '2px solid var(--red)';

      if (fbEl) {
        fbEl.textContent = isCorrect ? '✓ 正確!' : '✗ 再想想,正確答案已標出。';
        fbEl.style.color = isCorrect ? 'var(--green)' : 'var(--red)';
        fbEl.style.display = 'block';
      }
    });
  });

  resetOptimizers();
  render();
};
window.init_optimizer();
