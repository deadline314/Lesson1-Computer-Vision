window.init_annealing = function() {
  const CANVAS_W = 500;
  const CANVAS_H = 400;
  const SCHED_W = 600;
  const SCHED_H = 250;
  const VS_W = 300;
  const VS_H = 240;

  function costFunction(x, y) {
    const a = 10;
    const n = 2;
    return a * n
      + (x * x - a * Math.cos(2 * Math.PI * x))
      + (y * y - a * Math.cos(2 * Math.PI * y))
      + 3 * Math.sin(x * 1.5) * Math.sin(y * 1.5);
  }

  const X_MIN = -4, X_MAX = 4, Y_MIN = -4, Y_MAX = 4;

  function toCanvasX(x, w) { return (x - X_MIN) / (X_MAX - X_MIN) * w; }
  function toCanvasY(y, h) { return (1 - (y - Y_MIN) / (Y_MAX - Y_MIN)) * h; }
  function fromCanvasX(cx, w) { return X_MIN + cx / w * (X_MAX - X_MIN); }
  function fromCanvasY(cy, h) { return Y_MIN + (1 - cy / h) * (Y_MAX - Y_MIN); }

  function heatColor(val, minVal, maxVal) {
    const t = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
    const r = Math.round(30 + (1 - t) * 180);
    const g = Math.round(60 + (1 - t) * 140 - t * 30);
    const b = Math.round(80 + t * 150);
    return `rgb(${r},${g},${b})`;
  }

  // ========== MAIN DEMO ==========
  const landscape = document.getElementById('saLandscape');
  if (!landscape) return;
  const ctx = landscape.getContext('2d');

  let saT0 = 1000, saAlpha = 0.995, saStepSize = 0.30, saSpeed = 5;
  let saRunning = false, saAnimId = null;
  let saX = 0, saY = 0, saBestX = 0, saBestY = 0;
  let saTemp = 1000, saIter = 0, saAccepted = 0, saTotal = 0;
  let saTrail = [];

  const COST_MIN = -5, COST_MAX = 60;
  let landscapeImg = null;

  function renderLandscape(w, h) {
    const img = ctx.createImageData(w, h);
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const x = fromCanvasX(px, w);
        const y = fromCanvasY(py, h);
        const c = costFunction(x, y);
        const t = Math.max(0, Math.min(1, (c - COST_MIN) / (COST_MAX - COST_MIN)));
        const r = Math.round(20 + (1 - t) * 200);
        const g = Math.round(50 + (1 - t) * 170 - t * 20);
        const b = Math.round(70 + t * 160);
        const i = (py * w + px) * 4;
        img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
      }
    }
    landscapeImg = img;
  }

  function drawLandscape() {
    if (landscapeImg) ctx.putImageData(landscapeImg, 0, 0);
  }

  function drawTrail() {
    for (let i = 0; i < saTrail.length; i++) {
      const p = saTrail[i];
      const cx = toCanvasX(p.x, CANVAS_W);
      const cy = toCanvasY(p.y, CANVAS_H);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      if (p.type === 'accept') ctx.fillStyle = 'rgba(106, 176, 76, 0.7)';
      else if (p.type === 'uphill') ctx.fillStyle = 'rgba(232, 160, 50, 0.8)';
      else ctx.fillStyle = 'rgba(200, 85, 61, 0.5)';
      ctx.fill();
    }
  }

  function drawCurrent() {
    const cx = toCanvasX(saX, CANVAS_W);
    const cy = toCanvasY(saY, CANVAS_H);
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();

    const bx = toCanvasX(saBestX, CANVAS_W);
    const by = toCanvasY(saBestY, CANVAS_H);
    ctx.beginPath();
    ctx.arc(bx, by, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'gold';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '11px Patrick Hand';
    ctx.fillStyle = '#333';
    ctx.fillText('best', bx + 9, by + 4);
  }

  function updateStats() {
    const el = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    el('saStatTemp', saTemp.toFixed(1));
    el('saStatCur', costFunction(saX, saY).toFixed(3));
    el('saStatBest', costFunction(saBestX, saBestY).toFixed(3));
    el('saStatIter', saIter);
    el('saStatRate', saTotal > 0 ? (saAccepted / saTotal * 100).toFixed(1) + '%' : '—');
    const fill = document.getElementById('saTempFill');
    if (fill) fill.style.height = Math.max(0, Math.min(100, saTemp / saT0 * 100)) + '%';
  }

  function saStep() {
    const nx = saX + (Math.random() - 0.5) * 2 * saStepSize;
    const ny = saY + (Math.random() - 0.5) * 2 * saStepSize;
    const clamped_nx = Math.max(X_MIN, Math.min(X_MAX, nx));
    const clamped_ny = Math.max(Y_MIN, Math.min(Y_MAX, ny));
    const curCost = costFunction(saX, saY);
    const newCost = costFunction(clamped_nx, clamped_ny);
    const dE = newCost - curCost;
    saTotal++;

    let accepted = false;
    let trailType = 'reject';
    if (dE <= 0) {
      accepted = true;
      trailType = 'accept';
    } else if (Math.random() < Math.exp(-dE / saTemp)) {
      accepted = true;
      trailType = 'uphill';
    }

    if (accepted) {
      saAccepted++;
      saTrail.push({ x: clamped_nx, y: clamped_ny, type: trailType });
      saX = clamped_nx;
      saY = clamped_ny;
      if (costFunction(saX, saY) < costFunction(saBestX, saBestY)) {
        saBestX = saX;
        saBestY = saY;
      }
    } else {
      saTrail.push({ x: clamped_nx, y: clamped_ny, type: 'reject' });
    }

    if (saTrail.length > 800) saTrail = saTrail.slice(-500);
    saTemp *= saAlpha;
    saIter++;
  }

  function animate() {
    if (!saRunning) return;
    for (let i = 0; i < saSpeed; i++) saStep();
    drawLandscape();
    drawTrail();
    drawCurrent();
    updateStats();
    saAnimId = requestAnimationFrame(animate);
  }

  function resetSA() {
    saRunning = false;
    if (saAnimId) cancelAnimationFrame(saAnimId);
    saX = (Math.random() - 0.5) * 4;
    saY = (Math.random() - 0.5) * 4;
    saBestX = saX; saBestY = saY;
    saTemp = saT0; saIter = 0; saAccepted = 0; saTotal = 0;
    saTrail = [];
    drawLandscape();
    drawCurrent();
    updateStats();
    const startBtn = document.getElementById('saStart');
    if (startBtn) startBtn.textContent = '▶ Start';
  }

  renderLandscape(CANVAS_W, CANVAS_H);

  const t0Slider = document.getElementById('saT0');
  const alphaSlider = document.getElementById('saAlpha');
  const stepSlider = document.getElementById('saStep');
  const speedSlider = document.getElementById('saSpeed');

  if (t0Slider) t0Slider.addEventListener('input', function() {
    saT0 = parseInt(this.value);
    document.getElementById('saT0Val').textContent = saT0;
  });
  if (alphaSlider) alphaSlider.addEventListener('input', function() {
    saAlpha = parseInt(this.value) / 1000;
    document.getElementById('saAlphaVal').textContent = saAlpha.toFixed(3);
  });
  if (stepSlider) stepSlider.addEventListener('input', function() {
    saStepSize = parseInt(this.value) / 100;
    document.getElementById('saStepVal').textContent = saStepSize.toFixed(2);
  });
  if (speedSlider) speedSlider.addEventListener('input', function() {
    saSpeed = parseInt(this.value);
    document.getElementById('saSpeedVal').textContent = saSpeed;
  });

  const startBtn = document.getElementById('saStart');
  const pauseBtn = document.getElementById('saPause');
  const resetBtn = document.getElementById('saReset');

  if (startBtn) startBtn.addEventListener('click', function() {
    if (!saRunning) {
      saRunning = true;
      this.textContent = '● Running';
      animate();
    }
  });
  if (pauseBtn) pauseBtn.addEventListener('click', function() {
    saRunning = false;
    if (startBtn) startBtn.textContent = '▶ Resume';
  });
  if (resetBtn) resetBtn.addEventListener('click', resetSA);

  landscape.addEventListener('click', function(e) {
    const rect = landscape.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    saX = fromCanvasX(px, CANVAS_W);
    saY = fromCanvasY(py, CANVAS_H);
    saBestX = saX; saBestY = saY;
    saTrail = [];
    saIter = 0; saAccepted = 0; saTotal = 0;
    saTemp = saT0;
    drawLandscape();
    drawCurrent();
    updateStats();
  });

  resetSA();

  // ========== TEMPERATURE SCHEDULE ==========
  const schedCanvas = document.getElementById('schedCanvas');
  if (schedCanvas) {
    const sctx = schedCanvas.getContext('2d');
    const schedStep = document.getElementById('schedStep');
    const TOTAL_STEPS = 1000;
    const T_INIT = 1000;

    function drawSchedule() {
      const currentStep = parseInt(schedStep.value);
      document.getElementById('schedStepVal').textContent = currentStep;
      const w = SCHED_W, h = SCHED_H;
      sctx.clearRect(0, 0, w, h);

      sctx.strokeStyle = 'rgba(0,0,0,0.08)';
      sctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = h * i / 5;
        sctx.beginPath(); sctx.moveTo(0, y); sctx.lineTo(w, y); sctx.stroke();
      }

      const schedules = [
        { color: '#c8553d', fn: n => T_INIT * Math.pow(0.995, n) },
        { color: '#4a87a8', fn: n => Math.max(0, T_INIT - (T_INIT / TOTAL_STEPS) * n) },
        { color: '#6ab04c', fn: n => n === 0 ? T_INIT : T_INIT / Math.log(1 + n) },
      ];

      schedules.forEach(s => {
        sctx.beginPath();
        sctx.strokeStyle = s.color;
        sctx.lineWidth = 2.5;
        for (let n = 0; n <= TOTAL_STEPS; n++) {
          const x = n / TOTAL_STEPS * w;
          const temp = Math.max(0, s.fn(n));
          const y = h - (temp / T_INIT) * h * 0.9 - h * 0.05;
          if (n === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
        }
        sctx.stroke();
      });

      const markerX = currentStep / TOTAL_STEPS * w;
      sctx.strokeStyle = 'rgba(0,0,0,0.3)';
      sctx.lineWidth = 1.5;
      sctx.setLineDash([4, 4]);
      sctx.beginPath(); sctx.moveTo(markerX, 0); sctx.lineTo(markerX, h); sctx.stroke();
      sctx.setLineDash([]);

      schedules.forEach(s => {
        const temp = Math.max(0, s.fn(currentStep));
        const y = h - (temp / T_INIT) * h * 0.9 - h * 0.05;
        sctx.beginPath();
        sctx.arc(markerX, y, 5, 0, Math.PI * 2);
        sctx.fillStyle = s.color;
        sctx.fill();
        sctx.font = '11px JetBrains Mono';
        sctx.fillText(temp.toFixed(1), markerX + 8, y - 4);
      });

      sctx.font = '12px Patrick Hand';
      sctx.fillStyle = '#666';
      sctx.fillText('Step 0', 4, h - 4);
      sctx.fillText('Step ' + TOTAL_STEPS, w - 65, h - 4);
      sctx.fillText('T₀=' + T_INIT, 4, 16);
    }

    if (schedStep) schedStep.addEventListener('input', drawSchedule);
    drawSchedule();
  }

  // ========== SA vs GRADIENT DESCENT ==========
  const gdCanvas = document.getElementById('gdCanvas');
  const saCanvas2 = document.getElementById('saCanvas2');
  if (gdCanvas && saCanvas2) {
    const gctx = gdCanvas.getContext('2d');
    const sctx2 = saCanvas2.getContext('2d');
    let vsRunning = false, vsAnimId = null;

    function cost1D(x) {
      return 2 + Math.sin(x * 2.5) * 2 + Math.sin(x * 0.8) * 3 + 0.05 * (x - 1) * (x - 1);
    }
    function cost1DGrad(x) {
      return 2.5 * Math.cos(x * 2.5) * 2 + 0.8 * Math.cos(x * 0.8) * 3 + 0.1 * (x - 1);
    }

    const VS_XMIN = -5, VS_XMAX = 5;
    let gdX = -3.5, saX2 = -3.5;
    let sa2Temp = 500, sa2Best = -3.5;
    let gdTrail = [], sa2Trail = [];
    let vsIter = 0;

    function drawVsLandscape(context, w, h, posX, trail, label, isGD) {
      context.clearRect(0, 0, w, h);
      context.fillStyle = 'var(--paper-light, #faf7f0)';
      context.fillRect(0, 0, w, h);

      context.beginPath();
      context.strokeStyle = 'rgba(0,0,0,0.08)';
      context.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = h * 0.1 + (h * 0.8) * i / 4;
        context.moveTo(0, y); context.lineTo(w, y);
      }
      context.stroke();

      const costMin = -4, costMax = 10;
      context.beginPath();
      context.strokeStyle = '#555';
      context.lineWidth = 2;
      for (let px = 0; px < w; px++) {
        const x = VS_XMIN + px / w * (VS_XMAX - VS_XMIN);
        const c = cost1D(x);
        const y = h * 0.1 + (1 - (c - costMin) / (costMax - costMin)) * h * 0.8;
        if (px === 0) context.moveTo(px, y); else context.lineTo(px, y);
      }
      context.stroke();

      trail.forEach((tx, i) => {
        const px = (tx - VS_XMIN) / (VS_XMAX - VS_XMIN) * w;
        const c = cost1D(tx);
        const py = h * 0.1 + (1 - (c - costMin) / (costMax - costMin)) * h * 0.8;
        context.beginPath();
        context.arc(px, py, 2, 0, Math.PI * 2);
        context.fillStyle = isGD ? 'rgba(74,135,168,0.3)' : 'rgba(106,176,76,0.3)';
        context.fill();
      });

      const px = (posX - VS_XMIN) / (VS_XMAX - VS_XMIN) * w;
      const c = cost1D(posX);
      const py = h * 0.1 + (1 - (c - costMin) / (costMax - costMin)) * h * 0.8;
      context.beginPath();
      context.arc(px, py, 8, 0, Math.PI * 2);
      context.fillStyle = isGD ? '#4a87a8' : '#6ab04c';
      context.fill();
      context.strokeStyle = '#333';
      context.lineWidth = 1.5;
      context.stroke();
    }

    function vsStep() {
      const lr = 0.05;
      const grad = cost1DGrad(gdX);
      gdX -= lr * grad;
      gdX = Math.max(VS_XMIN, Math.min(VS_XMAX, gdX));
      gdTrail.push(gdX);

      const nx = saX2 + (Math.random() - 0.5) * 1.2;
      const clampedNx = Math.max(VS_XMIN, Math.min(VS_XMAX, nx));
      const dE = cost1D(clampedNx) - cost1D(saX2);
      if (dE <= 0 || Math.random() < Math.exp(-dE / sa2Temp)) {
        saX2 = clampedNx;
        if (cost1D(saX2) < cost1D(sa2Best)) sa2Best = saX2;
      }
      sa2Trail.push(saX2);
      sa2Temp *= 0.995;
      vsIter++;

      if (gdTrail.length > 300) gdTrail = gdTrail.slice(-200);
      if (sa2Trail.length > 300) sa2Trail = sa2Trail.slice(-200);
    }

    function vsAnimate() {
      if (!vsRunning) return;
      for (let i = 0; i < 3; i++) vsStep();
      drawVsLandscape(gctx, VS_W, VS_H, gdX, gdTrail, 'GD', true);
      drawVsLandscape(sctx2, VS_W, VS_H, saX2, sa2Trail, 'SA', false);

      const el = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
      el('gdCost', cost1D(gdX).toFixed(3));
      el('saCost2', cost1D(saX2).toFixed(3));

      const gdStuck = gdTrail.length > 20 && Math.abs(gdTrail[gdTrail.length-1] - gdTrail[gdTrail.length-20]) < 0.001;
      el('gdStatus', gdStuck ? 'Stuck!' : 'Running');
      el('saStatus2', vsIter > 500 ? 'Converging' : 'Exploring');

      vsAnimId = requestAnimationFrame(vsAnimate);
    }

    function vsReset() {
      vsRunning = false;
      if (vsAnimId) cancelAnimationFrame(vsAnimId);
      gdX = -3.5; saX2 = -3.5; sa2Best = -3.5;
      sa2Temp = 500; vsIter = 0;
      gdTrail = []; sa2Trail = [];
      drawVsLandscape(gctx, VS_W, VS_H, gdX, gdTrail, 'GD', true);
      drawVsLandscape(sctx2, VS_W, VS_H, saX2, sa2Trail, 'SA', false);
      const el = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
      el('gdCost', '—'); el('saCost2', '—');
      el('gdStatus', 'Ready'); el('saStatus2', 'Ready');
    }

    const vsStartBtn = document.getElementById('vsStart');
    const vsResetBtn = document.getElementById('vsReset');
    if (vsStartBtn) vsStartBtn.addEventListener('click', function() {
      if (!vsRunning) { vsRunning = true; vsAnimate(); }
    });
    if (vsResetBtn) vsResetBtn.addEventListener('click', vsReset);
    vsReset();
  }

  // ========== ACCEPTANCE PROBABILITY CALCULATOR ==========
  const probDE = document.getElementById('probDE');
  const probT = document.getElementById('probT');
  if (probDE && probT) {
    function updateProb() {
      const dE = parseInt(probDE.value);
      const T = parseInt(probT.value);
      const p = Math.exp(-dE / T);
      const pct = (p * 100).toFixed(2);
      const el = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
      el('probDEVal', dE);
      el('probTVal', T);
      el('probDEDisp', dE);
      el('probTDisp', T);
      el('probResult', pct + '%');
      const bar = document.getElementById('probBar');
      if (bar) bar.style.width = pct + '%';
    }
    probDE.addEventListener('input', updateProb);
    probT.addEventListener('input', updateProb);
    updateProb();
  }
};
window.init_annealing();
