window.init_gradient = function() {
  const depthSlider = document.getElementById('depthSlider');
  const factorSlider = document.getElementById('factorSlider');
  const plainLayers = document.getElementById('plainLayers');
  const resLayers = document.getElementById('resLayers');
  if (!depthSlider || !factorSlider || !plainLayers || !resLayers) return;
  let bpAnimating = false;

  function renderGradient(animProgress) {
    const depth = parseInt(depthSlider.value);
    const factor = parseInt(factorSlider.value) / 100;
    document.getElementById('depthVal').textContent = depth + ' layers';
    document.getElementById('factorVal').textContent = factor.toFixed(2);

    plainLayers.innerHTML = '';
    resLayers.innerHTML = '';

    let plainFirstGrad = 0, resFirstGrad = 0;
    let plainEffCount = 0, resEffCount = 0;

    for (let i = 0; i < depth; i++) {
      const distFromOutput = depth - 1 - i;
      let plainGrad = Math.pow(factor, distFromOutput);
      let resGrad = Math.min(1, plainGrad + (1 - Math.pow(0.92, distFromOutput)));

      if (typeof animProgress === 'number') {
        const wavePos = depth - 1 - animProgress * depth;
        if (i > wavePos) {
          // Already passed
        } else if (i > wavePos - 3) {
          plainGrad = Math.min(1, plainGrad * 2.5);
          resGrad = Math.min(1, resGrad * 1.5);
        } else {
          plainGrad = 0.05;
          resGrad = 0.05;
        }
      }

      if (i === 0) { plainFirstGrad = Math.pow(factor, depth - 1); resFirstGrad = Math.min(1, Math.pow(factor, depth - 1) + (1 - Math.pow(0.92, depth - 1))); }
      const realPlain = Math.pow(factor, depth - 1 - i);
      const realRes = Math.min(1, realPlain + (1 - Math.pow(0.92, depth - 1 - i)));
      if (realPlain > 0.01) plainEffCount++;
      if (realRes > 0.01) resEffCount++;

      const pBar = document.createElement('div');
      pBar.className = 'layer-bar';
      const pIntensity = Math.max(0.05, plainGrad);
      pBar.style.background = `rgb(${Math.round(106 + (200-106)*pIntensity)}, ${Math.round(155 + (85-155)*pIntensity)}, ${Math.round(94 + (61-94)*pIntensity)})`;
      pBar.style.opacity = Math.max(0.15, plainGrad);
      plainLayers.appendChild(pBar);

      const rBar = document.createElement('div');
      rBar.className = 'layer-bar';
      const rIntensity = Math.max(0.05, resGrad);
      rBar.style.background = `rgb(${Math.round(106 + (200-106)*rIntensity)}, ${Math.round(155 + (85-155)*rIntensity)}, ${Math.round(94 + (61-94)*rIntensity)})`;
      rBar.style.opacity = Math.max(0.15, resGrad);
      resLayers.appendChild(rBar);
    }

    const plainFirstEl = document.getElementById('plainFirst');
    if (plainFirstEl) plainFirstEl.textContent = plainFirstGrad < 0.001 ? '~0' : plainFirstGrad.toExponential(2);
    const resFirstEl = document.getElementById('resFirst');
    if (resFirstEl) resFirstEl.textContent = resFirstGrad.toFixed(3);
    const plainEffEl = document.getElementById('plainEff');
    if (plainEffEl) plainEffEl.textContent = plainEffCount + ' / ' + depth;
    const resEffEl = document.getElementById('resEff');
    if (resEffEl) resEffEl.textContent = resEffCount + ' / ' + depth;
  }

  depthSlider.addEventListener('input', () => renderGradient());
  factorSlider.addEventListener('input', () => renderGradient());

  const bpAnimBtn = document.getElementById('bpAnim');
  if (bpAnimBtn) {
    bpAnimBtn.addEventListener('click', () => {
      if (bpAnimating) return;
      bpAnimating = true;
      bpAnimBtn.textContent = '⏸ 播放中...';
      let progress = 0;
      const interval = setInterval(() => {
        progress += 0.03;
        if (progress >= 1.1) {
          clearInterval(interval);
          bpAnimating = false;
          bpAnimBtn.textContent = '⏵ 播放動畫';
          renderGradient();
        } else {
          renderGradient(progress);
        }
      }, 60);
    });
  }

  renderGradient();

  // ═══════════════════════════════════════════════════════════════
  // SECTION 2: Gradient Descent 2D Visualizer
  // ═══════════════════════════════════════════════════════════════
  const gdCanvas = document.getElementById('gdCanvas');
  if (gdCanvas) {
    const gdCtx = gdCanvas.getContext('2d');
    const GD_W = gdCanvas.width;
    const GD_H = gdCanvas.height;
    const GD_RANGE = 3.0;
    const GD_MAX_STEPS = 500;
    const GD_STEP_DELAY = 30;

    let gdFn = 'bowl';
    let gdTrail = [];
    let gdAnimId = null;
    let gdVx = 0, gdVy = 0;

    function lossFunction(x, y) {
      switch (gdFn) {
        case 'bowl': return x * x + y * y;
        case 'rosenbrock': return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        case 'saddle': return x * x - y * y;
        case 'rastrigin': return 20 + x*x - 10*Math.cos(2*Math.PI*x) + y*y - 10*Math.cos(2*Math.PI*y);
        default: return x * x + y * y;
      }
    }

    function lossGradient(x, y) {
      const h = 0.0001;
      const dfdx = (lossFunction(x + h, y) - lossFunction(x - h, y)) / (2 * h);
      const dfdy = (lossFunction(x, y + h) - lossFunction(x, y - h)) / (2 * h);
      return [dfdx, dfdy];
    }

    function worldToCanvas(wx, wy) {
      return [
        (wx + GD_RANGE) / (2 * GD_RANGE) * GD_W,
        (GD_RANGE - wy) / (2 * GD_RANGE) * GD_H
      ];
    }

    function canvasToWorld(cx, cy) {
      return [
        (cx / GD_W) * 2 * GD_RANGE - GD_RANGE,
        GD_RANGE - (cy / GD_H) * 2 * GD_RANGE
      ];
    }

    function drawContour() {
      const imgData = gdCtx.createImageData(GD_W, GD_H);
      let maxVal = 0;
      const vals = new Float32Array(GD_W * GD_H);

      for (let py = 0; py < GD_H; py++) {
        for (let px = 0; px < GD_W; px++) {
          const [wx, wy] = canvasToWorld(px, py);
          const v = lossFunction(wx, wy);
          vals[py * GD_W + px] = v;
          if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
        }
      }

      for (let i = 0; i < vals.length; i++) {
        const norm = Math.min(1, Math.log(1 + Math.abs(vals[i])) / Math.log(1 + maxVal));
        const r = Math.round(245 - norm * 160);
        const g = Math.round(236 - norm * 130);
        const b = Math.round(217 - norm * 100);
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = 255;
      }
      gdCtx.putImageData(imgData, 0, 0);

      gdCtx.strokeStyle = 'rgba(61, 47, 31, 0.25)';
      gdCtx.lineWidth = 1;
      const numContours = 15;
      const contourStep = maxVal / numContours;
      for (let level = 1; level <= numContours; level++) {
        const threshold = level * contourStep;
        gdCtx.beginPath();
        for (let py = 1; py < GD_H - 1; py += 2) {
          for (let px = 1; px < GD_W - 1; px += 2) {
            const v = vals[py * GD_W + px];
            const vr = vals[py * GD_W + px + 1];
            const vb = vals[(py + 1) * GD_W + px];
            if ((v < threshold && vr >= threshold) || (v >= threshold && vr < threshold) ||
                (v < threshold && vb >= threshold) || (v >= threshold && vb < threshold)) {
              gdCtx.rect(px, py, 1, 1);
            }
          }
        }
        gdCtx.stroke();
      }
    }

    function drawTrail() {
      if (gdTrail.length < 2) return;
      gdCtx.strokeStyle = 'rgba(200, 85, 61, 0.8)';
      gdCtx.lineWidth = 2.5;
      gdCtx.setLineDash([]);
      gdCtx.beginPath();
      const [sx, sy] = worldToCanvas(gdTrail[0][0], gdTrail[0][1]);
      gdCtx.moveTo(sx, sy);
      for (let i = 1; i < gdTrail.length; i++) {
        const [cx, cy] = worldToCanvas(gdTrail[i][0], gdTrail[i][1]);
        gdCtx.lineTo(cx, cy);
      }
      gdCtx.stroke();

      for (let i = 0; i < gdTrail.length; i++) {
        const [cx, cy] = worldToCanvas(gdTrail[i][0], gdTrail[i][1]);
        gdCtx.fillStyle = i === gdTrail.length - 1 ? '#c8553d' : 'rgba(200, 85, 61, 0.5)';
        gdCtx.beginPath();
        gdCtx.arc(cx, cy, i === gdTrail.length - 1 ? 6 : 3, 0, Math.PI * 2);
        gdCtx.fill();
        gdCtx.strokeStyle = '#3d2f1f';
        gdCtx.lineWidth = 1.5;
        gdCtx.stroke();
      }
    }

    function renderGD() {
      drawContour();
      drawTrail();
      const stepEl = document.getElementById('gdStep');
      const lossEl = document.getElementById('gdLoss');
      if (stepEl) stepEl.textContent = Math.max(0, gdTrail.length - 1);
      if (lossEl && gdTrail.length > 0) {
        const last = gdTrail[gdTrail.length - 1];
        lossEl.textContent = lossFunction(last[0], last[1]).toFixed(4);
      }
    }

    function startGD(startX, startY) {
      if (gdAnimId) { clearTimeout(gdAnimId); gdAnimId = null; }
      const lr = parseInt(document.getElementById('gdLrSlider').value) / 1000;
      const mom = parseInt(document.getElementById('gdMomSlider').value) / 100;
      gdTrail = [[startX, startY]];
      gdVx = 0; gdVy = 0;

      function step() {
        const last = gdTrail[gdTrail.length - 1];
        const [gx, gy] = lossGradient(last[0], last[1]);
        gdVx = mom * gdVx - lr * gx;
        gdVy = mom * gdVy - lr * gy;
        const nx = last[0] + gdVx;
        const ny = last[1] + gdVy;

        if (Math.abs(nx) > GD_RANGE * 2 || Math.abs(ny) > GD_RANGE * 2) {
          renderGD(); return;
        }
        gdTrail.push([nx, ny]);
        renderGD();

        const gradMag = Math.sqrt(gx * gx + gy * gy);
        if (gdTrail.length < GD_MAX_STEPS && gradMag > 1e-6) {
          gdAnimId = setTimeout(step, GD_STEP_DELAY);
        }
      }
      step();
    }

    gdCanvas.addEventListener('click', function(e) {
      const rect = gdCanvas.getBoundingClientRect();
      const scaleX = GD_W / rect.width;
      const scaleY = GD_H / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;
      const [wx, wy] = canvasToWorld(cx, cy);
      startGD(wx, wy);
    });

    document.getElementById('gdLrSlider').addEventListener('input', function() {
      document.getElementById('gdLrVal').textContent = (this.value / 1000).toFixed(3);
    });
    document.getElementById('gdMomSlider').addEventListener('input', function() {
      document.getElementById('gdMomVal').textContent = (this.value / 100).toFixed(2);
    });

    const fnBtns = ['gdFnBowl', 'gdFnRosen', 'gdFnSaddle', 'gdFnRastrigin'];
    const fnNames = ['bowl', 'rosenbrock', 'saddle', 'rastrigin'];
    fnBtns.forEach(function(id, i) {
      const btn = document.getElementById(id);
      if (!btn) return;
      btn.addEventListener('click', function() {
        fnBtns.forEach(function(bid) { document.getElementById(bid).dataset.active = 'false'; });
        btn.dataset.active = 'true';
        gdFn = fnNames[i];
        gdTrail = [];
        if (gdAnimId) { clearTimeout(gdAnimId); gdAnimId = null; }
        renderGD();
      });
    });

    document.getElementById('gdReset').addEventListener('click', function() {
      if (gdAnimId) { clearTimeout(gdAnimId); gdAnimId = null; }
      gdTrail = [];
      gdVx = 0; gdVy = 0;
      renderGD();
    });
    document.getElementById('gdClear').addEventListener('click', function() {
      if (gdAnimId) { clearTimeout(gdAnimId); gdAnimId = null; }
      gdTrail = [];
      renderGD();
    });

    renderGD();
  }

  // ═══════════════════════════════════════════════════════════════
  // SECTION 3: Learning Rate Effect Demo (3 parallel 1D animations)
  // ═══════════════════════════════════════════════════════════════
  const lrSmallCanvas = document.getElementById('lrSmallCanvas');
  const lrGoodCanvas = document.getElementById('lrGoodCanvas');
  const lrBigCanvas = document.getElementById('lrBigCanvas');

  if (lrSmallCanvas && lrGoodCanvas && lrBigCanvas) {
    const LR_CONFIGS = [
      { canvas: lrSmallCanvas, lr: 0.01, color: '#4a87a8', label: 'lr=0.01' },
      { canvas: lrGoodCanvas, lr: 0.1, color: '#6a9b5e', label: 'lr=0.1' },
      { canvas: lrBigCanvas, lr: 0.9, color: '#c8553d', label: 'lr=0.9' }
    ];
    const LR_START_X = 2.5;

    function lr1DLoss(x) { return x * x + 0.5 * Math.sin(3 * x); }
    function lr1DGrad(x) { return 2 * x + 1.5 * Math.cos(3 * x); }

    let lrPositions = LR_CONFIGS.map(function() { return LR_START_X; });
    let lrTrails = LR_CONFIGS.map(function() { return [LR_START_X]; });
    let lrAnimRunning = false;
    let lrAnimId = null;

    function drawLR1D(cfg, pos, trail, idx) {
      const ctx = cfg.canvas.getContext('2d');
      const W = cfg.canvas.width;
      const H = cfg.canvas.height;
      const xRange = [-3.5, 3.5];
      const padding = 20;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fff8e7';
      ctx.fillRect(0, 0, W, H);

      function xToCanvas(x) { return padding + (x - xRange[0]) / (xRange[1] - xRange[0]) * (W - 2 * padding); }
      function yToCanvas(y) { return H - padding - (y / 12) * (H - 2 * padding); }

      ctx.strokeStyle = 'rgba(61, 47, 31, 0.3)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      for (let px = padding; px <= W - padding; px++) {
        const x = xRange[0] + (px - padding) / (W - 2 * padding) * (xRange[1] - xRange[0]);
        const y = lr1DLoss(x);
        if (px === padding) ctx.moveTo(px, yToCanvas(y));
        else ctx.lineTo(px, yToCanvas(y));
      }
      ctx.stroke();

      if (trail.length > 1) {
        ctx.strokeStyle = cfg.color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
          const cx = xToCanvas(trail[i]);
          const cy = yToCanvas(lr1DLoss(trail[i]));
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;

        for (let i = 0; i < trail.length - 1; i++) {
          const cx = xToCanvas(trail[i]);
          const cy = yToCanvas(lr1DLoss(trail[i]));
          ctx.fillStyle = cfg.color;
          ctx.globalAlpha = 0.4;
          ctx.beginPath();
          ctx.arc(cx, cy, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      const cx = xToCanvas(pos);
      const cy = yToCanvas(lr1DLoss(pos));
      ctx.fillStyle = cfg.color;
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#3d2f1f';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function renderAllLR() {
      LR_CONFIGS.forEach(function(cfg, i) {
        drawLR1D(cfg, lrPositions[i], lrTrails[i], i);
      });
    }

    function lrStep() {
      let anyMoving = false;
      LR_CONFIGS.forEach(function(cfg, i) {
        const g = lr1DGrad(lrPositions[i]);
        const newX = lrPositions[i] - cfg.lr * g;
        if (Math.abs(newX) < 10 && Math.abs(g) > 0.001 && lrTrails[i].length < 200) {
          lrPositions[i] = newX;
          lrTrails[i].push(newX);
          anyMoving = true;
        }
      });
      renderAllLR();
      if (anyMoving && lrAnimRunning) {
        const speed = parseInt(document.getElementById('lrSpeedSlider').value);
        lrAnimId = setTimeout(lrStep, 150 / speed);
      } else {
        lrAnimRunning = false;
        document.getElementById('lrPlayBtn').textContent = '⏵ 播放';
      }
    }

    function resetLR() {
      if (lrAnimId) { clearTimeout(lrAnimId); lrAnimId = null; }
      lrAnimRunning = false;
      lrPositions = LR_CONFIGS.map(function() { return LR_START_X; });
      lrTrails = LR_CONFIGS.map(function() { return [LR_START_X]; });
      document.getElementById('lrPlayBtn').textContent = '⏵ 播放';
      renderAllLR();
    }

    document.getElementById('lrPlayBtn').addEventListener('click', function() {
      if (lrAnimRunning) {
        lrAnimRunning = false;
        if (lrAnimId) { clearTimeout(lrAnimId); lrAnimId = null; }
        this.textContent = '⏵ 播放';
      } else {
        lrAnimRunning = true;
        this.textContent = '⏸ 暫停';
        lrStep();
      }
    });
    document.getElementById('lrResetBtn').addEventListener('click', resetLR);
    document.getElementById('lrSpeedSlider').addEventListener('input', function() {
      document.getElementById('lrSpeedVal').textContent = (this.value * 0.5).toFixed(1) + 'x';
    });

    renderAllLR();
  }

  // ═══════════════════════════════════════════════════════════════
  // SECTION 4: Backpropagation Chain Rule Visualization
  // ═══════════════════════════════════════════════════════════════
  const bpChainVis = document.getElementById('bpChainVis');
  if (bpChainVis) {
    const BP_LOCAL_GRADS = [0.8, 0.6, 0.7, 0.5, 0.65, 0.55, 0.72, 0.48];
    let bpNumLayers = 5;
    let bpChainAnimId = null;

    function buildChainVis() {
      bpChainVis.innerHTML = '';
      bpNumLayers = parseInt(document.getElementById('bpLayerSlider').value);
      document.getElementById('bpLayerVal').textContent = bpNumLayers;

      for (let i = 0; i <= bpNumLayers; i++) {
        const layerDiv = document.createElement('div');
        layerDiv.style.cssText = 'display:flex;flex-direction:column;align-items:center;min-width:60px;cursor:pointer;z-index:1;';
        layerDiv.dataset.layerIdx = i;

        const box = document.createElement('div');
        const isLoss = (i === bpNumLayers);
        box.style.cssText = 'width:56px;height:56px;border-radius:10px;border:2.5px solid #3d2f1f;display:flex;align-items:center;justify-content:center;font-family:"JetBrains Mono",monospace;font-size:10px;font-weight:700;text-align:center;transition:all 0.3s;box-shadow:2px 2px 0 #3d2f1f;';
        box.style.background = isLoss ? '#c8553d' : '#fff8e7';
        box.style.color = isLoss ? '#f5ecd9' : '#3d2f1f';
        box.textContent = isLoss ? 'Loss' : 'L' + (i + 1);
        box.className = 'bp-layer-box';
        layerDiv.appendChild(box);

        const label = document.createElement('div');
        label.style.cssText = 'font-family:"Patrick Hand",cursive;font-size:11px;color:#a89878;margin-top:5px;';
        label.textContent = isLoss ? '∂L/∂L=1' : '∂=' + BP_LOCAL_GRADS[i].toFixed(2);
        layerDiv.appendChild(label);

        const gradVal = document.createElement('div');
        gradVal.style.cssText = 'font-family:"JetBrains Mono",monospace;font-size:10px;color:#c8553d;margin-top:3px;font-weight:700;opacity:0;transition:opacity 0.3s;';
        gradVal.className = 'bp-grad-value';
        layerDiv.appendChild(gradVal);

        layerDiv.addEventListener('click', function() { highlightBpLayer(i); });
        bpChainVis.appendChild(layerDiv);

        if (i < bpNumLayers) {
          const arrow = document.createElement('div');
          arrow.style.cssText = 'width:36px;height:4px;background:var(--line);position:relative;flex-shrink:0;border-radius:2px;transition:all 0.3s;';
          arrow.className = 'bp-arrow';
          const arrowHead = document.createElement('div');
          arrowHead.style.cssText = 'position:absolute;left:-2px;top:-4px;border:5px solid transparent;border-right:7px solid var(--line);transition:all 0.3s;';
          arrow.appendChild(arrowHead);
          bpChainVis.appendChild(arrow);
        }
      }
    }

    function highlightBpLayer(targetIdx) {
      const boxes = bpChainVis.querySelectorAll('.bp-layer-box');
      const arrows = bpChainVis.querySelectorAll('.bp-arrow');
      const gradVals = bpChainVis.querySelectorAll('.bp-grad-value');

      boxes.forEach(function(b) {
        b.style.transform = 'scale(1)';
        b.style.boxShadow = '2px 2px 0 #3d2f1f';
      });
      arrows.forEach(function(a) {
        a.style.background = 'var(--line)';
        a.style.height = '4px';
        a.firstChild.style.borderRightColor = 'var(--line)';
      });
      gradVals.forEach(function(g) { g.style.opacity = '0'; });

      let cumulativeGrad = 1.0;
      const infoEl = document.getElementById('bpChainInfo');
      let formula = '∂L/∂L = 1';

      for (let i = bpNumLayers; i >= targetIdx; i--) {
        if (i < bpNumLayers) {
          cumulativeGrad *= BP_LOCAL_GRADS[i];
          formula += ' × ' + BP_LOCAL_GRADS[i].toFixed(2);
        }
        boxes[i].style.transform = 'scale(1.1)';
        boxes[i].style.boxShadow = '0 0 12px rgba(232,197,71,0.6), 3px 3px 0 #3d2f1f';
        gradVals[i].style.opacity = '1';
        gradVals[i].textContent = i === bpNumLayers ? '1.000' : cumulativeGrad.toFixed(4);
      }
      for (let i = bpNumLayers - 1; i >= targetIdx; i--) {
        arrows[i].style.height = Math.min(8, 2 + 6 * (1 - (bpNumLayers - 1 - i) / bpNumLayers)) + 'px';
        arrows[i].style.background = '#c8553d';
        arrows[i].firstChild.style.borderRightColor = '#c8553d';
      }

      if (infoEl) {
        infoEl.innerHTML = '<strong>Layer ' + (targetIdx + 1) + '</strong> 的梯度: ' +
          formula + ' = <span style="color:var(--red);font-family:JetBrains Mono,monospace;font-weight:700;">' +
          cumulativeGrad.toFixed(6) + '</span>';
      }
    }

    function animateChainRule() {
      if (bpChainAnimId) return;
      let currentLayer = bpNumLayers;
      const playBtn = document.getElementById('bpChainPlay');
      playBtn.textContent = '⏸ 播放中...';

      function animStep() {
        highlightBpLayer(currentLayer);
        currentLayer--;
        if (currentLayer >= 0) {
          bpChainAnimId = setTimeout(animStep, 600);
        } else {
          bpChainAnimId = null;
          playBtn.textContent = '⏵ 播放 Chain Rule';
        }
      }
      animStep();
    }

    document.getElementById('bpLayerSlider').addEventListener('input', function() {
      if (bpChainAnimId) { clearTimeout(bpChainAnimId); bpChainAnimId = null; }
      document.getElementById('bpChainPlay').textContent = '⏵ 播放 Chain Rule';
      buildChainVis();
    });
    document.getElementById('bpChainPlay').addEventListener('click', function() {
      if (bpChainAnimId) {
        clearTimeout(bpChainAnimId);
        bpChainAnimId = null;
        this.textContent = '⏵ 播放 Chain Rule';
      } else {
        animateChainRule();
      }
    });

    buildChainVis();
  }

  // ═══════════════════════════════════════════════════════════════
  // SECTION 5: Quiz
  // ═══════════════════════════════════════════════════════════════
  document.querySelectorAll('.quiz-box').forEach(function(box) {
    const options = box.querySelectorAll('.quiz-option');
    const feedback = box.querySelector('.quiz-feedback');
    if (!feedback) return;
    let answered = false;

    options.forEach(function(opt) {
      opt.addEventListener('click', function() {
        if (answered) return;
        answered = true;
        const isCorrect = opt.dataset.correct === 'true';
        opt.classList.add(isCorrect ? 'correct' : 'wrong');
        if (!isCorrect) {
          options.forEach(function(o) {
            if (o.dataset.correct === 'true') o.classList.add('correct');
          });
        }
        feedback.textContent = isCorrect ? '正確! 做得好!' : '不對喔,正確答案已標出來了。';
        feedback.className = 'quiz-feedback show ' + (isCorrect ? 'correct' : 'wrong');
      });
    });
  });
};
window.init_gradient();
