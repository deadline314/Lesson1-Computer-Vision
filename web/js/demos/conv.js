window.init_conv = function() {
  const srcCanvas = document.getElementById('srcCanvas');
  const dstCanvas = document.getElementById('dstCanvas');
  if (!srcCanvas || !dstCanvas) return;
  const srcCtx = srcCanvas.getContext('2d');
  const dstCtx = dstCanvas.getContext('2d');
  const SIZE = 200;
  let convStride = 1;
  let convPad = 1;
  let rfAnimating = false;
  let rfFrame = 0;

  function generateImage(type) {
    const img = srcCtx.createImageData(SIZE, SIZE);
    const data = img.data;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const i = (y * SIZE + x) * 4;
        let v = 0;
        const cx = x - SIZE/2, cy = y - SIZE/2;
        const r = Math.sqrt(cx*cx + cy*cy);
        if (type === 'circles') v = Math.abs(Math.sin(r * 0.15)) * 200 + 30;
        else if (type === 'gradient') v = (x + y) / (2 * SIZE) * 255;
        else if (type === 'checker') { const cs = 25; v = ((Math.floor(x/cs) + Math.floor(y/cs)) % 2) * 200 + 30; }
        else if (type === 'face') {
          v = 240; if (r < 75) v = 200;
          const e1 = Math.sqrt((cx+25)**2 + (cy+15)**2);
          const e2 = Math.sqrt((cx-25)**2 + (cy+15)**2);
          if (e1 < 8 || e2 < 8) v = 60;
          if (cy > 20 && cy < 30 && Math.abs(cx) < 25) v = 60;
        } else if (type === 'lines') v = (Math.sin(x * 0.3) + Math.sin(y * 0.3)) * 60 + 128;
        data[i] = data[i+1] = data[i+2] = v;
        data[i+3] = 255;
      }
    }
    srcCtx.putImageData(img, 0, 0);
    applyConvolution();
  }

  const imgTypes = ['face', 'circles', 'checker', 'gradient', 'lines'];
  const imgPicker = document.getElementById('imgPicker');
  if (imgPicker) {
    imgTypes.forEach((type, idx) => {
      const thumb = document.createElement('canvas');
      thumb.width = 44; thumb.height = 44;
      thumb.className = 'img-thumb' + (idx === 0 ? ' active' : '');
      const tctx = thumb.getContext('2d');
      for (let y = 0; y < 44; y++) {
        for (let x = 0; x < 44; x++) {
          const cx = x - 22, cy = y - 22;
          const r = Math.sqrt(cx*cx + cy*cy);
          let v = 240;
          if (type === 'circles') v = Math.abs(Math.sin(r * 0.5)) * 200 + 30;
          else if (type === 'gradient') v = (x + y) / 88 * 255;
          else if (type === 'checker') v = ((Math.floor(x/6) + Math.floor(y/6)) % 2) * 200 + 30;
          else if (type === 'face') {
            v = 240; if (r < 16) v = 200;
            if (Math.sqrt((cx+5)**2 + (cy+3)**2) < 2) v = 60;
            if (Math.sqrt((cx-5)**2 + (cy+3)**2) < 2) v = 60;
            if (cy > 5 && cy < 7 && Math.abs(cx) < 5) v = 60;
          } else if (type === 'lines') v = (Math.sin(x * 0.5) + Math.sin(y * 0.5)) * 60 + 128;
          tctx.fillStyle = `rgb(${v|0},${v|0},${v|0})`;
          tctx.fillRect(x, y, 1, 1);
        }
      }
      thumb.addEventListener('click', () => {
        document.querySelectorAll('.img-thumb').forEach(t => t.classList.remove('active'));
        thumb.classList.add('active');
        generateImage(type);
      });
      imgPicker.appendChild(thumb);
    });
  }

  const kernelGrid = document.getElementById('kernelGrid');
  let kernel = [0, 0, 0, 0, 1, 0, 0, 0, 0];

  function buildKernelInputs() {
    if (!kernelGrid) return;
    kernelGrid.innerHTML = '';
    kernel.forEach((v, i) => {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.className = 'kernel-cell';
      inp.value = (Math.round(v * 100) / 100).toString();
      inp.step = '0.1';
      inp.addEventListener('input', (e) => {
        kernel[i] = parseFloat(e.target.value) || 0;
        applyConvolution();
      });
      kernelGrid.appendChild(inp);
    });
  }

  const presets = {
    identity: [0,0,0,0,1,0,0,0,0],
    edge: [-1,-1,-1,-1,8,-1,-1,-1,-1],
    'sobel-x': [-1,0,1,-2,0,2,-1,0,1],
    'sobel-y': [-1,-2,-1,0,0,0,1,2,1],
    blur: [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9],
    sharpen: [0,-1,0,-1,5,-1,0,-1,0],
    emboss: [-2,-1,0,-1,1,1,0,1,2]
  };
  document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
      kernel = [...presets[btn.dataset.preset]];
      buildKernelInputs();
      applyConvolution();
    });
  });

  function applyConvolution() {
    const src = srcCtx.getImageData(0, 0, SIZE, SIZE);
    const outW = Math.floor((SIZE + 2*convPad - 3) / convStride) + 1;
    const outH = outW;
    const dst = dstCtx.createImageData(SIZE, SIZE);

    for (let i = 0; i < dst.data.length; i += 4) {
      dst.data[i] = dst.data[i+1] = dst.data[i+2] = 30;
      dst.data[i+3] = 255;
    }

    const output = new Float32Array(outW * outH);
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const sy = oy * convStride + ky + (1 - convPad);
            const sx = ox * convStride + kx + (1 - convPad);
            if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE) continue;
            const si = (sy * SIZE + sx) * 4;
            const ki = (ky + 1) * 3 + (kx + 1);
            sum += src.data[si] * kernel[ki];
          }
        }
        output[oy * outW + ox] = sum;
      }
    }

    const scale = SIZE / outW;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const oy = Math.min(outH - 1, Math.floor(y / scale));
        const ox = Math.min(outW - 1, Math.floor(x / scale));
        let v = output[oy * outW + ox];
        v = Math.max(0, Math.min(255, v));
        const di = (y * SIZE + x) * 4;
        dst.data[di] = dst.data[di+1] = dst.data[di+2] = v;
        dst.data[di+3] = 255;
      }
    }
    dstCtx.putImageData(dst, 0, 0);

    const outShapeEl = document.getElementById('convOutShape');
    if (outShapeEl) outShapeEl.textContent = `${outW} × ${outH}`;
    const pyCodeEl = document.getElementById('convPyCode');
    if (pyCodeEl) pyCodeEl.textContent = `nn.Conv2d(1, 1, kernel_size=3, stride=${convStride}, padding=${convPad})`;

    if (rfAnimating) drawRF();
  }

  function drawRF() {
    const src = srcCtx.getImageData(0, 0, SIZE, SIZE);
    srcCtx.putImageData(src, 0, 0);
    const outW = Math.floor((SIZE + 2*convPad - 3) / convStride) + 1;
    const totalCells = outW * outW;
    const idx = rfFrame % totalCells;
    const oy = Math.floor(idx / outW);
    const ox = idx % outW;
    const sx = ox * convStride - convPad;
    const sy = oy * convStride - convPad;
    srcCtx.strokeStyle = '#c8553d';
    srcCtx.lineWidth = 3;
    srcCtx.strokeRect(sx, sy, 3, 3);
    srcCtx.fillStyle = 'rgba(232, 197, 71, 0.3)';
    srcCtx.fillRect(sx, sy, 3, 3);
  }

  function rfLoop() {
    if (!rfAnimating) return;
    rfFrame++;
    applyConvolution();
    setTimeout(rfLoop, 50);
  }

  const rfToggle = document.getElementById('rfToggle');
  if (rfToggle) {
    rfToggle.addEventListener('click', (e) => {
      rfAnimating = !rfAnimating;
      e.target.textContent = rfAnimating ? '⏸ 停止' : '啟動動畫';
      if (rfAnimating) rfLoop();
      else applyConvolution();
    });
  }

  const convStrideEl = document.getElementById('convStride');
  if (convStrideEl) {
    convStrideEl.addEventListener('input', (e) => {
      convStride = parseInt(e.target.value);
      document.getElementById('convStrideVal').textContent = convStride;
      applyConvolution();
    });
  }
  const convPadEl = document.getElementById('convPad');
  if (convPadEl) {
    convPadEl.addEventListener('input', (e) => {
      convPad = parseInt(e.target.value);
      document.getElementById('convPadVal').textContent = convPad;
      applyConvolution();
    });
  }

  buildKernelInputs();
  generateImage('face');

  // Output size calculator (enhanced)
  const calcInput = document.getElementById('calcInput');
  const calcKernel = document.getElementById('calcKernel');
  const calcStride = document.getElementById('calcStride');
  const calcPadding = document.getElementById('calcPadding');
  const calcResult = document.getElementById('calcResult');
  const calcResultNum = document.getElementById('calcResultNum');
  const calcSubstitution = document.getElementById('calcSubstitution');
  const calcComputation = document.getElementById('calcComputation');
  const calcGridInput = document.getElementById('calcGridInput');
  const calcGridOutput = document.getElementById('calcGridOutput');
  const calcGridInputSize = document.getElementById('calcGridInputSize');
  const calcGridOutputSize = document.getElementById('calcGridOutputSize');

  const calcInputSlider = document.getElementById('calcInputSlider');
  const calcKernelSlider = document.getElementById('calcKernelSlider');
  const calcStrideSlider = document.getElementById('calcStrideSlider');
  const calcPaddingSlider = document.getElementById('calcPaddingSlider');

  let prevCalcResult = 224;

  function buildVisualGrid(container, size, maxCells) {
    if (!container) return;
    const cells = Math.min(size, maxCells);
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${cells}, 1fr)`;
    container.style.width = Math.max(30, Math.min(120, cells * (120 / maxCells))) + 'px';
    container.style.height = container.style.width;
    for (let i = 0; i < cells * cells; i++) {
      const cell = document.createElement('div');
      cell.className = 'calc-cell';
      cell.style.animationDelay = (i * 10) + 'ms';
      container.appendChild(cell);
    }
    setTimeout(() => {
      const allCells = container.querySelectorAll('.calc-cell');
      let idx = 0;
      const interval = setInterval(() => {
        if (idx >= allCells.length) { clearInterval(interval); return; }
        allCells[idx].classList.add('highlight');
        if (idx > 0) allCells[idx - 1].classList.remove('highlight');
        idx++;
      }, 30);
      setTimeout(() => {
        allCells.forEach(c => c.classList.remove('highlight'));
        clearInterval(interval);
      }, allCells.length * 30 + 200);
    }, 100);
  }

  function updateCalc() {
    if (!calcInput) return;
    const i = parseInt(calcInput.value) || 0;
    const k = parseInt(calcKernel.value) || 1;
    const s = parseInt(calcStride.value) || 1;
    const p = parseInt(calcPadding.value) || 0;
    const out = Math.floor((i + 2*p - k) / s) + 1;

    if (calcSubstitution) {
      calcSubstitution.innerHTML = `floor((<span class="calc-hl-input">${i}</span> + 2×<span class="calc-hl-pad">${p}</span> - <span class="calc-hl-kernel">${k}</span>) / <span class="calc-hl-stride">${s}</span>) + 1`;
    }
    if (calcComputation) {
      const numerator = i + 2*p - k;
      calcComputation.innerHTML = `floor(<strong>${numerator}</strong> / <span class="calc-hl-stride">${s}</span>) + 1 = <strong>${out > 0 ? out : 'Invalid!'}</strong>`;
    }

    if (calcResultNum) {
      calcResultNum.textContent = out > 0 ? out : '!';
    }
    if (calcResult) {
      calcResult.classList.remove('invalid', 'pulse');
      if (out <= 0) {
        calcResult.classList.add('invalid');
      }
      if (out !== prevCalcResult) {
        void calcResult.offsetWidth;
        calcResult.classList.add('pulse');
        prevCalcResult = out;
      }
    }

    const MAX_VIS_CELLS = 10;
    const inputCells = Math.max(2, Math.min(MAX_VIS_CELLS, Math.round(i / (512 / MAX_VIS_CELLS))));
    const outputCells = out > 0 ? Math.max(2, Math.min(MAX_VIS_CELLS, Math.round(out / (512 / MAX_VIS_CELLS)))) : 1;

    buildVisualGrid(calcGridInput, inputCells, MAX_VIS_CELLS);
    buildVisualGrid(calcGridOutput, outputCells, MAX_VIS_CELLS);

    if (calcGridInputSize) calcGridInputSize.textContent = `${i}×${i}`;
    if (calcGridOutputSize) calcGridOutputSize.textContent = out > 0 ? `${out}×${out}` : 'Invalid';

    if (calcInputSlider) calcInputSlider.value = i;
    if (calcKernelSlider) calcKernelSlider.value = k;
    if (calcStrideSlider) calcStrideSlider.value = s;
    if (calcPaddingSlider) calcPaddingSlider.value = p;
  }

  if (calcInput) {
    [calcInput, calcKernel, calcStride, calcPadding].forEach(el => el.addEventListener('input', updateCalc));
    if (calcInputSlider) calcInputSlider.addEventListener('input', (e) => { calcInput.value = e.target.value; updateCalc(); });
    if (calcKernelSlider) calcKernelSlider.addEventListener('input', (e) => { calcKernel.value = e.target.value; updateCalc(); });
    if (calcStrideSlider) calcStrideSlider.addEventListener('input', (e) => { calcStride.value = e.target.value; updateCalc(); });
    if (calcPaddingSlider) calcPaddingSlider.addEventListener('input', (e) => { calcPadding.value = e.target.value; updateCalc(); });

    document.querySelectorAll('.calc-preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        calcInput.value = btn.dataset.i;
        calcKernel.value = btn.dataset.k;
        calcStride.value = btn.dataset.s;
        calcPadding.value = btn.dataset.p;
        updateCalc();
      });
    });

    updateCalc();
  }

  // Receptive Field Calculator
  (function() {
    const layersDiv = document.getElementById('rfLayers');
    const gridDiv = document.getElementById('rfGrid');
    const rfValEl = document.getElementById('rfValue');
    const rfStrideEl = document.getElementById('rfStride');
    const rfCountEl = document.getElementById('rfCount');

    if (!layersDiv) return;

    let rfLayers = [];
    const GRID_SIZE = 21;

    function compute() {
      let rf = 1, stride = 1;
      rfLayers.forEach(l => {
        if (l.type === '3x3') { rf += (3 - 1) * stride; }
        else if (l.type === 'pool') { rf += (2 - 1) * stride; stride *= 2; }
        else if (l.type === 'stride2') { rf += (3 - 1) * stride; stride *= 2; }
      });
      return { rf, stride };
    }

    function render() {
      layersDiv.innerHTML = '';
      rfLayers.forEach((l, i) => {
        const chip = document.createElement('span');
        chip.className = 'rf-calc-chip';
        const label = l.type === '3x3' ? '3×3 Conv' : l.type === 'pool' ? 'MaxPool 2×2' : 'Stride=2 Conv';
        chip.innerHTML = `${label}<span class="rf-remove" data-idx="${i}">×</span>`;
        layersDiv.appendChild(chip);
      });

      layersDiv.querySelectorAll('.rf-remove').forEach(btn => {
        btn.addEventListener('click', () => { rfLayers.splice(parseInt(btn.dataset.idx), 1); render(); });
      });

      const { rf, stride } = compute();
      rfValEl.textContent = `${rf} × ${rf}`;
      rfStrideEl.textContent = stride;
      rfCountEl.textContent = rfLayers.length;

      gridDiv.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 12px)`;
      gridDiv.innerHTML = '';
      const center = Math.floor(GRID_SIZE / 2);
      const half = Math.floor(rf / 2);
      for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
          const cell = document.createElement('div');
          cell.className = 'rf-cell';
          if (Math.abs(r - center) <= half && Math.abs(c - center) <= half) cell.classList.add('active');
          gridDiv.appendChild(cell);
        }
      }
    }

    const add3x3 = document.getElementById('rfAdd3x3');
    const addPool = document.getElementById('rfAddPool');
    const addStride = document.getElementById('rfAddStride');
    const clearBtn = document.getElementById('rfClear');
    if (add3x3) add3x3.addEventListener('click', () => { rfLayers.push({ type: '3x3' }); render(); });
    if (addPool) addPool.addEventListener('click', () => { rfLayers.push({ type: 'pool' }); render(); });
    if (addStride) addStride.addEventListener('click', () => { rfLayers.push({ type: 'stride2' }); render(); });
    if (clearBtn) clearBtn.addEventListener('click', () => { rfLayers = []; render(); });
    render();
  })();
};
window.init_conv();
