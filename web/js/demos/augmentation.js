window.init_augmentation = function() {
  const SIZE = 240;
  const MOSAIC_SIZE = 480;
  const COLORS = {
    red: '#c8553d', orange: '#e8c547', green: '#6b9f6b',
    blue: '#4a87a8', purple: '#8b6ba8', pink: '#c87da8'
  };

  function drawPattern(ctx, type, w, h) {
    w = w || ctx.canvas.width;
    h = h || ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (type === 'checker') {
      const cs = Math.floor(w / 8);
      for (let y = 0; y < h; y += cs) {
        for (let x = 0; x < w; x += cs) {
          ctx.fillStyle = ((Math.floor(x/cs) + Math.floor(y/cs)) % 2 === 0) ? '#f5e6d3' : COLORS.blue;
          ctx.fillRect(x, y, cs, cs);
        }
      }
    } else if (type === 'scene') {
      const grd = ctx.createLinearGradient(0, 0, 0, h);
      grd.addColorStop(0, '#87CEEB'); grd.addColorStop(0.6, '#E0F0FF'); grd.addColorStop(0.6, '#6b9f6b'); grd.addColorStop(1, '#3d6b3d');
      ctx.fillStyle = grd; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#e8c547'; ctx.beginPath(); ctx.arc(w*0.8, h*0.2, w*0.08, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#8B4513'; ctx.fillRect(w*0.25, h*0.35, w*0.04, h*0.25);
      ctx.fillStyle = '#228B22'; ctx.beginPath(); ctx.arc(w*0.27, h*0.32, w*0.08, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#c8553d'; ctx.fillRect(w*0.5, h*0.45, w*0.2, h*0.15);
      ctx.fillStyle = '#8B0000'; ctx.beginPath(); ctx.moveTo(w*0.48, h*0.45); ctx.lineTo(w*0.6, h*0.35); ctx.lineTo(w*0.72, h*0.45); ctx.fill();
    } else if (type === 'gradient') {
      const grd = ctx.createLinearGradient(0, 0, w, h);
      grd.addColorStop(0, COLORS.purple); grd.addColorStop(0.5, COLORS.pink); grd.addColorStop(1, COLORS.orange);
      ctx.fillStyle = grd; ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 5; i++) {
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(w*(0.2+i*0.15), h*0.5, w*0.06*(i+1), 0, Math.PI*2); ctx.stroke();
      }
    } else if (type === 'shapes') {
      ctx.fillStyle = '#f8f4e8'; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = COLORS.red; ctx.fillRect(w*0.1, h*0.1, w*0.25, h*0.25);
      ctx.fillStyle = COLORS.blue; ctx.beginPath(); ctx.arc(w*0.7, h*0.3, w*0.12, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = COLORS.green; ctx.beginPath(); ctx.moveTo(w*0.3, h*0.9); ctx.lineTo(w*0.5, h*0.55); ctx.lineTo(w*0.7, h*0.9); ctx.fill();
      ctx.fillStyle = COLORS.orange; ctx.beginPath();
      const cx = w*0.25, cy = h*0.7, r = w*0.1;
      for (let i = 0; i < 5; i++) { const a = Math.PI*2*i/5 - Math.PI/2; ctx.lineTo(cx+r*Math.cos(a), cy+r*Math.sin(a)); ctx.lineTo(cx+r*0.4*Math.cos(a+Math.PI/5), cy+r*0.4*Math.sin(a+Math.PI/5)); }
      ctx.fill();
    }
  }

  function getImageData(canvas) {
    return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  }

  // ========== BASIC AUGMENTATIONS ==========
  const srcCanvas = document.getElementById('augSrcCanvas');
  const dstCanvas = document.getElementById('augDstCanvas');
  if (!srcCanvas || !dstCanvas) return;
  const srcCtx = srcCanvas.getContext('2d');
  const dstCtx = dstCanvas.getContext('2d');
  let currentImg = 'checker';

  function drawSrc() { drawPattern(srcCtx, currentImg, SIZE, SIZE); }

  document.querySelectorAll('[data-aug-img]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      currentImg = btn.dataset.augImg;
      drawSrc();
      applyAug();
    });
  });

  const augTypeEl = document.getElementById('augType');
  const augParamEl = document.getElementById('augParam');
  const augParamValEl = document.getElementById('augParamVal');
  const augParamHelpEl = document.getElementById('augParamHelp');
  const augApplyBtn = document.getElementById('augApplyBtn');
  const augTypeInfo = document.getElementById('augTypeInfo');
  const augParamInfo = document.getElementById('augParamInfo');

  function applyAug() {
    const type = augTypeEl ? augTypeEl.value : 'flip';
    const param = augParamEl ? parseInt(augParamEl.value) / 100 : 0.5;
    const srcData = getImageData(srcCanvas);
    dstCtx.clearRect(0, 0, SIZE, SIZE);

    if (type === 'flip') {
      dstCtx.save(); dstCtx.translate(SIZE, 0); dstCtx.scale(-1, 1);
      dstCtx.drawImage(srcCanvas, 0, 0); dstCtx.restore();
      if (augTypeInfo) augTypeInfo.textContent = 'H-Flip';
      if (augParamInfo) augParamInfo.textContent = '—';
    } else if (type === 'crop') {
      const margin = Math.floor(param * SIZE * 0.4);
      const sx = margin, sy = margin;
      const sw = SIZE - 2*margin, sh = SIZE - 2*margin;
      dstCtx.drawImage(srcCanvas, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
      if (augTypeInfo) augTypeInfo.textContent = 'Crop';
      if (augParamInfo) augParamInfo.textContent = Math.round((sw/SIZE)*100) + '%';
    } else if (type === 'rotate') {
      const angle = (param - 0.5) * 90 * Math.PI / 180;
      dstCtx.save(); dstCtx.translate(SIZE/2, SIZE/2); dstCtx.rotate(angle);
      dstCtx.drawImage(srcCanvas, -SIZE/2, -SIZE/2); dstCtx.restore();
      if (augTypeInfo) augTypeInfo.textContent = 'Rotate';
      if (augParamInfo) augParamInfo.textContent = Math.round((param-0.5)*90) + '°';
    } else if (type === 'scale') {
      const s = 0.5 + param;
      const offset = SIZE * (1 - s) / 2;
      dstCtx.fillStyle = '#ddd'; dstCtx.fillRect(0, 0, SIZE, SIZE);
      dstCtx.drawImage(srcCanvas, offset, offset, SIZE*s, SIZE*s);
      if (augTypeInfo) augTypeInfo.textContent = 'Scale';
      if (augParamInfo) augParamInfo.textContent = s.toFixed(2) + '×';
    } else if (type === 'colorjitter') {
      dstCtx.drawImage(srcCanvas, 0, 0);
      const imgData = dstCtx.getImageData(0, 0, SIZE, SIZE);
      const d = imgData.data;
      const hueShift = (param - 0.5) * 60;
      const satMul = 0.5 + param;
      const valMul = 0.7 + param * 0.6;
      for (let i = 0; i < d.length; i += 4) {
        let r = d[i], g = d[i+1], b = d[i+2];
        const max = Math.max(r,g,b), min = Math.min(r,g,b);
        let h = 0, s = 0, v = max;
        const diff = max - min;
        if (max !== 0) s = diff / max;
        if (diff !== 0) {
          if (max === r) h = (g - b) / diff;
          else if (max === g) h = 2 + (b - r) / diff;
          else h = 4 + (r - g) / diff;
          h *= 60; if (h < 0) h += 360;
        }
        h = (h + hueShift + 360) % 360;
        s = Math.min(1, s * satMul);
        v = Math.min(255, v * valMul);
        const hi = Math.floor(h / 60) % 6;
        const f = h / 60 - Math.floor(h / 60);
        const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1-f) * s);
        if (hi === 0) { r=v; g=t; b=p; }
        else if (hi === 1) { r=q; g=v; b=p; }
        else if (hi === 2) { r=p; g=v; b=t; }
        else if (hi === 3) { r=p; g=q; b=v; }
        else if (hi === 4) { r=t; g=p; b=v; }
        else { r=v; g=p; b=q; }
        d[i] = r; d[i+1] = g; d[i+2] = b;
      }
      dstCtx.putImageData(imgData, 0, 0);
      if (augTypeInfo) augTypeInfo.textContent = 'Color Jitter';
      if (augParamInfo) augParamInfo.textContent = 'H:' + Math.round(hueShift) + '°';
    }
  }

  if (augParamEl) {
    augParamEl.addEventListener('input', function() {
      if (augParamValEl) augParamValEl.textContent = (parseInt(augParamEl.value)/100).toFixed(2);
      applyAug();
    });
  }
  if (augTypeEl) augTypeEl.addEventListener('change', function() { applyAug(); });
  if (augApplyBtn) augApplyBtn.addEventListener('click', function() { applyAug(); });

  drawSrc();
  applyAug();

  // ========== MOSAIC ==========
  const mosaicCanvas = document.getElementById('mosaicCanvas');
  if (mosaicCanvas) {
    const mCtx = mosaicCanvas.getContext('2d');
    let mcx = 0.5, mcy = 0.5;
    let dragging = false;
    const patterns = ['checker', 'scene', 'gradient', 'shapes'];

    function drawMosaic() {
      const W = MOSAIC_SIZE, H = MOSAIC_SIZE;
      const cx = Math.floor(mcx * W), cy = Math.floor(mcy * H);
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = W; tmpCanvas.height = H;
      const tmpCtx = tmpCanvas.getContext('2d');

      patterns.forEach(function(pat, i) {
        const tw = (i % 2 === 0) ? cx : W - cx;
        const th = (i < 2) ? cy : H - cy;
        const tx = (i % 2 === 0) ? 0 : cx;
        const ty = (i < 2) ? 0 : cy;
        tmpCtx.save(); tmpCtx.beginPath(); tmpCtx.rect(tx, ty, tw, th); tmpCtx.clip();
        const pc = document.createElement('canvas');
        pc.width = tw; pc.height = th;
        drawPattern(pc.getContext('2d'), pat, tw, th);
        tmpCtx.drawImage(pc, tx, ty);
        tmpCtx.restore();
      });

      mCtx.clearRect(0, 0, W, H);
      mCtx.drawImage(tmpCanvas, 0, 0);

      mCtx.strokeStyle = 'rgba(200,85,61,0.8)'; mCtx.lineWidth = 2.5; mCtx.setLineDash([8, 4]);
      mCtx.beginPath(); mCtx.moveTo(cx, 0); mCtx.lineTo(cx, H); mCtx.stroke();
      mCtx.beginPath(); mCtx.moveTo(0, cy); mCtx.lineTo(W, cy); mCtx.stroke();
      mCtx.setLineDash([]);

      mCtx.fillStyle = COLORS.red; mCtx.beginPath(); mCtx.arc(cx, cy, 8, 0, Math.PI*2); mCtx.fill();
      mCtx.strokeStyle = '#fff'; mCtx.lineWidth = 2; mCtx.beginPath(); mCtx.arc(cx, cy, 8, 0, Math.PI*2); mCtx.stroke();

      var cxEl = document.getElementById('mosaicCX');
      var cyEl = document.getElementById('mosaicCY');
      if (cxEl) cxEl.textContent = Math.round(mcx*100) + '%';
      if (cyEl) cyEl.textContent = Math.round(mcy*100) + '%';
    }

    function getMosaicPos(e) {
      const rect = mosaicCanvas.getBoundingClientRect();
      const scaleX = MOSAIC_SIZE / rect.width;
      const scaleY = MOSAIC_SIZE / rect.height;
      return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
    }

    mosaicCanvas.addEventListener('mousedown', function(e) { dragging = true; var p = getMosaicPos(e); mcx = Math.max(0.1, Math.min(0.9, p.x/MOSAIC_SIZE)); mcy = Math.max(0.1, Math.min(0.9, p.y/MOSAIC_SIZE)); drawMosaic(); });
    mosaicCanvas.addEventListener('mousemove', function(e) { if (!dragging) return; var p = getMosaicPos(e); mcx = Math.max(0.1, Math.min(0.9, p.x/MOSAIC_SIZE)); mcy = Math.max(0.1, Math.min(0.9, p.y/MOSAIC_SIZE)); drawMosaic(); });
    mosaicCanvas.addEventListener('mouseup', function() { dragging = false; });
    mosaicCanvas.addEventListener('mouseleave', function() { dragging = false; });
    mosaicCanvas.addEventListener('touchstart', function(e) { e.preventDefault(); dragging = true; var t = e.touches[0]; var p = getMosaicPos(t); mcx = Math.max(0.1, Math.min(0.9, p.x/MOSAIC_SIZE)); mcy = Math.max(0.1, Math.min(0.9, p.y/MOSAIC_SIZE)); drawMosaic(); });
    mosaicCanvas.addEventListener('touchmove', function(e) { e.preventDefault(); if (!dragging) return; var t = e.touches[0]; var p = getMosaicPos(t); mcx = Math.max(0.1, Math.min(0.9, p.x/MOSAIC_SIZE)); mcy = Math.max(0.1, Math.min(0.9, p.y/MOSAIC_SIZE)); drawMosaic(); });
    mosaicCanvas.addEventListener('touchend', function() { dragging = false; });
    drawMosaic();
  }

  // ========== CUTMIX ==========
  var cutmixCanvas = document.getElementById('cutmixCanvas');
  if (cutmixCanvas) {
    var cmCtx = cutmixCanvas.getContext('2d');
    var cutmixRatioEl = document.getElementById('cutmixRatio');
    var cutmixRatioValEl = document.getElementById('cutmixRatioVal');
    var cutmixLabelAEl = document.getElementById('cutmixLabelA');
    var cutmixLabelBEl = document.getElementById('cutmixLabelB');

    function drawCutmix() {
      var ratio = cutmixRatioEl ? parseInt(cutmixRatioEl.value) / 100 : 0.3;
      drawPattern(cmCtx, 'shapes', SIZE, SIZE);
      var cutW = Math.floor(SIZE * Math.sqrt(ratio));
      var cutH = Math.floor(SIZE * Math.sqrt(ratio));
      var cx = Math.floor((SIZE - cutW) / 2);
      var cy = Math.floor((SIZE - cutH) / 2);
      var tmpC = document.createElement('canvas');
      tmpC.width = cutW; tmpC.height = cutH;
      drawPattern(tmpC.getContext('2d'), 'gradient', cutW, cutH);
      cmCtx.drawImage(tmpC, cx, cy);
      cmCtx.strokeStyle = 'rgba(200,85,61,0.9)'; cmCtx.lineWidth = 2.5; cmCtx.setLineDash([6, 3]);
      cmCtx.strokeRect(cx, cy, cutW, cutH); cmCtx.setLineDash([]);
      cmCtx.fillStyle = 'rgba(200,85,61,0.15)'; cmCtx.fillRect(cx, cy, cutW, cutH);
      if (cutmixRatioValEl) cutmixRatioValEl.textContent = Math.round(ratio*100) + '%';
      if (cutmixLabelAEl) cutmixLabelAEl.textContent = (1 - ratio).toFixed(2);
      if (cutmixLabelBEl) cutmixLabelBEl.textContent = ratio.toFixed(2);
    }

    if (cutmixRatioEl) cutmixRatioEl.addEventListener('input', drawCutmix);
    drawCutmix();
  }

  // ========== MIXUP ==========
  var mixupCanvas = document.getElementById('mixupCanvas');
  if (mixupCanvas) {
    var muCtx = mixupCanvas.getContext('2d');
    var mixupAlphaEl = document.getElementById('mixupAlpha');
    var mixupAlphaValEl = document.getElementById('mixupAlphaVal');
    var mixupWeightAEl = document.getElementById('mixupWeightA');
    var mixupWeightBEl = document.getElementById('mixupWeightB');

    function drawMixup() {
      var alpha = mixupAlphaEl ? parseInt(mixupAlphaEl.value) / 100 : 0.5;
      var tmpA = document.createElement('canvas'); tmpA.width = SIZE; tmpA.height = SIZE;
      var tmpB = document.createElement('canvas'); tmpB.width = SIZE; tmpB.height = SIZE;
      drawPattern(tmpA.getContext('2d'), 'scene', SIZE, SIZE);
      drawPattern(tmpB.getContext('2d'), 'checker', SIZE, SIZE);
      var dataA = tmpA.getContext('2d').getImageData(0, 0, SIZE, SIZE);
      var dataB = tmpB.getContext('2d').getImageData(0, 0, SIZE, SIZE);
      var result = muCtx.createImageData(SIZE, SIZE);
      for (var i = 0; i < result.data.length; i += 4) {
        result.data[i]   = Math.round(alpha * dataA.data[i]   + (1-alpha) * dataB.data[i]);
        result.data[i+1] = Math.round(alpha * dataA.data[i+1] + (1-alpha) * dataB.data[i+1]);
        result.data[i+2] = Math.round(alpha * dataA.data[i+2] + (1-alpha) * dataB.data[i+2]);
        result.data[i+3] = 255;
      }
      muCtx.putImageData(result, 0, 0);
      if (mixupAlphaValEl) mixupAlphaValEl.textContent = alpha.toFixed(2);
      if (mixupWeightAEl) mixupWeightAEl.textContent = alpha.toFixed(2);
      if (mixupWeightBEl) mixupWeightBEl.textContent = (1-alpha).toFixed(2);
    }

    if (mixupAlphaEl) mixupAlphaEl.addEventListener('input', drawMixup);
    drawMixup();
  }

  // ========== RANDOM ERASING ==========
  var eraseCanvas = document.getElementById('eraseCanvas');
  if (eraseCanvas) {
    var eCtx = eraseCanvas.getContext('2d');
    var eraseResetBtn = document.getElementById('eraseResetBtn');
    var eraseRandomBtn = document.getElementById('eraseRandomBtn');
    var EW = 320, EH = 240;

    function drawEraseBase() {
      drawPattern(eCtx, 'scene', EW, EH);
    }

    function addEraseRect(x, y) {
      var w = 30 + Math.random() * 50;
      var h = 30 + Math.random() * 50;
      eCtx.fillStyle = 'rgba(40,40,40,0.85)';
      eCtx.fillRect(x - w/2, y - h/2, w, h);
      eCtx.strokeStyle = COLORS.red; eCtx.lineWidth = 1.5; eCtx.setLineDash([4,2]);
      eCtx.strokeRect(x - w/2, y - h/2, w, h); eCtx.setLineDash([]);
    }

    eraseCanvas.addEventListener('click', function(e) {
      var rect = eraseCanvas.getBoundingClientRect();
      var sx = EW / rect.width, sy = EH / rect.height;
      var x = (e.clientX - rect.left) * sx;
      var y = (e.clientY - rect.top) * sy;
      addEraseRect(x, y);
    });

    if (eraseResetBtn) eraseResetBtn.addEventListener('click', drawEraseBase);
    if (eraseRandomBtn) eraseRandomBtn.addEventListener('click', function() {
      for (var i = 0; i < 3; i++) { addEraseRect(40 + Math.random()*(EW-80), 40 + Math.random()*(EH-80)); }
    });
    drawEraseBase();
  }

  // ========== PIPELINE BUILDER ==========
  var pipeSrcCanvas = document.getElementById('pipeSrcCanvas');
  var pipeDstCanvas = document.getElementById('pipeDstCanvas');
  var pipelineStepsEl = document.getElementById('pipelineSteps');
  var pipeClearBtn = document.getElementById('pipeClearBtn');
  if (pipeSrcCanvas && pipeDstCanvas) {
    var pSrcCtx = pipeSrcCanvas.getContext('2d');
    var pDstCtx = pipeDstCanvas.getContext('2d');
    var PW = 180, PH = 180;
    var pipeline = [];

    function drawPipeSrc() { drawPattern(pSrcCtx, 'scene', PW, PH); }

    function applyPipeline() {
      var tmpC = document.createElement('canvas'); tmpC.width = PW; tmpC.height = PH;
      var tCtx = tmpC.getContext('2d');
      tCtx.drawImage(pipeSrcCanvas, 0, 0);

      pipeline.forEach(function(step) {
        var c2 = document.createElement('canvas'); c2.width = PW; c2.height = PH;
        var ctx2 = c2.getContext('2d');
        if (step === 'flip') {
          ctx2.save(); ctx2.translate(PW, 0); ctx2.scale(-1, 1); ctx2.drawImage(tmpC, 0, 0); ctx2.restore();
        } else if (step === 'rotate') {
          var a = (15 + Math.random()*30) * Math.PI / 180;
          ctx2.save(); ctx2.translate(PW/2, PH/2); ctx2.rotate(a); ctx2.drawImage(tmpC, -PW/2, -PH/2); ctx2.restore();
        } else if (step === 'crop') {
          var m = Math.floor(PW * 0.15);
          ctx2.drawImage(tmpC, m, m, PW-2*m, PH-2*m, 0, 0, PW, PH);
        } else if (step === 'colorjitter') {
          ctx2.drawImage(tmpC, 0, 0);
          var imgD = ctx2.getImageData(0, 0, PW, PH);
          var dd = imgD.data;
          var shift = 20 + Math.random()*40;
          for (var i = 0; i < dd.length; i += 4) {
            dd[i] = Math.min(255, dd[i] + shift);
            dd[i+2] = Math.max(0, dd[i+2] - shift*0.5);
          }
          ctx2.putImageData(imgD, 0, 0);
        } else if (step === 'erase') {
          ctx2.drawImage(tmpC, 0, 0);
          ctx2.fillStyle = 'rgba(40,40,40,0.85)';
          var ew = 30+Math.random()*40, eh = 30+Math.random()*40;
          ctx2.fillRect(Math.random()*(PW-ew), Math.random()*(PH-eh), ew, eh);
        } else if (step === 'blur') {
          ctx2.filter = 'blur(3px)'; ctx2.drawImage(tmpC, 0, 0); ctx2.filter = 'none';
        }
        tCtx.clearRect(0, 0, PW, PH);
        tCtx.drawImage(c2, 0, 0);
      });

      pDstCtx.clearRect(0, 0, PW, PH);
      pDstCtx.drawImage(tmpC, 0, 0);
    }

    function renderPipelineSteps() {
      if (!pipelineStepsEl) return;
      pipelineStepsEl.innerHTML = '';
      if (pipeline.length === 0) {
        pipelineStepsEl.innerHTML = '<span style="color:var(--ink-faint);">（尚未加入步驟）</span>';
        pDstCtx.clearRect(0, 0, PW, PH);
        pDstCtx.drawImage(pipeSrcCanvas, 0, 0);
        return;
      }
      pipeline.forEach(function(step, idx) {
        var chip = document.createElement('span');
        chip.style.cssText = 'background:var(--cream); border:1.5px solid var(--line); border-radius:12px; padding:3px 10px; display:inline-flex; align-items:center; gap:4px;';
        chip.innerHTML = (idx+1) + '. ' + step + ' <span style="cursor:pointer; color:var(--red); font-weight:700;" data-rm="'+idx+'">×</span>';
        pipelineStepsEl.appendChild(chip);
      });
      pipelineStepsEl.querySelectorAll('[data-rm]').forEach(function(btn) {
        btn.addEventListener('click', function() { pipeline.splice(parseInt(btn.dataset.rm), 1); renderPipelineSteps(); applyPipeline(); });
      });
      applyPipeline();
    }

    document.querySelectorAll('[data-pipe]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        if (pipeline.length >= 8) return;
        pipeline.push(btn.dataset.pipe);
        renderPipelineSteps();
        applyPipeline();
      });
    });

    if (pipeClearBtn) pipeClearBtn.addEventListener('click', function() { pipeline = []; renderPipelineSteps(); });

    drawPipeSrc();
    renderPipelineSteps();
  }
};
window.init_augmentation();
