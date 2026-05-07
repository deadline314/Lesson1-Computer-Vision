window.init_activation = function() {
  const activations = {
    relu: { name: 'ReLU', f: x => Math.max(0, x), df: x => x > 0 ? 1 : 0, color: '#c8553d' },
    leaky: { name: 'Leaky ReLU', f: x => x > 0 ? x : 0.1 * x, df: x => x > 0 ? 1 : 0.1, color: '#e89c50' },
    sigmoid: { name: 'Sigmoid', f: x => 1 / (1 + Math.exp(-x)), df: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, color: '#4a87a8' },
    tanh: { name: 'Tanh', f: x => Math.tanh(x), df: x => 1 - Math.tanh(x) ** 2, color: '#8b6db5' },
    silu: { name: 'SiLU/Swish', f: x => x / (1 + Math.exp(-x)), df: x => { const s = 1/(1+Math.exp(-x)); return s + x*s*(1-s); }, color: '#6a9b5e' },
    mish: { name: 'Mish', f: x => x * Math.tanh(Math.log(1 + Math.exp(Math.min(x, 30)))), df: x => { const sp = Math.log(1+Math.exp(Math.min(x,30))); const ts = Math.tanh(sp); const sech2 = 1 - ts*ts; const sigmoid = 1/(1+Math.exp(-x)); return ts + x*sech2*sigmoid; }, color: '#d68fa8' },
    gelu: { name: 'GELU', f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2/Math.PI) * (x + 0.044715 * x*x*x))), df: x => { const c = Math.sqrt(2/Math.PI); const inner = c*(x + 0.044715*x*x*x); const t = Math.tanh(inner); const sech2 = 1 - t*t; return 0.5*(1+t) + 0.5*x*sech2*c*(1 + 3*0.044715*x*x); }, color: '#3d2f1f' }
  };

  let activeActs = new Set(['relu']);

  function plotActivations() {
    const fnPlot = document.getElementById('actPlot');
    const gradPlot = document.getElementById('actGradPlot');
    if (!fnPlot || !gradPlot) return;
    const W = 400, H = 320;
    const xMin = -5, xMax = 5;
    const yMin = -2, yMax = 5;

    function plot(svg, fnKey) {
      svg.innerHTML = '';
      const grid = '<g stroke="rgba(180,150,90,0.2)" stroke-width="1">';
      let g = grid;
      for (let x = -4; x <= 4; x++) {
        const px = ((x - xMin) / (xMax - xMin)) * W;
        g += `<line x1="${px}" y1="20" x2="${px}" y2="${H-30}"/>`;
      }
      for (let y = -1; y <= 4; y++) {
        const py = H - 30 - ((y - yMin) / (yMax - yMin)) * (H - 50);
        g += `<line x1="0" y1="${py}" x2="${W}" y2="${py}"/>`;
      }
      g += '</g>';

      const ax = ((0 - xMin) / (xMax - xMin)) * W;
      const ay = H - 30 - ((0 - yMin) / (yMax - yMin)) * (H - 50);
      g += `<line x1="${ax}" y1="20" x2="${ax}" y2="${H-30}" stroke="#3d2f1f" stroke-width="1.5"/>`;
      g += `<line x1="0" y1="${ay}" x2="${W}" y2="${ay}" stroke="#3d2f1f" stroke-width="1.5"/>`;

      for (let x = -4; x <= 4; x += 2) {
        if (x === 0) continue;
        const px = ((x - xMin) / (xMax - xMin)) * W;
        g += `<text x="${px}" y="${H-12}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#6b5942">${x}</text>`;
      }
      for (let y = -1; y <= 4; y++) {
        if (y === 0) continue;
        const py = H - 30 - ((y - yMin) / (yMax - yMin)) * (H - 50);
        g += `<text x="${ax - 4}" y="${py + 3}" text-anchor="end" font-family="JetBrains Mono" font-size="10" fill="#6b5942">${y}</text>`;
      }

      activeActs.forEach(actKey => {
        const a = activations[actKey];
        const fn = fnKey === 'f' ? a.f : a.df;
        let path = '';
        const N = 200;
        for (let i = 0; i <= N; i++) {
          const x = xMin + (i / N) * (xMax - xMin);
          let y = fn(x);
          y = Math.max(yMin - 1, Math.min(yMax + 1, y));
          const px = (i / N) * W;
          const py = H - 30 - ((y - yMin) / (yMax - yMin)) * (H - 50);
          path += (i === 0 ? 'M' : ' L') + ` ${px.toFixed(1)} ${py.toFixed(1)}`;
        }
        g += `<path d="${path}" stroke="${a.color}" stroke-width="2.5" fill="none" stroke-linecap="round"/>`;
        const labelY = 28 + Array.from(activeActs).indexOf(actKey) * 18;
        g += `<rect x="${W - 110}" y="${labelY - 12}" width="100" height="16" rx="3" fill="${a.color}" opacity="0.9"/>`;
        g += `<text x="${W - 60}" y="${labelY}" text-anchor="middle" font-family="Kalam" font-size="11" fill="white" font-weight="700">${a.name}</text>`;
      });

      svg.innerHTML = g;
    }

    plot(fnPlot, 'f');
    plot(gradPlot, 'df');
    applyActivationToImage();
  }

  function applyActivationToImage() {
    const inCv = document.getElementById('actInput');
    const outCv = document.getElementById('actOutput');
    if (!inCv || !outCv) return;
    const inCtx = inCv.getContext('2d');
    const outCtx = outCv.getContext('2d');
    const SZ = 200;

    const inImg = inCtx.createImageData(SZ, SZ);
    for (let y = 0; y < SZ; y++) {
      for (let x = 0; x < SZ; x++) {
        const cx = x - SZ/2, cy = y - SZ/2;
        const v = (Math.sin(x * 0.05) + Math.cos(y * 0.05) + Math.sin((cx + cy) * 0.04)) * 1.5;
        const i = (y * SZ + x) * 4;
        if (v < 0) {
          inImg.data[i] = 200 + v * 30;
          inImg.data[i+1] = 100;
          inImg.data[i+2] = 100;
        } else {
          inImg.data[i] = 100;
          inImg.data[i+1] = 100 + v * 40;
          inImg.data[i+2] = 100;
        }
        inImg.data[i+3] = 255;
      }
    }
    inCtx.putImageData(inImg, 0, 0);

    const actKey = activeActs.values().next().value || 'relu';
    const act = activations[actKey];
    const labelEl = document.getElementById('actOutLabel');
    if (labelEl) labelEl.textContent = `After ${act.name}`;

    const outImg = outCtx.createImageData(SZ, SZ);
    for (let y = 0; y < SZ; y++) {
      for (let x = 0; x < SZ; x++) {
        const cx = x - SZ/2, cy = y - SZ/2;
        const v = (Math.sin(x * 0.05) + Math.cos(y * 0.05) + Math.sin((cx + cy) * 0.04)) * 1.5;
        const out = act.f(v);
        const i = (y * SZ + x) * 4;
        if (out < 0.05 && out > -0.05) {
          outImg.data[i] = 30; outImg.data[i+1] = 30; outImg.data[i+2] = 30;
        } else if (out < 0) {
          outImg.data[i] = 200 + out * 50;
          outImg.data[i+1] = 100;
          outImg.data[i+2] = 100;
        } else {
          outImg.data[i] = 100;
          outImg.data[i+1] = Math.min(255, 100 + out * 60);
          outImg.data[i+2] = 100;
        }
        outImg.data[i+3] = 255;
      }
    }
    outCtx.putImageData(outImg, 0, 0);
  }

  document.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.act;
      if (activeActs.has(key) && activeActs.size > 1) {
        activeActs.delete(key);
        btn.dataset.active = 'false';
      } else {
        activeActs.add(key);
        btn.dataset.active = 'true';
      }
      plotActivations();
    });
  });

  plotActivations();
};
window.init_activation();
