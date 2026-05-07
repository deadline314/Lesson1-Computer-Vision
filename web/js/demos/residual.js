window.init_residual = function() {
  let resAmp = 0.5;
  let resPhase = 0.2;

  function plotResidual() {
    const svg = document.getElementById('resPlot');
    if (!svg) return;
    const W = 500, H = 320;
    svg.innerHTML = '';

    let html = '';

    for (let x = 50; x <= W; x += 50) {
      html += `<line x1="${x}" y1="20" x2="${x}" y2="${H-30}" stroke="rgba(180,150,90,0.15)" stroke-width="1"/>`;
    }
    for (let y = 60; y < H - 30; y += 40) {
      html += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="rgba(180,150,90,0.15)" stroke-width="1"/>`;
    }

    const cy = H / 2 - 5;
    html += `<line x1="0" y1="${cy}" x2="${W}" y2="${cy}" stroke="#3d2f1f" stroke-width="1.5"/>`;
    html += `<text x="8" y="${cy - 6}" font-family="JetBrains Mono" font-size="10" fill="#6b5942">y = 0</text>`;

    function plotPath(fn, color, width) {
      let path = '';
      const N = 200;
      for (let i = 0; i <= N; i++) {
        const x = i / N;
        const y = fn(x);
        const px = x * W;
        const py = cy - y * 100;
        path += (i === 0 ? 'M' : ' L') + ` ${px.toFixed(1)} ${py.toFixed(1)}`;
      }
      return `<path d="${path}" stroke="${color}" stroke-width="${width}" fill="none" stroke-linecap="round"/>`;
    }

    const inputFn = t => Math.sin(t * Math.PI * 2) * 0.7;
    const residualFn = t => Math.sin(t * Math.PI * 4 + resPhase * Math.PI * 2) * resAmp;
    const outputFn = t => inputFn(t) + residualFn(t);

    html += plotPath(inputFn, '#4a87a8', 2.5);
    html += plotPath(residualFn, '#e89c50', 2.5);
    html += plotPath(outputFn, '#c8553d', 3);

    const endX = W - 12;
    html += `<text x="${endX}" y="${cy - inputFn(1) * 100 - 6}" text-anchor="end" font-family="Caveat" font-size="14" fill="#4a87a8" font-weight="700">x</text>`;
    html += `<text x="${endX}" y="${cy - residualFn(1) * 100 - 6}" text-anchor="end" font-family="Caveat" font-size="14" fill="#e89c50" font-weight="700">F(x)</text>`;
    html += `<text x="${endX}" y="${cy - outputFn(1) * 100 - 6}" text-anchor="end" font-family="Caveat" font-size="14" fill="#c8553d" font-weight="700">F(x)+x</text>`;

    svg.innerHTML = html;
  }

  const resAmpEl = document.getElementById('resAmp');
  if (resAmpEl) {
    resAmpEl.addEventListener('input', (e) => {
      resAmp = parseInt(e.target.value) / 100;
      document.getElementById('resAmpVal').textContent = resAmp.toFixed(1);
      plotResidual();
    });
  }
  const resPhaseEl = document.getElementById('resPhase');
  if (resPhaseEl) {
    resPhaseEl.addEventListener('input', (e) => {
      resPhase = parseInt(e.target.value) / 100;
      document.getElementById('resPhaseVal').textContent = resPhase.toFixed(1);
      plotResidual();
    });
  }
  plotResidual();

  // Residual gradient flow viz
  function drawResGradFlow(animProgress) {
    const svg = document.getElementById('resGradFlow');
    if (!svg) return;
    const W = 800, H = 240;
    svg.innerHTML = '';

    let html = '';

    const numLayers = 8;
    const startX = 60;
    const spacing = (W - 120) / (numLayers - 1);
    const topY = 60;
    const botY = 180;

    html += `<text x="20" y="${topY + 5}" font-family="Caveat" font-size="18" fill="#3d2f1f" font-weight="700">Plain</text>`;
    html += `<text x="20" y="${botY + 5}" font-family="Caveat" font-size="18" fill="#3d2f1f" font-weight="700">ResNet</text>`;

    const factor = 0.6;
    for (let i = 0; i < numLayers; i++) {
      const x = startX + i * spacing;
      const distFromOutput = numLayers - 1 - i;

      let plainGrad = Math.pow(factor, distFromOutput);
      let resGrad = Math.min(1, plainGrad + (1 - Math.pow(0.92, distFromOutput)));

      if (typeof animProgress === 'number') {
        const wavePos = numLayers - 1 - animProgress * numLayers;
        if (i > wavePos) {
          plainGrad *= 0.2;
          resGrad *= 0.2;
        }
      }

      const plainColor = `rgba(200, 85, 61, ${Math.max(0.15, plainGrad)})`;
      const resColor = `rgba(106, 155, 94, ${Math.max(0.15, resGrad)})`;

      html += `<circle cx="${x}" cy="${topY}" r="20" fill="${plainColor}" stroke="#3d2f1f" stroke-width="2"/>`;
      html += `<text x="${x}" y="${topY + 5}" text-anchor="middle" font-family="JetBrains Mono" font-size="11" fill="#3d2f1f" font-weight="700">L${i+1}</text>`;

      html += `<circle cx="${x}" cy="${botY}" r="20" fill="${resColor}" stroke="#3d2f1f" stroke-width="2"/>`;
      html += `<text x="${x}" y="${botY + 5}" text-anchor="middle" font-family="JetBrains Mono" font-size="11" fill="#3d2f1f" font-weight="700">L${i+1}</text>`;

      if (i < numLayers - 1) {
        const x2 = startX + (i + 1) * spacing;
        html += `<line x1="${x + 20}" y1="${topY}" x2="${x2 - 20}" y2="${topY}" stroke="#c8553d" stroke-width="2" opacity="${Math.max(0.2, plainGrad)}"/>`;
        html += `<line x1="${x + 20}" y1="${botY}" x2="${x2 - 20}" y2="${botY}" stroke="#6a9b5e" stroke-width="2" opacity="${Math.max(0.2, resGrad)}"/>`;
      }

      if (i % 2 === 0 && i + 2 < numLayers) {
        const x2 = startX + (i + 2) * spacing;
        const arcMidX = (x + x2) / 2;
        html += `<path d="M ${x + 20} ${botY - 5} Q ${arcMidX} ${botY - 35}, ${x2 - 20} ${botY - 5}" fill="none" stroke="#6a9b5e" stroke-width="2.5" stroke-dasharray="4,3" opacity="${Math.max(0.4, resGrad)}"/>`;
      }
    }

    html += `<text x="${W - 15}" y="30" text-anchor="end" font-family="Caveat" font-size="20" fill="#c8553d" font-weight="700">← backward</text>`;
    html += `<text x="${W - 15}" y="${H - 10}" text-anchor="end" font-family="Patrick Hand" font-size="13" fill="#6b5942">梯度從輸出層往輸入層傳</text>`;

    svg.innerHTML = html;
  }

  const resAnimBtn = document.getElementById('resAnimBtn');
  if (resAnimBtn) {
    resAnimBtn.addEventListener('click', () => {
      if (resAnimBtn.dataset.playing === 'true') return;
      resAnimBtn.dataset.playing = 'true';
      resAnimBtn.textContent = '⏸ 播放中...';
      let progress = 0;
      const interval = setInterval(() => {
        progress += 0.025;
        if (progress >= 1.1) {
          clearInterval(interval);
          resAnimBtn.dataset.playing = 'false';
          resAnimBtn.textContent = '⏵ 播放梯度反向傳播動畫';
          drawResGradFlow();
        } else {
          drawResGradFlow(progress);
        }
      }, 60);
    });
  }

  drawResGradFlow();
};
window.init_residual();
