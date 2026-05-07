window.init_bn = function() {
  let bnShift = 0;
  let bnScale = 1;
  let bnGamma = 1;
  let bnBeta = 0;

  const bnSamples = [];
  for (let i = 0; i < 100; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    bnSamples.push(z);
  }

  function plotBN() {
    const W = 400, H = 280;
    const xMin = -10, xMax = 10;

    function drawHist(svgId, data, meanLabel, stdLabel) {
      const svg = document.getElementById(svgId);
      if (!svg) return;
      svg.innerHTML = '';

      const mean = data.reduce((a,b) => a+b, 0) / data.length;
      const std = Math.sqrt(data.reduce((a,b) => a + (b-mean)**2, 0) / data.length);

      const meanEl = document.getElementById(meanLabel);
      if (meanEl) meanEl.textContent = mean.toFixed(2);
      const stdEl = document.getElementById(stdLabel);
      if (stdEl) stdEl.textContent = std.toFixed(2);

      const numBins = 30;
      const bins = new Array(numBins).fill(0);
      data.forEach(v => {
        const binIdx = Math.floor(((v - xMin) / (xMax - xMin)) * numBins);
        if (binIdx >= 0 && binIdx < numBins) bins[binIdx]++;
      });
      const maxBin = Math.max(...bins, 1);

      let html = '';

      for (let x = -8; x <= 8; x += 2) {
        const px = ((x - xMin) / (xMax - xMin)) * W;
        html += `<line x1="${px}" y1="20" x2="${px}" y2="${H-30}" stroke="rgba(180,150,90,0.2)" stroke-width="1"/>`;
        if (x !== 0) html += `<text x="${px}" y="${H-12}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#6b5942">${x}</text>`;
      }

      const cx = ((0 - xMin) / (xMax - xMin)) * W;
      html += `<line x1="${cx}" y1="20" x2="${cx}" y2="${H-30}" stroke="#3d2f1f" stroke-width="1.5"/>`;
      html += `<line x1="0" y1="${H-30}" x2="${W}" y2="${H-30}" stroke="#3d2f1f" stroke-width="1.5"/>`;

      const meanPx = ((mean - xMin) / (xMax - xMin)) * W;
      html += `<line x1="${meanPx}" y1="20" x2="${meanPx}" y2="${H-30}" stroke="#c8553d" stroke-width="2" stroke-dasharray="4,4"/>`;
      html += `<text x="${meanPx + 4}" y="32" font-family="Caveat" font-size="14" fill="#c8553d" font-weight="700">μ = ${mean.toFixed(2)}</text>`;

      const std1 = ((mean - std - xMin) / (xMax - xMin)) * W;
      const std2 = ((mean + std - xMin) / (xMax - xMin)) * W;
      html += `<rect x="${std1}" y="20" width="${std2-std1}" height="${H-50}" fill="rgba(232, 197, 71, 0.2)"/>`;

      bins.forEach((count, i) => {
        if (count === 0) return;
        const barH = (count / maxBin) * (H - 60);
        const barX = (i / numBins) * W;
        const barW = W / numBins - 1;
        html += `<rect x="${barX}" y="${H - 30 - barH}" width="${barW}" height="${barH}" fill="#6a9b5e" opacity="0.7" stroke="#3d2f1f" stroke-width="1" rx="1"/>`;
      });

      svg.innerHTML = html;
    }

    const original = bnSamples.map(v => v * bnScale + bnShift);
    const mean = original.reduce((a,b) => a+b, 0) / original.length;
    const std = Math.sqrt(original.reduce((a,b) => a + (b-mean)**2, 0) / original.length) || 1e-5;
    const bnOut = original.map(v => bnGamma * (v - mean) / std + bnBeta);

    drawHist('bnPlot1', original, 'bnMean1', 'bnStd1');
    drawHist('bnPlot2', bnOut, 'bnMean2', 'bnStd2');
  }

  const bnShiftEl = document.getElementById('bnShift');
  if (bnShiftEl) {
    bnShiftEl.addEventListener('input', (e) => {
      bnShift = parseInt(e.target.value) / 10;
      document.getElementById('bnShiftVal').textContent = bnShift.toFixed(1);
      plotBN();
    });
  }
  const bnScaleEl = document.getElementById('bnScale');
  if (bnScaleEl) {
    bnScaleEl.addEventListener('input', (e) => {
      bnScale = parseInt(e.target.value) / 10;
      document.getElementById('bnScaleVal').textContent = bnScale.toFixed(1);
      plotBN();
    });
  }
  const bnGammaEl = document.getElementById('bnGamma');
  if (bnGammaEl) {
    bnGammaEl.addEventListener('input', (e) => {
      bnGamma = parseInt(e.target.value) / 10;
      document.getElementById('bnGammaVal').textContent = bnGamma.toFixed(1);
      plotBN();
    });
  }
  const bnBetaEl = document.getElementById('bnBeta');
  if (bnBetaEl) {
    bnBetaEl.addEventListener('input', (e) => {
      bnBeta = parseInt(e.target.value) / 10;
      document.getElementById('bnBetaVal').textContent = bnBeta.toFixed(1);
      plotBN();
    });
  }
  plotBN();
};
window.init_bn();
