window.init_pool = function() {
  let poolType = 'max';
  let poolWin = 2;
  let poolData = [];

  function genPoolData() {
    poolData = [];
    for (let i = 0; i < 6 * 6; i++) {
      poolData.push(Math.floor(Math.random() * 9) + 1);
    }
  }
  genPoolData();

  function renderPool() {
    const isGAP = poolType === 'gap';
    const win = isGAP ? 6 : poolWin;
    const inSize = 6;
    const outSize = isGAP ? 1 : Math.floor(inSize / win);

    const poolWinValEl = document.getElementById('poolWinVal');
    if (poolWinValEl) poolWinValEl.textContent = isGAP ? '全部' : `${poolWin} × ${poolWin}`;
    const poolInEl = document.getElementById('poolInShape');
    if (poolInEl) poolInEl.textContent = `${inSize} × ${inSize}`;
    const poolOutEl = document.getElementById('poolOutShape');
    if (poolOutEl) poolOutEl.textContent = isGAP ? '1 × 1' : `${outSize} × ${outSize}`;
    const poolOpEl = document.getElementById('poolOp');
    if (poolOpEl) poolOpEl.textContent = poolType === 'max' ? 'Max' : poolType === 'avg' ? 'Average' : 'GAP';
    const poolPyEl = document.getElementById('poolPyCode');
    if (poolPyEl) poolPyEl.textContent =
      poolType === 'max' ? `nn.MaxPool2d(${poolWin}, ${poolWin})` :
      poolType === 'avg' ? `nn.AvgPool2d(${poolWin}, ${poolWin})` :
      `nn.AdaptiveAvgPool2d(1)`;

    const inputViz = document.getElementById('poolInputViz');
    if (!inputViz) return;
    inputViz.innerHTML = '';
    const inputGrid = document.createElement('div');
    inputGrid.className = 'pool-grid';
    inputGrid.style.gridTemplateColumns = `repeat(${inSize}, 1fr)`;
    for (let y = 0; y < inSize; y++) {
      for (let x = 0; x < inSize; x++) {
        const cell = document.createElement('div');
        cell.className = 'pool-cell';
        cell.textContent = poolData[y * inSize + x];
        cell.dataset.x = x;
        cell.dataset.y = y;
        inputGrid.appendChild(cell);
      }
    }
    inputViz.appendChild(inputGrid);

    const outputViz = document.getElementById('poolOutputViz');
    if (!outputViz) return;
    outputViz.innerHTML = '';
    const outputGrid = document.createElement('div');
    outputGrid.className = 'pool-grid';
    outputGrid.style.gridTemplateColumns = `repeat(${outSize}, 1fr)`;
    outputGrid.style.gap = '6px';

    for (let oy = 0; oy < outSize; oy++) {
      for (let ox = 0; ox < outSize; ox++) {
        let vals = [];
        for (let dy = 0; dy < win; dy++) {
          for (let dx = 0; dx < win; dx++) {
            const ix = ox * win + dx;
            const iy = oy * win + dy;
            if (ix < inSize && iy < inSize) vals.push(poolData[iy * inSize + ix]);
          }
        }
        let result;
        if (poolType === 'max') result = Math.max(...vals);
        else result = (vals.reduce((a,b) => a+b, 0) / vals.length).toFixed(1);

        const cell = document.createElement('div');
        cell.className = 'pool-output-cell';
        cell.textContent = result;
        cell.dataset.ox = ox;
        cell.dataset.oy = oy;
        cell.style.cursor = 'pointer';
        cell.addEventListener('mouseenter', () => highlightPoolWindow(ox, oy, win));
        cell.addEventListener('mouseleave', () => clearPoolHighlight());
        outputGrid.appendChild(cell);
      }
    }
    outputViz.appendChild(outputGrid);
    renderPoolCompare();
  }

  function highlightPoolWindow(ox, oy, win) {
    document.querySelectorAll('.pool-cell').forEach(c => {
      const x = parseInt(c.dataset.x), y = parseInt(c.dataset.y);
      if (x >= ox * win && x < (ox + 1) * win && y >= oy * win && y < (oy + 1) * win) {
        c.classList.add('in-window');
      }
    });
    if (poolType === 'max') {
      let maxVal = -Infinity, maxX = 0, maxY = 0;
      for (let dy = 0; dy < win; dy++) {
        for (let dx = 0; dx < win; dx++) {
          const ix = ox * win + dx, iy = oy * win + dy;
          if (ix < 6 && iy < 6) {
            const v = poolData[iy * 6 + ix];
            if (v > maxVal) { maxVal = v; maxX = ix; maxY = iy; }
          }
        }
      }
      document.querySelectorAll('.pool-cell').forEach(c => {
        if (parseInt(c.dataset.x) === maxX && parseInt(c.dataset.y) === maxY) c.classList.add('is-max');
      });
    }
  }

  function clearPoolHighlight() {
    document.querySelectorAll('.pool-cell').forEach(c => {
      c.classList.remove('in-window', 'is-max');
    });
  }

  function renderPoolCompare() {
    const compareDiv = document.getElementById('poolCompareViz');
    if (!compareDiv) return;
    compareDiv.innerHTML = '';
    const types = [
      { name: 'Max Pool 2×2', op: 'max', win: 2, color: 'var(--red)' },
      { name: 'Avg Pool 2×2', op: 'avg', win: 2, color: 'var(--blue)' },
      { name: 'Global Avg Pool', op: 'gap', win: 6, color: 'var(--green)' }
    ];

    types.forEach(t => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'background: rgba(255,255,255,0.5); border: 1.5px dashed var(--line); border-radius: 6px; padding: 12px; text-align: center;';
      wrap.innerHTML = `<div style="font-family: 'Caveat', cursive; font-size: 18px; color: ${t.color}; font-weight: 700; margin-bottom: 8px;">${t.name}</div>`;

      const grid = document.createElement('div');
      grid.style.cssText = 'display: grid; gap: 3px; background: var(--ink); padding: 3px; border-radius: 4px; width: fit-content; margin: 0 auto; border: 2px solid var(--ink);';

      const win = t.win;
      const outSize = t.op === 'gap' ? 1 : Math.floor(6 / win);
      grid.style.gridTemplateColumns = `repeat(${outSize}, 1fr)`;

      for (let oy = 0; oy < outSize; oy++) {
        for (let ox = 0; ox < outSize; ox++) {
          let vals = [];
          for (let dy = 0; dy < win; dy++) {
            for (let dx = 0; dx < win; dx++) {
              const ix = ox * win + dx, iy = oy * win + dy;
              if (ix < 6 && iy < 6) vals.push(poolData[iy * 6 + ix]);
            }
          }
          let result;
          if (t.op === 'max') result = Math.max(...vals);
          else result = (vals.reduce((a,b) => a+b, 0) / vals.length).toFixed(1);

          const cell = document.createElement('div');
          cell.style.cssText = `width: 38px; height: 38px; background: ${t.color}; color: white; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700; border-radius: 2px;`;
          cell.textContent = result;
          grid.appendChild(cell);
        }
      }
      wrap.appendChild(grid);
      compareDiv.appendChild(wrap);
    });
  }

  document.querySelectorAll('[data-pool]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-pool]').forEach(b => b.dataset.active = 'false');
      btn.dataset.active = 'true';
      poolType = btn.dataset.pool;
      renderPool();
    });
  });
  const poolWinEl = document.getElementById('poolWin');
  if (poolWinEl) {
    poolWinEl.addEventListener('input', (e) => {
      poolWin = parseInt(e.target.value);
      renderPool();
    });
  }
  const poolShuffleEl = document.getElementById('poolShuffle');
  if (poolShuffleEl) {
    poolShuffleEl.addEventListener('click', () => {
      genPoolData();
      renderPool();
    });
  }
  renderPool();
};
window.init_pool();
