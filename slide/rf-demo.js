// Receptive Field demo — animated grid showing how RF grows with depth
(function() {
  const root = document.getElementById('rfDemo');
  if (!root) return;

  const W = 9;
  const layers = [
    { label: 'Input · 9×9', size: 9, kernel: 0, center: [4,4] },
    { label: 'Layer 1 · 3×3 conv → RF 3×3', size: 9, kernel: 3, center: [4,4] },
    { label: 'Layer 2 · 3×3 conv → RF 5×5', size: 9, kernel: 5, center: [4,4] }
  ];

  let activeIdx = 0;

  function render() {
    root.innerHTML = '';
    const layer = layers[activeIdx];
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 18px;';

    const grid = document.createElement('div');
    grid.className = 'rf-grid';
    grid.style.gridTemplateColumns = `repeat(${W}, 32px)`;
    grid.style.gridTemplateRows = `repeat(${W}, 32px)`;

    const k = layer.kernel;
    const half = Math.floor(k / 2);
    const [cx, cy] = layer.center;

    for (let r = 0; r < W; r++) {
      for (let c = 0; c < W; c++) {
        const cell = document.createElement('div');
        cell.className = 'rf-cell';
        cell.style.width = '32px';
        cell.style.height = '32px';
        if (k > 0 && Math.abs(r - cy) <= half && Math.abs(c - cx) <= half) {
          cell.classList.add('active');
        }
        if (r === cy && c === cx && k > 0) {
          cell.classList.add('highlight');
        }
        grid.appendChild(cell);
      }
    }

    const label = document.createElement('div');
    label.style.cssText = 'font-family: "Patrick Hand", cursive; font-size: 22px; color: var(--ink-soft); text-align: center;';
    label.textContent = layer.label;

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display: flex; gap: 8px; justify-content: center;';
    layers.forEach((l, i) => {
      const btn = document.createElement('button');
      btn.style.cssText = `
        padding: 8px 14px; border: 2px solid var(--ink); border-radius: 6px;
        background: ${i === activeIdx ? 'var(--ink)' : 'var(--cream)'};
        color: ${i === activeIdx ? 'var(--paper)' : 'var(--ink)'};
        font-family: 'Kalam', cursive; font-weight: 700; cursor: pointer;
        font-size: 14px;
      `;
      btn.textContent = `Step ${i}`;
      btn.onclick = () => { activeIdx = i; render(); };
      btnRow.appendChild(btn);
    });

    wrap.appendChild(label);
    wrap.appendChild(grid);
    wrap.appendChild(btnRow);
    root.appendChild(wrap);
  }

  render();

  // auto-cycle when slide is active
  let timer = null;
  function startCycle() {
    if (timer) return;
    timer = setInterval(() => {
      activeIdx = (activeIdx + 1) % layers.length;
      render();
    }, 1800);
  }
  function stopCycle() { if (timer) { clearInterval(timer); timer = null; } }

  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) startCycle(); else stopCycle();
    });
  });
  observer.observe(root);
})();
