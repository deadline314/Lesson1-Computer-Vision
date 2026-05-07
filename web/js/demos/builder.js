window.init_builder = function() {
  const palette = document.getElementById('palette');
  const stack = document.getElementById('stack');
  const codeOutput = document.getElementById('codeOutput');
  const paramCountEl = document.getElementById('paramCount');
  const clearBtn = document.getElementById('clearBtn');
  const presetBtn = document.getElementById('presetBtn');
  const presetVgg = document.getElementById('presetVgg');
  if (!palette || !stack) return;

  let layers = [];
  const INPUT_SHAPE = [3, 224, 224]; // C, H, W

  const layerDefs = {
    Conv:     { label: 'Conv2d 3×3', color: 'var(--green)' },
    ConvDown: { label: 'Conv2d 3×3 ↓2', color: 'var(--green)' },
    BN:       { label: 'BatchNorm2d', color: 'var(--yellow)' },
    ReLU:     { label: 'ReLU', color: 'var(--orange)' },
    MaxPool:  { label: 'MaxPool2d 2×2', color: 'var(--blue)' },
    GAP:      { label: 'GlobalAvgPool', color: 'var(--purple)' },
    Flatten:  { label: 'Flatten', color: 'var(--ink-faint)' },
    FC:       { label: 'Linear → 1000', color: 'var(--red)' },
    Residual: { label: 'ResidualBlock', color: 'var(--pink)' }
  };

  function computeShapes() {
    let c = INPUT_SHAPE[0], h = INPUT_SHAPE[1], w = INPUT_SHAPE[2];
    let flat = false;
    let shapes = [{ c, h, w, flat }];
    let totalParams = 0;

    layers.forEach(layer => {
      let params = 0;
      if (flat) {
        if (layer.type === 'FC') {
          const inF = c;
          c = 1000;
          params = inF * c + c;
        }
      } else {
        switch (layer.type) {
          case 'Conv': {
            const outC = Math.min(c * 2, 512);
            params = c * outC * 9 + outC;
            c = outC;
            break;
          }
          case 'ConvDown': {
            const outC = Math.min(c * 2, 512);
            params = c * outC * 9 + outC;
            c = outC;
            h = Math.floor(h / 2);
            w = Math.floor(w / 2);
            break;
          }
          case 'BN': {
            params = c * 4;
            break;
          }
          case 'ReLU': break;
          case 'MaxPool': {
            h = Math.floor(h / 2);
            w = Math.floor(w / 2);
            break;
          }
          case 'GAP': {
            h = 1; w = 1;
            break;
          }
          case 'Flatten': {
            c = c * h * w;
            h = 1; w = 1;
            flat = true;
            break;
          }
          case 'FC': {
            const inF = c * h * w;
            c = 1000;
            h = 1; w = 1;
            flat = true;
            params = inF * c + c;
            break;
          }
          case 'Residual': {
            const outC = Math.min(c * 2, 512);
            params = c * outC * 9 + outC + outC * outC * 9 + outC;
            if (c !== outC) params += c * outC + outC;
            c = outC;
            break;
          }
        }
      }
      totalParams += params;
      shapes.push({ c, h, w, flat, params });
    });

    return { shapes, totalParams };
  }

  function formatNum(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
  }

  function render() {
    const { shapes, totalParams } = computeShapes();

    if (layers.length === 0) {
      stack.innerHTML = '<div class="stack-empty">把左邊的 layer 拖過來,或直接點擊</div>';
    } else {
      stack.innerHTML = layers.map((layer, idx) => {
        const s = shapes[idx + 1];
        const shapeStr = s.flat ? `(B, ${s.c})` : `(B, ${s.c}, ${s.h}, ${s.w})`;
        const def = layerDefs[layer.type] || { label: layer.type, color: 'var(--ink)' };
        return `<div class="stack-item" style="border-left-color: ${def.color};">
          <span>${def.label}</span>
          <span class="stack-item-shape">${shapeStr}</span>
          <button class="stack-item-remove" data-idx="${idx}">&times;</button>
        </div>`;
      }).join('');
    }

    paramCountEl.textContent = formatNum(totalParams);
    generateCode();

    stack.querySelectorAll('.stack-item-remove').forEach(btn => {
      btn.addEventListener('click', () => {
        layers.splice(parseInt(btn.dataset.idx), 1);
        render();
      });
    });
  }

  function generateCode() {
    if (layers.length === 0) {
      codeOutput.innerHTML = '<span class="com"># 拖 layer 進來,code 會自動生成</span>';
      return;
    }

    const { shapes } = computeShapes();
    let initLines = [];
    let fwdLines = [];
    let prevC = INPUT_SHAPE[0];

    layers.forEach((layer, idx) => {
      const s = shapes[idx + 1];
      const prev = shapes[idx];
      const i = idx + 1;

      switch (layer.type) {
        case 'Conv':
          initLines.push(`<span class="kw">self</span>.conv${i} = nn.Conv2d(${prev.c}, ${s.c}, 3, padding=1)`);
          fwdLines.push(`x = <span class="kw">self</span>.conv${i}(x)`);
          break;
        case 'ConvDown':
          initLines.push(`<span class="kw">self</span>.conv${i} = nn.Conv2d(${prev.c}, ${s.c}, 3, stride=2, padding=1)`);
          fwdLines.push(`x = <span class="kw">self</span>.conv${i}(x)`);
          break;
        case 'BN':
          initLines.push(`<span class="kw">self</span>.bn${i} = nn.BatchNorm2d(${s.c})`);
          fwdLines.push(`x = <span class="kw">self</span>.bn${i}(x)`);
          break;
        case 'ReLU':
          fwdLines.push(`x = F.relu(x)`);
          break;
        case 'MaxPool':
          fwdLines.push(`x = F.max_pool2d(x, 2)`);
          break;
        case 'GAP':
          fwdLines.push(`x = F.adaptive_avg_pool2d(x, 1)`);
          break;
        case 'Flatten':
          fwdLines.push(`x = x.view(x.size(0), -1)`);
          break;
        case 'FC': {
          const inF = prev.flat ? prev.c : prev.c * prev.h * prev.w;
          initLines.push(`<span class="kw">self</span>.fc${i} = nn.Linear(${inF}, 1000)`);
          fwdLines.push(`x = <span class="kw">self</span>.fc${i}(x)`);
          break;
        }
        case 'Residual':
          initLines.push(`<span class="kw">self</span>.res${i} = ResidualBlock(${prev.c}, ${s.c})`);
          fwdLines.push(`x = <span class="kw">self</span>.res${i}(x)`);
          break;
      }
    });

    let code = `<span class="kw">class</span> <span class="cls">MyModel</span>(nn.Module):\n`;
    code += `    <span class="kw">def</span> __init__(<span class="kw">self</span>):\n`;
    code += `        super().__init__()\n`;
    initLines.forEach(l => code += `        ${l}\n`);
    code += `\n    <span class="kw">def</span> forward(<span class="kw">self</span>, x):\n`;
    code += `        <span class="com"># x: (B, 3, 224, 224)</span>\n`;
    fwdLines.forEach(l => code += `        ${l}\n`);
    code += `        <span class="kw">return</span> x`;

    codeOutput.innerHTML = code;
  }

  function addLayer(type) {
    layers.push({ type });
    render();
  }

  // Click to add
  palette.querySelectorAll('.palette-item').forEach(item => {
    item.addEventListener('click', () => addLayer(item.dataset.type));
  });

  // Drag and drop
  palette.querySelectorAll('.palette-item').forEach(item => {
    item.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', item.dataset.type);
    });
  });

  stack.addEventListener('dragover', (e) => { e.preventDefault(); stack.classList.add('drag-over'); });
  stack.addEventListener('dragleave', () => stack.classList.remove('drag-over'));
  stack.addEventListener('drop', (e) => {
    e.preventDefault();
    stack.classList.remove('drag-over');
    const type = e.dataTransfer.getData('text/plain');
    if (type && layerDefs[type]) addLayer(type);
  });

  // Clear
  if (clearBtn) clearBtn.addEventListener('click', () => { layers = []; render(); });

  // Presets
  if (presetBtn) presetBtn.addEventListener('click', () => {
    layers = [
      { type: 'Conv' }, { type: 'BN' }, { type: 'ReLU' },
      { type: 'Residual' },
      { type: 'ConvDown' }, { type: 'BN' }, { type: 'ReLU' },
      { type: 'Residual' },
      { type: 'ConvDown' }, { type: 'BN' }, { type: 'ReLU' },
      { type: 'Residual' },
      { type: 'GAP' }, { type: 'Flatten' }, { type: 'FC' }
    ];
    render();
  });

  if (presetVgg) presetVgg.addEventListener('click', () => {
    layers = [
      { type: 'Conv' }, { type: 'ReLU' }, { type: 'Conv' }, { type: 'ReLU' }, { type: 'MaxPool' },
      { type: 'Conv' }, { type: 'ReLU' }, { type: 'Conv' }, { type: 'ReLU' }, { type: 'MaxPool' },
      { type: 'Conv' }, { type: 'ReLU' }, { type: 'Conv' }, { type: 'ReLU' }, { type: 'Conv' }, { type: 'ReLU' }, { type: 'MaxPool' },
      { type: 'Flatten' }, { type: 'FC' }
    ];
    render();
  });

  render();
};
window.init_builder();
