// PyTorch Model Builder
(function(){
  let stack = [];
  // shape tracker: [C, H, W]
  let inputShape = [3, 32, 32];

  function step(layer, shape){
    let [C,H,W] = shape;
    let code = '';
    let outShape = [C,H,W];
    if(layer.startsWith('Conv')){
      const newC = C * 2;
      code = `nn.Conv2d(${C}, ${newC}, 3, padding=1)`;
      outShape = [newC, H, W];
    } else if(layer.startsWith('MaxPool')){
      code = `nn.MaxPool2d(2, 2)`;
      outShape = [C, Math.floor(H/2), Math.floor(W/2)];
    } else if(layer === 'ReLU'){
      code = `nn.ReLU(inplace=True)`;
    } else if(layer === 'BatchNorm'){
      code = `nn.BatchNorm2d(${C})`;
    } else if(layer === 'GAP'){
      code = `nn.AdaptiveAvgPool2d(1)`;
      outShape = [C, 1, 1];
    } else if(layer === 'Linear'){
      const n = C * H * W;
      code = `nn.Linear(${n}, 10)`;
      outShape = [10, 1, 1];
    }
    return { code, outShape };
  }

  function render(){
    const stackEl = document.getElementById('mbStack');
    const codeEl = document.getElementById('mbCode');
    const statsEl = document.getElementById('mbStats');
    if(!stackEl) return;
    stackEl.innerHTML = '';
    let lines = ['class MyNet(nn.Module):', '    def __init__(self):', '        super().__init__()', '        self.net = nn.Sequential('];
    let shape = inputShape.slice();
    let totalParams = 0;
    stack.forEach((layer, i)=>{
      const item = document.createElement('div');
      item.className = 'stack-item';
      const r = step(layer, shape);
      item.innerHTML = `<span>${layer}</span><span style="font-family:JetBrains Mono;font-size:11px;color:#5a4f3a;">${r.outShape.join('×')}</span><button data-i="${i}">✕</button>`;
      item.querySelector('button').onclick = ()=>{ stack.splice(i,1); render(); };
      stackEl.appendChild(item);
      lines.push(`            ${r.code},  # ${r.outShape.join('×')}`);
      shape = r.outShape;
      // rough param count
      if(layer.startsWith('Conv')){
        const inC = inputShape[0];
        // ignore for now, just rough estimate
      }
    });
    lines.push('        )');
    if(stack.length === 0){
      codeEl.textContent = '# 點左邊加 layer';
    } else {
      codeEl.textContent = lines.join('\n');
    }
    if(statsEl){
      statsEl.innerHTML = `Input: <strong>${inputShape.join('×')}</strong> → Output: <strong>${shape.join('×')}</strong>`;
    }
  }

  function init(){
    const palette = document.getElementById('mbPalette');
    if(!palette) return;
    palette.querySelectorAll('.palette-item').forEach(btn=>{
      btn.onclick = ()=>{ stack.push(btn.dataset.layer); render(); };
    });
    const reset = document.getElementById('mbReset');
    if(reset) reset.onclick = ()=>{ stack = []; render(); };
    // seed with a default chain
    stack = ['Conv 3×3, +ch','BatchNorm','ReLU','MaxPool 2×2','Conv 3×3, +ch','ReLU','GAP','Linear'];
    render();
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
