// YOLO Anchor Visualizer
(function(){
  let grid = 7;
  let nAnchors = 3;
  let anchorMode = true; // anchor-based vs free
  let gtBoxes = [
    {cx:0.3, cy:0.3, w:0.2, h:0.3},
    {cx:0.65, cy:0.6, w:0.3, h:0.25}
  ];
  const ANCHOR_SHAPES = [
    {w:0.1,h:0.1},{w:0.2,h:0.15},{w:0.15,h:0.3},
    {w:0.4,h:0.25},{w:0.25,h:0.4},{w:0.5,h:0.5}
  ];

  function draw(){
    const canvas = document.getElementById('yoloDemoCanvas');
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    // background
    ctx.fillStyle = '#c8b89c';
    ctx.fillRect(0,0,W,H);
    // a faux scene
    ctx.fillStyle = 'rgba(105,140,90,0.5)';
    ctx.fillRect(0,H*0.55,W,H*0.45);
    ctx.fillStyle = 'rgba(180,170,150,0.5)';
    ctx.beginPath();
    ctx.arc(W*0.78, H*0.25, 35, 0, Math.PI*2);
    ctx.fill();

    // grid
    ctx.strokeStyle = 'rgba(41,38,27,0.45)';
    ctx.lineWidth = 1;
    const cellW = W/grid, cellH = H/grid;
    for(let i=1;i<grid;i++){
      ctx.beginPath(); ctx.moveTo(i*cellW,0); ctx.lineTo(i*cellW,H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,i*cellH); ctx.lineTo(W,i*cellH); ctx.stroke();
    }

    // responsible cells (yellow highlight)
    gtBoxes.forEach(b=>{
      const cx = Math.floor(b.cx * grid);
      const cy = Math.floor(b.cy * grid);
      ctx.fillStyle = 'rgba(232,197,71,0.55)';
      ctx.fillRect(cx*cellW, cy*cellH, cellW, cellH);
    });

    // GT boxes (orange)
    gtBoxes.forEach(b=>{
      const x = (b.cx - b.w/2)*W;
      const y = (b.cy - b.h/2)*H;
      ctx.strokeStyle = '#d97757';
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, b.w*W, b.h*H);
      // center dot
      ctx.fillStyle = '#d97757';
      ctx.beginPath();
      ctx.arc(b.cx*W, b.cy*H, 4, 0, Math.PI*2);
      ctx.fill();
    });

    // anchors (blue dashed)
    if(anchorMode && nAnchors > 0){
      gtBoxes.forEach(b=>{
        const cx = Math.floor(b.cx * grid);
        const cy = Math.floor(b.cy * grid);
        const cellCenterX = (cx + 0.5) * cellW;
        const cellCenterY = (cy + 0.5) * cellH;
        ctx.strokeStyle = '#4a7fb8';
        ctx.lineWidth = 2;
        ctx.setLineDash([5,4]);
        for(let i=0;i<nAnchors && i<ANCHOR_SHAPES.length;i++){
          const a = ANCHOR_SHAPES[i];
          ctx.strokeRect(cellCenterX - a.w*W/2, cellCenterY - a.h*H/2, a.w*W, a.h*H);
        }
        ctx.setLineDash([]);
      });
    } else {
      // anchor-free: just predict from cell center
      gtBoxes.forEach(b=>{
        const cx = Math.floor(b.cx * grid);
        const cy = Math.floor(b.cy * grid);
        const cellCenterX = (cx + 0.5) * cellW;
        const cellCenterY = (cy + 0.5) * cellH;
        ctx.fillStyle = '#4a7fb8';
        ctx.beginPath();
        ctx.arc(cellCenterX, cellCenterY, 6, 0, Math.PI*2);
        ctx.fill();
      });
    }
  }

  let dragging = null;
  function init(){
    const canvas = document.getElementById('yoloDemoCanvas');
    if(!canvas) return;
    const gridSlider = document.getElementById('ydGrid');
    const gridVal = document.getElementById('ydGridVal');
    const aSlider = document.getElementById('ydAnchors');
    const aVal = document.getElementById('ydAnchorsVal');
    const modeBtn = document.getElementById('ydMode');
    const resetBtn = document.getElementById('ydReset');
    if(gridSlider){
      gridSlider.oninput = ()=>{ grid = parseInt(gridSlider.value,10); gridVal.textContent = grid; draw(); };
    }
    if(aSlider){
      aSlider.oninput = ()=>{ nAnchors = parseInt(aSlider.value,10); aVal.textContent = nAnchors; draw(); };
    }
    if(modeBtn){
      modeBtn.onclick = ()=>{
        anchorMode = !anchorMode;
        modeBtn.textContent = anchorMode ? 'Anchor-Based' : 'Anchor-Free';
        modeBtn.className = anchorMode ? 'chip ink' : 'chip green';
        draw();
      };
    }
    if(resetBtn){
      resetBtn.onclick = ()=>{
        gtBoxes = [
          {cx:0.3, cy:0.3, w:0.2, h:0.3},
          {cx:0.65, cy:0.6, w:0.3, h:0.25}
        ];
        draw();
      };
    }
    canvas.addEventListener('mousedown', e=>{
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      dragging = {sx:x, sy:y, ex:x, ey:y};
    });
    canvas.addEventListener('mousemove', e=>{
      if(!dragging) return;
      const rect = canvas.getBoundingClientRect();
      dragging.ex = (e.clientX - rect.left) / rect.width;
      dragging.ey = (e.clientY - rect.top) / rect.height;
      draw();
      // preview
      const ctx = canvas.getContext('2d');
      ctx.strokeStyle = '#d97757';
      ctx.lineWidth = 3;
      ctx.setLineDash([4,4]);
      ctx.strokeRect(dragging.sx*canvas.width, dragging.sy*canvas.height,
        (dragging.ex-dragging.sx)*canvas.width, (dragging.ey-dragging.sy)*canvas.height);
      ctx.setLineDash([]);
    });
    canvas.addEventListener('mouseup', e=>{
      if(!dragging) return;
      const w = Math.abs(dragging.ex - dragging.sx);
      const h = Math.abs(dragging.ey - dragging.sy);
      if(w > 0.05 && h > 0.05){
        gtBoxes.push({
          cx: (dragging.sx + dragging.ex)/2,
          cy: (dragging.sy + dragging.ey)/2,
          w, h
        });
      }
      dragging = null;
      draw();
    });
    draw();
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
