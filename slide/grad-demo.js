// Gradient Vanishing demo - plain CNN vs ResNet
(function(){
  function color(intensity){
    // intensity 0..1 -> from dark red (0) to bright green (1)
    const r = Math.floor(80 + (1-intensity)*140);
    const g = Math.floor(60 + intensity*180);
    const b = 60;
    return `rgb(${r},${g},${b})`;
  }

  function render(depth){
    const plain = document.getElementById('plainBar');
    const resnet = document.getElementById('resnetBar');
    if(!plain || !resnet) return;
    plain.innerHTML = '';
    resnet.innerHTML = '';
    plain.style.cssText = 'display:flex;height:80px;border:2px solid #29261b;border-radius:6px;overflow:hidden;background:#0d0d0d;';
    resnet.style.cssText = 'display:flex;height:80px;border:2px solid #29261b;border-radius:6px;overflow:hidden;background:#0d0d0d;';
    for(let i=0;i<depth;i++){
      // depth-i = distance from output (gradient flows backward)
      const d = depth - i;
      const plainGrad = Math.pow(0.7, d);            // multiplicative decay
      const resnetGrad = Math.max(plainGrad, 0.4 + 0.6*Math.exp(-d/40));
      const p1 = document.createElement('div');
      p1.style.cssText = `flex:1;background:${color(plainGrad)};`;
      plain.appendChild(p1);
      const p2 = document.createElement('div');
      p2.style.cssText = `flex:1;background:${color(resnetGrad)};`;
      resnet.appendChild(p2);
    }
  }
  function init(){
    const slider = document.getElementById('depthSlider');
    const val = document.getElementById('depthVal');
    if(!slider || !val) return;
    slider.oninput = ()=>{ val.textContent = slider.value; render(parseInt(slider.value,10)); };
    render(parseInt(slider.value,10));
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }

  // Conv demo small canvas (slide 9)
  function convDemoInit(){
    const wrap = document.getElementById('convAnimDemo');
    if(!wrap) return;
    wrap.innerHTML = '<canvas id="cdCanvas" width="320" height="320" style="background:#fff;border:2px solid #29261b;image-rendering:pixelated;max-width:100%;"></canvas>';
    const canvas = document.getElementById('cdCanvas');
    const ctx = canvas.getContext('2d');
    const N = 6;
    const cell = canvas.width / N;
    let pos = 0;
    function draw(){
      ctx.clearRect(0,0,canvas.width,canvas.height);
      // background pattern (numbers)
      ctx.fillStyle = '#f6efe1';
      ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.strokeStyle = '#29261b';
      ctx.lineWidth = 1.5;
      ctx.font = '700 28px JetBrains Mono, monospace';
      ctx.fillStyle = '#29261b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for(let y=0;y<N;y++){
        for(let x=0;x<N;x++){
          ctx.strokeRect(x*cell, y*cell, cell, cell);
          const v = ((x*7 + y*13) % 9) + 1;
          ctx.fillStyle = '#29261b';
          ctx.fillText(v, x*cell + cell/2, y*cell + cell/2);
        }
      }
      // sliding kernel window
      const sx = pos % (N-2);
      const sy = Math.floor(pos / (N-2)) % (N-2);
      ctx.strokeStyle = '#c8553d';
      ctx.lineWidth = 4;
      ctx.fillStyle = 'rgba(232,197,71,0.35)';
      ctx.fillRect(sx*cell, sy*cell, cell*3, cell*3);
      ctx.strokeRect(sx*cell, sy*cell, cell*3, cell*3);
      pos = (pos + 1) % ((N-2)*(N-2));
    }
    draw();
    setInterval(draw, 600);
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', convDemoInit);
  } else { convDemoInit(); }
})();
