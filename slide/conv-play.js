// Convolution Playground
(function(){
  const PRESETS = {
    'Identity': [[0,0,0],[0,1,0],[0,0,0]],
    'Edge ↕': [[1,1,1],[0,0,0],[-1,-1,-1]],
    'Edge ↔': [[1,0,-1],[1,0,-1],[1,0,-1]],
    'Sobel': [[-1,-2,-1],[0,0,0],[1,2,1]],
    'Sharpen': [[0,-1,0],[-1,5,-1],[0,-1,0]],
    'Blur': [[1,1,1],[1,1,1],[1,1,1]].map(r=>r.map(v=>v/9)),
    'Emboss': [[-2,-1,0],[-1,1,1],[0,1,2]],
  };

  let kernel = PRESETS['Sobel'].map(r=>r.slice());
  const N = 30;

  function makeImage(){
    const img = [];
    for(let y=0;y<N;y++){
      const row=[];
      for(let x=0;x<N;x++){
        // a circle + diagonal stripes
        const cx=15,cy=15;
        const inCircle = (x-cx)**2 + (y-cy)**2 < 64;
        const stripe = ((x+y) % 6) < 3 ? 0.3 : 0.7;
        row.push(inCircle ? 1 - stripe*0.5 : stripe*0.4);
      }
      img.push(row);
    }
    return img;
  }
  const src = makeImage();

  function drawGrid(canvas, data){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, cell = w/N;
    for(let y=0;y<N;y++){
      for(let x=0;x<N;x++){
        const v = Math.max(0, Math.min(1, data[y][x]));
        const g = Math.floor(v*255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(x*cell, y*cell, cell+0.5, cell+0.5);
      }
    }
  }

  function convolve(){
    const out = [];
    for(let y=0;y<N;y++){
      const row=[];
      for(let x=0;x<N;x++){
        let s=0;
        for(let ky=-1;ky<=1;ky++){
          for(let kx=-1;kx<=1;kx++){
            const yy=y+ky, xx=x+kx;
            if(yy>=0&&yy<N&&xx>=0&&xx<N){
              s += src[yy][xx] * kernel[ky+1][kx+1];
            }
          }
        }
        row.push(Math.abs(s));
      }
      out.push(row);
    }
    // normalize
    let mx=0;
    for(const r of out) for(const v of r) if(v>mx) mx=v;
    if(mx>0) for(let y=0;y<N;y++) for(let x=0;x<N;x++) out[y][x]/=mx;
    return out;
  }

  function buildKernelUI(){
    const wrap = document.getElementById('cpKernel');
    if(!wrap) return;
    wrap.innerHTML = '';
    for(let y=0;y<3;y++){
      for(let x=0;x<3;x++){
        const inp = document.createElement('input');
        inp.type='number'; inp.step='0.1'; inp.value = kernel[y][x].toFixed(1);
        inp.style.cssText='width:50px;height:38px;text-align:center;border:2px solid #29261b;background:#f6efe1;font-family:JetBrains Mono,monospace;font-weight:700;font-size:15px;border-radius:4px;';
        inp.oninput = (e)=>{
          const v = parseFloat(e.target.value);
          if(!isNaN(v)){ kernel[y][x] = v; render(); }
        };
        wrap.appendChild(inp);
      }
    }
  }
  function buildPresets(){
    const wrap = document.getElementById('cpPresets');
    if(!wrap) return;
    wrap.innerHTML = '';
    Object.keys(PRESETS).forEach(name=>{
      const b = document.createElement('button');
      b.textContent = name;
      b.className = 'chip yellow';
      b.style.cssText = 'cursor:pointer;border:none;font-family:Kalam,cursive;font-weight:700;';
      b.onclick = ()=>{
        kernel = PRESETS[name].map(r=>r.slice());
        buildKernelUI();
        render();
      };
      wrap.appendChild(b);
    });
  }

  function render(){
    const srcC = document.getElementById('cpSrc');
    const dstC = document.getElementById('cpDst');
    if(!srcC || !dstC) return;
    drawGrid(srcC, src);
    drawGrid(dstC, convolve());
  }

  function init(){
    if(!document.getElementById('cpSrc')) return;
    buildKernelUI();
    buildPresets();
    render();
  }

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
