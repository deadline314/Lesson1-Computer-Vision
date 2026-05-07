// Gradient Descent Playground - 1D landscape animator
(function () {
  // Three landscapes: smooth bowl, multi-well (cubic-quartic), saddle-ish
  const FUNCS = {
    bowl: {
      label: 'x² (平滑碗)',
      f: (x) => x * x,
      g: (x) => 2 * x,
      domain: [-6, 6],
      yRange: [-2, 36],
      start: 5,
    },
    multi: {
      label: 'x⁴-10x²+5x (多谷)',
      f: (x) => Math.pow(x, 4) - 10 * x * x + 5 * x,
      g: (x) => 4 * Math.pow(x, 3) - 20 * x + 5,
      domain: [-3.5, 3.5],
      yRange: [-50, 30],
      start: 2.8,
    },
    plateau: {
      label: 'x²+ sin(3x)·3 (鞍/震盪)',
      f: (x) => 0.3 * x * x + Math.sin(3 * x) * 2.5,
      g: (x) => 0.6 * x + 3 * Math.cos(3 * x) * 2.5,
      domain: [-6, 6],
      yRange: [-5, 14],
      start: -5,
    },
  };

  const OPTS = ['SGD', 'Momentum', 'Adam', 'Anneal'];

  let state = {
    fn: 'multi',
    opt: 'SGD',
    lr: 0.05,
    running: false,
    x: 0,
    v: 0,
    m: 0,
    vAdam: 0,
    t: 0,
    T: 8,    // anneal temperature
    history: [],
    timer: null,
  };

  function reset() {
    const f = FUNCS[state.fn];
    state.x = f.start;
    state.v = 0;
    state.m = 0;
    state.vAdam = 0;
    state.t = 0;
    state.T = 8;
    state.history = [{ x: state.x, y: f.f(state.x) }];
  }

  function step() {
    const F = FUNCS[state.fn];
    const g = F.g(state.x);
    state.t += 1;
    if (state.opt === 'SGD') {
      state.x = state.x - state.lr * g;
    } else if (state.opt === 'Momentum') {
      state.v = 0.9 * state.v + state.lr * g;
      state.x = state.x - state.v;
    } else if (state.opt === 'Adam') {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      state.m = b1 * state.m + (1 - b1) * g;
      state.vAdam = b2 * state.vAdam + (1 - b2) * g * g;
      const mh = state.m / (1 - Math.pow(b1, state.t));
      const vh = state.vAdam / (1 - Math.pow(b2, state.t));
      state.x = state.x - state.lr * 8 * mh / (Math.sqrt(vh) + eps);
    } else if (state.opt === 'Anneal') {
      // simulated annealing: random jump, accept if better, else with prob
      const jump = (Math.random() - 0.5) * 1.4;
      const xNew = state.x + jump;
      const yNew = F.f(xNew);
      const yOld = F.f(state.x);
      const dE = yNew - yOld;
      if (dE < 0 || Math.random() < Math.exp(-dE / state.T)) {
        state.x = xNew;
      }
      state.T = Math.max(0.05, state.T * 0.97);
    }
    // clamp
    state.x = Math.max(F.domain[0], Math.min(F.domain[1], state.x));
    state.history.push({ x: state.x, y: F.f(state.x) });
    if (state.history.length > 200) state.history.shift();
  }

  function draw() {
    const cv = document.getElementById('gdCanvas');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    const F = FUNCS[state.fn];
    const [x0, x1] = F.domain;
    const [y0, y1] = F.yRange;
    const px = (x) => 40 + ((x - x0) / (x1 - x0)) * (W - 60);
    const py = (y) => H - 30 - ((y - y0) / (y1 - y0)) * (H - 60);

    // axes
    ctx.strokeStyle = '#a89878';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, py(0)); ctx.lineTo(W - 20, py(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // curve
    ctx.strokeStyle = '#3d2f1f';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = x0 + (x1 - x0) * i / 200;
      const y = F.f(x);
      if (i === 0) ctx.moveTo(px(x), py(y));
      else ctx.lineTo(px(x), py(y));
    }
    ctx.stroke();

    // history trace
    if (state.history.length > 1) {
      ctx.strokeStyle = 'rgba(200,85,61,0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      state.history.forEach((p, i) => {
        if (i === 0) ctx.moveTo(px(p.x), py(p.y));
        else ctx.lineTo(px(p.x), py(p.y));
      });
      ctx.stroke();
      // dots
      state.history.forEach((p, i) => {
        const a = i / state.history.length;
        ctx.fillStyle = `rgba(200,85,61,${0.3 + a * 0.7})`;
        ctx.beginPath();
        ctx.arc(px(p.x), py(p.y), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // current ball
    ctx.fillStyle = '#c8553d';
    ctx.strokeStyle = '#3d2f1f';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(px(state.x), py(F.f(state.x)), 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // labels
    ctx.fillStyle = '#3d2f1f';
    ctx.font = '600 16px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`step ${state.t}  |  x = ${state.x.toFixed(3)}  |  f(x) = ${F.f(state.x).toFixed(3)}`, 50, 24);
    if (state.opt === 'Anneal') {
      ctx.fillStyle = '#c8553d';
      ctx.fillText(`T = ${state.T.toFixed(2)}`, W - 130, 24);
    }
  }

  function tick() {
    if (!state.running) return;
    step();
    draw();
    if (state.t > 400) state.running = false;
  }

  function init() {
    const wrap = document.getElementById('gdDemoWrap');
    if (!wrap) return;
    if (wrap.dataset.inited) return;
    wrap.dataset.inited = '1';

    wrap.innerHTML = `
      <canvas id="gdCanvas" width="640" height="320"
        style="background:#fff8e7;border:2px solid #3d2f1f;display:block;margin:0 auto 12px;max-width:100%;"></canvas>
      <div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;align-items:center;font-size:18px;">
        <span><b>地形</b></span>
        <select id="gdFn" style="font:inherit;padding:4px 8px;border:2px solid #3d2f1f;background:#fff8e7;">
          ${Object.entries(FUNCS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('')}
        </select>
        <span style="margin-left:18px;"><b>優化器</b></span>
        <div id="gdOpts" style="display:flex;gap:6px;"></div>
        <span style="margin-left:18px;"><b>lr</b></span>
        <input id="gdLr" type="range" min="0.001" max="0.5" step="0.001" value="0.05" style="width:120px;">
        <span id="gdLrVal" style="font-family:JetBrains Mono,monospace;min-width:60px;">0.050</span>
      </div>
      <div style="display:flex;justify-content:center;gap:10px;margin-top:10px;">
        <button id="gdPlay" class="gd-btn">▶ 跑</button>
        <button id="gdStep" class="gd-btn">+1 步</button>
        <button id="gdReset" class="gd-btn">↺ 重置</button>
      </div>
      <style>
        .gd-btn{font:inherit;padding:6px 14px;border:2px solid #3d2f1f;background:#e8c547;cursor:pointer;font-weight:700;}
        .gd-btn:hover{background:#e89c50;}
        .gd-opt{font:inherit;padding:4px 10px;border:2px solid #3d2f1f;background:#fff8e7;cursor:pointer;}
        .gd-opt.active{background:#6a9b5e;color:#fff8e7;}
      </style>
    `;

    const optsBox = document.getElementById('gdOpts');
    OPTS.forEach((o) => {
      const b = document.createElement('button');
      b.className = 'gd-opt' + (o === state.opt ? ' active' : '');
      b.textContent = o;
      b.onclick = () => {
        state.opt = o;
        optsBox.querySelectorAll('.gd-opt').forEach((x) => x.classList.toggle('active', x.textContent === o));
        reset();
        draw();
      };
      optsBox.appendChild(b);
    });

    document.getElementById('gdFn').onchange = (e) => {
      state.fn = e.target.value;
      reset();
      draw();
    };
    const lr = document.getElementById('gdLr');
    const lrVal = document.getElementById('gdLrVal');
    lr.oninput = () => {
      state.lr = parseFloat(lr.value);
      lrVal.textContent = state.lr.toFixed(3);
    };
    document.getElementById('gdPlay').onclick = (e) => {
      state.running = !state.running;
      e.target.textContent = state.running ? '❚❚ 暫停' : '▶ 跑';
    };
    document.getElementById('gdStep').onclick = () => {
      state.running = false;
      step();
      draw();
    };
    document.getElementById('gdReset').onclick = () => {
      state.running = false;
      reset();
      draw();
      document.getElementById('gdPlay').textContent = '▶ 跑';
    };

    reset();
    draw();
    setInterval(tick, 80);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Re-init on slide change
  window.addEventListener('message', (e) => {
    if (e.data && e.data.slideIndexChanged !== undefined) {
      setTimeout(init, 50);
    }
  });
})();
