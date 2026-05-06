// 視覺模型基礎 — 迷你互動 demo 元件
// 用 React 寫成 Web Components 風格,可以塞進任何 slide section

const { useState, useEffect, useRef, useMemo } = React;

// ─────────────────────────────────────────────────────────
// 1. ImageNet Top-5 Error 柱狀圖 (slide 19)
// ─────────────────────────────────────────────────────────
function ImageNetChart() {
  const data = [
    { year: '2010', name: 'NEC', err: 28.2, color: 'var(--ink-faint)' },
    { year: '2011', name: 'XRCE', err: 25.8, color: 'var(--ink-faint)' },
    { year: '2012', name: 'AlexNet', err: 16.4, color: 'var(--orange)' },
    { year: '2013', name: 'ZFNet', err: 11.7, color: 'var(--orange)' },
    { year: '2014', name: 'VGG/GoogLeNet', err: 6.7, color: 'var(--yellow)' },
    { year: '2015', name: 'ResNet ★', err: 3.57, color: 'var(--red)' },
    { year: '2016', name: 'Trimps', err: 2.99, color: 'var(--green)' },
    { year: '2017', name: 'SENet', err: 2.25, color: 'var(--green)' },
    { year: 'Human', name: '人類基準', err: 5.1, color: 'var(--blue)' },
  ];
  const max = 30;
  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '24px 28px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)' }}>
      <div style={{ fontSize: 18, color: 'var(--ink-soft)', marginBottom: 12, fontWeight: 600 }}>ImageNet Top-5 Error Rate (%) · 越低越好</div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, height: 220, borderBottom: '2px solid var(--ink)', borderLeft: '2px solid var(--ink)', paddingLeft: 8 }}>
        {data.map((d, i) => {
          const isHuman = d.year === 'Human';
          return (
            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, position: 'relative' }}>
              <div style={{ fontSize: 14, color: d.color, fontWeight: 700, fontFamily: 'Caveat, cursive', fontSize: 18 }}>{d.err}%</div>
              <div style={{
                width: '70%',
                height: `${d.err / max * 180}px`,
                background: d.color,
                borderRadius: '4px 4px 0 0',
                border: '1.5px solid var(--ink)',
                position: 'relative',
                animation: `barGrow 0.8s cubic-bezier(.2,.8,.2,1) ${i * 0.08}s both`,
                transformOrigin: 'bottom'
              }}>
                {isHuman && <div style={{ position: 'absolute', top: -28, left: '50%', transform: 'translateX(-50%)', whiteSpace: 'nowrap', fontSize: 14, color: 'var(--blue)', fontStyle: 'italic' }}>← 5.1%</div>}
              </div>
              <div style={{ fontSize: 13, color: 'var(--ink-soft)', textAlign: 'center', lineHeight: 1.2, marginTop: 2 }}>{d.year}</div>
              <div style={{ fontSize: 12, color: 'var(--ink-faint)', textAlign: 'center', lineHeight: 1.2 }}>{d.name}</div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 14, fontSize: 16, color: 'var(--red)', fontFamily: 'Caveat, cursive', fontSize: 22 }}>
        ★ 2015 ResNet 首次超越人類基準 (3.57% &lt; 5.1%)
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 2. Conv 輸出尺寸計算器 (slide 12)
// ─────────────────────────────────────────────────────────
function ConvCalculator() {
  const [input, setInput] = useState(32);
  const [kernel, setKernel] = useState(3);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(1);

  const output = Math.floor((input + 2 * padding - kernel) / stride) + 1;
  const params = kernel * kernel;

  const Slider = ({ label, value, set, min, max, step = 1, color }) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 18 }}>
        <span style={{ color: 'var(--ink-soft)' }}>{label}</span>
        <span style={{ color, fontWeight: 700, fontFamily: 'Caveat, cursive', fontSize: 24 }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => set(+e.target.value)} style={{ accentColor: color }} />
    </div>
  );

  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '20px 24px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ fontSize: 18, color: 'var(--ink-soft)', fontWeight: 600, marginBottom: 4 }}>輸入尺寸計算機 ⟳ 拉拉看</div>
        <Slider label="Input H/W" value={input} set={setInput} min={8} max={224} step={4} color="var(--blue)" />
        <Slider label="Kernel" value={kernel} set={setKernel} min={1} max={11} step={2} color="var(--red)" />
        <Slider label="Stride" value={stride} set={setStride} min={1} max={4} color="var(--orange)" />
        <Slider label="Padding" value={padding} set={setPadding} min={0} max={5} color="var(--green)" />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <div style={{ fontSize: 16, color: 'var(--ink-faint)', fontStyle: 'italic' }}>output = ⌊(input + 2p - k) / s⌋ + 1</div>
        <div style={{ fontSize: 18, color: 'var(--ink-soft)', fontFamily: 'monospace' }}>
          ⌊({input} + 2×{padding} − {kernel}) / {stride}⌋ + 1
        </div>
        <div style={{ fontSize: 80, fontFamily: 'Caveat, cursive', color: 'var(--red)', lineHeight: 1, fontWeight: 700 }}>
          {output > 0 ? output : '?'}
        </div>
        <div style={{ fontSize: 16, color: 'var(--ink-soft)' }}>
          {input}×{input} → <span style={{ color: 'var(--red)', fontWeight: 700 }}>{output}×{output}</span>
        </div>
        <div style={{ fontSize: 14, color: 'var(--ink-faint)', marginTop: 4 }}>每個 kernel 參數量 = {params}</div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 3. Activation 函數曲線 (slide 18)
// ─────────────────────────────────────────────────────────
function ActivationViz() {
  const [enabled, setEnabled] = useState({ relu: true, sigmoid: true, tanh: true, leaky: false, gelu: false });
  const fns = {
    relu: { color: 'var(--red)', label: 'ReLU', f: x => Math.max(0, x) },
    sigmoid: { color: 'var(--blue)', label: 'Sigmoid', f: x => 1 / (1 + Math.exp(-x)) },
    tanh: { color: 'var(--green)', label: 'Tanh', f: x => Math.tanh(x) },
    leaky: { color: 'var(--orange)', label: 'LeakyReLU', f: x => x > 0 ? x : 0.1 * x },
    gelu: { color: 'var(--purple)', label: 'GELU', f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))) },
  };
  const W = 460, H = 240, range = 4;
  const sx = x => W / 2 + (x / range) * (W / 2);
  const sy = y => H / 2 - (y / 2.2) * (H / 2);
  const path = (f) => {
    let p = '';
    for (let i = 0; i <= 100; i++) {
      const x = -range + (2 * range * i / 100);
      const y = f(x);
      p += (i === 0 ? 'M' : 'L') + sx(x) + ',' + sy(Math.max(-2.2, Math.min(2.2, y))) + ' ';
    }
    return p;
  };

  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '20px 24px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 24, alignItems: 'center' }}>
      <svg width={W} height={H} style={{ background: 'rgba(255, 248, 231, 0.5)', borderRadius: 8 }}>
        {/* grid */}
        {[-3, -2, -1, 1, 2, 3].map(x => (
          <line key={'vx' + x} x1={sx(x)} y1={0} x2={sx(x)} y2={H} stroke="var(--line-soft)" strokeWidth={0.5} />
        ))}
        {[-2, -1, 1, 2].map(y => (
          <line key={'hy' + y} x1={0} y1={sy(y)} x2={W} y2={sy(y)} stroke="var(--line-soft)" strokeWidth={0.5} />
        ))}
        {/* axes */}
        <line x1={0} y1={H/2} x2={W} y2={H/2} stroke="var(--ink)" strokeWidth={1.5} />
        <line x1={W/2} y1={0} x2={W/2} y2={H} stroke="var(--ink)" strokeWidth={1.5} />
        {/* curves */}
        {Object.entries(fns).map(([k, v]) => enabled[k] && (
          <path key={k} d={path(v.f)} stroke={v.color} strokeWidth={3} fill="none" strokeLinecap="round" />
        ))}
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ fontSize: 18, color: 'var(--ink-soft)', fontWeight: 600, marginBottom: 4 }}>切換看曲線疊加</div>
        {Object.entries(fns).map(([k, v]) => (
          <label key={k} style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: 18 }}>
            <input type="checkbox" checked={enabled[k]} onChange={e => setEnabled({ ...enabled, [k]: e.target.checked })} style={{ accentColor: v.color, width: 18, height: 18 }} />
            <span style={{ display: 'inline-block', width: 24, height: 4, background: v.color, borderRadius: 2 }}></span>
            <span style={{ color: 'var(--ink)', fontWeight: 600 }}>{v.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 4. 電梯 vs 樓梯類比 (slide 25)
// ─────────────────────────────────────────────────────────
function ElevatorAnalogy() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 100), 60);
    return () => clearInterval(id);
  }, []);
  const stairY = Math.sin(tick / 100 * Math.PI * 2) * 40;
  const elevY = Math.sin(tick / 100 * Math.PI * 2) * 5;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginTop: 16 }}>
      {/* Plain CNN = 樓梯 */}
      <div style={{ background: 'rgba(200, 85, 61, 0.08)', border: '2px dashed var(--red)', borderRadius: 12, padding: '24px 28px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)' }}>
        <div style={{ fontFamily: 'Caveat, cursive', fontSize: 36, color: 'var(--red)', fontWeight: 700, marginBottom: 8 }}>Plain CNN</div>
        <div style={{ fontSize: 22, color: 'var(--ink-soft)', marginBottom: 16 }}>= 從 1 樓爬樓梯到 100 樓</div>
        <svg width="100%" height={180} viewBox="0 0 320 180">
          {/* stairs */}
          {[...Array(8)].map((_, i) => (
            <rect key={i} x={20 + i * 35} y={150 - i * 18} width={35} height={18} fill="var(--paper-dark)" stroke="var(--ink)" strokeWidth={1.5} />
          ))}
          {/* climber */}
          <circle cx={20 + 7 * 35 + 17 + stairY * 0.3} cy={150 - 7 * 18 - 12 + Math.abs(stairY) * 0.4} r={10} fill="var(--red)" stroke="var(--ink)" strokeWidth={1.5} />
          <text x={160} y={30} fontFamily="Caveat" fontSize={20} fill="var(--ink-soft)" textAnchor="middle">😰 累死了…</text>
        </svg>
        <div style={{ fontSize: 18, color: 'var(--ink-soft)', marginTop: 8, lineHeight: 1.5 }}>
          每層都要重新學整個 mapping <br/>
          梯度傳到底部已經消失 <br/>
          <span style={{ color: 'var(--red)', fontWeight: 700 }}>退化問題就是這樣來的</span>
        </div>
      </div>
      {/* ResNet = 電梯 */}
      <div style={{ background: 'rgba(106, 155, 94, 0.1)', border: '2px dashed var(--green)', borderRadius: 12, padding: '24px 28px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)' }}>
        <div style={{ fontFamily: 'Caveat, cursive', fontSize: 36, color: 'var(--green)', fontWeight: 700, marginBottom: 8 }}>ResNet</div>
        <div style={{ fontSize: 22, color: 'var(--ink-soft)', marginBottom: 16 }}>= 搭電梯到 100 樓 (中間隨時可下車)</div>
        <svg width="100%" height={180} viewBox="0 0 320 180">
          {/* elevator shaft */}
          <rect x={140} y={20} width={80} height={140} fill="none" stroke="var(--ink)" strokeWidth={2} strokeDasharray="4 3" rx={4} />
          {/* cable */}
          <line x1={180} y1={20} x2={180} y2={20 + (tick / 100) * 80} stroke="var(--ink-faint)" strokeWidth={1} />
          {/* floor markers */}
          {[40, 80, 120].map(y => <line key={y} x1={130} y1={y} x2={140} y2={y} stroke="var(--ink-faint)" strokeWidth={1} />)}
          {/* car */}
          <rect x={148} y={20 + (tick / 100) * 80 + elevY} width={64} height={50} fill="var(--green)" stroke="var(--ink)" strokeWidth={2} rx={4} />
          <circle cx={180} cy={20 + (tick / 100) * 80 + 25 + elevY} r={6} fill="var(--cream)" />
          <text x={250} y={50} fontFamily="Caveat" fontSize={20} fill="var(--ink-soft)">😎 輕鬆</text>
          <text x={250} y={75} fontFamily="Caveat" fontSize={16} fill="var(--ink-faint)">識別 = identity</text>
        </svg>
        <div style={{ fontSize: 18, color: 'var(--ink-soft)', marginTop: 8, lineHeight: 1.5 }}>
          只學「殘差」F(x) = H(x) − x <br/>
          梯度有 highway 直接回傳 <br/>
          <span style={{ color: 'var(--green)', fontWeight: 700 }}>1000 層也訓練得起來</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 5. Backbone-Neck-Head 流動動畫 (slide 41)
// ─────────────────────────────────────────────────────────
function BackboneNeckHead() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 100), 60);
    return () => clearInterval(id);
  }, []);

  const Block = ({ x, y, w, h, color, label, sub, t = 0 }) => {
    const active = (tick + t) % 100 < 33;
    return (
      <g transform={`translate(${x},${y})`}>
        <rect width={w} height={h} fill={active ? color : 'var(--cream)'} stroke="var(--ink)" strokeWidth={2} rx={6} style={{ transition: 'fill 0.4s' }} />
        <text x={w/2} y={h/2 - 4} fontFamily="Caveat" fontSize={22} fill={active ? 'var(--cream)' : 'var(--ink)'} textAnchor="middle" fontWeight="700" style={{ transition: 'fill 0.4s' }}>{label}</text>
        <text x={w/2} y={h/2 + 18} fontSize={13} fill={active ? 'var(--cream)' : 'var(--ink-soft)'} textAnchor="middle" style={{ transition: 'fill 0.4s' }}>{sub}</text>
      </g>
    );
  };

  // particles
  const pPos = (start, end) => {
    const p = (tick % 100) / 100;
    return { x: start.x + (end.x - start.x) * p, y: start.y + (end.y - start.y) * p };
  };

  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '20px 24px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', marginTop: 12 }}>
      <div style={{ fontSize: 18, color: 'var(--ink-soft)', fontWeight: 600, marginBottom: 8 }}>🎞 特徵流動 · YOLOv4 三段式架構</div>
      <svg width="100%" height={200} viewBox="0 0 900 200">
        {/* connections */}
        <line x1={130} y1={100} x2={250} y2={100} stroke="var(--ink-faint)" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={400} y1={100} x2={520} y2={100} stroke="var(--ink-faint)" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={670} y1={100} x2={790} y2={100} stroke="var(--ink-faint)" strokeWidth={2} strokeDasharray="4 3" />

        {/* input image */}
        <g transform="translate(40, 60)">
          <rect width={90} height={80} fill="var(--paper-dark)" stroke="var(--ink)" strokeWidth={2} rx={4} />
          <text x={45} y={42} fontFamily="Caveat" fontSize={20} fill="var(--ink)" textAnchor="middle" fontWeight={700}>輸入圖</text>
          <text x={45} y={62} fontSize={12} fill="var(--ink-soft)" textAnchor="middle">640×640×3</text>
        </g>

        <Block x={250} y={50} w={150} h={100} color="var(--blue)" label="Backbone" sub="特徵抽取 · CSPDarknet" t={0} />
        <Block x={520} y={50} w={150} h={100} color="var(--orange)" label="Neck" sub="尺度融合 · SPP+PANet" t={20} />
        <Block x={790} y={50} w={80} h={100} color="var(--red)" label="Head" sub="預測輸出" t={40} />

        {/* moving particles */}
        {(() => {
          const segs = [
            { from: { x: 130, y: 100 }, to: { x: 250, y: 100 }, t: 0 },
            { from: { x: 400, y: 100 }, to: { x: 520, y: 100 }, t: 25 },
            { from: { x: 670, y: 100 }, to: { x: 790, y: 100 }, t: 50 },
          ];
          return segs.map((s, i) => {
            const p = ((tick + s.t) % 100) / 100;
            const cx = s.from.x + (s.to.x - s.from.x) * p;
            return <circle key={i} cx={cx} cy={100} r={6} fill="var(--yellow)" stroke="var(--ink)" strokeWidth={1.5} />;
          });
        })()}
      </svg>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, marginTop: 12 }}>
        <div style={{ borderTop: '3px solid var(--blue)', paddingTop: 8, fontSize: 16 }}>
          <b style={{ color: 'var(--blue)' }}>Backbone</b><br/>把圖片變成 feature map<br/><span style={{ color: 'var(--ink-faint)' }}>例: CSPDarknet53</span>
        </div>
        <div style={{ borderTop: '3px solid var(--orange)', paddingTop: 8, fontSize: 16 }}>
          <b style={{ color: 'var(--orange)' }}>Neck</b><br/>不同層 feature 融合<br/><span style={{ color: 'var(--ink-faint)' }}>例: SPP + PANet</span>
        </div>
        <div style={{ borderTop: '3px solid var(--red)', paddingTop: 8, fontSize: 16 }}>
          <b style={{ color: 'var(--red)' }}>Head</b><br/>輸出 box + 類別 + 信心<br/><span style={{ color: 'var(--ink-faint)' }}>例: YOLO Head ×3 尺度</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 6. YOLO 對照表 (slide 51) — hover 高亮
// ─────────────────────────────────────────────────────────
function YoloCompareTable() {
  const [hoverRow, setHoverRow] = useState(null);
  const [hoverCol, setHoverCol] = useState(null);
  const cols = ['版本', 'Backbone', 'Neck', 'Head', 'Anchor', 'Loss'];
  const rows = [
    ['v1 (2016)', 'GoogLeNet 改', '—', 'Coupled', '無', 'SSE'],
    ['v2 (2017)', 'Darknet-19', '—', 'Coupled', 'k-means', 'SSE'],
    ['v3 (2018)', 'Darknet-53', 'FPN-like', 'Coupled', '9 個', 'BCE'],
    ['v4 (2020)', 'CSPDarknet53', 'SPP+PANet', 'Coupled', '9 個', 'CIoU'],
    ['v5 (2020)', 'CSPDarknet+Focus', 'SPP+PANet', 'Coupled', '自動 anchor', 'CIoU'],
    ['v6 (2022) ★', 'EfficientRep', 'Rep-PAN', 'Decoupled', 'Anchor-free', 'SIoU/CIoU'],
    ['v7 (2022)', 'E-ELAN', 'PANet+', 'Coupled+Aux', '9 個', 'CIoU'],
    ['v8 (2023) ★', 'C2f', 'PANet 改', 'Decoupled', 'Anchor-free', 'CIoU+DFL'],
  ];
  const getBg = (r, c) => {
    if (hoverRow === r && hoverCol === c) return 'rgba(200, 85, 61, 0.25)';
    if (hoverRow === r || hoverCol === c) return 'rgba(232, 197, 71, 0.18)';
    return 'transparent';
  };
  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '16px 20px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', overflow: 'hidden' }}>
      <div style={{ fontSize: 16, color: 'var(--ink-faint)', marginBottom: 8, fontStyle: 'italic' }}>滑鼠移上去看高亮 · ★ 表示重要轉折</div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 17 }}>
        <thead>
          <tr>
            {cols.map((c, ci) => (
              <th key={ci} style={{
                padding: '10px 8px',
                borderBottom: '2px solid var(--ink)',
                textAlign: 'left',
                background: hoverCol === ci ? 'rgba(200, 85, 61, 0.15)' : 'transparent',
                cursor: 'pointer',
                transition: 'background 0.15s'
              }} onMouseEnter={() => setHoverCol(ci)} onMouseLeave={() => setHoverCol(null)}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} onMouseEnter={() => setHoverRow(ri)} onMouseLeave={() => setHoverRow(null)}>
              {row.map((cell, ci) => (
                <td key={ci} style={{
                  padding: '8px',
                  borderBottom: '1px dashed var(--line-soft)',
                  background: getBg(ri, ci),
                  fontWeight: ci === 0 ? 700 : 400,
                  color: ci === 0 && cell.includes('★') ? 'var(--red)' : 'var(--ink)',
                  transition: 'background 0.15s'
                }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 7. 點圖辨識 mini demo (slide 3)
// ─────────────────────────────────────────────────────────
function PixelGuess() {
  const [revealed, setRevealed] = useState([]);
  const [showAnswer, setShowAnswer] = useState(false);

  // 用簡單的灰階 grid 表示一張貓圖
  const grid = useMemo(() => {
    // 16x16 簡化「貓」(用一些圖形啟發式)
    const g = [];
    for (let y = 0; y < 16; y++) {
      const row = [];
      for (let x = 0; x < 16; x++) {
        // 頭部圓形
        const dh = Math.sqrt((x - 8) ** 2 + (y - 8) ** 2);
        let v = 220;
        if (dh < 5.5) v = 180;
        // 耳朵
        if ((x >= 4 && x <= 6 && y >= 4 && y <= 6) || (x >= 10 && x <= 12 && y >= 4 && y <= 6)) v = 160;
        // 眼睛
        if ((x === 6 || x === 7) && y === 8) v = 50;
        if ((x === 10 || x === 11) && y === 8) v = 50;
        // 鼻子嘴
        if (x === 8 && y === 10) v = 100;
        if (y === 11 && x >= 7 && x <= 9) v = 130;
        row.push(v);
      }
      g.push(row);
    }
    return g;
  }, []);

  const toggle = (i) => {
    setRevealed(r => r.includes(i) ? r : [...r, i]);
  };
  const revealAll = () => {
    setRevealed([...Array(256).keys()]);
    setShowAnswer(true);
  };

  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '20px 24px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', display: 'grid', gridTemplateColumns: '320px 1fr', gap: 24 }}>
      <div>
        <div style={{ fontSize: 16, color: 'var(--ink-soft)', marginBottom: 8, fontWeight: 600 }}>點開像素 · 你看得出來嗎?</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: 1, background: 'var(--ink)', padding: 1, borderRadius: 4 }}>
          {grid.flat().map((v, i) => {
            const isRevealed = revealed.includes(i);
            return (
              <div key={i}
                onClick={() => toggle(i)}
                style={{
                  aspectRatio: '1',
                  background: isRevealed ? `rgb(${v},${v},${v})` : 'var(--paper-dark)',
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}
              >
                {!isRevealed && <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, color: 'var(--ink-faint)' }}>?</div>}
              </div>
            );
          })}
        </div>
        <button onClick={revealAll} style={{ marginTop: 12, padding: '8px 16px', background: 'var(--red)', color: 'var(--cream)', border: '2px solid var(--ink)', borderRadius: 8, fontSize: 16, cursor: 'pointer', fontFamily: 'inherit', boxShadow: '3px 3px 0 rgba(0,0,0,0.15)' }}>全部翻開</button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 16 }}>
        <div style={{ fontSize: 24, color: 'var(--ink)', lineHeight: 1.5 }}>
          這是一張 <b style={{ color: 'var(--red)' }}>16×16 = 256 個像素</b> 的圖
        </div>
        <div style={{ fontSize: 20, color: 'var(--ink-soft)', lineHeight: 1.5 }}>
          每個像素只是一個 0–255 的數字 <br/>
          電腦看到的就是 <span style={{ fontFamily: 'monospace', color: 'var(--blue)' }}>[[180, 220, ...], ...]</span>
        </div>
        {showAnswer && (
          <div style={{ fontSize: 32, fontFamily: 'Caveat, cursive', color: 'var(--green)', fontWeight: 700, animation: 'rise 0.5s both' }}>
            🐱 是一隻貓!
          </div>
        )}
        <div style={{ fontSize: 18, color: 'var(--ink-faint)', fontStyle: 'italic' }}>
          → 我們要做的事:用 256 個數字推斷「這是貓還是狗」
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// 8. 反向傳播粒子流動 (slide 24 梯度消失)
// ─────────────────────────────────────────────────────────
function GradientFlow() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 100), 50);
    return () => clearInterval(id);
  }, []);

  const layers = 8;
  return (
    <div style={{ background: 'var(--cream)', border: '2px dashed var(--ink)', borderRadius: 12, padding: '20px 24px', boxShadow: '4px 4px 0 rgba(0,0,0,0.08)', marginTop: 8 }}>
      <div style={{ fontSize: 18, color: 'var(--ink-soft)', fontWeight: 600, marginBottom: 8 }}>🌧 梯度反向傳播 · 越往前梯度越小</div>
      <svg width="100%" height={120} viewBox="0 0 800 120">
        {/* arrows backward */}
        <defs>
          <marker id="arrowback" viewBox="0 0 10 10" refX="3" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 10 0 L 0 5 L 10 10 z" fill="var(--ink)" />
          </marker>
        </defs>
        {[...Array(layers)].map((_, i) => {
          const x = 60 + i * 90;
          const opacity = 1 - i * 0.12;
          return (
            <g key={i}>
              <rect x={x} y={40} width={60} height={40} fill="var(--paper-dark)" stroke="var(--ink)" strokeWidth={2} rx={4} />
              <text x={x + 30} y={64} fontFamily="Caveat" fontSize={18} fill="var(--ink)" textAnchor="middle" fontWeight={700}>L{i+1}</text>
              {i > 0 && (
                <>
                  <line x1={x} y1={60} x2={x - 30} y2={60} stroke="var(--red)" strokeWidth={3} strokeOpacity={opacity} markerEnd="url(#arrowback)" />
                  {/* particle */}
                  {(() => {
                    const p = (tick + i * 10) % 100 / 100;
                    const cx = x - p * 30;
                    return <circle cx={cx} cy={60} r={5 - i * 0.4} fill="var(--red)" opacity={opacity} />;
                  })()}
                </>
              )}
              <text x={x + 30} y={104} fontSize={11} fill="var(--ink-faint)" textAnchor="middle" fontFamily="monospace">
                {(0.25 ** (layers - i)).toExponential(1)}
              </text>
            </g>
          );
        })}
        <text x={400} y={20} fontFamily="Caveat" fontSize={20} fill="var(--ink-soft)" textAnchor="middle">← Loss 從這邊回傳</text>
      </svg>
      <div style={{ fontSize: 16, color: 'var(--ink-faint)', marginTop: 4, textAlign: 'center', fontStyle: 'italic' }}>
        每層 sigmoid 最大梯度 0.25 → 8 層後只剩 1.5e-5 → 前層 weight 學不動
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Mount 元件到指定 mount points
// ─────────────────────────────────────────────────────────
const mounts = {
  'demo-imagenet': ImageNetChart,
  'demo-conv-calc': ConvCalculator,
  'demo-activation': ActivationViz,
  'demo-elevator': ElevatorAnalogy,
  'demo-bnh-flow': BackboneNeckHead,
  'demo-yolo-table': YoloCompareTable,
  'demo-pixel-guess': PixelGuess,
  'demo-gradient-flow': GradientFlow,
};

function mountAll() {
  Object.entries(mounts).forEach(([id, Comp]) => {
    document.querySelectorAll(`[data-demo="${id}"]`).forEach(el => {
      if (!el.dataset.mounted) {
        el.dataset.mounted = '1';
        ReactDOM.createRoot(el).render(<Comp />);
      }
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mountAll);
} else {
  mountAll();
}

// add bar grow keyframe
const sheet = document.createElement('style');
sheet.textContent = `
@keyframes barGrow {
  from { transform: scaleY(0); opacity: 0; }
  to { transform: scaleY(1); opacity: 1; }
}`;
document.head.appendChild(sheet);
