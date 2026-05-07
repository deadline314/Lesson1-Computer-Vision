window.init_fc = function() {
  let fcSize = 224;
  let fcOut = 1024;

  function fmt(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
  }

  function updateFCComparison() {
    const Cin = 3;
    const Cout = fcOut;

    const fcParams = fcSize * fcSize * Cin * Cout;
    const conv3Params = 3 * 3 * Cin * Cout;
    const conv1Params = 1 * 1 * Cin * Cout;

    const sizeValEl = document.getElementById('fcSizeVal');
    if (sizeValEl) sizeValEl.textContent = `${fcSize} × ${fcSize}`;
    const outValEl = document.getElementById('fcOutVal');
    if (outValEl) outValEl.textContent = fcOut;

    const fcFcCountEl = document.getElementById('fcFcCount');
    if (fcFcCountEl) fcFcCountEl.textContent = fmt(fcParams);
    const fcConvCountEl = document.getElementById('fcConvCount');
    if (fcConvCountEl) fcConvCountEl.textContent = fmt(conv3Params);
    const fcConv1CountEl = document.getElementById('fcConv1Count');
    if (fcConv1CountEl) fcConv1CountEl.textContent = fmt(conv1Params);

    const maxLog = Math.log10(fcParams + 1);
    const minLog = 0;
    function barWidth(n) {
      const l = Math.log10(n + 1);
      return Math.max(2, ((l - minLog) / (maxLog - minLog)) * 100);
    }

    const fcFcBar = document.getElementById('fcFcBar');
    if (fcFcBar) { fcFcBar.style.width = barWidth(fcParams) + '%'; fcFcBar.textContent = fmt(fcParams); }
    const fcConvBar = document.getElementById('fcConvBar');
    if (fcConvBar) { fcConvBar.style.width = barWidth(conv3Params) + '%'; fcConvBar.textContent = fmt(conv3Params); }
    const fcConv1Bar = document.getElementById('fcConv1Bar');
    if (fcConv1Bar) { fcConv1Bar.style.width = barWidth(conv1Params) + '%'; fcConv1Bar.textContent = fmt(conv1Params); }

    const ratio = (fcParams / conv3Params).toFixed(0);
    const fcRatioEl = document.getElementById('fcRatio');
    if (fcRatioEl) fcRatioEl.textContent = fmt(parseInt(ratio));
  }

  const fcSizeEl = document.getElementById('fcSize');
  if (fcSizeEl) {
    fcSizeEl.addEventListener('input', (e) => {
      fcSize = parseInt(e.target.value);
      updateFCComparison();
    });
  }
  const fcOutEl = document.getElementById('fcOut');
  if (fcOutEl) {
    fcOutEl.addEventListener('input', (e) => {
      fcOut = parseInt(e.target.value);
      updateFCComparison();
    });
  }
  updateFCComparison();
};
window.init_fc();
