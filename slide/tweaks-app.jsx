// Tweaks panel for 視覺模型基礎 deck
const { useEffect } = React;

function TweaksApp() {
  const [t, setTweak] = window.useTweaks(window.TWEAK_DEFAULTS);

  useEffect(() => {
    document.documentElement.dataset.theme = t.theme;
    document.documentElement.dataset.density = t.density;
    document.documentElement.style.setProperty('--anim-speed', t.animationSpeed);
  }, [t.theme, t.density, t.animationSpeed]);

  return (
    <window.TweaksPanel title="Tweaks · 簡報控制">
      <window.TweakSection title="主題">
        <window.TweakRadio
          label="配色"
          value={t.theme}
          onChange={v => setTweak('theme', v)}
          options={[
            { value: 'light', label: '紙' },
            { value: 'dark', label: '夜' },
            { value: 'bright', label: '白' }
          ]}
        />
        <window.TweakRadio
          label="密度"
          value={t.density}
          onChange={v => setTweak('density', v)}
          options={[
            { value: 'compact', label: '緊' },
            { value: 'normal', label: '中' },
            { value: 'loose', label: '鬆' }
          ]}
        />
      </window.TweakSection>
      <window.TweakSection title="動畫">
        <window.TweakSlider
          label="動畫速度"
          value={t.animationSpeed}
          min={0.3} max={2} step={0.1}
          onChange={v => setTweak('animationSpeed', v)}
        />
      </window.TweakSection>
      <window.TweakSection title="顯示">
        <window.TweakToggle
          label="講稿模式"
          value={t.showSpeakerNotes}
          onChange={v => setTweak('showSpeakerNotes', v)}
        />
      </window.TweakSection>
    </window.TweaksPanel>
  );
}

const root = document.createElement('div');
document.body.appendChild(root);
ReactDOM.createRoot(root).render(<TweaksApp />);
