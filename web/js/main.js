(function() {
  'use strict';

  const TAB_NAMES = ['intro','conv','pool','activation','bn','fc','gradient','residual','cnnhistory','optimizer','annealing','augmentation','yolo','yoloevo','builder','forwardpass','pytorch','assignment'];
  const DEMO_TABS = ['conv','pool','activation','bn','fc','gradient','residual','optimizer','annealing','augmentation','yolo','builder'];
  const tabCache = {};
  const loadedScripts = {};
  let currentTab = null;

  // ============ TAB SCROLL UX ============
  (function initTabScroll() {
    const wrapper = document.getElementById('tabsWrapper');
    const inner = document.getElementById('tabsInner');
    const leftBtn = document.getElementById('tabScrollLeft');
    const rightBtn = document.getElementById('tabScrollRight');
    if (!wrapper || !inner) return;

    function updateScrollState() {
      const sl = inner.scrollLeft;
      const maxScroll = inner.scrollWidth - inner.clientWidth;
      wrapper.classList.toggle('scroll-left', sl > 10);
      wrapper.classList.toggle('scroll-right', sl < maxScroll - 10);
      if (leftBtn) leftBtn.classList.toggle('visible', sl > 10);
      if (rightBtn) rightBtn.classList.toggle('visible', sl < maxScroll - 10);
    }

    inner.addEventListener('scroll', updateScrollState);
    window.addEventListener('resize', updateScrollState);
    setTimeout(updateScrollState, 100);

    if (leftBtn) leftBtn.addEventListener('click', function() { inner.scrollBy({ left: -200, behavior: 'smooth' }); });
    if (rightBtn) rightBtn.addEventListener('click', function() { inner.scrollBy({ left: 200, behavior: 'smooth' }); });

    // Mouse drag to scroll
    let isDragging = false;
    let startX = 0;
    let scrollStart = 0;
    let hasDragged = false;

    inner.addEventListener('mousedown', function(e) {
      if (e.target.closest('.tab')) {
        isDragging = true;
        hasDragged = false;
        startX = e.pageX;
        scrollStart = inner.scrollLeft;
        inner.style.cursor = 'grabbing';
        inner.style.userSelect = 'none';
        inner.style.scrollBehavior = 'auto';
        e.preventDefault();
      }
    });

    document.addEventListener('mousemove', function(e) {
      if (!isDragging) return;
      const dx = e.pageX - startX;
      if (Math.abs(dx) > 5) hasDragged = true;
      inner.scrollLeft = scrollStart - dx;
    });

    document.addEventListener('mouseup', function() {
      if (!isDragging) return;
      isDragging = false;
      inner.style.cursor = '';
      inner.style.userSelect = '';
      inner.style.scrollBehavior = '';
    });

    // Prevent tab click if user was dragging
    inner.addEventListener('click', function(e) {
      if (hasDragged) {
        e.stopPropagation();
        e.preventDefault();
        hasDragged = false;
      }
    }, true);
  })();

  // ============ TAB SYSTEM ============
  function switchTab(name) {
    if (!TAB_NAMES.includes(name)) name = 'intro';
    if (name === currentTab) return;
    currentTab = name;

    document.querySelectorAll('.tab').forEach(t => {
      const isActive = t.dataset.tab === name;
      t.classList.toggle('active', isActive);
      if (isActive) t.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    });
    window.location.hash = name === 'intro' ? '' : name;

    const container = document.getElementById('tab-container');

    if (tabCache[name]) {
      container.innerHTML = tabCache[name];
      afterTabLoad(name);
      return;
    }

    container.innerHTML = '<div class="tab-loading">載入中...</div>';
    fetch('tabs/' + name + '.html')
      .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(html => {
        tabCache[name] = html;
        container.innerHTML = html;
        afterTabLoad(name);
      })
      .catch(() => {
        container.innerHTML = '<div class="tab-loading" style="color:var(--red);">載入失敗</div>';
      });
  }

  function afterTabLoad(name) {
    initTooltips();
    initAccordions();
    initQuizzes();
    initChecklist();
    updateProgress();

    if (DEMO_TABS.includes(name) && !loadedScripts[name]) {
      const script = document.createElement('script');
      script.src = 'js/demos/' + name + '.js';
      script.onload = function() {
        loadedScripts[name] = true;
        if (window['init_' + name]) window['init_' + name]();
      };
      document.body.appendChild(script);
    } else if (window['init_' + name]) {
      window['init_' + name]();
    }
  }

  // ============ TOOLTIP SYSTEM ============
  const tip = document.getElementById('termTip');
  let tipTimeout;

  function initTooltips() {
    document.querySelectorAll('.term').forEach(el => {
      if (el.dataset.tipBound) return;
      el.dataset.tipBound = '1';
      el.addEventListener('mouseenter', showTip);
      el.addEventListener('mouseleave', hideTip);
    });
  }

  function showTip(e) {
    const key = e.target.dataset.term;
    const def = window.termDefs[key];
    if (!def) return;
    clearTimeout(tipTimeout);
    let html = '<div class="tip-title">' + def.title + '</div>' + def.body;
    if (def.link) html += '<div class="tip-link"><a href="' + def.link.url + '" target="_blank" rel="noopener">' + def.link.text + ' →</a></div>';
    tip.innerHTML = html;
    tip.classList.add('visible');
    const rect = e.target.getBoundingClientRect();
    let top = rect.bottom + 10;
    let left = rect.left;
    if (top + 200 > window.innerHeight) top = rect.top - tip.offsetHeight - 10;
    if (left + 380 > window.innerWidth) left = window.innerWidth - 390;
    tip.style.top = top + 'px';
    tip.style.left = Math.max(10, left) + 'px';
  }

  function hideTip() {
    tipTimeout = setTimeout(function() { tip.classList.remove('visible'); }, 100);
  }

  tip.addEventListener('mouseenter', function() { clearTimeout(tipTimeout); });
  tip.addEventListener('mouseleave', function() { tip.classList.remove('visible'); });

  // ============ ACCORDION ============
  function initAccordions() {
    document.querySelectorAll('.accordion-header').forEach(h => {
      if (h.dataset.bound) return;
      h.dataset.bound = '1';
      h.addEventListener('click', function() {
        const body = this.nextElementSibling;
        this.classList.toggle('open');
        body.classList.toggle('open');
      });
    });
  }

  // ============ QUIZ ============
  function initQuizzes() {
    document.querySelectorAll('.quiz-box').forEach(box => {
      if (box.dataset.bound) return;
      box.dataset.bound = '1';
      const options = box.querySelectorAll('.quiz-option');
      const feedback = box.querySelector('.quiz-feedback');
      options.forEach(opt => {
        opt.addEventListener('click', function() {
          if (box.dataset.answered) return;
          box.dataset.answered = '1';
          const correct = opt.dataset.correct === 'true';
          opt.classList.add(correct ? 'correct' : 'wrong');
          if (!correct) {
            options.forEach(o => { if (o.dataset.correct === 'true') o.classList.add('correct'); });
          }
          feedback.textContent = correct ? '✓ 正確! ' + opt.dataset.explanation : '✗ ' + opt.dataset.explanation;
          feedback.className = 'quiz-feedback show ' + (correct ? 'correct' : 'wrong');
          saveProgress(box.dataset.quiz, correct);
        });
      });
    });
  }

  // ============ CHECKLIST ============
  function initChecklist() {
    document.querySelectorAll('.checklist li').forEach(li => {
      if (li.dataset.bound) return;
      li.dataset.bound = '1';
      li.addEventListener('click', function() { this.classList.toggle('checked'); });
    });
  }

  // ============ PROGRESS ============
  function saveProgress(id, correct) {
    const progress = JSON.parse(localStorage.getItem('cv-lesson-progress') || '{}');
    progress[id] = correct;
    localStorage.setItem('cv-lesson-progress', JSON.stringify(progress));
    updateProgress();
  }

  function updateProgress() {
    const progress = JSON.parse(localStorage.getItem('cv-lesson-progress') || '{}');
    const total = document.querySelectorAll('.quiz-box').length || TAB_NAMES.length;
    const done = Object.keys(progress).length;
    const pct = Math.min(100, (done / total) * 100);
    document.getElementById('progressBar').style.width = pct + '%';
  }

  // ============ INIT ============
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', function() { switchTab(this.dataset.tab); });
  });

  const hash = window.location.hash.slice(1);
  switchTab(hash || 'intro');

  window.addEventListener('hashchange', function() {
    switchTab(window.location.hash.slice(1) || 'intro');
  });
})();
