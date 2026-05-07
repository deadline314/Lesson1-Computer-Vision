// Global helpers for interactive exercises
window.selectMatch = function(el, side) {
  const parent = el.closest('.match-columns');
  const col = side === 'left' ? parent.children[0] : parent.children[1];
  col.querySelectorAll('.match-item').forEach(i => i.classList.remove('selected'));
  el.classList.add('selected');
};

window.checkMatch = function() {
  const exercise = document.getElementById('introMatch');
  if (!exercise) return;
  const cols = exercise.querySelector('.match-columns');
  const left = cols.children[0].querySelectorAll('.match-item.selected');
  const right = cols.children[1].querySelectorAll('.match-item.selected');
  if (left.length === 0 || right.length === 0) {
    document.getElementById('matchResult').textContent = '請先各選一個!';
    return;
  }
  const l = left[0], r = right[0];
  if (l.dataset.match === r.dataset.match) {
    l.classList.add('correct'); r.classList.add('correct');
    l.classList.remove('selected'); r.classList.remove('selected');
    document.getElementById('matchResult').innerHTML = '<span style="color:var(--green)">✓ 配對正確!</span>';
  } else {
    l.classList.add('wrong'); r.classList.add('wrong');
    setTimeout(() => { l.classList.remove('wrong','selected'); r.classList.remove('wrong','selected'); }, 1000);
    document.getElementById('matchResult').innerHTML = '<span style="color:var(--red)">✗ 再試試看!</span>';
  }
};

window.revealNextStep = function(id) {
  const container = document.getElementById(id);
  if (!container) return;
  const items = container.querySelectorAll('.step-reveal-item:not(.revealed)');
  if (items.length > 0) items[0].classList.add('revealed');
};

window.fillBlank = function(el) {
  const answer = prompt('填入正確的程式碼:');
  if (!answer) return;
  if (answer.trim() === el.dataset.answer) {
    el.textContent = answer.trim();
    el.classList.add('filled');
    el.classList.remove('wrong-fill');
  } else {
    el.textContent = answer.trim();
    el.classList.add('wrong-fill');
    el.classList.remove('filled');
    setTimeout(() => { el.textContent = '???'; el.classList.remove('wrong-fill'); }, 1500);
  }
};
