(function () {
  var content = document.querySelector('.post-content');
  var toc = document.getElementById('toc');
  if (!content || !toc) return;

  var heads = content.querySelectorAll('h2, h3');
  if (!heads.length) {
    var box = document.querySelector('.post-toc');
    if (box) box.style.display = 'none';
    return;
  }

  var ul = document.createElement('ul');
  var items = [];
  heads.forEach(function (h) {
    if (!h.id) {
      h.id = h.textContent.trim().toLowerCase().replace(/[^\w]+/g, '-').replace(/^-+|-+$/g, '');
    }
    var li = document.createElement('li');
    li.className = 'toc-' + h.tagName.toLowerCase();
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent.replace(/^\d+\.\s*/, '');
    li.appendChild(a);
    ul.appendChild(li);
    items.push({ h: h, a: a });
  });
  toc.appendChild(ul);

  var ticking = false;
  function spy() {
    ticking = false;
    var y = window.scrollY + 100;
    var cur = items[0];
    for (var i = 0; i < items.length; i++) {
      if (items[i].h.getBoundingClientRect().top + window.scrollY <= y) cur = items[i];
    }
    items.forEach(function (it) { it.a.classList.toggle('active', it === cur); });
  }
  window.addEventListener('scroll', function () {
    if (!ticking) { window.requestAnimationFrame(spy); ticking = true; }
  }, { passive: true });
  spy();
})();

/* ---- light/dark theme toggle ---- */
(function () {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  function current() {
    var attr = document.documentElement.getAttribute('data-theme');
    if (attr) return attr;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  btn.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
})();
