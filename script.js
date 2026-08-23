document.documentElement.classList.add('js');

const header = document.querySelector('[data-header]');
const menuButton = document.querySelector('[data-menu-button]');
const mobileNav = document.querySelector('[data-mobile-nav]');

const closeMenu = () => {
  if (!menuButton || !mobileNav) return;
  menuButton.setAttribute('aria-expanded', 'false');
  mobileNav.hidden = true;
  document.body.classList.remove('menu-open');
};

menuButton?.addEventListener('click', () => {
  const opening = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(opening));
  mobileNav.hidden = !opening;
  document.body.classList.toggle('menu-open', opening);
});

mobileNav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));
window.addEventListener('resize', () => { if (window.innerWidth > 1100) closeMenu(); });
window.addEventListener('scroll', () => header?.classList.toggle('scrolled', window.scrollY > 12), { passive: true });

const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add('visible');
    observer.unobserve(entry.target);
  });
}, { threshold: 0.1, rootMargin: '0px 0px -30px' });

document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const slideViewer = document.querySelector('#slide-viewer');
document.querySelectorAll('[data-slide-page]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-slide-page]').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    const page = button.getAttribute('data-slide-page');
    if (slideViewer && page) slideViewer.src = `downloads/course-slides.pdf#page=${page}&view=FitH`;
  });
});

const copyButton = document.querySelector('[data-copy-code]');
const setupCode = document.querySelector('[data-setup-code]');
copyButton?.addEventListener('click', async () => {
  if (!setupCode) return;
  const text = setupCode.textContent.replace(/^\$\s?/gm, '').trim();
  try {
    await navigator.clipboard.writeText(text);
    const previous = copyButton.textContent;
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = previous; }, 1600);
  } catch {
    copyButton.textContent = 'Select text';
  }
});

const referenceList = document.querySelector('[data-reference-list]');
const referenceSearch = document.querySelector('[data-reference-search]');
const referenceCount = document.querySelector('[data-reference-count]');
const referenceUpdated = document.querySelector('[data-reference-updated]');
const referenceButtons = [...document.querySelectorAll('[data-reference-filter]')];
let referenceEntries = [];
let referenceFilter = 'all';

const renderReferences = () => {
  if (!referenceList) return;
  const query = (referenceSearch?.value || '').trim().toLowerCase();
  const visible = referenceEntries.filter((entry) => {
    const categoryMatch = referenceFilter === 'all' || entry.category === referenceFilter;
    const haystack = `${entry.title} ${entry.authors} ${entry.year} ${entry.venue} ${entry.category}`.toLowerCase();
    return categoryMatch && haystack.includes(query);
  });

  referenceList.replaceChildren();
  if (!visible.length) {
    const empty = document.createElement('p');
    empty.className = 'reference-empty';
    empty.textContent = 'No references match this search yet.';
    referenceList.append(empty);
  }

  visible.forEach((entry) => {
    const article = document.createElement('article');
    article.className = 'reference-item';

    const year = document.createElement('span');
    year.className = 'ref-year';
    year.textContent = entry.year;

    const content = document.createElement('div');
    const title = document.createElement('h3');
    title.textContent = entry.title;
    const citation = document.createElement('p');
    citation.textContent = `${entry.authors} · ${entry.venue}`;
    content.append(title, citation);

    const category = document.createElement('span');
    category.className = 'ref-category';
    category.textContent = entry.category;

    const link = document.createElement('a');
    link.href = entry.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.setAttribute('aria-label', `Open ${entry.title}`);
    link.textContent = '↗';

    article.append(year, content, category, link);
    referenceList.append(article);
  });

  if (referenceCount) referenceCount.textContent = `${visible.length} of ${referenceEntries.length} linked papers`;
};

referenceSearch?.addEventListener('input', renderReferences);
referenceButtons.forEach((button) => {
  button.addEventListener('click', () => {
    referenceButtons.forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    referenceFilter = button.dataset.referenceFilter || 'all';
    renderReferences();
  });
});

fetch('data/references.json')
  .then((response) => {
    if (!response.ok) throw new Error('Reference data unavailable');
    return response.json();
  })
  .then((data) => {
    referenceEntries = [...data.entries].sort((a, b) => Number(b.year) - Number(a.year) || a.title.localeCompare(b.title));
    if (referenceUpdated) referenceUpdated.textContent = `Updated ${data.lastUpdated}`;
    renderReferences();
  })
  .catch(() => {
    if (referenceList) referenceList.innerHTML = '<p class="reference-empty">The live list could not be loaded. Open the references PDF or BibTeX file below.</p>';
    if (referenceCount) referenceCount.textContent = 'Reference library';
  });
