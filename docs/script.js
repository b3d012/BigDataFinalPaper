const sections = [...document.querySelectorAll('[data-section]')];
const navLinks = [...document.querySelectorAll('.nav a')];
const revealNodes = [...document.querySelectorAll('.reveal')];
const modal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');
const modalCaption = document.getElementById('modal-caption');
const closeModalButtons = [...document.querySelectorAll('[data-close-modal]')];
const shotCards = [...document.querySelectorAll('.shot-card')];

function setActiveNav(id) {
  navLinks.forEach((link) => {
    const active = link.getAttribute('href') === `#${id}`;
    link.setAttribute('aria-current', active ? 'true' : 'false');
  });
}

if ('IntersectionObserver' in window) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (visible) {
        setActiveNav(visible.target.id);
      }
    },
    {
      rootMargin: '-35% 0px -45% 0px',
      threshold: [0.15, 0.3, 0.5],
    }
  );

  sections.forEach((section) => sectionObserver.observe(section));

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
    }
  );

  revealNodes.forEach((node) => revealObserver.observe(node));
} else {
  revealNodes.forEach((node) => node.classList.add('is-visible'));
  setActiveNav('home');
}

function openModal(src, alt, caption) {
  if (!modal || !modalImage || !modalCaption) return;
  modalImage.src = src;
  modalImage.alt = alt || '';
  modalCaption.textContent = caption || alt || '';
  modal.classList.add('is-open');
  modal.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  if (!modal || !modalImage || !modalCaption) return;
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  modalImage.src = '';
  modalImage.alt = '';
  modalCaption.textContent = '';
  document.body.style.overflow = '';
}

shotCards.forEach((card) => {
  const button = card.querySelector('.shot-button');
  const img = card.querySelector('img');
  const src = card.getAttribute('data-modal-src') || img?.getAttribute('src') || '';
  const alt = card.getAttribute('data-modal-alt') || img?.alt || '';
  const caption = card.querySelector('figcaption')?.innerText?.trim() || alt;

  if (button) {
    button.addEventListener('click', () => openModal(src, alt, caption));
  }
});

closeModalButtons.forEach((button) => {
  button.addEventListener('click', closeModal);
});

if (modal) {
  modal.addEventListener('click', (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeModal();
  }
});

navLinks.forEach((link) => {
  link.addEventListener('click', () => {
    if (window.innerWidth < 920) {
      setTimeout(() => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 0);
    }
  });
});

setActiveNav('home');
