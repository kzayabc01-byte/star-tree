const STAGES = [
  ['intro-stage-birth', 600],
  ['intro-stage-charge', 1000],
  ['intro-stage-ignite', 900],
  ['intro-stage-blast', 1100],
  ['intro-stage-galaxyReveal', 1000],
  ['intro-stage-done', 300]
];

const REDUCED_MOTION_STAGES = [
  ['intro-stage-galaxyReveal', 300],
  ['intro-stage-done', 100]
];

function clearTimers(timers) {
  timers.forEach((timer) => window.clearTimeout(timer));
  timers.length = 0;
}

function setIntroStage(introEl, stageClass) {
  introEl.classList.remove(...STAGES.map(([stage]) => stage));
  introEl.classList.add(stageClass);
  document.body.classList.toggle(
    'intro-galaxy-visible',
    stageClass === 'intro-stage-blast' ||
      stageClass === 'intro-stage-galaxyReveal' ||
      stageClass === 'intro-stage-done'
  );
}

function getTintClass(tint, baseClass) {
  if (tint < 0.16) return `${baseClass} ${baseClass}-cool`;
  if (tint < 0.32) return `${baseClass} ${baseClass}-warm`;
  return baseClass;
}

function buildStarField(container, particleCount = 88) {
  const fragment = document.createDocumentFragment();

  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('span');
    const angle = (Math.PI * 2 * i) / particleCount + (Math.random() - 0.5) * 0.12;
    const startRadius = 18 + Math.random() * 26;
    const radius = 240 + Math.random() * 420;
    const size = 0.8 + Math.random() * 2.2;
    const delay = Math.random() * 0.2;
    const tint = Math.random();
    const drift = (Math.random() - 0.5) * 86;
    const targetX = Math.cos(angle) * radius - Math.sin(angle) * drift;
    const targetY = Math.sin(angle) * radius + Math.cos(angle) * drift;

    particle.style.setProperty('--from-x', `${Math.cos(angle) * startRadius}px`);
    particle.style.setProperty('--from-y', `${Math.sin(angle) * startRadius}px`);
    particle.style.setProperty('--to-x', `${targetX}px`);
    particle.style.setProperty('--to-y', `${targetY}px`);
    particle.style.setProperty('--size', `${size}px`);
    particle.style.setProperty('--delay', `${delay}s`);
    particle.style.setProperty('--spin', `${(-22 + Math.random() * 44).toFixed(2)}deg`);
    particle.className = getTintClass(tint, 'intro-particle');

    fragment.appendChild(particle);
  }

  container.replaceChildren(fragment);
}

function buildShockwaveField(container, particleCount = 30) {
  const fragment = document.createDocumentFragment();

  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('span');
    const angle = (Math.PI * 2 * i) / particleCount + (Math.random() - 0.5) * 0.14;
    const startRadius = 44 + Math.random() * 26;
    const shellRadius = 150 + Math.random() * 150;
    const tangentDrift = (Math.random() - 0.5) * 110;
    const toX = Math.cos(angle) * shellRadius - Math.sin(angle) * tangentDrift;
    const toY = Math.sin(angle) * shellRadius + Math.cos(angle) * tangentDrift;
    const width = 52 + Math.random() * 110;
    const height = width * (0.28 + Math.random() * 0.2);
    const delay = Math.random() * 0.16;
    const tint = Math.random();

    particle.style.setProperty('--from-x', `${Math.cos(angle) * startRadius}px`);
    particle.style.setProperty('--from-y', `${Math.sin(angle) * startRadius}px`);
    particle.style.setProperty('--to-x', `${toX}px`);
    particle.style.setProperty('--to-y', `${toY}px`);
    particle.style.setProperty('--width', `${width}px`);
    particle.style.setProperty('--height', `${height}px`);
    particle.style.setProperty('--delay', `${delay}s`);
    particle.style.setProperty('--angle', `${(angle * 180) / Math.PI}deg`);
    particle.style.setProperty('--twist', `${(-22 + Math.random() * 44).toFixed(2)}deg`);
    particle.className = getTintClass(tint, 'intro-wave-particle');

    fragment.appendChild(particle);
  }

  container.replaceChildren(fragment);
}

function buildDustClouds(container, cloudCount = 26) {
  const fragment = document.createDocumentFragment();

  for (let i = 0; i < cloudCount; i++) {
    const cloud = document.createElement('span');
    const angle = Math.random() * Math.PI * 2;
    const startRadius = 22 + Math.random() * 28;
    const radius = 150 + Math.random() * 280;
    const width = 120 + Math.random() * 220;
    const height = width * (0.42 + Math.random() * 0.3);
    const delay = Math.random() * 0.18;
    const tint = Math.random();

    cloud.style.setProperty('--from-x', `${Math.cos(angle) * startRadius}px`);
    cloud.style.setProperty('--from-y', `${Math.sin(angle) * startRadius}px`);
    cloud.style.setProperty('--to-x', `${Math.cos(angle) * radius}px`);
    cloud.style.setProperty('--to-y', `${Math.sin(angle) * radius}px`);
    cloud.style.setProperty('--width', `${width}px`);
    cloud.style.setProperty('--height', `${height}px`);
    cloud.style.setProperty('--delay', `${delay}s`);
    cloud.style.setProperty('--rotation', `${(angle * 180) / Math.PI}deg`);
    cloud.className = getTintClass(tint, 'intro-dust-cloud');

    fragment.appendChild(cloud);
  }

  container.replaceChildren(fragment);
}

export function createIntroSequence({ onComplete } = {}) {
  const introEl = document.getElementById('intro');
  const skipButton = document.getElementById('intro-skip');
  const particleContainer = introEl?.querySelector('.intro-particles');
  const waveParticleContainer = introEl?.querySelector('.intro-wave-particles');
  const dustCloudContainer = introEl?.querySelector('.intro-dust-clouds');

  if (!introEl || !skipButton || !particleContainer || !waveParticleContainer || !dustCloudContainer) {
    onComplete?.();
    return { finish() {}, destroy() {} };
  }

  const timers = [];
  let finished = false;
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  buildStarField(particleContainer);
  buildShockwaveField(waveParticleContainer);
  buildDustClouds(dustCloudContainer);

  function finish() {
    if (finished) return;
    finished = true;
    clearTimers(timers);
    skipButton.removeEventListener('click', finish);
    window.removeEventListener('keydown', handleKeydown);
    setIntroStage(introEl, 'intro-stage-done');
    document.body.classList.remove('intro-active');
    document.body.classList.remove('intro-galaxy-visible');
    document.body.classList.add('intro-complete');

    window.setTimeout(() => {
      introEl.setAttribute('aria-hidden', 'true');
      introEl.remove();
      onComplete?.();
    }, prefersReducedMotion ? 120 : 520);
  }

  function handleKeydown(event) {
    if (event.key === 'Escape' || event.key === 'Enter' || event.code === 'Space') {
      event.preventDefault();
      finish();
    }
  }

  function runStages() {
    document.body.classList.add('intro-active');

    let elapsed = 0;
    const activeStages = prefersReducedMotion ? REDUCED_MOTION_STAGES : STAGES;

    activeStages.forEach(([stageClass, duration]) => {
      timers.push(window.setTimeout(() => setIntroStage(introEl, stageClass), elapsed));
      elapsed += duration;
    });

    timers.push(window.setTimeout(finish, elapsed));
  }

  skipButton.addEventListener('click', finish);
  window.addEventListener('keydown', handleKeydown);
  runStages();

  return {
    finish,
    destroy() {
      clearTimers(timers);
      skipButton.removeEventListener('click', finish);
      window.removeEventListener('keydown', handleKeydown);
    }
  };
}
