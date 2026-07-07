import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { GalaxySimulation } from './galaxy.js';
import { GalaxyUI } from './ui.js';
import { CameraController } from './cameraController.js';
import { GalaxyGestures } from './gestures.js';
import { createIntroSequence } from './intro.js';

// Configuration
const config = {
  starCount: 750000,
  rotationSpeed: 0.1,
  spiralTightness: 1.75,
  mouseForce: 7.0,
  mouseRadius: 10.0,
  galaxyRadius: 13.0,
  galaxyThickness: 3,
  armCount: 2,
  armWidth: 2.25,
  randomness: 1.8,
  particleSize: 0.04,
  starBrightness: 0.15,
  denseStarColor: '#1885ff',
  sparseStarColor: '#ffb28a',
  bloomStrength: 0.15,
  bloomRadius: 0.3,
  bloomThreshold: 0.5,
  cloudCount: 5000,
  cloudSize: 1.0,
  cloudOpacity: 0.008,
  cloudTintColor: '#ffdace',
  leafCount: 15000,
  // Growth mode configuration (阿凡达光纤生命树配色)
  growthCoreColor: '#00ffff',    // 青色/亮蓝（主干核心）
  growthArmColor: '#0055ff',     // 深邃蓝（光纤旋臂）
  growthTipColor: '#ffffff',     // 纯白发光点（树冠尖端）
  gatheringThreshold: 2.0        // 聚集触发阈值（~2 秒凝聚）
};

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 12, 17);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);
renderer.domElement.classList.add('galaxy-canvas');

createIntroSequence();

// Camera controller (replaces OrbitControls — supports POI nav + gestures)
const viewController = new CameraController(camera, renderer.domElement, config);

// Post-processing
let postProcessing = null;
let bloomPassNode = null;

// Mouse tracking
const mouse3D = new THREE.Vector3(0, 0, 0);
const raycaster = new THREE.Raycaster();
const intersectionPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let mousePressed = false;

// 🌟 NEW: Smooth hand spread & depth transitions (借鉴 gem4)
let targetHandSpread = 1.0;
let currentHandSpread = 1.0;
let targetHandDepth = 1.0;
let currentHandDepth = 1.0;

window.addEventListener('mousedown', () => mousePressed = true);
window.addEventListener('mouseup', () => mousePressed = false);
window.addEventListener('mousemove', (event) => {
  const mouse = new THREE.Vector2(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1
  );
  raycaster.setFromCamera(mouse, camera);
  raycaster.ray.intersectPlane(intersectionPlane, mouse3D);
});

/**
 * Creates a starry background with random colored stars distributed on a sphere
 * @param {THREE.Scene} scene - Scene to add stars to
 * @param {number} count - Number of background stars
 * @returns {THREE.Points} - The star points object
 */
function createStarryBackground(scene, count = 5000) {
  const starGeometry = new THREE.BufferGeometry();
  const starPositions = new Float32Array(count * 3);
  const starColors = new Float32Array(count * 3);

  // Distribute stars randomly on a sphere
  for (let i = 0; i < count; i++) {
    // Spherical coordinates for uniform distribution
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const radius = 100 + Math.random() * 100;

    // Convert to Cartesian coordinates
    starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3 + 2] = radius * Math.cos(phi);

    // Add color variation (mostly white, some blue/orange tinted)
    const color = 0.8 + Math.random() * 0.2;
    const tint = Math.random();
    if (tint < 0.1) {
      // Blue tint
      starColors[i * 3] = color * 0.8;
      starColors[i * 3 + 1] = color * 0.9;
      starColors[i * 3 + 2] = color;
    } else if (tint < 0.2) {
      // Orange tint
      starColors[i * 3] = color;
      starColors[i * 3 + 1] = color * 0.8;
      starColors[i * 3 + 2] = color * 0.6;
    } else {
      // White
      starColors[i * 3] = color;
      starColors[i * 3 + 1] = color;
      starColors[i * 3 + 2] = color;
    }
  }

  starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

  const starMaterial = new THREE.PointsMaterial({
    size: 0.3,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true
  });

  const stars = new THREE.Points(starGeometry, starMaterial);
  scene.add(stars);

  return stars;
}

// Preload cloud texture
const textureLoader = new THREE.TextureLoader();
const cloudTexture = textureLoader.load('cloud.png');

// Create galaxy simulation with preloaded texture
const galaxySimulation = new GalaxySimulation(scene, config, cloudTexture);
galaxySimulation.createGalaxySystem();
galaxySimulation.createClouds();
galaxySimulation.createLeaves();

// Create starry background
createStarryBackground(scene);

// ---- Gesture tracking ----
const videoEl = document.getElementById('webcam');
const gestures = new GalaxyGestures({
  onStatus(msg) {
    document.getElementById('status').textContent = msg;
  },
  onMode(mode) {
    const indicator = document.getElementById('mode-indicator');
    if (mode === 'gesture') {
      indicator.textContent = '🖐 手势模式';
      indicator.className = 'gesture';
    } else {
      indicator.textContent = '🖱 鼠标模式';
      indicator.className = 'mouse';
    }
  },
  onSwipeLeft() {
    viewController.previousPoi();
  },
  onSwipeRight() {
    viewController.nextPoi();
  },
  onFist() {
    viewController.toggleView();
  },
  onHandsTogether(active, center) {
    // 更新星云聚集状态
    galaxySimulation.updateHandsTogether(active, center);
  },
  // 🌟 NEW: Hand spread control (借鉴 gem4)
  onHandSpread(spread) {
    // 平滑过渡：target → current
    targetHandSpread = spread;
  },
  // 🌟 NEW: Hand depth control (借鉴 gem4)
  onHandDepth(depth) {
    // 平滑过渡：target → current
    targetHandDepth = depth;
  },
  onFallback(reason) {
    document.getElementById('status').textContent = reason + ' — 🖱 鼠标模式';
    document.getElementById('mode-indicator').textContent = '🖱 鼠标模式';
    document.getElementById('mode-indicator').className = 'mouse';
  }
});

// Start gesture pipeline asynchronously (non-blocking)
gestures.init(videoEl).then((ok) => {
  if (ok) {
    document.getElementById('status').textContent = '就绪 (请伸出手掌)';
  }
});

// ---- HUD update helper (only updates when values change) ----
let _lastPoiName = '';
let _lastViewStateName = '';
let _lastGrowthState = '';

function updateHUD() {
  const poiName = viewController.getPoiName();
  const viewName = viewController.getViewStateName();

  if (poiName !== _lastPoiName) {
    document.getElementById('poi-name').textContent = poiName;
    _lastPoiName = poiName;
  }
  if (viewName !== _lastViewStateName) {
    document.getElementById('view-name').textContent = viewName;
    _lastViewStateName = viewName;
  }

  // 更新生长状态显示
  if (galaxySimulation.growthMode) {
    const progressPct = Math.round(galaxySimulation.growthProgress * 100);

    // 🌳 NEW: 三阶段状态提示
    let stageText;
    if (galaxySimulation.growthProgress < 0.3) {
      stageText = '萌芽期 (积蓄能量)';
    } else if (galaxySimulation.growthProgress < 0.8) {
      stageText = '生长期 (向上生长)';
    } else {
      stageText = '绽放期 (树枝展开) ✨';
    }

    const growthState = `${stageText} - ${progressPct}%`;
    if (growthState !== _lastGrowthState) {
      document.getElementById('status').textContent = `🌱 ${growthState}`;
      _lastGrowthState = growthState;
    }
  } else if (_lastGrowthState !== '') {
    _lastGrowthState = '';
  }
}

// Setup bloom
function setupBloom() {
  if (!postProcessing) return;

  const scenePass = pass(scene, camera);
  const scenePassColor = scenePass.getTextureNode();

  bloomPassNode = bloom(scenePassColor);
  bloomPassNode.threshold.value = config.bloomThreshold;
  bloomPassNode.strength.value = config.bloomStrength;
  bloomPassNode.radius.value = config.bloomRadius;

  postProcessing.outputNode = scenePassColor.add(bloomPassNode);
}

// Create UI with callbacks
const ui = new GalaxyUI(config, {
  onUniformChange: (key, value) => galaxySimulation.updateUniforms({ [key]: value }),

  onBloomChange: (property, value) => {
    if (bloomPassNode) bloomPassNode[property].value = value;
  },

  onStarCountChange: (newCount) => {
    galaxySimulation.updateStarCount(newCount);
    document.getElementById('star-count').textContent = newCount.toLocaleString();
  },

  onCloudCountChange: (newCount) => {
    galaxySimulation.updateUniforms({ cloudCount: newCount });
    galaxySimulation.createClouds();
    galaxySimulation.createLeaves();
  },

  onCloudTintChange: (color) => {
    galaxySimulation.updateUniforms({ cloudTintColor: color });
    galaxySimulation.createClouds();
    galaxySimulation.createLeaves();
  },

  onGrowthThresholdChange: (threshold) => {
    galaxySimulation.gatheringThreshold = threshold;
  },

  onRegenerate: () => {
    galaxySimulation.updateUniforms(config);
    galaxySimulation.createClouds();
    galaxySimulation.createLeaves();
    galaxySimulation.regenerate();
    viewController.regeneratePois();
  }
});

// FPS counter
let frameCount = 0;
let lastTime = performance.now();
let fps = 60;

function updateFPS() {
  frameCount++;
  const currentTime = performance.now();
  const deltaTime = currentTime - lastTime;

  if (deltaTime >= 1000) {
    fps = Math.round((frameCount * 1000) / deltaTime);
    frameCount = 0;
    lastTime = currentTime;

    document.getElementById('fps').textContent = fps;
    ui.updateFPS(fps);
  }
}

// Animation loop
let lastFrameTime = performance.now();

async function animate() {
  requestAnimationFrame(animate);

  const currentTime = performance.now();
  const deltaTime = Math.min((currentTime - lastFrameTime) / 1000, 0.033);
  lastFrameTime = currentTime;

  // Feed gesture rotation into camera controller (树锁定后禁用，交还鼠标)
  if (gestures.handActive && !galaxySimulation.isTreeLocked) {
    viewController.setExternalRotation(gestures.handRot.x, gestures.handRot.y);
  } else {
    viewController.clearExternalRotation();
  }

  // 🌟 NEW: Smooth lerp for hand spread & depth
  if (!galaxySimulation.isTreeLocked) {
    currentHandSpread += (targetHandSpread - currentHandSpread) * 0.08;
    currentHandDepth += (targetHandDepth - currentHandDepth) * 0.08;
  } else {
    // 💡 树锁定后平滑归位到 0.6，保持凝聚紧凑的树形
    currentHandSpread += (0.6 - currentHandSpread) * 0.08;
    currentHandDepth += (1.0 - currentHandDepth) * 0.08;
  }

  galaxySimulation.uniforms.compute.handSpread.value = currentHandSpread;
  galaxySimulation.uniforms.compute.handDepth.value = currentHandDepth;

  // Update camera (smooth transitions, orbit, keyboard nav)
  viewController.update();

  // Dynamic bloom adjustment for growth mode
  // 宇宙树三阶段动态发光效果（降低强度避免过曝）
  if (galaxySimulation.growthMode && bloomPassNode) {
    const growthProg = galaxySimulation.growthProgress;

    // 三阶段不同的发光效果
    let bloomStrength, bloomThreshold;

    if (growthProg < 0.3) {
      // 萌芽期：微弱发光
      bloomStrength = config.bloomStrength + growthProg * 0.1;
      bloomThreshold = config.bloomThreshold;
    } else if (growthProg < 0.8) {
      // 生长期：逐渐增强
      bloomStrength = config.bloomStrength + growthProg * 0.15;
      bloomThreshold = config.bloomThreshold - growthProg * 0.05;
    } else {
      // 绽放期：柔和定型
      bloomStrength = config.bloomStrength + 0.1;
      bloomThreshold = config.bloomThreshold - 0.05;
    }

    bloomPassNode.strength.value = bloomStrength;
    bloomPassNode.threshold.value = bloomThreshold;
  } else if (bloomPassNode) {
    // 恢复默认 bloom
    bloomPassNode.strength.value = config.bloomStrength;
    bloomPassNode.threshold.value = config.bloomThreshold;
  }

  // Update galaxy
  await galaxySimulation.update(renderer, deltaTime, mouse3D, mousePressed);

  // Render
  if (postProcessing) {
    postProcessing.render();
  } else {
    renderer.render(scene, camera);
  }

  updateFPS();
  updateHUD();
}

// Handle resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize
renderer.init().then(() => {
  postProcessing = new THREE.PostProcessing(renderer);
  setupBloom();
  ui.setBloomNode(bloomPassNode);

  document.getElementById('star-count').textContent = config.starCount.toLocaleString();
  animate();
}).catch(err => {
  console.error('Failed to initialize renderer:', err);
});
