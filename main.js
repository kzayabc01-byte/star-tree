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
  particleSize: 0.06,
  starBrightness: 0.6,
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
  gatheringThreshold: 2.0,        // 聚集触发阈值（~2 秒凝聚）
  colorTheme: 'classic'           // 色彩风格预设
};

// ✌️ 比耶快速自转全局状态
let isFastRotating = false;

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

// ==========================================
// 屏幕中央的醒目状态提示 UI
// ==========================================
const statusDiv = document.createElement('div');
statusDiv.style.cssText = `
  position: absolute;
  top: 15%;
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  font-size: 28px;
  font-weight: bold;
  font-family: sans-serif;
  text-shadow: 0 0 15px rgba(0, 255, 255, 0.9), 0 0 30px rgba(0, 255, 255, 0.6);
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s ease-in-out;
  z-index: 9999;
`;
document.body.appendChild(statusDiv);

let statusTimer = null;
function showStatus(text, duration = 2500) {
  statusDiv.innerText = text;
  statusDiv.style.opacity = 1;
  if (statusTimer) clearTimeout(statusTimer);
  statusTimer = setTimeout(() => {
    statusDiv.style.opacity = 0;
  }, duration);
}

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
  onMode: (mode) => {
    if (mode === 'gesture') showStatus('🖐️ 已捕捉到手势控制', 2000);
  },
  onStatus: (msg) => {
    showStatus(msg, 2000);
  },
  onHandSpread: (spread) => {
    targetHandSpread = spread;
  },
  onHandDepth: (depth) => {
    targetHandDepth = depth;
  },
  onSwipeLeft: () => {
    viewController.previousPoi();
    showStatus('👋 视角向左切换');
  },
  onSwipeRight: () => {
    viewController.nextPoi();
    showStatus('👋 视角向右切换');
  },
  onFist: () => {
    viewController.toggleView();
    showStatus('✊ 切换: 全景 / 特写');
  },
  onHandsTogether: (isActive, center) => {
    galaxySimulation.updateHandsTogether(isActive, center);
  },
  onVPose: (isActive) => {
    isFastRotating = isActive;
    if (isActive) {
      // 当比耶时，确保不在树形态，让它展现纯净的星系旋转
      galaxySimulation.isTreeLocked = false;
    }
  },
  onFallback: (reason) => {
    showStatus('⚠️ 鼠标模式 (' + reason + ')');
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

// ==========================================
// 色彩风格预设
// ==========================================
const THEMES = {
  classic: {
    denseStarColor: '#1885ff', sparseStarColor: '#ffb28a',
    growthCoreColor: '#00ffff', growthArmColor: '#0055ff', growthTipColor: '#ffffff',
    cloudTintColor: '#ffdace', starBrightness: 0.6, bloomStrength: 0.15
  },
  ice: {
    denseStarColor: '#00bfff', sparseStarColor: '#87ceeb',
    growthCoreColor: '#b0e0e6', growthArmColor: '#4169e1', growthTipColor: '#f0f8ff',
    cloudTintColor: '#e0f0ff', starBrightness: 0.7, bloomStrength: 0.2
  },
  cyber: {
    denseStarColor: '#ff00ff', sparseStarColor: '#00ffff',
    growthCoreColor: '#ff1493', growthArmColor: '#00ff7f', growthTipColor: '#ffff00',
    cloudTintColor: '#ff6ec7', starBrightness: 0.8, bloomStrength: 0.35
  },
  golden: {
    denseStarColor: '#ff8c00', sparseStarColor: '#ffd700',
    growthCoreColor: '#ff6600', growthArmColor: '#cc5500', growthTipColor: '#fff8dc',
    cloudTintColor: '#ffe4b5', starBrightness: 0.5, bloomStrength: 0.1
  }
};

function applyTheme(themeName) {
  const t = THEMES[themeName];
  if (!t) return;
  Object.assign(config, t);
  config.colorTheme = themeName;
  galaxySimulation.updateUniforms(t);
  if (bloomPassNode) {
    bloomPassNode.strength.value = t.bloomStrength;
  }
  showStatus(`🎨 切换至「${['深空经典','冰冷深蓝','赛博霓虹','暗金余晖'][['classic','ice','cyber','golden'].indexOf(themeName)]}」`, 2000);
}

// ==========================================
// 随机重构星系
// ==========================================
function randomizeGalaxy() {
  config.armCount = Math.floor(Math.random() * 3) + 1;     // 1-3
  config.galaxyRadius = 8 + Math.random() * 10;             // 8-18
  config.spiralTightness = 1 + Math.random() * 3;           // 1-4
  config.randomness = 0.5 + Math.random() * 3;              // 0.5-3.5
  config.rotationSpeed = 0.05 + Math.random() * 0.3;        // 0.05-0.35
  config.starCount = Math.round(200000 + Math.random() * 600000); // 200k-800k

  galaxySimulation.updateUniforms(config);
  galaxySimulation.createClouds();
  galaxySimulation.createLeaves();
  galaxySimulation.updateStarCount(config.starCount);
  galaxySimulation.regenerate();
  viewController.regeneratePois();
  document.getElementById('star-count').textContent = config.starCount.toLocaleString();
  showStatus('🎲 星系已随机重构!', 2500);
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
  },

  onThemeChange: (theme) => applyTheme(theme),

  onRandomizeGalaxy: () => randomizeGalaxy()
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

  // 【新增】✌️ 比耶快速自转：持续每帧增加相机的水平旋转角度
  if (isFastRotating) {
    viewController._tTheta -= 0.03;
  }

  // 【核心修复】：如果手势处于互斥锁状态（isPoseLocked），彻底停止相机的外部旋转干扰！
  if (gestures.handActive && !gestures.isPoseLocked && !galaxySimulation.isTreeLocked) {
    viewController.setExternalRotation(gestures.handRot.x, gestures.handRot.y);
  } else {
    viewController.clearExternalRotation();
  }

  // 处理手掌张开与深度的平滑过渡（同样受互斥锁和树形模式限制）
  if (!galaxySimulation.isTreeLocked && !gestures.isPoseLocked) {
    currentHandSpread += (targetHandSpread - currentHandSpread) * 0.08;
    currentHandDepth += (targetHandDepth - currentHandDepth) * 0.08;
  } else {
    // 树锁定或结印状态下，平滑归位到安全默认值，防止镜头乱飙
    currentHandSpread += (0.6 - currentHandSpread) * 0.08;
    currentHandDepth += (1.0 - currentHandDepth) * 0.08;
  }

  // 传递给着色器
  if (galaxySimulation.uniforms) {
    galaxySimulation.uniforms.compute.handSpread.value = currentHandSpread;
    galaxySimulation.uniforms.compute.handDepth.value = currentHandDepth;
  }

  // Update camera (smooth transitions, orbit, keyboard nav)
  viewController.update();

  // Dynamic bloom adjustment for growth mode
  if (galaxySimulation.growthMode && bloomPassNode) {
    const growthProg = galaxySimulation.growthProgress;
    let bloomStrength, bloomThreshold;

    if (growthProg < 0.3) {
      bloomStrength = config.bloomStrength + growthProg * 0.1;
      bloomThreshold = config.bloomThreshold;
    } else if (growthProg < 0.8) {
      bloomStrength = config.bloomStrength + growthProg * 0.15;
      bloomThreshold = config.bloomThreshold - growthProg * 0.05;
    } else {
      bloomStrength = config.bloomStrength + 0.1;
      bloomThreshold = config.bloomThreshold - 0.05;
    }

    bloomPassNode.strength.value = bloomStrength;
    bloomPassNode.threshold.value = bloomThreshold;
  } else if (bloomPassNode) {
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
