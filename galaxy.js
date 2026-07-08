/**
 * Galaxy Simulation - Main Orchestrator
 *
 * This module contains the main GalaxySimulation class that manages:
 * - Uniform initialization and updates
 * - Star particle system creation and physics
 * - Cloud particle system to simulate dust
 * - WebGPU compute shader execution
 */

import * as THREE from 'three/webgpu';
import {
  uniform,
  instancedArray,
  instanceIndex,
  vec3,
  vec4,
  float,
  Fn,
  mix,
  length,
  sin,
  cos,
  acos,
  uv,
  smoothstep,
  step,
  floor,
  texture,
  normalize,
  pow
} from 'three/tsl';

import {
  hash,
  applyDifferentialRotation,
  applyMouseForce,
  applySpringForce,
  applyAttractionForce
} from './helpers.js';

/**
 * Spiral Galaxy Position Generation
 *
 * Note: TSL Fn() functions can only return single TSL types (vec3, float, etc.),
 * not JavaScript objects. Since spiral position generation needs to return multiple
 * values (position, normalizedRadius, angle, etc.), we inline this logic in both
 * the star and cloud initialization shaders below.
 *
 * The pattern is consistent between stars and clouds:
 * 1. Generate radius using hash with power function (controls distribution)
 * 2. Select spiral arm and calculate spiral angle
 * 3. Add randomness for natural appearance
 * 4. Convert to Cartesian coordinates
 * 5. Apply vertical thickness (thicker at center, thinner at edges)
 */

// ==============================================================================
// GALAXY SIMULATION CLASS
// ==============================================================================

/**
 * GPU-accelerated galaxy simulation with stars and dust clouds
 * Uses WebGPU compute shaders for particle physics and rendering
 */
export class GalaxySimulation {
  constructor(scene, config, cloudTexture = null) {
    this.scene = scene;
    this.config = config;
    this.COUNT = config.starCount;
    this.cloudTexture = cloudTexture;

    // Storage buffers
    this.spawnPositionBuffer = null;
    this.originalPositionBuffer = null;
    this.velocityBuffer = null;
    this.densityFactorBuffer = null;

    // Compute shaders
    this.computeInit = null;
    this.computeUpdate = null;
    this.cloudInit = null;
    this.cloudUpdate = null;

    // Scene objects
    this.galaxy = null;
    this.cloudPlane = null;

    // Initialize uniforms organized by category
    this.initializeUniforms(config);

    // State
    this.initialized = false;
    this.cloudInitialized = false;

    // Alternate sword-array state
    this.growthMode = false;
    this.growthProgress = 0.0;
    this.gatheringThreshold = 2.0; // ~2 秒凝聚时间
    this.lastGatheringLevel = 0.0;
    this.swordPoseActive = false;
    this.isTreeLocked = false; // 🌟 一键触发后永久定型
    this.scatteringMode = false;   // 🌌 散落回星系动画
    this.scatterProgress = 0.0;

    // Leaf particle system state
    this.LEAF_COUNT = config.leafCount || 15000;
    this.leafInit = null;
    this.leafUpdate = null;
    this.leafPlane = null;
    this.leafInitialized = false;
  }

  /**
   * Initialize all shader uniforms organized into logical groups
   */
  initializeUniforms(config) {
    // Compute state uniforms (time, mouse interaction)
    this.uniforms = {
      compute: {
        time: uniform(0),
        deltaTime: uniform(0.016),
        mouse: uniform(new THREE.Vector3(0, 0, 0)),
        mouseActive: uniform(0.0),
        mouseForce: uniform(config.mouseForce),
        mouseRadius: uniform(config.mouseRadius),
        rotationSpeed: uniform(config.rotationSpeed),
        // Hands together (galaxy gathering)
        handsCenter: uniform(new THREE.Vector3(0, 0, 0)),
        handsTogetherActive: uniform(0.0),
        attractionStrength: uniform(15.0),  // 强吸引力
        attractionRadius: uniform(25.0),    // 大范围影响
        // 🌟 NEW: Hand spread & depth (借鉴 gem4)
        handSpread: uniform(1.0),          // 五指张开程度（控制粒子扩散）
        handDepth: uniform(1.0),           // 手掌深度（控制整体缩放）
        // Growth mode (螺旋植物生长)
        growthModeActive: uniform(0.0),
        growthProgress: uniform(0.0),
        growthSpiralIntensity: uniform(0.0)  // 螺旋强度（控制旋臂展开程度）
      },

      // Galaxy structure uniforms (shape, size, distribution)
      galaxy: {
        radius: uniform(config.galaxyRadius),
        thickness: uniform(config.galaxyThickness || 0.1),
        spiralTightness: uniform(config.spiralTightness),
        armCount: uniform(config.armCount),
        armWidth: uniform(config.armWidth),
        randomness: uniform(config.randomness)
      },

      // Visual appearance uniforms (colors, sizes, opacity)
      visual: {
        particleSize: uniform(config.particleSize),
        cloudSize: uniform(config.cloudSize),
        cloudOpacity: uniform(config.cloudOpacity !== undefined ? config.cloudOpacity : 0.5),
        starBrightness: uniform(config.starBrightness !== undefined ? config.starBrightness : 1.0),
        denseStarColor: uniform(new THREE.Color(config.denseStarColor || '#99ccff')),
        sparseStarColor: uniform(new THREE.Color(config.sparseStarColor || '#ffb380')),
        cloudTintColor: uniform(new THREE.Color(config.cloudTintColor || '#6ba8cc')),
        // Tree effect colors (阿凡达光纤生命树配色)
        growthCoreColor: uniform(new THREE.Color(config.growthCoreColor || '#00ffff')),
        growthArmColor: uniform(new THREE.Color(config.growthArmColor || '#0055ff')),
        growthTipColor: uniform(new THREE.Color(config.growthTipColor || '#ffffff')),
        // Leaf effect colors
        leafColorDark: uniform(new THREE.Color('#001133')),     // 极暗夜空蓝
        leafColorLight: uniform(new THREE.Color('#00e5ff')),    // 明亮荧光青
        leafOpacity: uniform(0.08),
        leafSize: uniform(0.8)
      }
    };
  }

  /**
   * Creates the star particle system with spiral galaxy structure
   */
  createGalaxySystem() {
    // Clean up old galaxy
    if (this.galaxy) {
      this.scene.remove(this.galaxy);
      if (this.galaxy.material) {
        this.galaxy.material.dispose();
      }
    }

    // Create storage buffers for star particles
    this.spawnPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.originalPositionBuffer = instancedArray(this.COUNT, 'vec3');
    this.velocityBuffer = instancedArray(this.COUNT, 'vec3');
    this.densityFactorBuffer = instancedArray(this.COUNT, 'float');
    // Alternate formation target buffer
    this.growthTargetBuffer = instancedArray(this.COUNT, 'vec3');
    this.growthAngleBuffer = instancedArray(this.COUNT, 'float');
    this.growthTreeLevelBuffer = instancedArray(this.COUNT, 'float');

    // Initialize stars with spiral arm distribution
    this.computeInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat();

      // Distance from center (square root for even distribution)
      const radius = hash(seed.add(1)).pow(0.5).mul(this.uniforms.galaxy.radius);
      const normalizedRadius = radius.div(this.uniforms.galaxy.radius);

      // Choose which spiral arm this particle belongs to
      const armIndex = hash(seed.add(2)).mul(this.uniforms.galaxy.armCount).floor();
      const armAngle = armIndex.mul(6.28318).div(this.uniforms.galaxy.armCount);

      // Spiral angle based on distance (logarithmic spiral)
      const spiralAngle = normalizedRadius.mul(this.uniforms.galaxy.spiralTightness).mul(6.28318);

      // Add randomness to create natural appearance
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.uniforms.galaxy.randomness);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.uniforms.galaxy.armWidth);

      // Final angle and radius
      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Convert to Cartesian coordinates
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Vertical position: thicker at center, thinner at edges
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.2); // 1.2 at center, 0.2 at edge
      const y = hash(seed.add(5)).sub(0.5).mul(this.uniforms.galaxy.thickness).mul(thicknessFactor);

      const position = vec3(x, y, z);

      // Store initial positions
      this.spawnPositionBuffer.element(idx).assign(position);
      this.originalPositionBuffer.element(idx).assign(position);

      // ═══════════════════════════════════════════════════════
      // 🌳🌳🌳 分块组装大法：高耸主干与扁平化层次树冠 (Layered Crown)
      // ═══════════════════════════════════════════════════════
      const growthSeed = hash(seed.add(10.0));
      const R = this.uniforms.galaxy.radius;
      const treeMaxHeight = R.mul(float(3.8)); // [增强]：拔高整棵树的上限，主干会更修长挺拔！

      // 决定命运的抛硬币（约15%去拼树干，剩下的去做叶片）
      const partType = growthSeed;
      const isLeaf = step(float(0.15), partType);

      // ==========================================
      // 🧩 零件 1：绝对不糊住任何东西的"参天巨木组"
      // ==========================================
      const trunkH = hash(seed.add(20.0)); // 在高度0.0 ~ 1.0之间游走
      const trunkY = trunkH.mul(treeMaxHeight);

      // 树干结构设计：根盘极剧收紧，主轴非常之细 (绝不去遮视野！)
      const rootFlare = float(1.0).sub(trunkH).pow(float(6.0)).mul(R.mul(float(0.7)));
      const trunkRadius = rootFlare.add(R.mul(float(0.06)));

      const trunkAngle = hash(seed.add(21.0)).mul(6.28318);
      const trunkPos = vec3(
        cos(trunkAngle).mul(trunkRadius),
        trunkY,
        sin(trunkAngle).mul(trunkRadius)
      );

      // ==========================================
      // 🧩 零件 2：拒绝塌方的 "高层悬浮松叶/扁平迎客松 云盘"
      // ==========================================
      const CLUSTER_COUNT = float(22.0);
      const clusterId = floor(hash(seed.add(30.0)).mul(CLUSTER_COUNT));
      const clusterSeed = clusterId.mul(1.234);

      // --- 核心改动 1：锁定最高空高度领域！严禁往树根跑！---
      // 球心必须停留在顶部：高度基础为总高的 70% ~ 105% 范围间
      const clBaseY = treeMaxHeight.mul(float(0.7)).add(
         hash(clusterSeed.add(2.0)).mul(treeMaxHeight.mul(float(0.35)))
      );

      // 巨大的开花展开度（巨伞感）
      const clSpreadRatio = pow(hash(clusterSeed.add(1.0)), float(0.6));
      const clAngle = hash(clusterSeed).mul(6.28318);
      const clDist = clSpreadRatio.mul(R.mul(float(2.8)));

      // 【关键切除垂坠感】: 最多只下降 0.15，悬得很稳！
      const clDroop = clSpreadRatio.pow(float(2.0)).mul(treeMaxHeight.mul(float(0.15)));
      const clY = clBaseY.sub(clDroop);

      const clusterCenter = vec3(cos(clAngle).mul(clDist), clY, sin(clAngle).mul(clDist));

      // --- 核心改动 2：压缩Y厚度，改成迎客松般的平云盘(扁平层次感) ---
      const lRadius = pow(hash(seed.add(43.0)), float(0.5)).mul(R.mul(float(1.5))); // 小树云的外放极限
      const lTheta = hash(seed.add(41.0)).mul(6.28318);
      const lPhi = hash(seed.add(42.0)).mul(3.14159);

      // 给 XYZ 画球，但疯狂压缩 0.3 的高度，变松油薄饼块：
      const leafX = sin(lPhi).mul(cos(lTheta)).mul(lRadius);
      const leafY = cos(lPhi).mul(lRadius).mul(float(0.3));  // 💥 扁平化
      const leafZ = sin(lPhi).mul(sin(lTheta)).mul(lRadius);

      const leafPos = clusterCenter.add(vec3(leafX, leafY, leafZ));

      // ==========================================
      // 🛠️ 合体并归还至天空正上方
      // ==========================================
      const finalRawPos = mix(trunkPos, leafPos, isLeaf);

      // ✅ 深植入地下 1.5*R，让上部云层留出足够视野！
      const growthPosition = finalRawPos.sub(vec3(float(0.0), R.mul(float(1.5)), float(0.0)));

      this.growthTargetBuffer.element(idx).assign(growthPosition);
      this.growthTreeLevelBuffer.element(idx).assign(isLeaf);

      const swordBladeCount = this.uniforms.galaxy.armCount.mul(3.0).add(6.0);
      const swordBladeIndex = hash(seed.add(21)).mul(swordBladeCount).floor();
      const swordHeightRatio = growthSeed.pow(0.62);
      const swordSection = swordHeightRatio.mul(4.0).floor().min(float(3.0));
      const swordBaseAngle = swordBladeIndex.mul(6.28318).div(swordBladeCount);
      const swordSweep = swordHeightRatio.pow(1.3).mul(0.22);
      const swordJitter = hash(seed.add(22)).sub(0.5).mul(0.05);
      const swordAngleFinal = swordBaseAngle.add(swordSweep).add(swordJitter);

      const swordBaseRing = this.uniforms.galaxy.radius.mul(0.26)
        .add(hash(seed.add(23)).sub(0.5).mul(this.uniforms.galaxy.radius.mul(0.08)));
      const swordReach = swordHeightRatio.pow(1.6).mul(this.uniforms.galaxy.radius.mul(0.46));
      const swordRadius = swordBaseRing.add(swordReach);

      const swordThickness = float(1.0)
        .sub(swordHeightRatio.mul(0.82))
        .max(0.05)
        .mul(this.uniforms.galaxy.radius.mul(0.14));
      const swordLateralAngle = swordAngleFinal.add(1.5707963);
      const swordLateralOffset = hash(seed.add(24)).sub(0.5).mul(swordThickness);
      const swordForwardOffset = hash(seed.add(25)).sub(0.35).mul(swordThickness.mul(0.55));
      const swordVerticalOffset = hash(seed.add(26)).sub(0.5).mul(swordThickness.mul(0.75));

      const swordX = cos(swordAngleFinal).mul(swordRadius)
        .add(cos(swordLateralAngle).mul(swordLateralOffset))
        .add(cos(swordAngleFinal).mul(swordForwardOffset));
      const swordZ = sin(swordAngleFinal).mul(swordRadius)
        .add(sin(swordLateralAngle).mul(swordLateralOffset))
        .add(sin(swordAngleFinal).mul(swordForwardOffset));
      const swordY = swordHeightRatio.mul(this.uniforms.galaxy.radius.mul(2.15))
        .sub(this.uniforms.galaxy.radius.mul(0.7))
        .add(swordVerticalOffset);
      const swordPosition = vec3(swordX, swordY, swordZ);

      this.growthAngleBuffer.element(idx).assign(swordAngleFinal);

      // Calculate orbital velocity (faster closer to center)
      const orbitalSpeed = float(1.0).div(offsetRadius.add(0.5)).mul(5.0);
      const vx = sin(angle).mul(orbitalSpeed).negate();
      const vz = cos(angle).mul(orbitalSpeed);
      this.velocityBuffer.element(idx).assign(vec3(vx, 0, vz));

      // Calculate density factor for coloring (0 = dense/center, 1 = sparse/edge)
      const radialSparsity = radiusOffset.abs().div(this.uniforms.galaxy.armWidth.mul(0.5).add(0.01));
      const angularSparsity = angleOffset.abs().div(this.uniforms.galaxy.randomness.mul(0.5).add(0.01));
      const sparsityFactor = radialSparsity.add(angularSparsity).mul(0.5).min(1.0);

      this.densityFactorBuffer.element(idx).assign(sparsityFactor);
    })().compute(this.COUNT);

    // Update shader: applies rotation, mouse interaction, spring forces, and growth animation
    this.computeUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = this.spawnPositionBuffer.element(idx).toVar();
      const originalPos = this.originalPositionBuffer.element(idx);
      const growthTargetPos = this.growthTargetBuffer.element(idx);

      // Apply differential rotation (reduce during growth mode)
      const rotationSpeed = this.uniforms.compute.rotationSpeed.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive.mul(this.uniforms.compute.growthProgress.mul(1.0)))
      );
      const rotatedPos = applyDifferentialRotation(
        position,
        rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      position.assign(rotatedPos);

      // Rotate original position to maintain spring force target
      const rotatedOriginal = applyDifferentialRotation(
        originalPos,
        rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      this.originalPositionBuffer.element(idx).assign(rotatedOriginal);

      // Apply mouse repulsion force (disabled during growth)
      const mouseForceActive = this.uniforms.compute.mouseActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const mouseForce = applyMouseForce(
        position,
        this.uniforms.compute.mouse,
        mouseForceActive,
        this.uniforms.compute.mouseForce,
        this.uniforms.compute.mouseRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(mouseForce);

      // Apply hands together attraction force (only when not in growth mode)
      const attractionActive = this.uniforms.compute.handsTogetherActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const attractionForce = applyAttractionForce(
        position,
        this.uniforms.compute.handsCenter,
        attractionActive,
        this.uniforms.compute.attractionStrength,
        this.uniforms.compute.attractionRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(attractionForce);

      // ---- Growth mode: smooth transition to spiral plant shape ----
      const growthActive = this.uniforms.compute.growthModeActive;
      const growthProg = this.uniforms.compute.growthProgress;
      const smoothProgress = pow(growthProg, float(0.7)); // ease-out 曲线

      // 🌟 【修复压扁问题】缩放系数只影响水平的 X 和 Z，保留 Y 轴挺拔的高度！
      const spreadScale = this.uniforms.compute.handSpread;
      const scaledGrowthTarget = vec3(
        growthTargetPos.x.mul(spreadScale),
        growthTargetPos.y, // 保持高度不变
        growthTargetPos.z.mul(spreadScale)
      );

      // 过渡力：当前位置 → 缩放后的目标位置
      const transitionForce = scaledGrowthTarget.sub(position).mul(smoothProgress.mul(8.0)).mul(this.uniforms.compute.deltaTime);
      position.addAssign(transitionForce.mul(growthActive));

      // 生长过程中的额外旋转
      const growthRotationSpeed = float(0.4).mul(growthActive).mul(float(1.0).sub(growthProg));
      const growthRotatedPos = applyDifferentialRotation(position, growthRotationSpeed, this.uniforms.compute.deltaTime);
      position.assign(mix(position, growthRotatedPos, growthActive.mul(float(1.0).sub(growthProg))));

      // 🌟 弹簧目标也同样只缩放 X 和 Z
      const springTarget = mix(rotatedOriginal, scaledGrowthTarget, growthActive.mul(growthProg));
      const springStrength = float(2.0)
        .sub(this.uniforms.compute.handsTogetherActive.mul(1.5))
        .sub(growthActive.mul(growthProg.mul(1.8)));
      const springForce = applySpringForce(
        position,
        springTarget,
        springStrength.max(float(0.2)),
        this.uniforms.compute.deltaTime
      );
      position.addAssign(springForce);

      this.spawnPositionBuffer.element(idx).assign(position);
    })().compute(this.COUNT);

    // Create star visualization material
    const spriteMaterial = new THREE.SpriteNodeMaterial();
    spriteMaterial.transparent = false;
    spriteMaterial.depthWrite = false;
    spriteMaterial.blending = THREE.AdditiveBlending;

    const starPos = this.spawnPositionBuffer.toAttribute();
    const densityFactor = this.densityFactorBuffer.toAttribute();
    const treeLevel = this.growthTreeLevelBuffer.toAttribute(); // 🌳 树的层级

    // Smooth circular star shape
    const circleShape = Fn(() => {
      const center = uv().sub(0.5).mul(2.0);
      const dist = length(center);
      const alpha = smoothstep(1.0, 0.0, dist).mul(smoothstep(1.0, 0.3, dist));
      return alpha;
    })();

    // Dynamic color based on growth mode
    const starColorNode = Fn(() => {
      const growthActive = this.uniforms.compute.growthModeActive;
      const growthProg = this.uniforms.compute.growthProgress;

      // Default galaxy colors
      const defaultColor = mix(
        vec3(this.uniforms.visual.denseStarColor),
        vec3(this.uniforms.visual.sparseStarColor),
        densityFactor
      );

      // 🌳 NEW: Tree level-based colors (更丰富的层次)
      // Level 0 (主干) = 最亮金色
      // Level 1 (主分支) = 橙色
      // Level 2 (次级分支) = 金橙色
      // Level 3 (树叶/花朵) = 柔和金粉色

      const trunkColor = vec3(this.uniforms.visual.growthCoreColor); // 主干：青蓝
      const mainBranchColor = vec3(this.uniforms.visual.growthArmColor); // 主分支：深邃蓝
      const subBranchColor = mix(
        vec3(this.uniforms.visual.growthCoreColor),
        vec3(this.uniforms.visual.growthArmColor),
        float(0.5)
      ); // 次级分支：青蓝混合
      const leafColor = vec3(this.uniforms.visual.growthTipColor).mul(1.2); // 树叶：纯白发光

      // 根据树的层级选择颜色
      const levelColor = mix(
        mix(
          mix(trunkColor, mainBranchColor, treeLevel.sub(float(0.0)).max(0.0).min(1.0)),
          subBranchColor,
          treeLevel.sub(float(1.0)).max(0.0).min(1.0)
        ),
        leafColor,
        treeLevel.sub(float(2.0)).max(0.0).min(1.0)
      );

      // Blend between default and tree colors
      const finalColor = mix(defaultColor, levelColor, growthActive.mul(growthProg));

      // 根据分件上色：主干中等亮(1.2)，高悬落叶爆亮(2.0)
      const brightnessBoost = growthActive.mul(growthProg).mul(
        mix(
          float(1.2), // 主干中等程度照亮，别闪瞎
          float(2.0), // 把所有的极昼特效资源分配给孤悬的高空落叶
          treeLevel
        )
      );

      return finalColor.mul(this.uniforms.visual.starBrightness.add(brightnessBoost));
    })();

    spriteMaterial.positionNode = starPos;
    spriteMaterial.colorNode = vec4(starColorNode.x, starColorNode.y, starColorNode.z, float(1.0));
    spriteMaterial.opacityNode = circleShape.mul(0.15); // 降低基础透明度，避免加法混合过曝

    // ---- 🌳 大小分离渲染：细心打磨分块颗粒度 ----
    const dynamicSizeNode = Fn(() => {
      const baseSize = this.uniforms.visual.particleSize;

      // treeLevel (是主干 = 0.0; 是漂浮树叶 = 1.0)
      // 0.07 × 2.0 ≈ 0.14 (主干) / 0.07 × 2.2 ≈ 0.15 (树冠)
      const typeSize = mix(float(2.0), float(2.2), treeLevel);

      const growthActive = this.uniforms.compute.growthModeActive;
      const growthProg = this.uniforms.compute.growthProgress;

      return baseSize.mul(mix(float(1.0), typeSize, growthActive.mul(growthProg)));
    })();

    spriteMaterial.scaleNode = dynamicSizeNode;

    this.galaxy = new THREE.Sprite(spriteMaterial);
    this.galaxy.count = this.COUNT;
    this.galaxy.frustumCulled = false;

    this.scene.add(this.galaxy);
  }

  /**
   * Creates cloud particles that follow the galaxy structure
   */
  createClouds() {
    // Clean up old clouds
    if (this.cloudPlane) {
      this.scene.remove(this.cloudPlane);
      if (this.cloudPlane.material) this.cloudPlane.material.dispose();
    }

    const CLOUD_COUNT = this.config.cloudCount;

    // Create cloud particle buffers
    const cloudPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudOriginalPositionBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudColorBuffer = instancedArray(CLOUD_COUNT, 'vec3');
    const cloudSizeBuffer = instancedArray(CLOUD_COUNT, 'float');
    const cloudRotationBuffer = instancedArray(CLOUD_COUNT, 'float');
    // Cloud growth target buffer (螺旋植物目标位置)
    const cloudGrowthTargetBuffer = instancedArray(CLOUD_COUNT, 'vec3');

    // Initialize cloud particles
    this.cloudInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat().add(10000); // Offset seed from stars

      // Distance from center (power = 0.7 for more even distribution to avoid center oversaturation)
      const radius = hash(seed.add(1)).pow(0.7).mul(this.uniforms.galaxy.radius);
      const normalizedRadius = radius.div(this.uniforms.galaxy.radius);

      // Choose spiral arm
      const armIndex = hash(seed.add(2)).mul(this.uniforms.galaxy.armCount).floor();
      const armAngle = armIndex.mul(6.28318).div(this.uniforms.galaxy.armCount);

      // Spiral angle based on distance (logarithmic spiral)
      const spiralAngle = normalizedRadius.mul(this.uniforms.galaxy.spiralTightness).mul(6.28318);

      // Add randomness (same pattern as stars)
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.uniforms.galaxy.randomness);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.uniforms.galaxy.armWidth);

      // Final angle and radius
      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      // Convert to Cartesian coordinates
      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      // Vertical position: slightly thinner than stars
      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.15); // 1.15 at center, 0.15 at edge
      const y = hash(seed.add(5)).sub(0.5).mul(this.uniforms.galaxy.thickness).mul(thicknessFactor);

      const position = vec3(x, y, z);

      // Store positions
      cloudPositionBuffer.element(idx).assign(position);
      cloudOriginalPositionBuffer.element(idx).assign(position);

      // ---- Calculate cloud growth target position (云层附着在树冠上) ----
        const growthSeed = hash(seed.add(10));

        // 修复高度对齐：让云层仅在树冠部分 (0.6 ~ 1.0) 产生，但高度映射与主树严格一致
        const h = float(0.6).add(growthSeed.pow(float(1.5)).mul(float(0.4)));

        const MAX_TREE_HEIGHT = this.uniforms.galaxy.radius.mul(3.5);
        const rawTreeY = h.mul(MAX_TREE_HEIGHT);

        // 🌟 云层同样使用抛物线穹顶轮廓
        const canopyFactor = smoothstep(float(0.4), float(1.0), h);
        const canopyCurve = float(1.0).sub(pow(canopyFactor.mul(2.0).sub(1.0), float(2.0)));
        const crownSpread = canopyCurve.mul(float(5.8)); // 比星星稍微大一点，形成包裹

        const cloudFuzziness = hash(seed.add(19)).pow(float(2.0)).mul(float(1.5));
        const baseSpread = crownSpread.add(cloudFuzziness);

        const branchArmIndex = hash(seed.add(11)).mul(this.uniforms.galaxy.armCount.mul(5.0)).floor();
        const branchBaseAngle = branchArmIndex.mul(6.28318).div(this.uniforms.galaxy.armCount.mul(5.0));
        const cloudGrowthSpiralAngle = h.pow(float(1.2)).mul(6.28318).mul(float(1.5));
        const swayAngle = hash(seed.add(12)).sub(0.5).mul(1.0).mul(canopyFactor);
        const growthAngleFinal = branchBaseAngle.add(cloudGrowthSpiralAngle).add(swayAngle);

        const growthRadiusFinal = baseSpread.max(float(0.0)).mul(this.uniforms.galaxy.radius);
        const growthX = cos(growthAngleFinal).mul(growthRadiusFinal);
        const growthZ = sin(growthAngleFinal).mul(growthRadiusFinal);

        // 🌟 云层的上下膨胀感也受到 canopyCurve 约束
        const droopAmount = crownSpread.mul(float(0.2));
        const volumetricYLift = hash(seed.add(14)).sub(0.5).mul(float(3.0)).mul(canopyCurve);

        const growthY = rawTreeY
          .sub(droopAmount.mul(this.uniforms.galaxy.radius))
          .add(volumetricYLift.mul(this.uniforms.galaxy.radius))
          .sub(this.uniforms.galaxy.radius.mul(1.5));

        const growthPosition = vec3(growthX, growthY, growthZ);

      const swordBladeCount = this.uniforms.galaxy.armCount.mul(3.0).add(6.0);
      const swordBladeIndex = hash(seed.add(21)).mul(swordBladeCount).floor();
      const swordHeightRatio = growthSeed.pow(0.55);
      const swordBaseAngle = swordBladeIndex.mul(6.28318).div(swordBladeCount);
      const swordAngleFinal = swordBaseAngle
        .add(swordHeightRatio.pow(1.2).mul(0.28))
        .add(hash(seed.add(22)).sub(0.5).mul(0.08));

      const auraBaseRadius = this.uniforms.galaxy.radius.mul(0.22)
        .add(swordHeightRatio.pow(1.45).mul(this.uniforms.galaxy.radius.mul(0.62)));
      const auraThickness = float(1.0)
        .sub(swordHeightRatio.mul(0.68))
        .max(0.12)
        .mul(this.uniforms.galaxy.radius.mul(0.24));
      const auraLateralAngle = swordAngleFinal.add(1.5707963);
      const auraLateral = hash(seed.add(23)).sub(0.5).mul(auraThickness.mul(1.6));
      const auraForward = hash(seed.add(24)).sub(0.5).mul(auraThickness.mul(0.8));
      const auraVertical = hash(seed.add(25)).sub(0.5).mul(auraThickness.mul(0.9));

      const auraX = cos(swordAngleFinal).mul(auraBaseRadius)
        .add(cos(auraLateralAngle).mul(auraLateral))
        .add(cos(swordAngleFinal).mul(auraForward));
      const auraZ = sin(swordAngleFinal).mul(auraBaseRadius)
        .add(sin(auraLateralAngle).mul(auraLateral))
        .add(sin(swordAngleFinal).mul(auraForward));
      const auraY = swordHeightRatio.mul(this.uniforms.galaxy.radius.mul(2.25))
        .sub(this.uniforms.galaxy.radius.mul(0.55))
        .add(auraVertical);
      const auraPosition = vec3(auraX, auraY, auraZ);

      const haloAngle = hash(seed.add(26)).mul(6.28318);
      const haloRadius = this.uniforms.galaxy.radius.mul(0.32)
        .add(hash(seed.add(27)).pow(0.65).mul(this.uniforms.galaxy.radius.mul(1.1)));
      const haloY = hash(seed.add(28)).sub(0.5).mul(this.uniforms.galaxy.radius.mul(0.5))
        .add(growthSeed.pow(1.8).mul(this.uniforms.galaxy.radius.mul(0.22)));
      const haloPosition = vec3(
        cos(haloAngle).mul(haloRadius),
        haloY,
        sin(haloAngle).mul(haloRadius)
      );

      const haloBlend = hash(seed.add(29)).pow(2.1);
      const swordCloudPosition = mix(auraPosition, haloPosition, haloBlend);

      cloudGrowthTargetBuffer.element(idx).assign(growthPosition);

      // Cloud color: tinted and darker towards edges
      const tintColor = vec3(this.uniforms.visual.cloudTintColor);
      const cloudColor = tintColor.mul(float(1.0).sub(normalizedRadius.mul(0.3)));
      cloudColorBuffer.element(idx).assign(cloudColor);

      // Size variation: larger clouds in denser regions
      const densityFactor = float(1.0).sub(normalizedRadius.mul(0.5));
      const size = hash(seed.add(6)).mul(0.5).add(0.7).mul(densityFactor);
      cloudSizeBuffer.element(idx).assign(size);

      // Random rotation for visual variation
      const rotation = hash(seed.add(7)).mul(6.28318); // 0 to 2π
      cloudRotationBuffer.element(idx).assign(rotation);
    })().compute(CLOUD_COUNT);

    // Update cloud particles (same physics as stars but weaker spring)
    this.cloudUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = cloudPositionBuffer.element(idx).toVar();
      const originalPos = cloudOriginalPositionBuffer.element(idx);
      const growthTargetPos = cloudGrowthTargetBuffer.element(idx);

      // Apply differential rotation (reduce during growth mode)
      const rotationSpeed = this.uniforms.compute.rotationSpeed.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive.mul(this.uniforms.compute.growthProgress.mul(1.0)))
      );
      const rotatedPos = applyDifferentialRotation(
        position,
        rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      position.assign(rotatedPos);

      // Rotate original position
      const rotatedOriginal = applyDifferentialRotation(
        originalPos,
        rotationSpeed,
        this.uniforms.compute.deltaTime
      );
      cloudOriginalPositionBuffer.element(idx).assign(rotatedOriginal);

      // Apply mouse force (disabled during growth)
      const mouseForceActive = this.uniforms.compute.mouseActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const mouseForce = applyMouseForce(
        position,
        this.uniforms.compute.mouse,
        mouseForceActive,
        this.uniforms.compute.mouseForce,
        this.uniforms.compute.mouseRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(mouseForce);

      // Apply hands together attraction force (only when not in growth mode)
      const attractionActive = this.uniforms.compute.handsTogetherActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const attractionForce = applyAttractionForce(
        position,
        this.uniforms.compute.handsCenter,
        attractionActive,
        this.uniforms.compute.attractionStrength.mul(1.5),
        this.uniforms.compute.attractionRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(attractionForce);

      // ---- Growth mode: smooth transition to spiral plant shape ----
      const growthActive = this.uniforms.compute.growthModeActive;
      const growthProg = this.uniforms.compute.growthProgress;

      // 🌟 只缩放云层的 X 和 Z，保留 Y 轴高度
      const spreadScale = this.uniforms.compute.handSpread;
      const scaledGrowthTarget = vec3(
        growthTargetPos.x.mul(spreadScale),
        growthTargetPos.y,
        growthTargetPos.z.mul(spreadScale)
      );

      const smoothProgress = pow(growthProg, float(0.7));
      const transitionForce = scaledGrowthTarget.sub(position).mul(smoothProgress.mul(9.0)).mul(this.uniforms.compute.deltaTime);
      position.addAssign(transitionForce.mul(growthActive));

      // 生长过程中的额外旋转（树长满后自动归零）
      const growthRotationSpeed = float(0.5).mul(growthActive).mul(float(1.0).sub(growthProg));
      const growthRotatedPos = applyDifferentialRotation(position, growthRotationSpeed, this.uniforms.compute.deltaTime);
      position.assign(mix(position, growthRotatedPos, growthActive.mul(float(1.0).sub(growthProg))));

      // Apply spring force (weaker than stars for more fluid movement)
      const springTarget = mix(rotatedOriginal, scaledGrowthTarget, growthActive.mul(growthProg));
      const springStrength = float(1.0)
        .sub(this.uniforms.compute.handsTogetherActive.mul(0.7))
        .sub(growthActive.mul(growthProg.mul(1.5)));
      const springForce = applySpringForce(
        position,
        springTarget,
        springStrength.max(float(0.15)),
        this.uniforms.compute.deltaTime
      );
      position.addAssign(springForce);

      cloudPositionBuffer.element(idx).assign(position);
    })().compute(CLOUD_COUNT);

    // Store cloud state
    this.cloudCount = CLOUD_COUNT;

    // Create cloud sprite material
    const cloudMaterial = new THREE.SpriteNodeMaterial();
    cloudMaterial.transparent = true;
    cloudMaterial.depthWrite = false;
    cloudMaterial.blending = THREE.AdditiveBlending; // Efficient for overlapping particles

    const cloudPos = cloudPositionBuffer.toAttribute();
    const cloudColor = cloudColorBuffer.toAttribute();
    const cloudSize = cloudSizeBuffer.toAttribute();
    const cloudRotation = cloudRotationBuffer.toAttribute();

    cloudMaterial.positionNode = cloudPos;
    cloudMaterial.colorNode = vec4(cloudColor.x, cloudColor.y, cloudColor.z, float(1.0));
    cloudMaterial.scaleNode = cloudSize.mul(this.uniforms.visual.cloudSize);
    cloudMaterial.rotationNode = cloudRotation;

    // Use texture for soft cloud appearance
    if (this.cloudTexture) {
      const cloudTextureNode = texture(this.cloudTexture, uv());
      cloudMaterial.opacityNode = cloudTextureNode.a.mul(this.uniforms.visual.cloudOpacity);
    } else {
      cloudMaterial.opacityNode = this.uniforms.visual.cloudOpacity;
    }

    this.cloudPlane = new THREE.Sprite(cloudMaterial);
    this.cloudPlane.count = CLOUD_COUNT;
    this.cloudPlane.frustumCulled = false;
    this.cloudPlane.renderOrder = -1; // Render clouds before stars

    this.scene.add(this.cloudPlane);

    // Reset initialization flag so clouds get initialized on next update
    this.cloudInitialized = false;
  }

  /**
   * Creates leaf particles that form volumetric foliage around tree branches
   * Leaves concentrate at mid-to-upper canopy with clumping and gravity droop
   */
  createLeaves() {
    // Clean up old leaves
    if (this.leafPlane) {
      this.scene.remove(this.leafPlane);
      if (this.leafPlane.material) this.leafPlane.material.dispose();
    }

    const LEAF_COUNT = this.LEAF_COUNT;

    // Create leaf particle buffers
    const leafPositionBuffer = instancedArray(LEAF_COUNT, 'vec3');
    const leafOriginalPositionBuffer = instancedArray(LEAF_COUNT, 'vec3');
    const leafColorBuffer = instancedArray(LEAF_COUNT, 'vec3');
    const leafSizeBuffer = instancedArray(LEAF_COUNT, 'float');
    const leafGrowthTargetBuffer = instancedArray(LEAF_COUNT, 'vec3');

    // Initialize leaf particles with spiral galaxy distribution + tree growth targets
    this.leafInit = Fn(() => {
      const idx = instanceIndex;
      const seed = idx.toFloat().add(50000); // Offset seed from stars and clouds

      // Base galaxy distribution (same pattern as stars/clouds)
      const radius = hash(seed.add(1)).pow(0.5).mul(this.uniforms.galaxy.radius);
      const normalizedRadius = radius.div(this.uniforms.galaxy.radius);

      const armIndex = hash(seed.add(2)).mul(this.uniforms.galaxy.armCount).floor();
      const armAngle = armIndex.mul(6.28318).div(this.uniforms.galaxy.armCount);

      const spiralAngle = normalizedRadius.mul(this.uniforms.galaxy.spiralTightness).mul(6.28318);
      const angleOffset = hash(seed.add(3)).sub(0.5).mul(this.uniforms.galaxy.randomness);
      const radiusOffset = hash(seed.add(4)).sub(0.5).mul(this.uniforms.galaxy.armWidth);

      const angle = armAngle.add(spiralAngle).add(angleOffset);
      const offsetRadius = radius.add(radiusOffset);

      const x = cos(angle).mul(offsetRadius);
      const z = sin(angle).mul(offsetRadius);

      const thicknessFactor = float(1.0).sub(normalizedRadius).add(0.15);
      const y = hash(seed.add(5)).sub(0.5).mul(this.uniforms.galaxy.thickness).mul(thicknessFactor);

      const position = vec3(x, y, z);
      leafPositionBuffer.element(idx).assign(position);
      leafOriginalPositionBuffer.element(idx).assign(position);

      // ---- Calculate Leaf growth target position (树叶粒子世界树形态) ----
      // Leaves concentrate at mid-to-upper branches, forming volumetric clumps
      const growthSeed = hash(seed.add(10));

      // 🌟 树叶倾向于聚集在中上部 (使用 pow(0.8) 稍微降低重心)
      const treeHeightRatio = float(0.4).add(growthSeed.pow(float(0.8)).mul(0.6));
      const treeY = treeHeightRatio.mul(this.uniforms.galaxy.radius.mul(3.5));

      // 🌟 完美抛物线穹顶
      const canopyFactor = smoothstep(float(0.4), float(1.0), treeHeightRatio);
      const canopyCurve = float(1.0).sub(pow(canopyFactor.mul(2.0).sub(1.0), float(2.0)));
      const crownSpread = canopyCurve.mul(float(5.5));

      const branchArmIndex = hash(seed.add(11)).mul(this.uniforms.galaxy.armCount.mul(3.0)).floor();
      const branchBaseAngle = branchArmIndex.mul(6.28318).div(this.uniforms.galaxy.armCount.mul(3.0));

      // 团簇感 (受 canopyCurve 约束)
      const clumpNoise = sin(branchBaseAngle.mul(15.0)).mul(sin(treeHeightRatio.mul(25.0))).mul(float(1.2)).mul(canopyCurve);

      const growthSpiralAngle = treeHeightRatio.pow(float(1.5)).mul(6.28318).mul(float(1.5));
      const swayAngle = hash(seed.add(12)).sub(0.5).mul(1.2).mul(canopyFactor);
      const growthAngleFinal = branchBaseAngle.add(growthSpiralAngle).add(swayAngle);

      // 🌟 树叶特有的 3D "云团" 体积 (全部乘以 canopyCurve)
      const leafVolumeX = hash(seed.add(20)).sub(0.5).mul(float(2.0)).mul(canopyCurve);
      const leafVolumeY = hash(seed.add(21)).sub(0.5).mul(float(2.5)).mul(canopyCurve);
      const leafVolumeZ = hash(seed.add(22)).sub(0.5).mul(float(2.0)).mul(canopyCurve);

      const gravityDroop = crownSpread.mul(float(0.2));

      const growthRadiusFinal = crownSpread.add(clumpNoise).add(leafVolumeX).max(float(0.0)).mul(this.uniforms.galaxy.radius);

      const growthX = cos(growthAngleFinal).mul(growthRadiusFinal);
      const growthZ = sin(growthAngleFinal).mul(growthRadiusFinal).add(leafVolumeZ.mul(this.uniforms.galaxy.radius));

      const growthY = treeY
        .sub(this.uniforms.galaxy.radius.mul(1.5))
        .sub(gravityDroop.mul(this.uniforms.galaxy.radius))
        .add(leafVolumeY.mul(this.uniforms.galaxy.radius));

      const growthPosition = vec3(growthX, growthY, growthZ);
      leafGrowthTargetBuffer.element(idx).assign(growthPosition);

      // Leaf color: darker green at bottom, brighter yellow-green at top (sunlit)
      const leafColorDark = vec3(this.uniforms.visual.leafColorDark);
      const leafColorLight = vec3(this.uniforms.visual.leafColorLight);
      const leafColor = mix(leafColorDark, leafColorLight, treeHeightRatio);
      leafColorBuffer.element(idx).assign(leafColor);

      // Leaf size: smaller particles, varied
      const size = hash(seed.add(6)).mul(0.3).add(0.5);
      leafSizeBuffer.element(idx).assign(size);
    })().compute(LEAF_COUNT);

    // Update leaf particles (same physics as clouds)
    this.leafUpdate = Fn(() => {
      const idx = instanceIndex;
      const position = leafPositionBuffer.element(idx).toVar();
      const originalPos = leafOriginalPositionBuffer.element(idx);
      const growthTargetPos = leafGrowthTargetBuffer.element(idx);

      // Apply differential rotation (reduce during growth mode)
      const rotationSpeed = this.uniforms.compute.rotationSpeed.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive.mul(this.uniforms.compute.growthProgress.mul(1.0)))
      );
      const rotatedPos = applyDifferentialRotation(position, rotationSpeed, this.uniforms.compute.deltaTime);
      position.assign(rotatedPos);

      const rotatedOriginal = applyDifferentialRotation(originalPos, rotationSpeed, this.uniforms.compute.deltaTime);
      leafOriginalPositionBuffer.element(idx).assign(rotatedOriginal);

      // Apply mouse force (disabled during growth)
      const mouseForceActive = this.uniforms.compute.mouseActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const mouseForce = applyMouseForce(
        position, this.uniforms.compute.mouse, mouseForceActive,
        this.uniforms.compute.mouseForce, this.uniforms.compute.mouseRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(mouseForce);

      // Apply hands together attraction force
      const attractionActive = this.uniforms.compute.handsTogetherActive.mul(
        float(1.0).sub(this.uniforms.compute.growthModeActive)
      );
      const attractionForce = applyAttractionForce(
        position, this.uniforms.compute.handsCenter, attractionActive,
        this.uniforms.compute.attractionStrength.mul(1.2), this.uniforms.compute.attractionRadius,
        this.uniforms.compute.deltaTime
      );
      position.addAssign(attractionForce);

      // ---- Growth mode: transition to leaf canopy shape ----
      const growthActive = this.uniforms.compute.growthModeActive;
      const growthProg = this.uniforms.compute.growthProgress;

      // 🌟 只缩放树叶的 X 和 Z，保留 Y 轴高度
      const spreadScale = this.uniforms.compute.handSpread;
      const scaledGrowthTarget = vec3(
        growthTargetPos.x.mul(spreadScale),
        growthTargetPos.y,
        growthTargetPos.z.mul(spreadScale)
      );

      const smoothProgress = pow(growthProg, float(0.7));
      const transitionForce = scaledGrowthTarget.sub(position).mul(smoothProgress.mul(7.0)).mul(this.uniforms.compute.deltaTime);
      position.addAssign(transitionForce.mul(growthActive));

      const growthRotationSpeed = float(0.6).mul(growthActive).mul(float(1.0).sub(growthProg));
      const growthRotatedPos = applyDifferentialRotation(position, growthRotationSpeed, this.uniforms.compute.deltaTime);
      position.assign(mix(position, growthRotatedPos, growthActive.mul(float(1.0).sub(growthProg))));

      // Spring force (weaker for delicate leaf movement)
      const springTarget = mix(rotatedOriginal, scaledGrowthTarget, growthActive.mul(growthProg));
      const springStrength = float(0.8)
        .sub(this.uniforms.compute.handsTogetherActive.mul(0.5))
        .sub(growthActive.mul(growthProg.mul(1.2)));
      const springForce = applySpringForce(
        position, springTarget, springStrength.max(float(0.1)),
        this.uniforms.compute.deltaTime
      );
      position.addAssign(springForce);

      leafPositionBuffer.element(idx).assign(position);
    })().compute(LEAF_COUNT);

    // Create leaf sprite material
    const leafMaterial = new THREE.SpriteNodeMaterial();
    leafMaterial.transparent = true;
    leafMaterial.depthWrite = false;
    leafMaterial.blending = THREE.AdditiveBlending;

    const leafPos = leafPositionBuffer.toAttribute();
    const leafColor = leafColorBuffer.toAttribute();
    const leafSize = leafSizeBuffer.toAttribute();

    leafMaterial.positionNode = leafPos;
    leafMaterial.colorNode = vec4(leafColor.x, leafColor.y, leafColor.z, float(1.0));
    leafMaterial.scaleNode = leafSize.mul(this.uniforms.visual.leafSize).mul(this.uniforms.visual.particleSize);

    // Use cloud texture for soft leaf appearance
    if (this.cloudTexture) {
      const leafTextureNode = texture(this.cloudTexture, uv());
      leafMaterial.opacityNode = leafTextureNode.a.mul(this.uniforms.visual.leafOpacity);
    } else {
      leafMaterial.opacityNode = this.uniforms.visual.leafOpacity;
    }

    this.leafPlane = new THREE.Sprite(leafMaterial);
    this.leafPlane.count = LEAF_COUNT;
    this.leafPlane.frustumCulled = false;
    this.leafPlane.renderOrder = -2; // Render leaves behind clouds

    this.scene.add(this.leafPlane);

    this.leafInitialized = false;
  }

  /**
   * Updates star count and regenerates galaxy
   */
  updateStarCount(newCount) {
    this.COUNT = newCount;
    this.config.starCount = newCount;
    this.createGalaxySystem();
    this.initialized = false;
  }

  /**
   * Updates uniform values from config changes
   */
  updateUniforms(configUpdate) {
    // Galaxy structure uniforms
    if (configUpdate.galaxyRadius !== undefined)
      this.uniforms.galaxy.radius.value = configUpdate.galaxyRadius;
    if (configUpdate.galaxyThickness !== undefined)
      this.uniforms.galaxy.thickness.value = configUpdate.galaxyThickness;
    if (configUpdate.spiralTightness !== undefined)
      this.uniforms.galaxy.spiralTightness.value = configUpdate.spiralTightness;
    if (configUpdate.armCount !== undefined)
      this.uniforms.galaxy.armCount.value = configUpdate.armCount;
    if (configUpdate.armWidth !== undefined)
      this.uniforms.galaxy.armWidth.value = configUpdate.armWidth;
    if (configUpdate.randomness !== undefined)
      this.uniforms.galaxy.randomness.value = configUpdate.randomness;

    // Compute uniforms
    if (configUpdate.rotationSpeed !== undefined)
      this.uniforms.compute.rotationSpeed.value = configUpdate.rotationSpeed;
    if (configUpdate.mouseForce !== undefined)
      this.uniforms.compute.mouseForce.value = configUpdate.mouseForce;
    if (configUpdate.mouseRadius !== undefined)
      this.uniforms.compute.mouseRadius.value = configUpdate.mouseRadius;

    // Visual uniforms
    if (configUpdate.particleSize !== undefined)
      this.uniforms.visual.particleSize.value = configUpdate.particleSize;
    if (configUpdate.cloudSize !== undefined)
      this.uniforms.visual.cloudSize.value = configUpdate.cloudSize;
    if (configUpdate.cloudOpacity !== undefined)
      this.uniforms.visual.cloudOpacity.value = configUpdate.cloudOpacity;
    if (configUpdate.starBrightness !== undefined)
      this.uniforms.visual.starBrightness.value = configUpdate.starBrightness;
    if (configUpdate.denseStarColor !== undefined)
      this.uniforms.visual.denseStarColor.value.set(configUpdate.denseStarColor);
    if (configUpdate.sparseStarColor !== undefined)
      this.uniforms.visual.sparseStarColor.value.set(configUpdate.sparseStarColor);
    if (configUpdate.cloudTintColor !== undefined)
      this.uniforms.visual.cloudTintColor.value.set(configUpdate.cloudTintColor);
    // Tree growth colors
    if (configUpdate.growthCoreColor !== undefined)
      this.uniforms.visual.growthCoreColor.value.set(configUpdate.growthCoreColor);
    if (configUpdate.growthArmColor !== undefined)
      this.uniforms.visual.growthArmColor.value.set(configUpdate.growthArmColor);
    if (configUpdate.growthTipColor !== undefined)
      this.uniforms.visual.growthTipColor.value.set(configUpdate.growthTipColor);

    // Leaf visual
    if (configUpdate.leafColorDark !== undefined)
      this.uniforms.visual.leafColorDark.value.set(configUpdate.leafColorDark);
    if (configUpdate.leafColorLight !== undefined)
      this.uniforms.visual.leafColorLight.value.set(configUpdate.leafColorLight);
    if (configUpdate.leafOpacity !== undefined)
      this.uniforms.visual.leafOpacity.value = configUpdate.leafOpacity;
    if (configUpdate.leafSize !== undefined)
      this.uniforms.visual.leafSize.value = configUpdate.leafSize;

    // Config state
    if (configUpdate.cloudCount !== undefined) {
      this.config.cloudCount = configUpdate.cloudCount;
    }
    if (configUpdate.leafCount !== undefined) {
      this.config.leafCount = configUpdate.leafCount;
    }
  }

  /**
   * Updates hands together state from gesture detection
   */
  updateHandsTogether(active, center) {
    // 树锁定后不再响应凝聚手势，避免吸引力与树形态打架
    if (this.isTreeLocked) return;
    this.uniforms.compute.handsTogetherActive.value = active ? 1.0 : 0.0;
    this.uniforms.compute.handsCenter.value.set(center.x, center.y, center.z);
  }

  /**
   * Keyboard-triggered: starts growth mode immediately (no gesture needed).
   * Idempotent once tree is locked.
   */
  startGrowth() {
    if (this.isTreeLocked) return;
    this.growthMode = true;
    this.growthProgress = 0.0;
    this.uniforms.compute.growthModeActive.value = 1.0;
    this.uniforms.compute.handsTogetherActive.value = 0.0;
    this.isTreeLocked = true;
    console.log('🌱 Tree growth triggered via keyboard (T).');
  }

  /**
   * 🌌 将树形态散落回星系形态
   * Smoothly reverses the growth animation, particles spring back to galaxy.
   */
  scatterToGalaxy() {
    if (!this.isTreeLocked && !this.growthMode) return;
    if (this.scatteringMode) return; // already scattering
    this.scatteringMode = true;
    this.scatterProgress = this.growthProgress;
    console.log('🌌 Scattering tree back to galaxy...');
  }

  /**
   * Main update loop - runs compute shaders and updates uniforms
   */
  async update(renderer, deltaTime, mouse3D, mousePressed) {
    // Initialize stars on first frame
    if (!this.initialized) {
      await renderer.computeAsync(this.computeInit);
      this.initialized = true;
    }

    // Initialize clouds on first frame
    if (!this.cloudInitialized && this.cloudInit) {
      await renderer.computeAsync(this.cloudInit);
      this.cloudInitialized = true;
    }

    // Initialize leaves on first frame
    if (!this.leafInitialized && this.leafInit) {
      await renderer.computeAsync(this.leafInit);
      this.leafInitialized = true;
    }

    // Update compute uniforms
    this.uniforms.compute.time.value += deltaTime;
    this.uniforms.compute.deltaTime.value = deltaTime;
    this.uniforms.compute.mouse.value.copy(mouse3D);
    this.uniforms.compute.mouseActive.value = mousePressed ? 1.0 : 0.0;

    // ---- Growth mode detection ----
    const handsActive = this.uniforms.compute.handsTogetherActive.value > 0.5;

    // 触发检测（锁定后不再重复触发）
    if (handsActive && !this.growthMode && !this.isTreeLocked) {
      this.lastGatheringLevel += deltaTime * 1.0; // ~2 秒蓄力
      if (this.lastGatheringLevel >= this.gatheringThreshold) {
        this.growthMode = true;
        this.growthProgress = 0.0;
        this.uniforms.compute.growthModeActive.value = 1.0;
        this.isTreeLocked = true;
        console.log('🌱 Tree locked! One-shot growth initiated.');
      }
    } else if (!handsActive && !this.isTreeLocked) {
      this.lastGatheringLevel = Math.max(0, this.lastGatheringLevel - deltaTime * 0.5);
    }

    // ---- 🌌 散落动画：树 → 星系 (反向过渡) ----
    if (this.scatteringMode) {
      // 平滑递减 progress：树冠先缩、树干后散
      const scatterSpeed = this.scatterProgress > 0.6 ? 0.55 : (this.scatterProgress > 0.2 ? 0.35 : 0.2);
      this.scatterProgress -= deltaTime * scatterSpeed;

      if (this.scatterProgress <= 0.0) {
        this.scatterProgress = 0.0;
        this.scatteringMode = false;
        this.growthMode = false;
        this.isTreeLocked = false;
        this.uniforms.compute.growthModeActive.value = 0.0;
        this.uniforms.compute.growthProgress.value = 0.0;
        this.uniforms.compute.growthSpiralIntensity.value = 0.0;
        console.log('🌌 Scatter complete. Galaxy restored.');
      } else {
        this.growthProgress = this.scatterProgress;
        this.uniforms.compute.growthProgress.value = this.growthProgress;
        // 螺旋强度随进度递减
        this.uniforms.compute.growthSpiralIntensity.value = Math.pow(this.growthProgress, 0.5);
      }
    }

    // ---- 三阶段生长动画 (保持不变) ----
    if (this.growthMode && !this.scatteringMode) {
      let growthSpeed;
      if (this.growthProgress < 0.3) {
        growthSpeed = 0.6;
      } else if (this.growthProgress < 0.8) {
        growthSpeed = 1.2;
      } else {
        growthSpeed = 0.6;
      }

      this.growthProgress += deltaTime * growthSpeed;
      this.growthProgress = Math.min(this.growthProgress, 1.0);
      this.uniforms.compute.growthProgress.value = this.growthProgress;

      // 螺旋强度随进度变化
      if (this.growthProgress >= 1.0) {
        this.uniforms.compute.growthSpiralIntensity.value = 1.0;
      } else {
        const spiralIntensity = this.growthProgress < 0.8
          ? Math.pow(this.growthProgress, 1.0)
          : Math.pow(this.growthProgress, 0.5);
        this.uniforms.compute.growthSpiralIntensity.value = spiralIntensity;
      }
    }

    // Run physics computations
    await renderer.computeAsync(this.computeUpdate);

    if (this.cloudUpdate) {
      await renderer.computeAsync(this.cloudUpdate);
    }

    if (this.leafUpdate) {
      await renderer.computeAsync(this.leafUpdate);
    }
  }

  /**
   * Marks galaxy for regeneration on next update
   */
  regenerate() {
    this.initialized = false;
    this.cloudInitialized = false;
    this.leafInitialized = false;
  }
}
