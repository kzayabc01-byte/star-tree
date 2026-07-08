import { Pane } from 'tweakpane';

export class GalaxyUI {
  constructor(config, callbacks) {
    this.config = config;
    this.callbacks = callbacks;
    // 汉化主标题
    this.pane = new Pane({ title: '🌌 星系控制面板' });
    this.bloomPassNode = null;
    this.perfParams = { fps: 60 };

    this.setupUI();
  }

  setupUI() {
    // 只调用精简后的四大核心面板
    this.setupPerformanceFolder();
    this.setupAppearanceFolder();
    this.setupGalaxyFolder();
    this.setupEffectsFolder();
  }

  // 1. 性能监控
  setupPerformanceFolder() {
    const perfFolder = this.pane.addFolder({ title: '⚡ 性能参数' });
    perfFolder.addBinding(this.perfParams, 'fps', { readonly: true, label: '帧率 (FPS)' });

    perfFolder.addBinding(this.config, 'starCount', {
      min: 1000, max: 1000000, step: 1000, label: '粒子总数'
    }).on('change', () => this.callbacks.onStarCountChange(this.config.starCount));
  }

  // 2. 基本外观
  setupAppearanceFolder() {
    const appearanceFolder = this.pane.addFolder({ title: '✨ 视觉外观' });

    appearanceFolder.addBinding(this.config, 'particleSize', {
      min: 0.05, max: 0.5, step: 0.01, label: '粒子大小'
    }).on('change', () => this.callbacks.onUniformChange('particleSize', this.config.particleSize));

    appearanceFolder.addBinding(this.config, 'starBrightness', {
      min: 0.0, max: 2.0, step: 0.01, label: '星光亮度'
    }).on('change', () => this.callbacks.onUniformChange('starBrightness', this.config.starBrightness));

    // 新增：色彩风格预设下拉菜单
    appearanceFolder.addBinding(this.config, 'colorTheme', {
      label: '色彩风格',
      options: {
        '深空经典 (Classic)': 'classic',
        '冰冷深蓝 (Ice Blue)': 'ice',
        '赛博霓虹 (Cyberpunk)': 'cyber',
        '暗金余晖 (Golden)': 'golden'
      }
    }).on('change', (ev) => {
      if (this.callbacks.onThemeChange) {
        this.callbacks.onThemeChange(ev.value);
      }
    });
  }

  // 3. 星系大体结构
  setupGalaxyFolder() {
    const galaxyFolder = this.pane.addFolder({ title: '🌀 星系形态' });

    galaxyFolder.addBinding(this.config, 'rotationSpeed', {
      min: 0, max: 2, step: 0.01, label: '自转速度'
    }).on('change', () => this.callbacks.onUniformChange('rotationSpeed', this.config.rotationSpeed));

    galaxyFolder.addBinding(this.config, 'armCount', {
      min: 1, max: 4, step: 1, label: '旋臂数量'
    }).on('change', () => this.callbacks.onRegenerate());

    galaxyFolder.addBinding(this.config, 'galaxyRadius', {
      min: 5, max: 20, step: 0.01, label: '星系范围'
    }).on('change', () => this.callbacks.onRegenerate());
  }

  // 4. 合并光效与鼠标交互
  setupEffectsFolder() {
    const effectsFolder = this.pane.addFolder({ title: '🌟 特效与交互' });

    effectsFolder.addBinding(this.config, 'bloomStrength', {
      min: 0, max: 3, step: 0.01, label: '泛光强度'
    }).on('change', () => this.callbacks.onBloomChange('strength', this.config.bloomStrength));

    effectsFolder.addBinding(this.config, 'mouseForce', {
      min: 0, max: 10, step: 0.01, label: '鼠标排斥力'
    }).on('change', () => this.callbacks.onUniformChange('mouseForce', this.config.mouseForce));

    // 视觉分隔符
    effectsFolder.addBlade({ view: 'separator' });

    // 互动按钮：一键随机重构
    effectsFolder.addButton({
      title: '🎲 随机重构星系',
    }).on('click', () => {
      if (this.callbacks.onRandomizeGalaxy) {
        this.callbacks.onRandomizeGalaxy();
      }
    });
  }

  updateFPS(fps) {
    this.perfParams.fps = fps;
    this.pane.refresh();
  }

  setBloomNode(bloomNode) {
    this.bloomPassNode = bloomNode;
  }
}
