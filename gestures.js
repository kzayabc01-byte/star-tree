/**
 * Galaxy Gestures - 优化版 (包含互斥锁、防抖与宽容判定)
 */

export class GalaxyGestures {
  constructor(callbacks = {}) {
    this.cb = callbacks;
    this.handActive = false;
    this.handRot = { x: 0, y: 0 };
    this.handSpread = 1.0;
    this.handDepth = 1.0;
    this.initialized = false;
    this.camera = null;

    // ---- 核心优化：防抖与缓冲计时器 ----
    this._lastX = null;
    this._lastSw = 0;

    // 握拳状态
    this._fistOn = false;
    this._lastFistTime = 0;

    // 双手合十状态
    this.handsTogether = false;
    this.handsCenter = { x: 0, y: 0, z: 0 };
    this._handsTogetherLastSeen = 0; // 缓冲防断

    // 互斥锁：当处于施法状态时，锁定相机的缩放和旋转
    this.isPoseLocked = false;

    // ✌️ 比耶状态
    this.vPoseActive = false;
    this._vPoseLastSeen = 0;
  }

  async init(videoElement) {
    if (this.initialized) return true;
    if (typeof window.Hands === 'undefined' || typeof window.Camera === 'undefined') {
      this._fallback('手势库未加载');
      return false;
    }

    let ok = false;
    const timeoutId = setTimeout(() => {
      if (!ok) this._fallback('摄像头超时 (5s)');
    }, 5000);

    try {
      const hands = new window.Hands({
        locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
      });
      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      hands.onResults((results) => this._onResults(results));

      this.camera = new window.Camera(videoElement, {
        onFrame: async () => { await hands.send({ image: videoElement }); },
        width: 640, height: 480
      });

      await this.camera.start();
      clearTimeout(timeoutId);
      ok = true;
      this.initialized = true;
      return true;
    } catch (e) {
      clearTimeout(timeoutId);
      this._fallback('摄像头不可用');
      return false;
    }
  }

  _onResults(results) {
    const now = performance.now();

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      if (this.handActive) {
        this.handActive = false;
        this.handsTogether = false;
        this.isPoseLocked = false;
        this.cb.onMode?.('mouse');
        this.cb.onHandsTogether?.(false, { x: 0, y: 0, z: 0 });
      }
      return;
    }

    if (!this.handActive) {
      this.handActive = true;
      this.cb.onMode?.('gesture');
    }

    const numHands = results.multiHandLandmarks.length;

    // ==========================================
    // 1. 双手合十检测 (星云聚集) - 加入防抖与互斥锁
    // ==========================================
    if (numHands === 2) {
      const [lm1, lm2] = results.multiHandLandmarks;
      const h1c = { x: (lm1[0].x+lm1[9].x)/2, y: (lm1[0].y+lm1[9].y)/2, z: (lm1[0].z+lm1[9].z)/2 };
      const h2c = { x: (lm2[0].x+lm2[9].x)/2, y: (lm2[0].y+lm2[9].y)/2, z: (lm2[0].z+lm2[9].z)/2 };
      const dist = Math.hypot(h1c.x-h2c.x, h1c.y-h2c.y, h1c.z-h2c.z);

      const isCurrentlyTogether = dist < 0.20; // 放宽判定距离

      if (isCurrentlyTogether) {
        this._handsTogetherLastSeen = now;
        if (!this.handsTogether) {
          this.handsTogether = true;
          this.isPoseLocked = true; // 开启互斥锁
          this.cb.onStatus?.('🔒 双手合十锁定 - 停止晃动');
        }
        this.handsCenter = {
          x: ((h1c.x+h2c.x)/2 - 0.5) * 2,
          y: ((h1c.y+h2c.y)/2 - 0.5) * 1.5,
          z: ((h1c.z+h2c.z)/2) * 10
        };
        this.cb.onHandsTogether?.(true, this.handsCenter);
      } else {
        // 缓冲 400ms，防止手抖突然断开
        if (this.handsTogether && now - this._handsTogetherLastSeen > 400) {
          this.handsTogether = false;
          this.isPoseLocked = false; // 解除互斥锁
          this.cb.onHandsTogether?.(false, { x: 0, y: 0, z: 0 });
          this.cb.onStatus?.('🔓 解除锁定');
        }
      }

      return;
    }

    // ==========================================
    // 2. 单手模式
    // ==========================================
    const lm = results.multiHandLandmarks[0];

    // --- A. 握拳检测 (视图切换) ---
    let fistDist = 0;
    [8, 12, 16, 20].forEach(t => fistDist += Math.hypot(lm[t].x-lm[0].x, lm[t].y-lm[0].y, lm[t].z-lm[0].z));
    fistDist /= 4;
    const isFist = fistDist < 0.30; // 稍微放宽一点点握拳判定

    if (isFist) {
      if (!this._fistOn && now - this._lastFistTime > 1500) {
        this.cb.onFist?.(); // 触发切换
        this._lastFistTime = now;
      }
      this._fistOn = true;
      this.isPoseLocked = true; // 握拳时锁住屏幕不让动
    } else if (fistDist > 0.45) {
      this._fistOn = false;
    }

    // --- B. ✌️ 比耶检测 (触发星系快速自转) ---
    const isVPose = this._detectVPose(lm);
    if (isVPose) {
      this._vPoseLastSeen = now;
      if (!this.vPoseActive) {
        this.vPoseActive = true;
        this.isPoseLocked = true; // 开启互斥锁，防止缩放和乱晃
        this.cb.onVPose?.(true);
        this.cb.onStatus?.('🌀 ✌️ 锁定 - 星系快速环绕');
      }
    } else {
      // 缓冲 500ms：手指稍微弯一下不会立刻断开
      if (this.vPoseActive && now - this._vPoseLastSeen > 500) {
        this.vPoseActive = false;
        this.isPoseLocked = false; // 解除互斥锁
        this.cb.onVPose?.(false);
        this.cb.onStatus?.('🔓 解除环绕');
      }
    }

    // ==========================================
    // 3. 基础移动与缩放 (互斥锁逻辑同步修改)
    // ==========================================

    // 【核心修复】如果当前正在结印（握拳、✌️比耶、合十），直接跳过旋转和缩放的计算！
    if (!this.vPoseActive && !this.handsTogether && !this._fistOn) {
      this.isPoseLocked = false;

      // -- 手部位置 → 旋转角度 --
      this.handRot.y = ((lm[0].x + lm[9].x) / 2 - 0.5) * 1.8;
      this.handRot.x = ((lm[0].y + lm[9].y) / 2 - 0.5) * 1.2;

      // -- 五指张开 → 扩散 --
      const tips = [4, 8, 12, 16, 20];
      let accDist = 0;
      tips.forEach(t => {
        accDist += Math.hypot(lm[t].x - lm[0].x, lm[t].y - lm[0].y, lm[t].z - lm[0].z);
      });
      const openMetric = accDist / 5.0;
      const targetSpread = this._clamp(this._mapLinear(openMetric, 0.25, 0.65, 0.35, 2.2), 0.3, 2.5);

      // 死区：变化大于 0.05 才更新，防止微微抽搐
      if (Math.abs(targetSpread - this.handSpread) > 0.05) {
        this.handSpread += (targetSpread - this.handSpread) * 0.5; // 平滑过渡
        this.cb.onHandSpread?.(this.handSpread);
      }

      // -- 手掌前后移动 (宽度) → 缩放 --
      const palmWidth = Math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y);
      const targetDepth = this._clamp(this._mapLinear(palmWidth, 0.08, 0.28, 0.5, 2.2), 0.4, 2.5);

      // 死区：前后移动幅度太小时忽略，修复一动手指就放大缩小的 Bug
      if (Math.abs(targetDepth - this.handDepth) > 0.06) {
        this.handDepth += (targetDepth - this.handDepth) * 0.4; // 平滑过渡
        this.cb.onHandDepth?.(this.handDepth);
      }
    }

    // --- C. 水平挥手切换视角 (挥手动作不受互斥锁限制，因为它是一瞬间的) ---
    const cx = lm[9].x;
    if (this._lastX !== null && now - this._lastSw > 800 && Math.abs(cx - this._lastX) > 0.08) {
      cx > this._lastX ? this.cb.onSwipeLeft?.() : this.cb.onSwipeRight?.();
      this._lastSw = now;
      this.cb.onStatus?.('👋 挥手切换视角');
    }
    this._lastX = cx;
  }

  // ==========================================
  // 工具函数
  // ==========================================

  _distance3(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
  }

  _mapLinear(x, a1, a2, b1, b2) {
    return b1 + (x - a1) * (b2 - b1) / (a2 - a1);
  }

  _clamp(x, min, max) {
    return Math.max(min, Math.min(max, x));
  }

  _fallback(reason) {
    this.handActive = false;
    this.cb.onFallback?.(reason);
    this.cb.onMode?.('mouse');
  }

  // ✌️ 比耶判定：食指中指伸直，无名指小拇指弯曲
  _detectVPose(lm) {
    const idxExt = this._fingerExtended(lm, 8, 6);
    const midExt = this._fingerExtended(lm, 12, 10);
    const ringCur = this._fingerCurled(lm, 16, 14);
    const pinCur = this._fingerCurled(lm, 20, 18);
    return idxExt && midExt && ringCur && pinCur;
  }

  _fingerExtended(lm, tip, pip) {
    return lm[tip].y < lm[pip].y;
  }

  _fingerCurled(lm, tip, pip) {
    return lm[tip].y > lm[pip].y;
  }
}
