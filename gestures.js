/**
 * Galaxy Gestures - MediaPipe Hand Tracking Module
 *
 * Provides hand gesture control for the galaxy simulation:
 * - Hand position  → camera orbit rotation
 * - Wave left/right → switch point of interest
 * - Fist gesture    → toggle galaxy / focus view mode
 * - Auto-fallback to mouse mode on timeout or error (5s)
 *
 * Requires MediaPipe Hands + Camera utils loaded via CDN in index.html.
 * Globals used: window.Hands, window.Camera
 */

export class GalaxyGestures {
  /**
   * @param {Object} callbacks
   * @param {(msg:string)=>void} [callbacks.onStatus]    - status message update
   * @param {(mode:string)=>void} [callbacks.onMode]     - 'gesture' | 'mouse'
   * @param {()=>void}            [callbacks.onSwipeLeft]  - wave left detected
   * @param {()=>void}            [callbacks.onSwipeRight] - wave right detected
   * @param {()=>void}            [callbacks.onFist]       - fist gesture detected
 * @param {(msg:string)=>void} [callbacks.onFallback]  - fallback to mouse, with reason
 * @param {(active:boolean, center:{x:number,y:number,z:number})=>void} [callbacks.onHandsTogether] - hands together gesture
 * @param {(spread:number)=>void} [callbacks.onHandSpread] - hand spread (five fingers open/close)
 * @param {(depth:number)=>void} [callbacks.onHandDepth] - hand depth (palm width for zoom)
 * @param {(active:boolean)=>void} [callbacks.onSwordPose] - two-finger sword gesture
   */
  constructor(callbacks = {}) {
    this.cb = callbacks;

    // Public state
    this.handActive = false;
    this.handRot = { x: 0, y: 0 };
    this.handSpread = 1.0;  // 五指张开程度（控制扩散）
    this.handDepth = 1.0;   // 手掌深度（控制缩放）
    this.initialized = false;
    this.camera = null;

    // Internal swipe detection
    this._lastX = null;
    this._lastSw = 0;

    // Internal fist detection
    this._fistOn = false;
    this._lastFist = 0;

    // Hands together detection
    this.handsTogether = false;
    this.handsCenter = { x: 0, y: 0, z: 0 };
    this._lastHandsTogether = false;

    // Sword pose detection
    this.swordPoseActive = false;
  }

  /**
   * Start the webcam + MediaPipe Hands pipeline.
   * @param {HTMLVideoElement} videoElement - already in DOM, autoplay playsinline
   * @returns {Promise<boolean>} true if started successfully
   */
  async init(videoElement) {
    if (this.initialized) return true;

    // Guard: MediaPipe globals not loaded
    if (typeof window.Hands === 'undefined' || typeof window.Camera === 'undefined') {
      this._fallback('手势库未加载');
      return false;
    }

    const Hands = window.Hands;
    const Camera = window.Camera;

    let ok = false;
    const timeoutId = setTimeout(() => {
      if (!ok) this._fallback('摄像头超时 (5s)');
    }, 5000);

    try {
      const hands = new Hands({
        locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
      });

      hands.setOptions({
        maxNumHands: 2,  // 改为双手追踪以支持双手合拢手势
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      hands.onResults((results) => this._onResults(results));

      this.camera = new Camera(videoElement, {
        onFrame: async () => {
          await hands.send({ image: videoElement });
        },
        width: 640,
        height: 480
      });

      await this.camera.start();
      clearTimeout(timeoutId);
      ok = true;
      this.initialized = true;

      if (this.cb.onStatus) this.cb.onStatus('就绪 (请伸出手掌)');
      return true;
    } catch (e) {
      clearTimeout(timeoutId);
      console.warn('Gesture init error:', e);
      this._fallback('摄像头不可用');
      return false;
    }
  }

  // ---- internals ----

  _fallback(reason) {
    this.handActive = false;
    if (this.cb.onFallback) this.cb.onFallback(reason);
    if (this.cb.onMode) this.cb.onMode('mouse');
  }

  _onResults(results) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      if (this.handActive) {
        this.handActive = false;
        this.handsTogether = false;
        if (this.cb.onMode) this.cb.onMode('mouse');
        // 停止星云聚集
        if (this.cb.onHandsTogether) {
          this.cb.onHandsTogether(false, { x: 0, y: 0, z: 0 });
        }
        // 重置 spread 和 depth
        if (this.cb.onHandSpread) this.cb.onHandSpread(1.0);
        if (this.cb.onHandDepth) this.cb.onHandDepth(1.0);
        if (this.swordPoseActive && this.cb.onSwordPose) this.cb.onSwordPose(false);
        this.swordPoseActive = false;
      }
      return;
    }

    if (!this.handActive) {
      this.handActive = true;
      if (this.cb.onMode) this.cb.onMode('gesture');
    }

    const now = performance.now();
    const numHands = results.multiHandLandmarks.length;

    // ---- 双手合拢检测 ----
    if (numHands === 2) {
      const lm1 = results.multiHandLandmarks[0];
      const lm2 = results.multiHandLandmarks[1];

      // 计算两手的中心点（使用手腕和中指MCP的平均）
      const hand1Center = {
        x: (lm1[0].x + lm1[9].x) / 2,
        y: (lm1[0].y + lm1[9].y) / 2,
        z: (lm1[0].z + lm1[9].z) / 2
      };
      const hand2Center = {
        x: (lm2[0].x + lm2[9].x) / 2,
        y: (lm2[0].y + lm2[9].y) / 2,
        z: (lm2[0].z + lm2[9].z) / 2
      };

      // 计算双手之间的距离
      const dx = hand1Center.x - hand2Center.x;
      const dy = hand1Center.y - hand2Center.y;
      const dz = hand1Center.z - hand2Center.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      // 合拢阈值：距离小于 0.18（两手靠近）- 调整为更稳定
      const togetherThreshold = 0.18;
      const isHandsTogether = distance < togetherThreshold;

      // 计算双手合拢的中心点（转换到星系坐标系）
      if (isHandsTogether) {
        const centerX = ((hand1Center.x + hand2Center.x) / 2 - 0.5) * 2;  // [-1, 1] -> [-20, 20]
        const centerY = ((hand1Center.y + hand2Center.y) / 2 - 0.5) * 1.5; // [-1, 1] -> [-15, 15]
        const centerZ = ((hand1Center.z + hand2Center.z) / 2) * 10;        // z 值较小，放大到合理范围

        this.handsCenter = { x: centerX, y: centerY, z: centerZ };
        this.handsTogether = true;

        // 触发双手合拢回调（星云聚集）
        if (this.cb.onHandsTogether && !this._lastHandsTogether) {
          this.cb.onHandsTogether(true, this.handsCenter);
          if (this.cb.onStatus) this.cb.onStatus('双手合拢 - 星云聚集');
        }
      } else {
        this.handsTogether = false;
        if (this.cb.onHandsTogether && this._lastHandsTogether) {
          this.cb.onHandsTogether(false, { x: 0, y: 0, z: 0 });
          if (this.cb.onStatus) this.cb.onStatus('就绪 (请伸出手掌)');
        }
      }
      this._lastHandsTogether = isHandsTogether;

      // 双手模式下不处理单手手势（挥手、握拳）
      if (this.swordPoseActive && this.cb.onSwordPose) this.cb.onSwordPose(false);
      this.swordPoseActive = false;
      return;
    }

    // ---- 单手模式：手势控制 ----
    const lm = results.multiHandLandmarks[0];

    // ---- hand position → rotation ----
    // Average of wrist (0) and middle-finger MCP (9) gives stable hand center
    // 降低旋转灵敏度（从 2.5/1.5 改为 1.8/1.2），更稳定
    this.handRot.y = ((lm[0].x + lm[9].x) / 2 - 0.5) * 1.8;
    this.handRot.x = ((lm[0].y + lm[9].y) / 2 - 0.5) * 1.2;

    // ---- 🌟 NEW: Five fingers spread → galaxy spread ----
    // 计算所有指尖到手腕的距离（借鉴 gem4）
    const tips = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky tips
    let accumulationDist = 0;
    tips.forEach(tip => {
      const dx = lm[tip].x - lm[0].x;
      const dy = lm[tip].y - lm[0].y;
      const dz = lm[tip].z - lm[0].z;
      accumulationDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
    });
    const openMetric = accumulationDist / 5.0;

    // 优化映射区间，降低灵敏度（0.25 → 0.65 映射到 0.35 → 2.2）
    const mappedSpread = this._mapLinear(openMetric, 0.25, 0.65, 0.35, 2.2);
    this.handSpread = this._clamp(mappedSpread, 0.3, 2.5);

    // 触发回调
    if (this.cb.onHandSpread) {
      this.cb.onHandSpread(this.handSpread);
    }

    // ---- 🌟 NEW: Palm width → Z-axis depth (视差缩放) ----
    // 使用掌骨间距（index MCP → pinky MCP）代替不稳定的绝对 Z 坐标
    const indexMCP = lm[5];
    const pinkyMCP = lm[17];
    const palmWidth = Math.sqrt(
      Math.pow(indexMCP.x - pinkyMCP.x, 2) +
      Math.pow(indexMCP.y - pinkyMCP.y, 2)
    );

    // 掌心离镜头越近，二维投射宽度越大 → 视差距离缩放
    const mappedDepth = this._mapLinear(palmWidth, 0.08, 0.28, 0.5, 2.2);
    this.handDepth = this._clamp(mappedDepth, 0.4, 2.5);

    // 触发回调
    if (this.cb.onHandDepth) {
      this.cb.onHandDepth(this.handDepth);
    }

    // ---- two-finger sword pose detection ----
    const swordPose = this._detectSwordPose(lm);
    if (swordPose !== this.swordPoseActive) {
      this.swordPoseActive = swordPose;
      if (this.cb.onSwordPose) this.cb.onSwordPose(swordPose);
      if (this.cb.onStatus && swordPose) {
        this.cb.onStatus('双指剑诀 - 剑阵启动');
      }
    }

    // ---- horizontal wave → switch POI ----
    const cx = lm[9].x; // middle-finger MCP x
    if (this._lastX !== null && now - this._lastSw > 600 && Math.abs(cx - this._lastX) > 0.07) {
      if (cx > this._lastX) {
        if (this.cb.onSwipeLeft) this.cb.onSwipeLeft();   // hand moved right → previous POI
      } else {
        if (this.cb.onSwipeRight) this.cb.onSwipeRight(); // hand moved left → next POI
      }
      this._lastSw = now;
    }
    this._lastX = cx;

    // ---- fist detection → toggle view ----
    // Average distance of 4 fingertip landmarks from wrist
    let d = 0;
    [8, 12, 16, 20].forEach(t => {
      const dx = lm[t].x - lm[0].x;
      const dy = lm[t].y - lm[0].y;
      const dz = lm[t].z - lm[0].z;
      d += Math.sqrt(dx * dx + dy * dy + dz * dz);
    });
    d /= 4;

    if (d < 0.26 && !this._fistOn && now - this._lastFist > 1200) {
      if (this.cb.onFist) this.cb.onFist();
      this._fistOn = true;
      this._lastFist = now;
    } else if (d > 0.4) {
      this._fistOn = false;
    }
  }

  // ---- Helper methods (借鉴 Three.js MathUtils) ----
  _mapLinear(value, a1, a2, b1, b2) {
    return b1 + (value - a1) * (b2 - b1) / (a2 - a1);
  }

  _clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  _distance3(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  _fingerExtended(lm, tip, pip, mcp) {
    const wrist = lm[0];
    const tipDist = this._distance3(lm[tip], wrist);
    const pipDist = this._distance3(lm[pip], wrist);
    const mcpDist = this._distance3(lm[mcp], wrist);
    return tipDist > pipDist * 1.08 && pipDist > mcpDist * 0.98;
  }

  _fingerCurled(lm, tip, pip, mcp) {
    const wrist = lm[0];
    const tipDist = this._distance3(lm[tip], wrist);
    const pipDist = this._distance3(lm[pip], wrist);
    const mcpDist = this._distance3(lm[mcp], wrist);
    return tipDist < pipDist * 1.08 || tipDist < mcpDist * 1.02;
  }

  _detectSwordPose(lm) {
    const indexExtended = this._fingerExtended(lm, 8, 6, 5);
    const middleExtended = this._fingerExtended(lm, 12, 10, 9);
    const ringCurled = this._fingerCurled(lm, 16, 14, 13);
    const pinkyCurled = this._fingerCurled(lm, 20, 18, 17);
    const fingertipGap = this._distance3(lm[8], lm[12]);

    return indexExtended &&
      middleExtended &&
      ringCurled &&
      pinkyCurled &&
      fingertipGap > 0.02 &&
      fingertipGap < 0.18;
  }
}
