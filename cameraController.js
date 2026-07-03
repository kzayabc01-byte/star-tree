/**
 * Camera Controller - Spherical-orbit camera with POI navigation
 *
 * Replaces Three.js OrbitControls with a custom controller that supports:
 * - Spherical coordinate orbit (theta / phi / radius) around a target point
 * - Mouse drag to rotate, scroll to zoom
 * - Predefined Points of Interest (POIs) in the galaxy
 * - Smooth exponential-ease transitions between views
 * - Keyboard shortcuts for navigation
 * - External rotation input (from gesture tracking)
 * - GALAXY (overview) and FOCUS (close-up) dual view modes
 *
 * All transitions go through a shared lerp so manual input and automatic
 * POI switching never fight — the latest input always wins smoothly.
 */

export const VIEW_STATE = Object.freeze({ GALAXY: 'galaxy', FOCUS: 'focus' });

export class CameraController {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLCanvasElement} domElement - the renderer canvas
   * @param {Object} config - galaxy config (galaxyRadius, armCount)
   */
  constructor(camera, domElement, config) {
    this.camera = camera;
    this.domElement = domElement;
    this.config = config;

    // ---- Points of Interest ----
    /** @type {Array<{name:string, target:{x:number,y:number,z:number}, radius:number, theta:number, phi:number}>} */
    this.pois = [];
    this.poiIndex = 0;
    this.viewState = VIEW_STATE.GALAXY;

    // ---- Current spherical state (smoothly lerped each frame) ----
    this._target = { x: 0, y: -2, z: 0 };
    this._theta = 0;
    this._phi = 0.882;    // ~50 deg elevation
    this._radius = 22;

    // ---- Target spherical state (what we're lerping toward) ----
    this._tTarget = { x: 0, y: -2, z: 0 };
    this._tTheta = 0;
    this._tPhi = 0.882;
    this._tRadius = 22;

    // ---- Smoothing factor (higher = snappier) ----
    this.smoothFactor = 0.06;

    // ---- Mouse drag ----
    this._dragging = false;
    this._prevMouse = { x: 0, y: 0 };
    this._rotSensitivity = 0.005;

    // ---- Zoom ----
    this._zoom = 1.0;
    this._minZoom = 0.2;
    this._maxZoom = 4.0;

    // ---- External rotation (gestures) ----
    this._extRotX = 0;
    this._extRotY = 0;
    this._useExtRot = false;

    // ---- Distance limits ----
    this._minRadius = 1.5;
    this._maxRadius = 40;

    // ---- Generate POIs ----
    this._generatePois();

    // ---- Set initial state ----
    const poi = this.pois[0];
    this._target.x = poi.target.x;
    this._target.y = poi.target.y;
    this._target.z = poi.target.z;
    this._tTarget.x = poi.target.x;
    this._tTarget.y = poi.target.y;
    this._tTarget.z = poi.target.z;
    this._theta = poi.theta;
    this._phi = poi.phi;
    this._radius = poi.radius;
    this._tTheta = poi.theta;
    this._tPhi = poi.phi;
    this._tRadius = poi.radius;

    // ---- Bind & install listeners ----
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);

    this.domElement.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mouseup', this._onMouseUp);
    window.addEventListener('mousemove', this._onMouseMove);
    this.domElement.addEventListener('wheel', this._onWheel, { passive: false });
    window.addEventListener('keydown', this._onKeyDown);

    // Initial camera placement (instant, no lerp)
    this._applyCamera(false);
  }

  // ==========================================================================
  //  PUBLIC API
  // ==========================================================================

  /** Set external rotation source (from hand gestures). Overrides mouse orbit. */
  setExternalRotation(x, y) {
    this._extRotX = x;
    this._extRotY = y;
    this._useExtRot = true;
  }

  /** Clear external rotation so mouse orbit resumes. */
  clearExternalRotation() {
    this._useExtRot = false;
  }

  /** Switch to the next POI. */
  nextPoi() {
    this.poiIndex = (this.poiIndex + 1) % this.pois.length;
    this._applyPoi();
  }

  /** Switch to the previous POI. */
  previousPoi() {
    this.poiIndex = (this.poiIndex - 1 + this.pois.length) % this.pois.length;
    this._applyPoi();
  }

  /** Toggle between GALAXY overview and FOCUS close-up. */
  toggleView() {
    this.viewState = this.viewState === VIEW_STATE.GALAXY
      ? VIEW_STATE.FOCUS
      : VIEW_STATE.GALAXY;
    this._applyPoi();
  }

  /** Reset to first POI, galaxy view, default zoom. */
  resetView() {
    this.poiIndex = 0;
    this.viewState = VIEW_STATE.GALAXY;
    this._zoom = 1.0;
    this._extRotX = 0;
    this._extRotY = 0;
    this._applyPoi();
  }

  /** Toggle fullscreen. */
  toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen();
    }
  }

  /** Rebuild POIs after galaxy structure changes (arm count, radius, etc.). */
  regeneratePois() {
    this._generatePois();
    this.poiIndex = Math.min(this.poiIndex, this.pois.length - 1);
    this._applyPoi();
  }

  /** @returns {string} current POI name */
  getPoiName() {
    return this.pois[this.poiIndex]
      ? this.pois[this.poiIndex].name
      : '星系全景';
  }

  /** @returns {string} current view state display name */
  getViewStateName() {
    return this.viewState === VIEW_STATE.GALAXY ? '星系全景' : '近距离特写';
  }

  /** @returns {string} 'gesture' | 'mouse' based on external rotation state */
  getModeName() {
    return this._useExtRot ? 'gesture' : 'mouse';
  }

  /**
   * Main update — call once per frame.
   * Lerps current spherical state toward target, then applies to camera.
   */
  update() {
    const sp = this.smoothFactor;

    // Lerp target position
    this._target.x += (this._tTarget.x - this._target.x) * sp;
    this._target.y += (this._tTarget.y - this._target.y) * sp;
    this._target.z += (this._tTarget.z - this._target.z) * sp;

    // Lerp spherical coords
    this._theta += (this._tTheta - this._theta) * sp;
    this._phi   += (this._tPhi   - this._phi)   * sp;
    this._radius += (this._tRadius - this._radius) * sp;

    this._applyCamera(true);
  }

  /** Clean up event listeners. */
  dispose() {
    this.domElement.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mouseup', this._onMouseUp);
    window.removeEventListener('mousemove', this._onMouseMove);
    this.domElement.removeEventListener('wheel', this._onWheel);
    window.removeEventListener('keydown', this._onKeyDown);
  }

  // ==========================================================================
  //  INTERNALS
  // ==========================================================================

  /** Build POI list from current galaxy config. */
  _generatePois() {
    const R = this.config.galaxyRadius || 13;
    const arms = this.config.armCount || 2;

    const list = [];

    // 0 — Galaxy Overview (default, camera above & back)
    list.push({
      name: '星系全景',
      target: { x: 0, y: -2, z: 0 },
      radius: 22,
      theta: 0,
      phi: 0.882
    });

    // 1 — Galactic Core
    list.push({
      name: '星系核心',
      target: { x: 0, y: 0, z: 0 },
      radius: 8,
      theta: 0,
      phi: 0.7
    });

    // 2+ — Spiral arm viewpoints
    for (let i = 0; i < arms; i++) {
      const angle = (i / arms) * Math.PI * 2;
      const tx = Math.cos(angle) * R * 0.55;
      const tz = Math.sin(angle) * R * 0.55;
      // camera position is offset outward from the arm tip
      const cx = Math.cos(angle) * R * 0.75;
      const cz = Math.sin(angle) * R * 0.75;
      const dx = cx - tx;
      const dz = cz - tz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      list.push({
        name: `旋臂 ${String.fromCharCode(65 + i)}`,
        target: { x: tx, y: -1, z: tz },
        radius: Math.sqrt(dist * dist + 5 * 5), // elevated above arm
        theta: Math.atan2(dx, dz),
        phi: Math.atan2(dist, 5)
      });
    }

    // Bird's-eye view (top-down)
    list.push({
      name: '俯瞰星系',
      target: { x: 0, y: 0, z: 0 },
      radius: R * 0.5,
      theta: 0,
      phi: 0.05  // nearly straight down
    });

    // Edge-on view
    list.push({
      name: '侧视星系',
      target: { x: 0, y: 0, z: 0 },
      radius: R * 1.2,
      theta: 0,
      phi: Math.PI / 2  // at the equatorial plane
    });

    this.pois = list;
    if (this.poiIndex >= list.length) this.poiIndex = 0;
  }

  /** Set _tTarget / _tTheta / _tPhi / _tRadius from current POI + view state. */
  _applyPoi() {
    const poi = this.pois[this.poiIndex];
    if (!poi) return;

    this._tTarget.x = poi.target.x;
    this._tTarget.y = poi.target.y;
    this._tTarget.z = poi.target.z;

    if (this.viewState === VIEW_STATE.FOCUS) {
      // Close-up: tighter radius, keep current orbit angle if user was rotating
      this._tRadius = Math.max(2.5, poi.radius * 0.35 * this._zoom);
      // In focus mode keep user's current theta/phi for continuity
    } else {
      this._tRadius = poi.radius * this._zoom;
      this._tTheta = poi.theta;
      this._tPhi = poi.phi;
    }
  }

  /**
   * Convert spherical → Cartesian and apply to the real camera.
   * @param {boolean} smooth - if true, lerp is already applied; if false, snap
   */
  _applyCamera(smooth) {
    const t = smooth ? this._target : this._tTarget;
    const theta = smooth ? this._theta : this._tTheta;
    const phi   = smooth ? this._phi   : this._tPhi;
    const radius = smooth ? this._radius : this._tRadius;

    // Clamp phi away from poles
    const safePhi = Math.max(0.05, Math.min(Math.PI - 0.05, phi));

    const sinPhi = Math.sin(safePhi);
    const x = t.x + radius * sinPhi * Math.sin(theta);
    const y = t.y + radius * Math.cos(safePhi);
    const z = t.z + radius * sinPhi * Math.cos(theta);

    this.camera.position.set(x, y, z);
    this.camera.lookAt(t.x, t.y, t.z);
  }

  // ---- Mouse events ----

  _onMouseDown(e) {
    // Only respond to left click on the canvas itself
    if (e.button !== 0) return;
    this._dragging = true;
    this._prevMouse.x = e.clientX;
    this._prevMouse.y = e.clientY;
  }

  _onMouseUp() {
    this._dragging = false;
  }

  _onMouseMove(e) {
    // Apply external rotation (gesture) if active — it overrides mouse
    if (this._useExtRot) {
      this._tTheta = -this._extRotY * 2.5;
      this._tPhi   = 0.882 + this._extRotX * 1.2;
      this._tPhi   = Math.max(0.1, Math.min(Math.PI - 0.1, this._tPhi));
      return;
    }

    if (!this._dragging) return;

    const dx = e.clientX - this._prevMouse.x;
    const dy = e.clientY - this._prevMouse.y;

    this._tTheta -= dx * this._rotSensitivity;
    this._tPhi   -= dy * this._rotSensitivity;

    // Clamp phi to avoid gimbal lock
    this._tPhi = Math.max(0.1, Math.min(Math.PI - 0.1, this._tPhi));

    this._prevMouse.x = e.clientX;
    this._prevMouse.y = e.clientY;
  }

  _onWheel(e) {
    e.preventDefault();
    this._zoom += e.deltaY * 0.001;
    this._zoom = Math.max(this._minZoom, Math.min(this._maxZoom, this._zoom));

    // Re-apply POI with new zoom
    this._applyPoi();
  }

  // ---- Keyboard ----

  _onKeyDown(e) {
    // Ignore if user is typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
      case 'ArrowLeft':  case 'a': case 'A': this.previousPoi();  break;
      case 'ArrowRight': case 'd': case 'D': this.nextPoi();      break;
      case ' ':
        e.preventDefault();
        this.toggleView();
        break;
      case 'r': case 'R': this.resetView();         break;
      case 'f': case 'F': this.toggleFullscreen();  break;
    }
  }
}
