// Runs after warp-data.js (both deferred, execution order preserved).
document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  if (typeof WARP_FIELD_PNG === 'undefined') return;

  // ── Metadata (kept in sync with warp-data.js by generate_warp_data.py) ──
  const T = WARP_T, FH = WARP_FH, FW = WARP_FW, COLS = WARP_COLS, ROWS = WARP_ROWS, N_HEADS = WARP_N_HEADS;
  const FLOW_SCALE = WARP_FLOW_SCALE;
  // Brand palette — distinct, dark enough to read on light background
  const HEAD_COLORS = ['#A83E46','#D67C3B','#2A5058','#434A6B','#859E87','#6C96A3']
    .concat(['#A83E46','#D67C3B','#2A5058','#434A6B','#859E87','#6C96A3']); // repeat if >6 heads
  const HEAD_ALPHAS = WARP_HEAD_ALPHAS;

  // ── Canvas setup ──────────────────────────────────────────────────
  const canvas = document.getElementById('warp-canvas');
  if (!canvas) return;
  const ctx  = canvas.getContext('2d');
  const hint = document.getElementById('warp-hint');

  const LW = 960, LH = 360;  // logical drawing coordinate space

  // Resize canvas to exactly match its rendered CSS size × device pixel ratio.
  // Called once at init and again whenever the element resizes (including browser zoom).
  function resizeCanvas() {
    const dpr  = window.devicePixelRatio || 1;
    const cssW = canvas.parentElement.getBoundingClientRect().width || LW;
    const cssH = cssW * LH / LW;
    canvas.style.height = cssH + 'px';
    canvas.width  = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(canvas.parentElement);

  const TILE_W = Math.round(LH * WARP_FW / WARP_FH);  // aspect-ratio-correct tile width

  // ── Off-screen canvas: holds one decoded field frame ──────────────
  const offscreen = document.createElement('canvas');
  offscreen.width  = FW;
  offscreen.height = FH;
  const oc = offscreen.getContext('2d');

  // ── Mouse ──────────────────────────────────────────────────────────
  let mx = -9999, my = -9999, everHovered = false;
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    mx = (e.clientX - r.left) / r.width  * LW;
    my = (e.clientY - r.top)  / r.height * LH;
    if (!everHovered) { everHovered = true; if (hint) hint.style.opacity = '0'; }
  });
  canvas.addEventListener('mouseleave', () => { mx = my = -9999; });

  // ── Flow lookup ───────────────────────────────────────────────────
  const ST = ROWS * COLS * N_HEADS * 2;
  const SR = COLS * N_HEADS * 2;
  const SC = N_HEADS * 2;
  const FS = FLOW_SCALE / 127;
  let FLOW = null;

  function getFlow(fi, row, col, head) {
    const o = fi * ST + row * SR + col * SC + head * 2;
    return [FLOW[o] * FS, FLOW[o + 1] * FS];
  }

  // ── Drawing helpers ───────────────────────────────────────────────
  function drawArrow(x0, y0, x1, y1, col, alpha, lw) {
    const dx = x1 - x0, dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;
    const ang = Math.atan2(dy, dx);
    const hl  = Math.min(len * 0.45, 14);
    const ux  = dx / len, uy = dy / len;
    ctx.save();
    ctx.globalAlpha   = alpha;
    ctx.strokeStyle   = col;
    ctx.fillStyle     = col;
    ctx.lineWidth     = lw;
    ctx.lineCap       = 'butt';
    ctx.lineJoin      = 'miter';
    ctx.shadowBlur    = 0;
    ctx.shadowColor   = 'transparent';
    // Shaft stops at arrowhead base so the butt end doesn't bleed into the head
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1 - ux * hl * 0.85, y1 - uy * hl * 0.85);
    ctx.stroke();
    // Filled arrowhead triangle
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hl * Math.cos(ang - 0.65), y1 - hl * Math.sin(ang - 0.65));
    ctx.lineTo(x1 - hl * Math.cos(ang + 0.65), y1 - hl * Math.sin(ang + 0.65));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function glowDot(x, y, col, alpha, r, blur) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle   = col;
    ctx.shadowColor = col;
    ctx.shadowBlur  = blur;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // ── Field sprite: Image + hidden canvas to blit individual frames ─
  let spriteCanvas = null;   // set after PNG loads

  function drawField(fi, alpha) {
    if (!spriteCanvas) return;
    // Correct linear blend: draw fA at full opacity, then fB fades in on top
    const fA = fi, fB = (fi + 1) % T;
    oc.clearRect(0, 0, FW, FH);
    oc.globalAlpha = 1;
    oc.drawImage(spriteCanvas, 0, fA * FH, FW, FH, 0, 0, FW, FH);
    oc.globalAlpha = alpha;
    oc.drawImage(spriteCanvas, 0, fB * FH, FW, FH, 0, 0, FW, FH);
    oc.globalAlpha = 1;

    ctx.save();
    ctx.globalAlpha = 0.88;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    for (let tx = 0; tx * TILE_W < LW + TILE_W; tx++) {
      ctx.drawImage(offscreen, tx * TILE_W, 0, TILE_W, LH);
    }
    ctx.restore();
  }

  // ── Local mouse-hover dim (softens field beneath arrows) ──────────
  function drawMouseDim(mx, my, radius) {
    if (mx < -100) return;   // cursor not over canvas
    const grad = ctx.createRadialGradient(mx, my, 0, mx, my, radius);
    grad.addColorStop(0, 'rgba(255,255,255,0.78)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, LW, LH);
    ctx.restore();
  }

  // ── Main loop ─────────────────────────────────────────────────────
  const HOVER_CELLS = 1.2, FPS = 6;
  let t0 = null;

  function frame(ts) {
    if (!t0) t0 = ts;
    const t      = (ts - t0) * 0.001;
    const frameT = t * FPS;
    const fi     = Math.floor(frameT) % T;
    const alpha  = frameT % 1;

    const cellW = LW / (COLS + 1);
    const cellH = LH / (ROWS + 1);
    const hR    = HOVER_CELLS * Math.min(cellW, cellH);

    // 0 — Sync transform to actual canvas size (stays correct across zoom/resize)
    const sx = canvas.width  / LW;   // physical px per logical unit
    const sy = canvas.height / LH;
    ctx.setTransform(sx, 0, 0, sy, 0, 0);
    ctx.globalAlpha = 1;
    ctx.shadowBlur  = 0;
    ctx.shadowColor = 'transparent';
    ctx.fillStyle   = '#ffffff';
    ctx.fillRect(0, 0, LW, LH);

    // 1 — Field (drawn at reduced opacity over white base)
    drawField(fi, alpha);

    // 2 — Re-sync transform, then locally dim background near cursor
    ctx.setTransform(sx, 0, 0, sy, 0, 0);
    ctx.globalAlpha = 1;
    ctx.shadowBlur  = 0;
    ctx.shadowColor = 'transparent';
    drawMouseDim(mx, my, hR * 2.2);

    // Re-sync after dim overlay
    ctx.setTransform(sx, 0, 0, sy, 0, 0);
    ctx.globalAlpha = 1;
    ctx.shadowBlur  = 0;
    ctx.shadowColor = 'transparent';

    // 3 — Arrows (skipped until flow data is ready)
    if (FLOW) {
      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
          const px = Math.round((col + 1) * cellW);
          const py = Math.round((row + 1) * cellH);
          const bl    = Math.max(0, 1 - Math.hypot(px - mx, py - my) / hR);
          const bloom = bl * bl;
          const hA = Math.min(bloom * 1.50, 1.00) * 0.95;

          if (hA > 0.01) {
            // Heads 1..N_HEADS-1 = PCA components (drawn first, behind mean)
            for (let h = 1; h < N_HEADS; h++) {
              const [dx, dy] = getFlow(fi, row, col, h);
              const ex = px + dx * (TILE_W/2);
              const ey = py + dy * (LH /2);
              const ha = HEAD_ALPHAS[h] * hA * 0.85;
              drawArrow(px, py, ex, ey, HEAD_COLORS[(h-1) % HEAD_COLORS.length], ha, 3.0);
              if (bloom > 0.5 && HEAD_ALPHAS[h] > 0.5) {
                glowDot(ex, ey, HEAD_COLORS[(h-1) % HEAD_COLORS.length], (bloom-0.5)*2*HEAD_ALPHAS[h]*0.85, 2.5, 0);
              }
            }

            // Head 0 = mean of all original heads — drawn on top, fades in with bloom
            const [adx, ady] = getFlow(fi, row, col, 0);
            drawArrow(px, py, px + adx*(TILE_W/2), py + ady*(LH/2), '#111111', Math.min(hA * 1.1, 0.88), 2.8);
          }
        }
      }
    }

    requestAnimationFrame(frame);
  }

  // ── Load PNG from object URL (no fetch, works on static sites) ────
  FLOW = WARP_FLOW;   // already decoded by warp-data.js

  const img = new Image();
  img.onload = () => {
    spriteCanvas = document.createElement('canvas');
    spriteCanvas.width  = FW;
    spriteCanvas.height = T * FH;
    spriteCanvas.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(img.src);
    requestAnimationFrame(frame);
  };
  img.src = WARP_FIELD_PNG;
});
