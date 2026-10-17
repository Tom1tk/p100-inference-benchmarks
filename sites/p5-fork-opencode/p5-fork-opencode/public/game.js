/* p5-fork-opencode — ice fishing mini-game
 * Self-contained: drives #ice-canvas and #ice-score.
 *
 * Mechanics:
 *   - the hook hangs from the rod and slowly sinks while idle
 *   - holding the mouse button (or touch) reels it up; releasing drops it
 *   - fish swim left → right at varying depths and speeds
 *   - touching the hook catches a fish; it then follows the hook
 *   - a caught fish that reaches the ice is collected (+1 score)
 *   - with nothing on the line, the hook clamps at the ice until released
 */
(function () {
  'use strict';

  var canvas = document.getElementById('ice-canvas');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var scoreEl = document.getElementById('ice-score');

  /* ---------- fixed logical size, DPR-sharp backing store ---------- */
  var W = 520, H = 340;
  var DPR = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  canvas.width = W * DPR;
  canvas.height = H * DPR;
  ctx.scale(DPR, DPR);

  /* ---------- geometry ---------- */
  var ICE_H = 52;            // height of the ice band
  var holeX = W / 2;         // the hole, and the hook's x position
  var topY = ICE_H;          // hook clamps here (top of the water / the ice)
  var bottomY = H - 12;      // hook clamps here (lake floor)
  var rodTip = { x: holeX - 24, y: ICE_H - 8 };

  /* ---------- tuning ---------- */
  var FALL_SPEED = 34;       // px/s — the slow default sink
  var REEL_SPEED = 135;      // px/s while the player holds
  var FISH_COUNT = 6;

  var FISH_TYPES = [
    { e: '🐟', w: 3 },
    { e: '🐠', w: 3 },
    { e: '🐡', w: 2 },
    { e: '🦈', w: 1 }
  ];

  function rand(a, b) { return a + Math.random() * (b - a); }
  function pickType() {
    var total = 0, i;
    for (i = 0; i < FISH_TYPES.length; i++) total += FISH_TYPES[i].w;
    var r = Math.random() * total;
    for (i = 0; i < FISH_TYPES.length; i++) {
      r -= FISH_TYPES[i].w;
      if (r <= 0) return FISH_TYPES[i];
    }
    return FISH_TYPES[0];
  }
  function newDepth() { return rand(topY + 28, bottomY - 14); }
  function newSpeed(e) { return e === '🦈' ? rand(55, 85) : rand(22, 65); }
  function spawnFish(atLeft) {
    var t = pickType();
    return {
      e: t.e,
      s: t.e === '🦈' ? 30 : rand(17, 25),   // glyph size in px
      x: atLeft ? -34 : rand(0, W),
      y: newDepth(),
      v: newSpeed(t.e),
      phase: Math.random() * Math.PI * 2,
      caught: false,
      slot: 0
    };
  }

  var fish = [];
  for (var i = 0; i < FISH_COUNT; i++) fish.push(spawnFish(false));

  var bubbles = [];
  for (var b = 0; b < 12; b++) {
    bubbles.push({ x: rand(0, W), y: rand(topY, H), r: rand(1, 3), v: rand(9, 22) });
  }

  var hook = { y: topY + 54 };
  var attached = [];          // caught fish currently on the line
  var holding = false;        // the two hook states are mutually exclusive
  var score = 0;

  function setScore(n) {
    score = n;
    if (!scoreEl) return;
    scoreEl.textContent = String(score);
    scoreEl.classList.remove('bump');
    void scoreEl.offsetWidth; // restart the bump animation
    scoreEl.classList.add('bump');
  }

  /* ---------- input: hold to reel, release to drop ----------
   * Pointer events unify mouse and touch.
   */
  function press(e) { holding = true; if (e && e.cancelable) e.preventDefault(); }
  function release() { holding = false; }
  canvas.addEventListener('pointerdown', press);
  window.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', release);
  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });

  /* ---------- update ---------- */
  function step(dt) {
    // hook: falling by default, reeling while held — never both
    hook.y += (holding ? -REEL_SPEED : FALL_SPEED) * dt;
    hook.y = Math.max(topY, Math.min(bottomY, hook.y));

    for (var i = fish.length - 1; i >= 0; i--) {
      var f = fish[i];

      if (f.caught) {
        // follow the hook, thrashing, stacked in slots
        f.phase += dt * 13;
        f.x = holeX + Math.sin(f.phase) * 2;
        f.y = hook.y + 12 + f.slot * 16;
        if (f.y - f.s / 2 <= topY + 4) {
          // reached the ice layer → collected
          setScore(score + 1);
          attached.splice(attached.indexOf(f), 1);
          fish.splice(i, 1);
          fish.push(spawnFish(true)); // restock the pond from the left
        }
        continue;
      }

      // free fish: swim left → right with a gentle depth wobble
      f.x += f.v * dt;
      f.phase += dt * (2.5 + f.v / 28);
      f.y += Math.sin(f.phase) * 6 * dt;
      f.y = Math.max(topY + 22, Math.min(bottomY - 8, f.y));

      if (f.x > W + 34) { // swam off the right edge → back around the left
        f.x = -34;
        f.y = newDepth();
        f.v = newSpeed(f.e);
      }

      // hook collision
      var dx = f.x - holeX;
      var dy = f.y - hook.y;
      var rr = f.s / 2 + 8;
      if (dx * dx + dy * dy < rr * rr) {
        f.caught = true;
        attached.push(f);
      }
    }

    for (var k = 0; k < attached.length; k++) attached[k].slot = k;

    // ambient bubbles rising
    for (var j = 0; j < bubbles.length; j++) {
      var bb = bubbles[j];
      bb.y -= bb.v * dt;
      if (bb.y < topY + 6) { bb.y = H - 6; bb.x = rand(0, W); }
    }
  }

  /* ---------- draw ---------- */
  function drawHook(x, y) {
    ctx.save();
    ctx.translate(x, y);
    ctx.strokeStyle = '#e9f2ff';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(0, -8);
    ctx.lineTo(0, 3);
    ctx.arc(4, 3, 4, Math.PI, -Math.PI / 4, true);
    ctx.stroke();
    ctx.beginPath(); // barb
    ctx.moveTo(6.8, 0.2);
    ctx.lineTo(5.4, -1.4);
    ctx.stroke();
    ctx.restore();
  }

  function draw(now) {
    // water
    var wg = ctx.createLinearGradient(0, topY, 0, H);
    wg.addColorStop(0, '#14406f');
    wg.addColorStop(0.7, '#0a2348');
    wg.addColorStop(1, '#061831');
    ctx.fillStyle = wg;
    ctx.fillRect(0, topY - 1, W, H - topY + 1);

    // drifting light shafts
    ctx.save();
    ctx.globalAlpha = 0.05;
    ctx.fillStyle = '#bfe3ff';
    for (var s = 0; s < 3; s++) {
      var sx = ((s * 190 + now * 0.012) % (W + 140)) - 70;
      ctx.beginPath();
      ctx.moveTo(sx, topY);
      ctx.lineTo(sx + 46, topY);
      ctx.lineTo(sx - 30, H);
      ctx.lineTo(sx - 76, H);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();

    // bubbles
    ctx.fillStyle = 'rgba(214, 236, 255, 0.22)';
    for (var j = 0; j < bubbles.length; j++) {
      var bb = bubbles[j];
      ctx.beginPath();
      ctx.arc(bb.x, bb.y, bb.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // fish (flipped so the glyph faces right, their direction of travel)
    for (var i = 0; i < fish.length; i++) {
      var f = fish[i];
      ctx.save();
      ctx.translate(f.x, f.y);
      ctx.scale(-1, 1);
      if (f.caught) ctx.rotate(Math.sin(f.phase) * 0.3); // thrashing
      ctx.font = f.s + 'px serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(f.e, 0, 0);
      ctx.restore();
    }

    // fishing line: rod tip → hook (slight slack while sinking, taut while reeling)
    ctx.strokeStyle = 'rgba(226, 240, 255, 0.8)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(rodTip.x, rodTip.y);
    if (holding) {
      ctx.lineTo(holeX, hook.y - 8);
    } else {
      ctx.quadraticCurveTo(holeX - 8, (rodTip.y + hook.y) / 2, holeX, hook.y - 8);
    }
    ctx.stroke();

    drawHook(holeX, hook.y);

    // ice band
    var ig = ctx.createLinearGradient(0, 0, 0, ICE_H);
    ig.addColorStop(0, '#eaf6ff');
    ig.addColorStop(1, '#a8cff0');
    ctx.fillStyle = ig;
    ctx.fillRect(0, 0, W, ICE_H);

    // snow drifts
    ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.beginPath();
    ctx.ellipse(70, ICE_H - 6, 90, 10, 0, 0, Math.PI * 2);
    ctx.ellipse(W - 60, ICE_H - 8, 110, 12, 0, 0, Math.PI * 2);
    ctx.fill();

    // the hole
    ctx.fillStyle = '#050f1e';
    ctx.beginPath();
    ctx.ellipse(holeX, ICE_H - 5, 27, 9, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // the fisher (emoji), rod aimed at the hole
    ctx.save();
    ctx.translate(rodTip.x - 10, ICE_H - 15);
    ctx.rotate(-0.6);
    ctx.font = '26px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🎣', 0, 0);
    ctx.restore();

    // subtle state indicator on the ice
    if (holding) {
      ctx.fillStyle = 'rgba(198, 255, 0, 0.9)';
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText('● reeling', W - 10, 16);
    }
  }

  /* ---------- main loop ---------- */
  var last = performance.now();
  function frame(now) {
    var dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    step(dt);
    draw(now);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
