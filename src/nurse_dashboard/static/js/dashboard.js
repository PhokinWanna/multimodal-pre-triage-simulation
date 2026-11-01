// Pre-Triage Dashboard — small, dependency-free script.
// 1) keeps waiting times current, 2) refreshes the queue when the server reports a change,
// 3) plays an alert sound for NEW urgent cases / nurse-assistance alerts (after the nurse turns it on).
(function () {
  "use strict";

  // ---------- waiting-time ticker ----------
  function updateWaits(root) {
    var now = Date.now();
    (root || document).querySelectorAll(".waited[data-since]").forEach(function (el) {
      var since = Date.parse(el.getAttribute("data-since"));
      if (!isNaN(since)) el.textContent = Math.max(0, Math.floor((now - since) / 60000)) + " min";
    });
  }
  updateWaits();
  setInterval(updateWaits, 30000);

  var queue = document.getElementById("queue");
  if (!queue) return; // the rest is for the queue page only

  // ---------- alert sound ----------
  var SOUND_KEY = "triage.sound";
  var soundOn = false;
  try { soundOn = localStorage.getItem(SOUND_KEY) === "1"; } catch (e) { /* storage blocked */ }
  var audioCtx = null;
  var toggle = document.getElementById("sound-toggle");

  function renderToggle() {
    if (!toggle) return;
    toggle.textContent = soundOn ? "Alert sound: on" : "Turn on alert sound";
    toggle.setAttribute("aria-pressed", soundOn ? "true" : "false");
    toggle.classList.toggle("btn-on", soundOn);
  }

  function beep() {
    if (!soundOn) return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      [0, 0.35].forEach(function (offset) {
        var osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.value = offset ? 660 : 880;
        gain.gain.setValueAtTime(0.0001, audioCtx.currentTime + offset);
        gain.gain.exponentialRampToValueAtTime(0.3, audioCtx.currentTime + offset + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + offset + 0.3);
        osc.connect(gain).connect(audioCtx.destination);
        osc.start(audioCtx.currentTime + offset);
        osc.stop(audioCtx.currentTime + offset + 0.32);
      });
    } catch (e) { /* audio not available */ }
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      soundOn = !soundOn;
      try { localStorage.setItem(SOUND_KEY, soundOn ? "1" : "0"); } catch (e) {}
      renderToggle();
      if (soundOn) beep(); // browsers only allow sound after a click: this also unlocks audio
    });
    renderToggle();
  }

  function alertKeys() {
    var inner = document.getElementById("queue-inner");
    return new Set(((inner && inner.getAttribute("data-alert-keys")) || "").split(/\s+/).filter(Boolean));
  }
  var known = alertKeys();

  // ---------- refresh ----------
  var refreshing = false;
  function refresh() {
    if (refreshing) return;
    refreshing = true;
    fetch(queue.getAttribute("data-fragment-url"), { headers: { "X-Requested-With": "fetch" }, credentials: "same-origin" })
      .then(function (r) {
        if (r.redirected || r.status === 401 || r.status === 303) { window.location = "/login"; return null; }
        return r.ok ? r.text() : null;
      })
      .then(function (html) {
        if (html === null || html === undefined) return;
        queue.innerHTML = html;
        updateWaits(queue);
        var now = alertKeys(), isNew = false;
        now.forEach(function (k) { if (!known.has(k)) isNew = true; });
        known = now;
        if (isNew) beep();
      })
      .catch(function () {})
      .finally(function () { refreshing = false; });
  }

  // ---------- live connection (Server-Sent Events, with polling fallback) ----------
  var status = document.getElementById("live-status");
  function setLive(on) {
    if (!status) return;
    status.textContent = on ? "Live" : "Reconnecting…";
    status.classList.toggle("live-on", on);
    status.classList.toggle("live-off", !on);
  }

  if (window.EventSource) {
    var es = new EventSource("/events");
    es.addEventListener("hello", function () { setLive(true); });
    es.addEventListener("changed", function () { setLive(true); refresh(); });
    es.onerror = function () { setLive(false); };
  } else {
    setLive(true);
  }
  // Safety net: refresh every 30 s even if the live connection silently drops.
  setInterval(refresh, 30000);
})();
