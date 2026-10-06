/* FreshEye AI — common.js
   Shared across every page: tiny DOM helpers + the backend health check
   that drives the header status chip. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Sets/clears the `hidden` attribute directly via the DOM attribute API.
 * SVG elements don't reliably reflect `.hidden` as a JS property in every
 * browser (assigning el.hidden = true silently does nothing on <svg> in
 * some Chromium versions) - toggle icons in this app are all <svg>, so
 * this is the only form that's actually guaranteed to work everywhere. */
function setHidden(el, hide) {
  if (hide) el.setAttribute("hidden", "");
  else el.removeAttribute("hidden");
}

// Kept in sync with Config.ALLOWED_IMAGE_EXTENSIONS in config.py. Client-side
// validation is a UX convenience (instant feedback at selection time) - the
// backend re-validates every file regardless, so this list falling slightly
// out of sync is never a security issue, just a UX one worth keeping tidy.
const ALLOWED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "bmp"];

function getFileExtension(filename) {
  const parts = (filename || "").split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

/** Splits a FileList/array of File objects into {valid, invalid}, checking
 * both the file extension and (when the browser provides one) the MIME
 * type, so a renamed non-image file doesn't slip through on extension
 * alone. Each invalid entry is {file, reason}. */
function validateImageFiles(files) {
  const valid = [];
  const invalid = [];
  for (const file of files) {
    const ext = getFileExtension(file.name);
    const extOk = ALLOWED_IMAGE_EXTENSIONS.includes(ext);
    const mimeOk = !file.type || file.type.startsWith("image/");
    if (extOk && mimeOk) {
      valid.push(file);
    } else if (!extOk) {
      invalid.push({ file, reason: `unsupported format ".${ext || "unknown"}"` });
    } else {
      invalid.push({ file, reason: "not an image file" });
    }
  }
  return { valid, invalid };
}

/** Generates a v4-style UUID for identifying a chat conversation.
 * `crypto.randomUUID()` is the modern way to do this, but browsers
 * restrict it to "secure contexts" (HTTPS, or http://localhost) - it's
 * simply undefined when the app is opened over plain http:// on a LAN
 * IP (e.g. http://192.168.x.x:5000), which is a normal way to reach a
 * Flask dev server from another device. This isn't used for anything
 * security-sensitive (just a client-side id to group chat turns into
 * one saved conversation), so a Math.random-based fallback is fine. */
function generateUUID() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
    const hex = [...bytes].map(b => b.toString(16).padStart(2, "0"));
    return `${hex.slice(0,4).join("")}-${hex.slice(4,6).join("")}-${hex.slice(6,8).join("")}-${hex.slice(8,10).join("")}-${hex.slice(10,16).join("")}`;
  }
  // Last-resort fallback (very old browsers): not cryptographically
  // random, but sufficient for a client-side conversation grouping id.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Returns (creating if needed) a per-tab-type conversation id, persisted
 * in localStorage so repeated turns on the same page append to the same
 * saved ChatSession server-side instead of starting a new one every
 * message. `kind` is "chat" or "agent" - each gets its own id. Call
 * newConversationId(kind) to explicitly start a fresh saved thread
 * (e.g. when the AI Agent page starts a new image inspection). */
function getConversationId(kind) {
  const key = `fresheye_session_${kind}`;
  let id = localStorage.getItem(key);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(key, id);
  }
  return id;
}
function newConversationId(kind) {
  const id = generateUUID();
  localStorage.setItem(`fresheye_session_${kind}`, id);
  return id;
}

/** Adds an animated "thinking" bubble to a chat window (three bouncing
 * dots, in place of a static "Thinking…" label) and returns the element
 * so the caller can .remove() it once the real response arrives. Shared
 * by chat.js and agent.js so both surfaces animate identically. */
function addTypingBubble(container) {
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-bot typing-msg";
  div.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

/** Looks up a saved conversation for this browser tab's session id, if
 * one exists on the server - this is how "continue from history" works:
 * history.js points a tab's saved localStorage id at an existing
 * conversation and redirects here; this fetch is what lets the chat
 * page notice there's already a transcript and rehydrate instead of
 * starting fresh. Returns the session dict ({chat_type, title, context,
 * messages: [...]}) or null if this is a brand-new conversation. */
async function fetchSavedSession(kind) {
  const sessionId = getConversationId(kind);
  try {
    const res = await fetch(`/api/history/session?type=${kind}&session_id=${sessionId}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.session || null;
  } catch {
    return null; // fine to fail quiet here - worst case the page just starts fresh
  }
}

async function checkHealth() {
  const dot = $("#statusDot"), text = $("#statusText");
  if (!dot || !text) return;
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.yolo_weights_found && data.llm_config_valid) {
      dot.classList.add("ok");
      text.textContent = "All systems ready";
    } else {
      dot.classList.add("bad");
      const issues = [];
      if (!data.yolo_weights_found) issues.push("model weights missing");
      if (!data.llm_config_valid) issues.push("LLM key missing");
      text.textContent = issues.join(" · ");
    }
  } catch {
    dot.classList.add("bad");
    text.textContent = "Backend unreachable";
  }
}
checkHealth();

/* ---------------------------------------------------------------------
   AUTH FORM UX — password show/hide toggles + name-field digit filter
   Runs on every page (this file is loaded site-wide), but does nothing
   unless the relevant elements are actually present, so it's a no-op
   outside login/register.
   --------------------------------------------------------------------- */
$$(".toggle-password").forEach(btn => {
  const input = document.getElementById(btn.dataset.target);
  if (!input) return;
  const eyeIcon = btn.querySelector(".icon-eye");
  const eyeOffIcon = btn.querySelector(".icon-eye-off");
  btn.addEventListener("click", () => {
    const showing = input.type === "password";
    input.type = showing ? "text" : "password";
    btn.setAttribute("aria-pressed", String(showing));
    btn.setAttribute("aria-label", showing ? "Hide password" : "Show password");
    setHidden(eyeIcon, showing);
    setHidden(eyeOffIcon, !showing);
  });
});

// The name field only accepts letters, spaces, apostrophes and hyphens -
// strip anything else as the person types rather than only rejecting it
// on submit, so the mistake is obvious immediately instead of after a
// failed form post. Actual enforcement still happens server-side too.
const nameField = document.getElementById("name");
if (nameField) {
  nameField.addEventListener("input", () => {
    const cleaned = nameField.value.replace(/[^A-Za-z .'-]/g, "");
    if (cleaned !== nameField.value) nameField.value = cleaned;
  });
}

/* ---------------------------------------------------------------------
   RESPONSIVE NAV — hamburger toggle for the header drawer on narrow
   screens (see .nav-drawer / .nav-toggle in style.css). No-op on pages
   without a nav (there aren't any, but kept defensive).
   --------------------------------------------------------------------- */
(function initNavToggle() {
  const toggle = $("#navToggle");
  const drawer = $("#navDrawer");
  if (!toggle || !drawer) return;

  const burgerIcon = toggle.querySelector(".icon-burger");
  const closeIcon = toggle.querySelector(".icon-close");

  function setOpen(open) {
    drawer.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    setHidden(burgerIcon, open);
    setHidden(closeIcon, !open);
  }

  toggle.addEventListener("click", () => setOpen(!drawer.classList.contains("open")));
  // Tapping a link inside the drawer should close it, not leave it open
  // while the new page loads underneath.
  drawer.querySelectorAll("a").forEach(link => link.addEventListener("click", () => setOpen(false)));
  // Clicking outside the open drawer closes it too.
  document.addEventListener("click", e => {
    if (drawer.classList.contains("open") && !drawer.contains(e.target) && e.target !== toggle && !toggle.contains(e.target)) {
      setOpen(false);
    }
  });
})();
