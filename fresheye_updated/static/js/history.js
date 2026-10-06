/* FreshEye AI — history.js
   Fetches this user's saved conversations (both Ask FreshEye and AI
   Agent) and renders them as expandable cards, newest first. */

const historyList = $("#historyList");
let allSessions = [];
let currentFilter = "all";

const TYPE_LABEL = { chat: "Ask FreshEye", agent: "AI Agent" };
const TYPE_PAGE = { chat: "/chat", agent: "/agent" };

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  });
}

/** Points this browser's saved conversation id (for that chat surface)
 * at the chosen session, then sends the user to the matching page -
 * which on load notices the id already has a transcript (see
 * fetchSavedSession() in common.js) and rehydrates instead of starting
 * a new conversation. This is the entire "continue" mechanism: no
 * separate resume endpoint needed, just pointing the same id the normal
 * chat flow already uses at an existing conversation. */
function continueSession(session) {
  localStorage.setItem(`fresheye_session_${session.chat_type}`, session.client_session_id);
  window.location.href = TYPE_PAGE[session.chat_type] || "/";
}

function renderSessions() {
  const sessions = currentFilter === "all"
    ? allSessions
    : allSessions.filter(s => s.chat_type === currentFilter);

  if (sessions.length === 0) {
    historyList.innerHTML = `<p class="history-empty">No conversations here yet.</p>`;
    return;
  }

  historyList.innerHTML = "";
  sessions.forEach(session => {
    const card = document.createElement("div");
    card.className = "history-card";

    const head = document.createElement("button");
    head.className = "history-card-head";
    head.innerHTML = `
      <span class="history-type-tag history-type-${session.chat_type}">${TYPE_LABEL[session.chat_type] || session.chat_type}</span>
      <span class="history-title">${session.title}</span>
      <span class="history-meta">${session.messages.length} messages · ${formatDate(session.started_at)}</span>
      <span class="history-chevron">▾</span>
    `;

    const body = document.createElement("div");
    body.className = "history-card-body";
    body.hidden = true;
    session.messages.forEach(m => {
      const msg = document.createElement("div");
      msg.className = `chat-msg chat-msg-${m.role === "bot" ? "bot" : "user"} history-msg`;
      const p = document.createElement("p");
      p.textContent = m.content;
      msg.appendChild(p);
      body.appendChild(msg);
    });

    const continueBtn = document.createElement("button");
    continueBtn.type = "button";
    continueBtn.className = "btn btn-primary btn-sm history-continue-btn";
    continueBtn.textContent = "Continue this conversation →";
    continueBtn.addEventListener("click", e => {
      e.stopPropagation(); // don't also toggle the card's expand/collapse
      continueSession(session);
    });
    body.appendChild(continueBtn);

    head.addEventListener("click", () => {
      body.hidden = !body.hidden;
      head.classList.toggle("open", !body.hidden);
    });

    card.appendChild(head);
    card.appendChild(body);
    historyList.appendChild(card);
  });
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history/sessions");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not load history.");
    allSessions = data.sessions || [];
    renderSessions();
  } catch (err) {
    historyList.innerHTML = `<p class="history-empty">Couldn't load your history — ${err.message}</p>`;
  }
}

$$(".history-filter").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".history-filter").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.type;
    renderSessions();
  });
});

loadHistory();
