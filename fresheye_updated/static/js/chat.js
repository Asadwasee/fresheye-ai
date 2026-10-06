/* FreshEye AI — chat.js
   The standalone "Ask FreshEye" chat page. Answers are grounded in the
   knowledge base on the backend, but raw file paths / sources are never
   surfaced in the UI - only the plain-language answer is shown.

   Short-term memory: this page keeps a running `chatHistory` array and
   sends the recent turns with every request, so follow-up questions like
   "is it safe to eat?" are understood in context instead of the assistant
   asking "what's 'it'?" every time.

   Continuing from /history: on load, this page checks whether its saved
   conversation id (see getConversationId() in common.js) already has a
   transcript on the server - if so (either a reload of an ongoing chat,
   or the user just clicked "Continue" on a past conversation in
   /history), it rehydrates the window and chatHistory from that instead
   of showing the default greeting. */

const chatWindow = $("#chatWindow");
const chatForm = $("#chatForm");
const chatInput = $("#chatInput");

let chatHistory = []; // [{role: "user"|"bot", content: string}, ...]

function addChatMessage(text, who = "bot") {
  const div = document.createElement("div");
  div.className = `chat-msg chat-msg-${who}`;
  const p = document.createElement("p");
  p.textContent = text;
  div.appendChild(p);
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

async function sendChat(message) {
  addChatMessage(message, "user");
  chatInput.value = "";

  const thinking = addTypingBubble(chatWindow);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: chatHistory, session_id: getConversationId("chat") }),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    addChatMessage(data.answer, "bot");
    chatHistory.push({ role: "user", content: message }, { role: "bot", content: data.answer });
    // Keep the client-side transcript bounded too (server also trims, but
    // no reason to keep growing the request payload indefinitely).
    if (chatHistory.length > 12) chatHistory = chatHistory.slice(-12);
  } catch (err) {
    thinking.remove();
    addChatMessage(`Sorry — ${err.message}`, "bot");
  }
}

chatForm.addEventListener("submit", e => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (msg) sendChat(msg);
});
$$(".chip-btn").forEach(btn => btn.addEventListener("click", () => sendChat(btn.dataset.q)));

(async function hydrateFromHistory() {
  const saved = await fetchSavedSession("chat");
  if (!saved || !saved.messages || saved.messages.length === 0) return; // brand-new conversation, keep the default greeting

  chatWindow.innerHTML = "";
  saved.messages.forEach(m => addChatMessage(m.content, m.role === "bot" ? "bot" : "user"));
  chatHistory = saved.messages
    .slice(-12)
    .map(m => ({ role: m.role, content: m.content }));
})();
