/* FreshEye AI — agent.js
   The AI Agent page: upload a single photo, get an immediate
   classification + instructions + confidence score, then keep chatting -
   the agent grounds every follow-up answer in that item's classification
   AND remembers the recent conversation, so references like "it" or
   "that one" resolve correctly instead of the agent asking what you mean.

   Two things are sent with every follow-up:
   - `context`: a fixed one-line summary of the current item (what was
     classified, at what confidence, what the verdict was) - this is the
     agent's grounding and doesn't change turn to turn.
   - `history`: the recent back-and-forth (see MAX turns below) - this is
     the agent's short-term memory, giving it the actual conversational
     thread so it can resolve references and avoid repeating itself. */

const agentDropzone = $("#agentDropzone");
const agentImageInput = $("#agentImageInput");
const agentDropzoneError = $("#agentDropzoneError");
const agentResultCol = $("#agentResultCol");
const agentChatSub = $("#agentChatSub");
const agentChatWindow = $("#agentChatWindow");
const agentChatForm = $("#agentChatForm");
const agentChatInput = $("#agentChatInput");
const agentChatSend = $("#agentChatSend");

const MAX_HISTORY_TURNS = 12; // messages, not exchanges (~6 back-and-forths)

let itemContext = null;   // plain-text summary of the current classification, sent every turn
let chatHistory = [];     // [{role: "user"|"bot", content: string}, ...] - the agent's memory
let agentSessionId = null; // id of the saved conversation thread for this item, see resetConversation()

agentDropzone.addEventListener("click", () => agentImageInput.click());
["dragover", "dragleave", "drop"].forEach(evt =>
  agentDropzone.addEventListener(evt, e => {
    e.preventDefault();
    agentDropzone.classList.toggle("drag", evt === "dragover");
  })
);
agentDropzone.addEventListener("drop", e => {
  handleAgentFileSelection([...e.dataTransfer.files]);
});
agentImageInput.addEventListener("change", () => {
  handleAgentFileSelection([...agentImageInput.files]);
  agentImageInput.value = "";
});

function handleAgentFileSelection(files) {
  if (!files.length) return;
  const { valid, invalid } = validateImageFiles(files.slice(0, 1)); // agent page is single-item only

  if (invalid.length) {
    const { file, reason } = invalid[0];
    agentDropzoneError.hidden = false;
    agentDropzoneError.innerHTML = `<strong>${file.name}</strong> was skipped: ${reason}. Allowed formats: ${ALLOWED_IMAGE_EXTENSIONS.join(", ").toUpperCase()}.`;
    return;
  }
  agentDropzoneError.hidden = true;
  agentDropzoneError.innerHTML = "";
  if (valid.length) runAgentPredict(valid[0]);
}

async function runAgentPredict(file) {
  renderLoading(agentResultCol, "Classifying and preparing instructions…");
  resetConversation();

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");

    agentResultCol.innerHTML = "";
    agentResultCol.appendChild(buildResultCard({
      filename: data.filename,
      label: data.classification.label,
      confidence: data.classification.confidence,
      advice: data.advice,
      imageUrl: data.image_url,
    }));

    startConversation(data);
  } catch (err) {
    renderError(agentResultCol, err.message);
  }
}

function resetConversation() {
  itemContext = null;
  chatHistory = [];
  // A new photo = a new saved conversation thread, not a continuation of
  // whatever the previous item's chat history was.
  agentSessionId = newConversationId("agent");
  agentChatWindow.innerHTML = "";
  agentChatInput.disabled = true;
  agentChatSend.disabled = true;
  agentChatSub.textContent = "Upload a photo to start — the agent will ground its answers in that item.";
}

function startConversation(data) {
  const label = data.classification.label;
  const confidence = (data.classification.confidence * 100).toFixed(1);
  const verdict = data.advice?.verdict || "UNKNOWN";
  const notApplicable = !!data.advice?.not_applicable;

  itemContext = notApplicable
    ? `The uploaded image was classified as "${label}" (${confidence}% confidence), which the vision model treats as not a fruit, so no freshness verdict applies.`
    : `The uploaded item was classified as "${label}" (${confidence}% confidence). Advisor verdict: ${verdict}.`;

  agentChatInput.disabled = false;
  agentChatSend.disabled = false;
  agentChatSub.textContent = `Grounded in this reading: ${label.replace(/[_-]/g, " ")} (${confidence}%)`;

  const openingMessage = notApplicable
    ? `I couldn't identify a fruit in that photo, so I can't give freshness instructions for it — but feel free to ask me anything else, like what a good photo for inspection looks like.`
    : `This item reads as "${label.replace(/[_-]/g, " ")}" with ${confidence}% confidence — verdict: ${verdict}. Ask me anything about it — storage, shelf life, or what to do next.`;

  addAgentMessage(openingMessage, "bot");
  // The opening message becomes part of the agent's own memory too, so it
  // won't repeat itself if asked something like "what did you just say?".
  chatHistory.push({ role: "bot", content: openingMessage });
}

function addAgentMessage(text, who = "bot") {
  const div = document.createElement("div");
  div.className = `chat-msg chat-msg-${who}`;
  const p = document.createElement("p");
  p.textContent = text;
  div.appendChild(p);
  agentChatWindow.appendChild(div);
  agentChatWindow.scrollTop = agentChatWindow.scrollHeight;
  return div;
}

agentChatForm.addEventListener("submit", async e => {
  e.preventDefault();
  const question = agentChatInput.value.trim();
  if (!question || !itemContext) return;

  addAgentMessage(question, "user");
  agentChatInput.value = "";

  const thinking = addTypingBubble(agentChatWindow);

  try {
    const res = await fetch("/api/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: question,
        context: itemContext,
        history: chatHistory,
        session_id: agentSessionId,
      }),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) throw new Error(data.error || "The agent couldn't answer that.");
    addAgentMessage(data.agent_output, "bot");
    chatHistory.push({ role: "user", content: question }, { role: "bot", content: data.agent_output });
    if (chatHistory.length > MAX_HISTORY_TURNS) chatHistory = chatHistory.slice(-MAX_HISTORY_TURNS);
  } catch (err) {
    thinking.remove();
    addAgentMessage(`Sorry — ${err.message}`, "bot");
  }
});

(async function hydrateFromHistory() {
  const saved = await fetchSavedSession("agent");
  if (!saved || !saved.messages || saved.messages.length === 0) return; // fresh page, nothing to continue

  // Keep replying in the same saved thread instead of starting a new one.
  agentSessionId = getConversationId("agent");
  itemContext = saved.context || null;

  agentChatWindow.innerHTML = "";
  saved.messages.forEach(m => addAgentMessage(m.content, m.role === "bot" ? "bot" : "user"));
  chatHistory = saved.messages
    .slice(-MAX_HISTORY_TURNS)
    .map(m => ({ role: m.role, content: m.content }));

  agentResultCol.innerHTML = `
    <div class="empty-state continue-banner">
      <p><strong>Continuing a saved conversation.</strong></p>
      <p>${itemContext ? itemContext : "Upload a new photo any time to switch items."}</p>
    </div>
  `;

  if (itemContext) {
    agentChatInput.disabled = false;
    agentChatSend.disabled = false;
    agentChatSub.textContent = "Continuing your previous conversation about this item.";
  } else {
    // Older saved sessions (from before context was persisted) won't have
    // grounding to resume with - safest is to ask for a fresh photo
    // rather than let the agent answer ungrounded.
    agentChatSub.textContent = "Upload the photo again to keep grounding answers in this item.";
  }
})();
