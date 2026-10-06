/* FreshEye AI — results.js
   Shared result-card rendering, used by both the Inspect page (batch of
   photos) and the AI Agent page (single photo). No file paths, directory
   listings, or raw source citations are ever rendered here by design. */

function buildResultCard({ filename, label, confidence, advice, imageUrl, error }) {
  if (error) {
    const div = document.createElement("div");
    div.className = "result-card";
    div.innerHTML = `
      ${imageUrl ? `<div class="result-media"><img class="result-thumb" src="${imageUrl}" alt=""></div>` : ""}
      <div class="result-body">
        <p class="result-filename">${filename || ""}</p>
        <p style="color:#8B4432;margin:0">${error}</p>
      </div>`;
    return div;
  }

  const tpl = $("#resultTemplate").content.cloneNode(true);
  const verdict = advice?.verdict || "UNKNOWN";
  const notApplicable = !!advice?.not_applicable;

  const thumb = tpl.querySelector(".result-thumb");
  if (imageUrl) {
    thumb.src = imageUrl;
  } else {
    tpl.querySelector(".result-media").remove();
  }

  tpl.querySelector(".result-filename").textContent = filename || "";

  const stamp = tpl.querySelector(".stamp");
  stamp.dataset.verdict = verdict;
  tpl.querySelector(".stamp-text").textContent = verdict;

  tpl.querySelector(".predicted-label").textContent = label.replace(/[_-]/g, " ");
  tpl.querySelector(".confidence-fill").style.width = `${Math.round(confidence * 100)}%`;
  tpl.querySelector(".confidence-label").textContent = `Model confidence: ${(confidence * 100).toFixed(1)}%`;

  if (advice) {
    if (notApplicable) {
      // Non-fruit prediction: no reasoning/recommendation split, just the
      // honest apology message, in full.
      tpl.querySelector(".advice-reasoning").remove();
      tpl.querySelector(".advice-recommendation").textContent = advice.advice_text;
    } else {
      const lines = advice.advice_text.split("\n").filter(Boolean);
      const reasoning = lines.find(l => l.toUpperCase().startsWith("REASONING:"))?.split(":").slice(1).join(":").trim()
        || lines.find(l => l.toUpperCase().startsWith("CONFIDENCE NOTE:"))?.split(":").slice(1).join(":").trim() || "";
      const recommendation = lines.find(l => l.toUpperCase().startsWith("RECOMMENDATION:"))?.split(":").slice(1).join(":").trim() || "";
      tpl.querySelector(".advice-reasoning").textContent = reasoning;
      tpl.querySelector(".advice-recommendation").textContent = recommendation || advice.advice_text;
    }
  } else {
    tpl.querySelector(".advice-block").remove();
  }

  const wrapper = document.createElement("div");
  wrapper.appendChild(tpl);
  return wrapper.firstElementChild;
}

function renderLoading(container, message = "Running inspection…") {
  container.innerHTML = `<div class="empty-state"><p class="loading-text">${message}</p></div>`;
}
function renderError(container, message) {
  container.innerHTML = `<div class="empty-state"><p style="color:#8B4432">${message}</p></div>`;
}
