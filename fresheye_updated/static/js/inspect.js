/* FreshEye AI — inspect.js
   Multi-image (or single-image) batch inspection flow for the Inspect page. */

const imageDropzone = $("#imageDropzone");
const imageInput = $("#imageInput");
const imageDropzoneError = $("#imageDropzoneError");
const selectedRow = $("#selectedRow");
const selectedThumbs = $("#selectedThumbs");
const selectedCount = $("#selectedCount");
const runImageBtn = $("#runImageBtn");
const clearSelectionBtn = $("#clearSelectionBtn");
const imageResultCol = $("#imageResultCol");

let selectedFiles = []; // array of File

imageDropzone.addEventListener("click", () => imageInput.click());
["dragover", "dragleave", "drop"].forEach(evt =>
  imageDropzone.addEventListener(evt, e => {
    e.preventDefault();
    imageDropzone.classList.toggle("drag", evt === "dragover");
  })
);
imageDropzone.addEventListener("drop", e => {
  addFiles([...e.dataTransfer.files]);
});
imageInput.addEventListener("change", () => {
  addFiles([...imageInput.files]);
  imageInput.value = ""; // allow re-selecting the same file(s) later
});

function showSelectionError(invalid) {
  if (!invalid.length) {
    imageDropzoneError.hidden = true;
    imageDropzoneError.innerHTML = "";
    return;
  }
  const names = invalid.map(({ file, reason }) => `${file.name} (${reason})`);
  const intro = invalid.length === 1 ? "1 file was skipped:" : `${invalid.length} files were skipped:`;
  imageDropzoneError.hidden = false;
  imageDropzoneError.innerHTML = `<strong>${intro}</strong> ${names.join(", ")}. Allowed formats: ${ALLOWED_IMAGE_EXTENSIONS.join(", ").toUpperCase()}.`;
}

function addFiles(files) {
  if (!files.length) return;
  const { valid, invalid } = validateImageFiles(files);
  showSelectionError(invalid);
  if (valid.length) {
    selectedFiles = [...selectedFiles, ...valid];
    renderSelection();
  }
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderSelection();
}

function renderSelection() {
  if (!selectedFiles.length) {
    selectedRow.hidden = true;
    runImageBtn.disabled = true;
    return;
  }
  selectedRow.hidden = false;
  runImageBtn.disabled = false;
  selectedCount.textContent = selectedFiles.length === 1
    ? "1 photo selected"
    : `${selectedFiles.length} photos selected`;

  selectedThumbs.innerHTML = "";
  selectedFiles.forEach((file, idx) => {
    const thumb = document.createElement("div");
    thumb.className = "selected-thumb";
    thumb.innerHTML = `<img src="${URL.createObjectURL(file)}" alt=""><button class="thumb-remove" type="button" title="Remove">×</button>`;
    thumb.querySelector(".thumb-remove").addEventListener("click", () => removeFile(idx));
    selectedThumbs.appendChild(thumb);
  });
}

clearSelectionBtn.addEventListener("click", () => {
  selectedFiles = [];
  renderSelection();
  showSelectionError([]);
});

runImageBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  runImageBtn.disabled = true;

  if (selectedFiles.length === 1) {
    renderLoading(imageResultCol, "Classifying and consulting the knowledge base…");
    const formData = new FormData();
    formData.append("image", selectedFiles[0]);
    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed.");
      imageResultCol.innerHTML = "";
      imageResultCol.appendChild(buildResultCard({
        filename: data.filename,
        label: data.classification.label,
        confidence: data.classification.confidence,
        advice: data.advice,
        imageUrl: data.image_url,
      }));
    } catch (err) {
      renderError(imageResultCol, err.message);
    } finally {
      runImageBtn.disabled = false;
    }
    return;
  }

  // Multiple images -> batch endpoint, one card per image.
  renderLoading(imageResultCol, `Classifying ${selectedFiles.length} photos and consulting the knowledge base…`);
  const formData = new FormData();
  selectedFiles.forEach(file => formData.append("images", file));

  try {
    const res = await fetch("/api/predict/batch", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Batch prediction failed.");

    imageResultCol.innerHTML = "";
    data.results.forEach(r => {
      if (r.error) {
        imageResultCol.appendChild(buildResultCard({ filename: r.filename, error: r.error, imageUrl: r.image_url }));
      } else {
        imageResultCol.appendChild(buildResultCard({
          filename: r.filename,
          label: r.classification.label,
          confidence: r.classification.confidence,
          advice: r.advice,
          imageUrl: r.image_url,
        }));
      }
    });
  } catch (err) {
    renderError(imageResultCol, err.message);
  } finally {
    runImageBtn.disabled = false;
  }
});
