"""routes/predict.py - image upload -> classify -> RAG advice.

Supports both a single image and multiple images in one request. When
multiple files are sent, each one is classified and given its own
independent advice (based on its own predicted label + confidence), so a
batch of 10 photos returns 10 separate verdicts.

If the classifier predicts a "non-fruit" class (see
Config.NON_FRUIT_KEYWORDS), the RAG advisor is skipped entirely and an
honest "can't classify this" response is returned instead of forcing a
freshness verdict onto something that isn't fruit.
"""
import traceback

from flask import Blueprint, request, jsonify

from config import Config
from core.classifier import get_classifier, is_non_fruit_label
from core.rag.chain import generate_freshness_advice
from utils.helpers import allowed_file, save_upload, verdict_from_advice

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

NON_FRUIT_MESSAGE = (
    "Sorry - this doesn't look like a fruit, so I can't generate freshness "
    "instructions or a suggestion for it. Try uploading a clear, well-lit "
    "photo of a single piece of fruit."
)


def _build_advice(result, include_advice: bool):
    """Returns the advice payload for a classification result, or None if
    advice was not requested. Non-fruit predictions skip the RAG/LLM call
    entirely and return a fixed, honest response instead."""
    if not include_advice:
        return None

    if is_non_fruit_label(result.label):
        return {
            "advice_text": NON_FRUIT_MESSAGE,
            "sources": [],
            "verdict": "UNCLASSIFIED",
            "not_applicable": True,
        }

    advice_payload = generate_freshness_advice(result.label, result.confidence, result.is_fresh)
    advice_payload["verdict"] = verdict_from_advice(advice_payload["advice_text"])
    advice_payload["not_applicable"] = False
    return advice_payload


def _classify_and_advise(file_storage, include_advice: bool):
    """Runs the full pipeline for a single uploaded file and returns a
    JSON-serialisable dict, or a dict with an "error" key on failure."""
    if not allowed_file(file_storage.filename, Config.ALLOWED_IMAGE_EXTENSIONS):
        return {
            "filename": file_storage.filename,
            "error": f"Unsupported file type. Allowed: {sorted(Config.ALLOWED_IMAGE_EXTENSIONS)}",
        }

    try:
        abs_path, public_path = save_upload(file_storage, "images")

        clf = get_classifier()
        result = clf.predict_image(abs_path)
        advice_payload = _build_advice(result, include_advice)

        return {
            "filename": file_storage.filename,
            "image_url": public_path,
            "classification": result.to_dict(),
            "advice": advice_payload,
        }
    except Exception as e:
        traceback.print_exc()
        return {"filename": file_storage.filename, "error": str(e)}


@predict_bp.route("/predict", methods=["POST"])
def predict_image():
    """Single-image endpoint - kept for backwards compatibility / simple
    integrations. Accepts one file under the 'image' field."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name 'image')."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    include_advice = request.args.get("advice", "true").lower() != "false"
    result = _classify_and_advise(file, include_advice)
    if "error" in result and "classification" not in result:
        return jsonify(result), 400 if "Unsupported" in result["error"] else 500

    return jsonify(result)


@predict_bp.route("/predict/batch", methods=["POST"])
def predict_images_batch():
    """Multi-image endpoint. Accepts one or more files under the 'images'
    field (e.g. FormData.append('images', file) once per file) and returns
    an independent classification + advice for each one."""
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No image files provided (field name 'images')."}), 400

    if len(files) > Config.MAX_BATCH_IMAGES:
        return jsonify({
            "error": f"Too many images in one request (max {Config.MAX_BATCH_IMAGES})."
        }), 400

    include_advice = request.args.get("advice", "true").lower() != "false"

    results = []
    for file_storage in files:
        if file_storage.filename == "":
            continue
        results.append(_classify_and_advise(file_storage, include_advice))

    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    return jsonify({
        "total": len(results),
        "succeeded": len(successes),
        "failed": len(failures),
        "results": results,
    })
