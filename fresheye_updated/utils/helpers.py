"""utils/helpers.py - small shared utilities used across routes."""
import os
import uuid
from datetime import datetime

from werkzeug.utils import secure_filename

from config import Config


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def save_upload(file_storage, subfolder: str) -> tuple[str, str]:
    """Saves an uploaded file with a collision-safe name.
    Returns (absolute_path, public_relative_path)."""
    original = secure_filename(file_storage.filename)
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else "bin"
    unique_name = f"{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.{ext}"

    folder_abs = os.path.join(Config.UPLOAD_FOLDER, subfolder)
    os.makedirs(folder_abs, exist_ok=True)

    abs_path = os.path.join(folder_abs, unique_name)
    file_storage.save(abs_path)

    public_path = f"/static/uploads/{subfolder}/{unique_name}"
    return abs_path, public_path


def verdict_from_advice(advice_text: str) -> str:
    """Extracts the VERDICT line from the LLM's structured advice text so
    the UI can color-code the result without re-parsing the whole thing."""
    for line in advice_text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"
