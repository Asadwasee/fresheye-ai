"""
config.py
=========
Central application configuration. Everything here is read from
environment variables (see .env.example) with sane defaults, so the
app is configurable per-environment (dev / staging / prod) without any
code changes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ---- Flask ----
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))

    # ---- Auth / Database ----
    # SQLite is intentionally the only supported store here - this app has
    # no need for a heavier DB, and SQLite needs zero setup for grading /
    # running locally. Swap SQLALCHEMY_DATABASE_URI for a Postgres URL later
    # if this ever needs to run with concurrent writers.
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "fresheye.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ---- Uploads ----
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "static/uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", 100)) * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    # Cap how many images can be classified in a single batch request.
    MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", 25))

    # ---- YOLO model ----
    YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, os.getenv("YOLO_WEIGHTS_PATH", "models/best.pt"))
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.35))

    # ---- Freshness label mapping ----
    # Your YOLO classification model's class names are read dynamically at
    # runtime (model.names). This mapping tells the app which of those class
    # names count as "fresh" vs "rotten/spoiled" so the advice engine and UI
    # can react correctly regardless of your exact class naming convention
    # (e.g. "freshapple"/"rottenapple", "fresh_banana"/"stale_banana", etc).
    # Edit this if your class names don't contain these keywords.
    FRESH_KEYWORDS = ["fresh", "good", "healthy", "ripe"]
    ROTTEN_KEYWORDS = ["rotten", "rot", "stale", "spoiled", "spoilt", "bad", "moldy", "mold"]
    # Class names that mean "this isn't a fruit at all" (e.g. a "Non Fruits"
    # class some datasets include as a catch-all/negative class). Matched
    # as a normalized substring, so "Non Fruits", "non_fruit", "NonFruit"
    # all match. Edit this if your model's negative class is named differently.
    NON_FRUIT_KEYWORDS = ["non fruit", "nonfruit", "not a fruit", "not fruit", "no fruit", "unknown", "background"]

    # ---- RAG / Vector store ----
    CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fruit_freshness_knowledge")
    KNOWLEDGE_DIR = os.path.join(BASE_DIR, "core", "rag", "knowledge")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Text splitting
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

    # Retrieval
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 6))
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", 20))
    RETRIEVER_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.55))

    # ---- LLM ----
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))


def ensure_directories():
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.UPLOAD_FOLDER, "images"), exist_ok=True)
    os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
