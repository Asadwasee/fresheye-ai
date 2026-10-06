"""
app.py
======
FreshEye AI - Flask application entry point.

Wires together:
  - Authentication (Flask-Login) + SQLite persistence (Flask-SQLAlchemy)
  - The YOLO fruit-freshness classifier (single image + multi-image batch)
  - The LangChain RAG pipeline (ChromaDB + advanced retriever + open-source LLM)
  - The LangChain agent (tool-calling orchestration)
  - The web UI (templates/ + static/)

Run:
    python app.py
"""
import os
import traceback

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
from sqlalchemy import inspect, text

from config import Config, ensure_directories
from core.api_keys import validate_llm_config
from db_models import db, User

from routes.predict import predict_bp
from routes.chat import chat_bp
from routes.auth import auth_bp
from routes.history import history_bp


def _run_light_migrations():
    """Tiny additive-only auto-migration for SQLite: if `fresheye.db`
    already exists from before a model change (e.g. the `context` column
    added to ChatSession), add the missing column instead of erroring out.
    `db.create_all()` only creates missing *tables*, not missing *columns*
    on tables that already exist - this covers that gap without pulling
    in a full migration framework (out of scope for this project's size).
    Only ever ADDs columns; never renames or drops anything."""
    inspector = inspect(db.engine)
    if "chat_sessions" not in inspector.get_table_names():
        return  # brand-new DB - db.create_all() already built it correctly
    existing_cols = {c["name"] for c in inspector.get_columns("chat_sessions")}
    if "context" not in existing_cols:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN context TEXT"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    ensure_directories()

    # ---- Database + auth ----
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _run_light_migrations()

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to continue to FreshEye AI."
    login_manager.login_message_category = "info"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    app.register_blueprint(predict_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(history_bp)

    @app.route("/")
    @login_required
    def inspect_page():
        return render_template("inspect.html", active_page="inspect")

    @app.route("/agent")
    @login_required
    def agent_page():
        return render_template("agent.html", active_page="agent")

    @app.route("/chat")
    @login_required
    def chat_page():
        return render_template("chat.html", active_page="chat")

    @app.route("/about")
    def about_page():
        # Public on purpose - a visitor deciding whether to sign up should
        # be able to see how the system works without logging in first.
        return render_template("about.html", active_page="about")

    @app.route("/api/health")
    def health():
        llm_ok, llm_msg = validate_llm_config()
        model_loaded = os.path.exists(Config.YOLO_WEIGHTS_PATH)
        return jsonify({
            "status": "ok",
            "yolo_weights_found": model_loaded,
            "yolo_weights_path": Config.YOLO_WEIGHTS_PATH,
            "llm_config_valid": llm_ok,
            "llm_config_message": llm_msg,
            "authenticated": current_user.is_authenticated,
        })

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large."}), 413

    @app.errorhandler(500)
    def server_error(e):
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    print("=" * 70)
    print(" FreshEye AI - Fruit Freshness Detection System")
    print("=" * 70)
    ok, msg = validate_llm_config()
    print(f" LLM config check: {'OK' if ok else 'WARNING'} - {msg}")
    print(f" YOLO weights: {Config.YOLO_WEIGHTS_PATH} "
          f"({'found' if os.path.exists(Config.YOLO_WEIGHTS_PATH) else 'MISSING'})")
    print(f" Database: {Config.SQLALCHEMY_DATABASE_URI}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
