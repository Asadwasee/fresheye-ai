"""
routes/history.py
==================
Lets a logged-in user browse their own past conversations (both the
"Ask FreshEye" knowledge chat and the AI Agent chat). Read-only - actual
saving happens inline in routes/chat.py as each turn is answered.
"""
from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user

from db_models import ChatSession

history_bp = Blueprint("history", __name__)


@history_bp.route("/history")
@login_required
def history_page():
    return render_template("history.html", active_page="history")


@history_bp.route("/api/history/sessions")
@login_required
def list_sessions():
    sessions = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.started_at.desc())
        .all()
    )
    return jsonify({"sessions": [s.to_dict(include_messages=True) for s in sessions]})


@history_bp.route("/api/history/session")
@login_required
def get_session():
    """Fetches one conversation by (chat_type, client_session_id), scoped
    to the logged-in user. Used by chat.js / agent.js on page load to
    check "does this browser tab's conversation id already have saved
    messages?" - which is how continuing a conversation from /history
    actually works: history.js points the tab's saved id at an existing
    conversation and redirects here, and this endpoint is what lets the
    chat page notice and rehydrate instead of starting fresh."""
    chat_type = request.args.get("type")
    client_session_id = request.args.get("session_id")
    if chat_type not in ("chat", "agent") or not client_session_id:
        return jsonify({"error": "type and session_id are required"}), 400

    session = ChatSession.query.filter_by(
        user_id=current_user.id, chat_type=chat_type, client_session_id=client_session_id,
    ).first()
    if session is None:
        return jsonify({"session": None})
    return jsonify({"session": session.to_dict(include_messages=True)})
