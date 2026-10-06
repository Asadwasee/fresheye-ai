"""routes/chat.py - RAG-powered chat assistant + agent entry point.

Every turn on both chat surfaces is now persisted per logged-in user:
each browser tab keeps a `session_id` (a UUID, see static/js/chat.js /
agent.js) so repeated turns append to the same ChatSession row instead
of creating a new one each time; the very first user message becomes
that session's title, for the history page.
"""
import traceback

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from core.rag.chain import answer_chat_question
from core.agent import run_agent_chat, run_agent_on_image
from db_models import db, ChatSession, ChatMessage

chat_bp = Blueprint("chat", __name__, url_prefix="/api")


def _get_or_create_session(
    chat_type: str, client_session_id: str, title_hint: str, context: str = None
) -> ChatSession:
    """Fetches this user's ChatSession for the given (chat_type,
    client_session_id) pair, creating it if this is the first turn.
    `context` (agent sessions only) is stored/refreshed so a conversation
    can be resumed later from /history with the agent still grounded in
    the right item, without needing the original image again."""
    session = ChatSession.query.filter_by(
        user_id=current_user.id,
        chat_type=chat_type,
        client_session_id=client_session_id,
    ).first()
    if session is None:
        title = (title_hint or "New conversation").strip()
        if len(title) > 80:
            title = title[:77] + "..."
        session = ChatSession(
            user_id=current_user.id,
            chat_type=chat_type,
            client_session_id=client_session_id,
            title=title,
            context=context,
        )
        db.session.add(session)
        db.session.flush()  # get session.id without a full commit yet
    elif context and session.context != context:
        session.context = context
    return session


def _save_turn(chat_type: str, client_session_id: str, user_text: str, bot_text: str, context: str = None):
    if not client_session_id:
        return  # caller didn't send one (e.g. an older client) - skip persistence, don't fail the request
    session = _get_or_create_session(chat_type, client_session_id, title_hint=user_text, context=context)
    db.session.add(ChatMessage(session_id=session.id, role="user", content=user_text))
    db.session.add(ChatMessage(session_id=session.id, role="bot", content=bot_text))
    db.session.commit()


@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """Direct RAG Q&A - fast path, used by the standalone 'Ask FreshEye'
    chat page for most turns. Accepts optional `history` (list of
    {"role": "user"|"bot", "content": str}, oldest first, NOT including
    the current `message`) so follow-up questions like "is it safe to
    eat?" can be resolved against what was discussed earlier."""
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    history = data.get("history") or []
    client_session_id = (data.get("session_id") or "").strip()
    if not question:
        return jsonify({"error": "message is required"}), 400
    try:
        result = answer_chat_question(question, history=history)
        _save_turn("chat", client_session_id, question, result.get("answer", ""))
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/agent/chat", methods=["POST"])
@login_required
def agent_chat():
    """Agentic path - the LLM decides which tools to call (knowledge
    search, classification, advice generation). Used by the AI Agent page:

    - First turn on that page: the image is classified via /api/predict,
      and its result is sent here as free-form `context` on every
      follow-up turn, so the agent can reason about "this item" without
      re-running the vision model on every message.
    - `history` (list of {"role": "user"|"bot", "content": str}, oldest
      first, NOT including the current `message`) gives the agent
      short-term memory so it can resolve "it" / "that one" / "the fruit
      we discussed" against earlier turns instead of asking the user to
      re-explain themselves.
    - `image_path` is still supported for a single-shot "classify this
      image and advise" agentic call if a caller wants the full
      classify -> advise flow driven entirely by the agent.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    image_path = data.get("image_path")
    context = data.get("context")
    history = data.get("history") or []
    client_session_id = (data.get("session_id") or "").strip()
    if not question and not image_path:
        return jsonify({"error": "message or image_path is required"}), 400
    try:
        if image_path:
            result = run_agent_on_image(image_path, user_message=question or None, chat_history=history)
            logged_user_text = question or "[uploaded an image for inspection]"
        else:
            result = run_agent_chat(question, context=context, chat_history=history)
            logged_user_text = question
        _save_turn(
            "agent", client_session_id, logged_user_text,
            result.get("agent_output", result.get("answer", "")),
            context=context,
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
