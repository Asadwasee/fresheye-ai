"""
models.py
=========
SQLAlchemy models backing authentication and per-user chat history.

FreshEye AI is otherwise stateless (images live on disk, knowledge lives
in ChromaDB) - this is the one place actual relational data is persisted:
who the user is, and what they've asked the assistant / agent over time.

Kept deliberately small for an FYP scope: no email verification, no
password reset flow, no OAuth. Just enough to (a) gate the app behind a
login and (b) let each user come back and see their own past
conversations.
"""
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


def _utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")  # "user" | "admin"
    created_at = db.Column(db.DateTime, default=_utcnow)

    chat_sessions = db.relationship(
        "ChatSession", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "email": self.email, "role": self.role}


class ChatSession(db.Model):
    """One conversation thread. `chat_type` distinguishes the two chat
    surfaces (`chat` = Ask FreshEye knowledge assistant, `agent` = AI
    Agent page). `client_session_id` is a UUID the browser generates and
    keeps in localStorage, so the same open tab keeps appending to the
    same thread instead of starting a new row on every message."""
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    chat_type = db.Column(db.String(10), nullable=False)  # "chat" | "agent"
    client_session_id = db.Column(db.String(64), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=True)  # first user message, for the history list
    # For "agent" sessions only: the one-line classification summary the
    # agent grounds every answer in (see itemContext in static/js/agent.js).
    # Saved so a conversation can be resumed from /history without having
    # re-uploaded the photo - the agent still knows what item it was
    # discussing.
    context = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime, default=_utcnow)

    messages = db.relationship(
        "ChatMessage", backref="session", lazy=True,
        cascade="all, delete-orphan", order_by="ChatMessage.timestamp",
    )

    def to_dict(self, include_messages: bool = False) -> dict:
        d = {
            "id": self.id,
            "chat_type": self.chat_type,
            "client_session_id": self.client_session_id,
            "title": self.title or "New conversation",
            "context": self.context,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "message_count": len(self.messages) if include_messages else None,
        }
        if include_messages:
            d["messages"] = [m.to_dict() for m in self.messages]
        return d


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)  # "user" | "bot"
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=_utcnow)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
