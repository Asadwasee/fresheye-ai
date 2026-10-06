"""
routes/auth.py
===============
Minimal authentication: register / login / logout. No email verification,
no password reset, no OAuth - deliberately out of scope for this project.
Passwords are hashed with Werkzeug's PBKDF2 helper (already a Flask
dependency, no extra package needed).
"""
import re

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from db_models import db, User

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Letters (incl. accented), spaces, apostrophes and hyphens only - no
# digits or other symbols. Client-side there's also a live filter
# (see common.js) that strips disallowed characters as you type, but
# this is the actual enforcement: never trust the browser alone.
NAME_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|[ .'\-])*$", re.UNICODE)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("inspect_page"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        error = None
        if not name or not email or not password:
            error = "All fields are required."
        elif len(name) > 120 or not NAME_RE.match(name):
            error = "Name can only contain letters, spaces, apostrophes and hyphens — no numbers or symbols."
        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif User.query.filter_by(email=email).first():
            error = "An account with that email already exists."

        if error:
            flash(error, "error")
            return render_template("register.html", name=name, email=email), 400

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("inspect_page"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("inspect_page"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(email=email).first()
        if user is None or not user.check_password(password):
            flash("Incorrect email or password.", "error")
            return render_template("login.html", email=email), 401

        login_user(user, remember=remember)
        next_url = request.args.get("next")
        # Only follow same-site relative redirects (open-redirect guard).
        if next_url and next_url.startswith("/"):
            return redirect(next_url)
        return redirect(url_for("inspect_page"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
