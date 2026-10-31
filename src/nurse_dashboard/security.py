"""Password hashing, user creation and a simple login-attempt guard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import bcrypt
from sqlmodel import Session, select

from . import audit
from .models import Role, User

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72          # bcrypt ignores anything after 72 bytes
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60


class PasswordPolicyError(ValueError):
    pass


def check_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")


def hash_password(password: str) -> str:
    check_password_policy(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


# A real hash of a random string: used so that "unknown user" takes as long as
# "wrong password" (does not reveal which usernames exist).
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt()).decode("ascii")


def authenticate(session: Session, username: str, password: str) -> Optional[User]:
    user = session.exec(select(User).where(User.username == normalise_username(username))).first()
    if user is None:
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user.password_hash) or not user.is_active:
        return None
    return user


def session_version(user: User) -> str:
    """Changes whenever the password changes, which logs out old sessions."""
    return user.password_hash[-12:]


def normalise_username(username: str) -> str:
    return username.strip().lower()


def create_user(
    session: Session,
    username: str,
    password: str,
    role: Role = Role.NURSE,
    display_name: str = "",
    actor: str = "system",
) -> User:
    username = normalise_username(username)
    if not username or not username.replace(".", "").replace("_", "").replace("-", "").isalnum():
        raise ValueError("Username may contain only letters, digits, '.', '_' and '-'.")
    if session.exec(select(User).where(User.username == username)).first():
        raise ValueError(f"User '{username}' already exists.")
    user = User(username=username, password_hash=hash_password(password), role=role,
                display_name=display_name.strip() or username)
    session.add(user)
    session.flush()
    audit.record(session, actor, "user_created", "user", username, role=role.value)
    session.commit()
    session.refresh(user)
    return user


def set_password(session: Session, user: User, new_password: str, actor: str = "system") -> None:
    user.password_hash = hash_password(new_password)
    session.add(user)
    audit.record(session, actor, "password_changed", "user", user.username)
    session.commit()


@dataclass
class LoginGuard:
    """Locks a username for a few minutes after repeated wrong passwords (in memory)."""

    max_attempts: int = MAX_FAILED_ATTEMPTS
    lockout_seconds: float = LOCKOUT_SECONDS
    _fails: dict[str, list[float]] = field(default_factory=dict)

    def is_locked(self, username: str) -> bool:
        key = normalise_username(username)
        now = time.monotonic()
        recent = [t for t in self._fails.get(key, []) if now - t < self.lockout_seconds]
        self._fails[key] = recent
        return len(recent) >= self.max_attempts

    def failed(self, username: str) -> None:
        self._fails.setdefault(normalise_username(username), []).append(time.monotonic())

    def succeeded(self, username: str) -> None:
        self._fails.pop(normalise_username(username), None)
