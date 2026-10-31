"""Command-line tools for managing dashboard accounts.

    PYTHONPATH=src python -m nurse_dashboard create-admin
    PYTHONPATH=src python -m nurse_dashboard create-user  [--admin]
    PYTHONPATH=src python -m nurse_dashboard set-password USERNAME
    PYTHONPATH=src python -m nurse_dashboard deactivate   USERNAME
    PYTHONPATH=src python -m nurse_dashboard activate     USERNAME
    PYTHONPATH=src python -m nurse_dashboard list-users
"""

from __future__ import annotations

import argparse
import getpass
import sys

from sqlmodel import Session, select

from . import audit
from .db import init_db, make_engine
from .models import Role, User
from .security import PasswordPolicyError, check_password_policy, create_user, normalise_username, set_password
from .settings import get_settings


def _ask_password() -> str:
    while True:
        pw = getpass.getpass("Password (min 8 characters): ")
        try:
            check_password_policy(pw)
        except PasswordPolicyError as e:
            print(e)
            continue
        if getpass.getpass("Repeat password: ") != pw:
            print("Passwords do not match.")
            continue
        return pw


def _find(session: Session, username: str) -> User:
    user = session.exec(select(User).where(User.username == normalise_username(username))).first()
    if user is None:
        sys.exit(f"No user named '{username}'.")
    return user


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nurse_dashboard", description="Manage dashboard accounts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create-admin", help="create an admin account")
    cu = sub.add_parser("create-user", help="create a nurse account")
    cu.add_argument("--admin", action="store_true", help="give the account admin rights")
    for name in ("set-password", "deactivate", "activate"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list-users")
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = make_engine(settings)
    init_db(engine)

    with Session(engine) as session:
        if args.command in ("create-admin", "create-user"):
            role = Role.ADMIN if args.command == "create-admin" or args.admin else Role.NURSE
            username = input("Username: ")
            display = input("Display name (e.g. 'Nurse Somchai'): ")
            try:
                user = create_user(session, username, _ask_password(), role=role, display_name=display, actor="cli")
            except ValueError as e:
                sys.exit(str(e))
            print(f"Created {user.role.value} account '{user.username}'.")
        elif args.command == "set-password":
            user = _find(session, args.username)
            set_password(session, user, _ask_password(), actor="cli")
            print("Password changed. Existing logins for this user are signed out.")
        elif args.command in ("deactivate", "activate"):
            user = _find(session, args.username)
            user.is_active = args.command == "activate"
            session.add(user)
            audit.record(session, "cli", f"user_{args.command}d", "user", user.username)
            session.commit()
            print(f"User '{user.username}' is now {'active' if user.is_active else 'deactivated'}.")
        elif args.command == "list-users":
            users = session.exec(select(User).order_by(User.username)).all()
            if not users:
                print("No users yet. Run: python -m nurse_dashboard create-admin")
            for u in users:
                print(f"{u.username:<20} {u.role.value:<6} {'active' if u.is_active else 'INACTIVE':<9} {u.display_name}")


if __name__ == "__main__":
    main()
