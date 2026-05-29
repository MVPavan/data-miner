"""Bootstrap LS reviewer users for collaborative review.

Looks up the requested reviewer emails in Label Studio. With
``--create-missing`` it POSTs new accounts to ``/api/users/`` (works on
LS Community 1.23+) so a fresh deployment can bootstrap its 5-person
team in one shot. Without that flag the script is read-only and just
reports who's missing.

Token retrieval is intentionally one-way: LS Community exposes
``/api/current-user/token`` only to the bearer of that token, so the
admin cannot fetch other users' tokens. After this script creates the
accounts, each reviewer logs in once at ``/user/login/`` to mint their
personal token via Account & Settings.

Usage::

    python -m manual_reviewer.scripts.create_collaborators \\
        --ls-url http://localhost:8080 \\
        --ls-token "$LS_TOKEN" \\
        --users pavan:pavan@jci.com,sree:sree@jci.com \\
        --create-missing --default-password review2026

The script never deletes or overwrites users. It is safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("manual_reviewer.create_collaborators")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = args.ls_token or os.environ.get("LS_TOKEN")
    if not token:
        logger.error("--ls-token or LS_TOKEN env var required")
        return 2

    try:
        users = _parse_user_specs(args.users)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        import httpx
    except ImportError:
        logger.error("httpx required: pip install httpx")
        return 2

    if args.create_missing and not args.default_password:
        logger.error("--create-missing requires --default-password")
        return 2

    base_url = args.ls_url.rstrip("/")
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        try:
            existing = _fetch_users(client, base_url)
        except Exception as exc:  # noqa: BLE001
            logger.error("LS user list failed: %s", exc)
            return 3

        rows: list[tuple[str, str, str, int | None, str | None, str]] = []
        missing: list[tuple[str, str]] = []
        for assigned_to, email in users:
            match = existing.get(email.lower())
            if match is None and args.create_missing:
                created = _create_user(
                    client, base_url, email=email, name=assigned_to,
                    password=args.default_password,
                    ls_data_dir=args.ls_data_dir,
                )
                if created is not None:
                    match = created
                    rows.append(
                        (assigned_to, email, _fmt_name(match), match.get("id"),
                         None, "CREATED")
                    )
                    continue
            if match is None:
                missing.append((assigned_to, email))
                rows.append((assigned_to, email, "—", None, None, "MISSING"))
                continue
            user_id = match.get("id")
            user_token = _resolve_user_token(client, base_url, match)
            rows.append(
                (assigned_to, email, _fmt_name(match), user_id, user_token, "OK")
            )

    _print_table(rows)

    if missing:
        print("")
        print(f"{len(missing)} user(s) not yet in LS — create them via:")
        print(
            f"  1) Open {base_url}/organization in a browser as the admin "
            "user (the one whose token is LS_TOKEN)."
        )
        print(
            "  2) Members → 'Add Member' → enter the email; LS sends a"
            " signup link the user opens to set their password."
        )
        if args.invite:
            print("")
            print("Or generate signup links via the API (best-effort):")
            for assigned_to, email in missing:
                _try_generate_invite(base_url, headers, args.timeout, email, assigned_to)
        print("")
        print("After creating the missing users, re-run this script to capture"
              " their API tokens.")
        return 1

    print("")
    print("All requested reviewers exist. Use the assigned_to column with:")
    print(
        f"  python -m manual_reviewer.scripts.build_tasks "
        f"--assignees {','.join(a for a, _ in users)} ..."
    )
    return 0


def _parse_user_specs(raw: str) -> list[tuple[str, str]]:
    """Parse ``name:email,name:email,...`` into ``[(assigned_to, email), ...]``."""
    out: list[tuple[str, str]] = []
    if not raw:
        raise ValueError("--users is required")
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--users entry {chunk!r} missing ':'; expected 'name:email'"
            )
        name, email = chunk.split(":", 1)
        name = name.strip()
        email = email.strip()
        if not name or not email or "@" not in email:
            raise ValueError(f"--users entry {chunk!r} is malformed")
        out.append((name, email))
    if {a for a, _ in out} != {a for a, _ in out}:  # invariant placeholder
        raise ValueError("duplicate assigned_to names in --users")
    seen: set[str] = set()
    for a, _ in out:
        if a in seen:
            raise ValueError(f"duplicate assigned_to name: {a}")
        seen.add(a)
    return out


def _fetch_users(client: Any, base_url: str) -> dict[str, dict[str, Any]]:
    """Index existing users by lowercase email.

    LS Community returns the org's full user list at /api/users/. The
    response shape varies across LS versions: sometimes a flat list,
    sometimes ``{"results": [...]}``.
    """
    resp = client.get(f"{base_url}/api/users/")
    resp.raise_for_status()
    payload = resp.json()
    items = payload if isinstance(payload, list) else (payload.get("results") or [])
    out: dict[str, dict[str, Any]] = {}
    for u in items:
        if not isinstance(u, dict):
            continue
        email = u.get("email") or u.get("username") or ""
        if email:
            out[email.lower()] = u
    return out


def _create_user(
    client: Any, base_url: str, *, email: str, name: str, password: str,
    ls_data_dir: str | None,
) -> dict[str, Any] | None:
    """POST /api/users/ to create a new account, then set the password
    via Django ORM.

    LS Community 1.23+ accepts ``POST /api/users/`` for admin-token
    requests, but the serializer **silently drops the ``password``
    field** — the row is created with an empty hash and the user can
    never log in. The fix is to call ``User.set_password()`` directly
    after creation, which requires importing LS's Django app with the
    matching ``LABEL_STUDIO_BASE_DATA_DIR`` env var so we hit the right
    sqlite. ``ls_data_dir`` carries that path; without it we still
    create the row but warn that login won't work.
    """
    payload = {
        "email": email,
        "username": email,
        "password": password,
        "first_name": name,
        "last_name": "",
    }
    try:
        resp = client.post(f"{base_url}/api/users/", json=payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("create user %s failed: %s", email, exc)
        return None
    if resp.status_code >= 300:
        logger.warning(
            "create user %s returned %d: %s",
            email, resp.status_code, resp.text[:200],
        )
        return None
    body = resp.json()
    if not isinstance(body, dict):
        return None
    if ls_data_dir:
        if not _set_password_via_orm(ls_data_dir, email, password):
            logger.warning(
                "user %s created but password set failed — login won't work "
                "until you run set_password() manually", email,
            )
    else:
        logger.warning(
            "user %s created but --ls-data-dir not set — password silently "
            "dropped by LS, login won't work until set_password() is run",
            email,
        )
    return body


_DJANGO_READY = False


def _set_password_via_orm(ls_data_dir: str, email: str, password: str) -> bool:
    """Lazy-import LS's Django app and run ``User.set_password()``.

    Cached after first call: Django setup() is one-shot per process.
    """
    global _DJANGO_READY
    try:
        if not _DJANGO_READY:
            os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = ls_data_dir
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
            import sys
            import site
            for site_pkg in site.getsitepackages():
                ls_pkg = os.path.join(site_pkg, "label_studio")
                if os.path.isdir(ls_pkg) and ls_pkg not in sys.path:
                    sys.path.insert(0, ls_pkg)
                    break
            import django  # noqa: WPS433 — lazy
            django.setup()
            _DJANGO_READY = True
        from users.models import User  # noqa: WPS433 — lazy
        user = User.objects.get(email=email)
        user.set_password(password)
        user.save()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ORM set_password for %s failed: %s", email, exc)
        return False


def _resolve_user_token(client: Any, base_url: str, user: dict[str, Any]) -> str | None:
    """Return the user's legacy auth token if discoverable.

    LS includes ``token`` directly on /api/users/ rows in some builds;
    otherwise it's at /api/users/<id>/. Older Community releases hide
    it behind /api/current-user/token (only the user themselves can see
    their token), in which case we return None and the operator must
    fetch it from each reviewer's Account & Settings page.
    """
    direct = user.get("token")
    if isinstance(direct, str) and direct:
        return direct
    user_id = user.get("id")
    if user_id is None:
        return None
    try:
        resp = client.get(f"{base_url}/api/users/{user_id}/")
        if resp.status_code >= 300:
            return None
        body = resp.json()
        tok = body.get("token") if isinstance(body, dict) else None
        return tok if isinstance(tok, str) and tok else None
    except Exception:  # noqa: BLE001
        return None


def _try_generate_invite(
    base_url: str, headers: dict[str, str], timeout: float, email: str, assigned_to: str
) -> None:
    """Best-effort: invoke /api/invite/ to print a one-off signup link.

    LS Community has historically routed invites through several paths;
    we hit the most common one and just print the response or the
    failure for the operator to act on. No retries — this is a UX hint.
    """
    try:
        import httpx
    except ImportError:
        return
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.post(
                f"{base_url}/api/invite/", json={"email": email}
            )
            if resp.status_code < 300:
                body = resp.json() if resp.text else {}
                link = body.get("invite_url") or body.get("url") or body
                print(f"  - {assigned_to} ({email}): {link}")
            else:
                print(
                    f"  - {assigned_to} ({email}): invite API returned "
                    f"{resp.status_code} — use the UI flow"
                )
    except Exception as exc:  # noqa: BLE001
        print(f"  - {assigned_to} ({email}): invite API unreachable ({exc})")


def _fmt_name(user: dict[str, Any]) -> str:
    fn = (user.get("first_name") or "").strip()
    ln = (user.get("last_name") or "").strip()
    return (fn + " " + ln).strip() or user.get("username") or "—"


def _print_table(
    rows: list[tuple[str, str, str, int | None, str | None, str]],
) -> None:
    cols = ("assigned_to", "email", "ls_user_name", "ls_user_id", "ls_token", "status")
    widths = [len(c) for c in cols]
    for r in rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(str(val) if val is not None else "—"))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*[str(v) if v is not None else "—" for v in r]))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ls-url", default="http://localhost:8080",
                   help="Label Studio base URL")
    p.add_argument("--ls-token", default=None,
                   help="Admin LS API token (defaults to $LS_TOKEN)")
    p.add_argument(
        "--users",
        required=True,
        help="Comma-separated assigned_to:email pairs, e.g. "
             "'pavan:pavan@example.com,sree:sree@example.com'",
    )
    p.add_argument("--invite", action="store_true",
                   help="Try POST /api/invite/ to generate signup links for "
                        "missing users (LS-version-specific; best-effort)")
    p.add_argument("--create-missing", action="store_true",
                   help="POST /api/users/ to create accounts for missing "
                        "emails. Requires --default-password.")
    p.add_argument("--default-password", default=None,
                   help="Initial password applied to every newly created "
                        "user. They should rotate it on first login.")
    p.add_argument(
        "--ls-data-dir",
        default=os.environ.get("LS_DATA_DIR"),
        help="Path to LS's data dir (LABEL_STUDIO_BASE_DATA_DIR). "
             "Required with --create-missing because LS Community 1.23 "
             "silently drops the password field on POST /api/users/; the "
             "script invokes Django ORM set_password() to fix it. "
             "Defaults to $LS_DATA_DIR.",
    )
    p.add_argument("--timeout", type=float, default=15.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
