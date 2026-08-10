"""Daily runner for the college research agents.

Usage:
    python run_daily.py                # run workflow, write digest, (optionally) email
    python run_daily.py --no-email     # skip email
    python run_daily.py --print        # print digest to stdout

Emails via Gmail SMTP using an App Password (works headless in GitHub Actions):
    GMAIL_SMTP_USER = your@gmail.com
    GMAIL_SMTP_PASS = the 16-char app password
    (falls back to data/college-config.json digest.email_to if unset)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import smtplib
import ssl
from email.message import EmailMessage

from college_agents.agents import run
from college_agents import memory as mem


def _root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _config() -> dict:
    return json.loads((_root() / "data" / "college-config.json").read_text())


def write_digest(digest: str) -> pathlib.Path:
    path = _root() / "data" / "college-digest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest)
    return path


def build_email(digest: str, to: str) -> EmailMessage:
    msg = EmailMessage()
    cfg = _config().get("digest", {})
    msg["Subject"] = f"{cfg.get('subject_prefix', 'College Scout')} - " + mem.today()
    msg["From"] = True  # placeholder, replaced in send
    msg["To"] = to
    msg.set_content(digest)
    msg.add_alternative(
        f"<html><body><pre style='font-family:Menlo,monospace'>{digest}</pre></body></html>",
        subtype="html",
    )
    return msg


def send_email(digest: str) -> bool:
    cfg = _config().get("digest", {})
    to = os.environ.get("GMAIL_SMTP_TO") or cfg.get("email_to") or ""
    smtp_user = os.environ.get("GMAIL_SMTP_USER") or ""
    smtp_pass = os.environ.get("GMAIL_SMTP_PASS") or ""
    if not (to and smtp_user and smtp_pass):
        return False

    msg = build_email(digest, to)
    msg["From"] = smtp_user
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg, from_addr=smtp_user, to_addrs=[to])
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    result = run()
    digest = result.get("digest", "").strip()
    if not digest:
        digest = "No digest produced today."

    path = write_digest(digest)
    if args.print:
        print(digest)

    emailed = False
    if not args.no_email:
        emailed = send_email(digest)

    store = mem.MemoryStore(_root() / "data" / "college-memory.json")
    stat = {"with_digest": bool(digest), "emailed": emailed, "coverage": result.get("coverage", [])}
    print(json.dumps({**stat, "digest_file": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())