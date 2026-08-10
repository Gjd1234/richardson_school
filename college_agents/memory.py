"""Self-improving memory store for the college research agents.

Design goals:
- *Relevant*: knowledge is weighted; frequently corroborated facts stay, stale
  entries decay unless refreshed.
- *Up-to-date*: every entry carries `last_seen`; the daily run refreshes via
  new searches and the analyst scores each item.
- *Self-improving*: what the agents researched yesterday shapes today's plan
  (interests, open questions, coverage gaps) -> planner reads memory.

Stored in `data/college-memory.json` (git-tracked so the GitHub action's
runs build on yesterday's memory).
"""
from __future__ import annotations

import json
import os
import pathlib
from datetime import date, datetime, timedelta, timezone

DEFAULT_WEIGHT = 1.0
DECAY_DAYS = 14
MIN_RELEVANCE = 0.4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _parse(s: str):
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class MemoryStore:
    def __init__(self, path: str | os.PathLike | None = None):
        if path is None:
            here = pathlib.Path(__file__).resolve().parent.parent
            path = here / "data" / "college-memory.json"
        self.path = pathlib.Path(path)
        self.data: dict = {
            "meta": {"last_run": None, "run_count": 0},
            "profile": {},
            "interests": [],       # [{name, strength, last_seen}]
            "open_questions": [],  # [str] unresolved questions for the son
            "knowledge": [],       # [{id, topic, fact, source, weight, first_seen, last_seen}]
            "deadlines": [],       # [{id, title, date, url, weight, first_seen, last_seen}]
            "scholarships": [],    # [{id, name, amount, deadline, url, eligibility, weight, last_seen}]
            "coverage": {},        # topic -> {last_researched, search_count}
            "digest_history": [],  # [{date, topics_covered}]
        }
        self.load()

    # ---- persistence ---------------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    for k in self.data:
                        if k in loaded:
                            self.data[k] = loaded[k]
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.data["meta"]["last_run"] = now()
        self.data["meta"]["run_count"] = int(self.data.get("meta", {}).get("run_count", 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    # ---- upsert helper ---------------------------------------------------------
    def _upsert(self, key: str, matches: dict, entry: dict) -> bool:
        """"Insert or bump an existing entry. Returns True when newly inserted."""
        for e in self.data[key]:
            if all(e.get(k) == v for k, v in matches.items()):
                e["weight"] = min(3.0, e.get("weight", DEFAULT_WEIGHT) + 0.5)
                e["last_seen"] = now()
                return False
        entry["weight"] = DEFAULT_WEIGHT
        entry["first_seen"] = now()
        entry["last_seen"] = now()
        self.data[key].append(entry)
        return True

    # ---- writers ---------------------------------------------------------------
    def add_fact(self, topic: str, fact: str, source: str = "") -> bool:
        entry = {"id": str(abs(hash(f"{topic}|{fact}"))), "topic": topic,
                 "fact": fact, "source": source}
        return self._upsert("knowledge", {"topic": topic.casefold(), "fact": fact}, entry)

    def add_deadline(self, title: str, date: str, url: str = "") -> bool:
        entry = {"id": str(abs(hash(title))), "title": title, "date": date, "url": url}
        return self._upsert("deadlines", {"title": title.casefold()}, entry)

    def add_scholarship(self, name: str, amount: str, deadline: str, url: str, eligibility: str = "") -> bool:
        entry = {"id": str(abs(hash(name))), "name": name.title(), "amount": amount,
                 "deadline": deadline, "url": url, "eligibility": eligibility}
        return self._upsert("scholarships", {"name": name.title().casefold()}, entry)

    def add_interest(self, interest: str) -> bool:
        key = interest.strip().casefold()
        for i in self.data["interests"]:
            if i.get("name", "").casefold() == key:
                i["strength"] = min(3, i.get("strength", 1) + 1)
                i["last_seen"] = now()
                return False
        self.data["interests"].append({"name": interest.strip(), "strength": 1, "last_seen": now()})
        return True

    def mark_coverage(self, topic: str) -> None:
        c = self.data["coverage"].get(topic, {"last_researched": None, "search_count": 0})
        c["last_researched"] = today()
        c["search_count"] = int(c.get("search_count", 0)) + 1
        self.data["coverage"][topic] = c

    def log_digest(self, topics: list[str]) -> None:
        self.data["digest_history"].append({"date": today(), "topics_covered": topics})
        self.data["digest_history"] = self.data["digest_history"][-30:]  # keep a month

    # ---- pruning (keeps memory relevant) ----------------------------------------
    def prune(self) -> dict:
        cutoff = datetime.now(timezone.utc)
        removed = {"facts": 0, "deadlines": 0, "scholarships": 0}

        kept = []
        for f in self.data["knowledge"]:
            seen = _parse(f.get("last_seen", ""))
            age_days = (cutoff - seen).days
            if f.get("weight", 1) >= 2.0 or age_days <= DECAY_DAYS:
                kept.append(f)
            else:
                removed["facts"] += 1
        self.data["knowledge"] = kept

        for label in ("deadlines", "scholarships"):
            kept = []
            for d in self.data[label]:
                dl = d.get("date") or d.get("deadline") or ""
                try:
                    dt = datetime.fromisoformat(dl).date() if dl else None
                except ValueError:
                    dt = None
                if dt is None or dt >= (datetime.now(timezone.utc).date() - timedelta(days=60)):
                    kept.append(d)
                else:
                    removed[label] += 1
            self.data[label] = kept
        return removed

    # ---- planner reads ------------------------------------------------------------
    def top_interests(self, limit: int = 8) -> list[str]:
        return [i["name"] for i in sorted(self.data["interests"], key=lambda x: -x.get("strength", 1))][:limit]

    def recent_facts(self, topic: str | None = None, limit: int = 10) -> list[dict]:
        items = self.data["knowledge"]
        if topic:
            t = topic.casefold()
            items = [f for f in items if t in f.get("topic", "").casefold()
                     or t in f.get("fact", "").casefold()]
        items = sorted(items, key=lambda x: (-x.get("weight", 0), x.get("last_seen", "")))
        return items[:limit]

    def upcoming_deadlines(self, days: int = 90) -> list[dict]:
        out = []
        today_d = datetime.now(timezone.utc).date()
        for d in self.data["deadlines"]:
            try:
                dl = datetime.fromisoformat(d["date"]).date()
            except (ValueError, KeyError, TypeError):
                continue
            delta = (dl - today_d).days
            if -14 <= delta <= days:
                item = dict(d)
                item["days_left"] = delta
                out.append(item)
        return sorted(out, key=lambda x: x["days_left"])

    def upcoming_scholarships(self, days: int = 90) -> list[dict]:
        out = []
        today_d = datetime.now(timezone.utc).date()
        for s in self.data["scholarships"]:
            try:
                dl = datetime.fromisoformat(s["deadline"]).date()
                delta = (dl - today_d).days
            except (ValueError, KeyError, TypeError):
                delta = (today_d + timedelta(days=999)).toordinal() - today_d.toordinal()
            if -7 <= delta <= days:
                item = dict(s)
                item["days_left"] = delta
                item["amount_or_aha"] = item.get("amount", "")
                out.append(item)
        return sorted(out, key=lambda x: x["days_left"])