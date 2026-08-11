"""LangGraph workflow orchestrating the college research agents.

Flow (graph):
    planner -> ({studies, scholarships, admissions} in parallel)
             -> analyst (compiles digest)
             -> memory (prune + save, planning inputs feed next run)

Self-improvement loop:
    - planner reads memory (interests, coverage, open questions)
    - researchers update memory with new facts/deadlines/scholarships
    - analyst writes digest and logs what was covered
    - prune() removes stale entries so memory stays relevant
"""
from __future__ import annotations

import json
import operator
import pathlib
import re
from datetime import datetime
from typing import Annotated, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from college_agents import memory as mem
from college_agents.llm import get_model


def _merge_briefs(left: dict, right: dict) -> dict:
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = v
    return merged


class CollegeState(TypedDict, total=False):
    profile: dict
    questions: list[str]
    briefs: Annotated[dict, _merge_briefs]
    digest: str
    new_news: list[str]
    coverage: Annotated[list[str], operator.add]
    store: object


def _profile() -> dict:
    here = __file__
    import pathlib
    root = pathlib.Path(here).resolve().parent.parent
    cfg = json.loads((root / "data" / "college-config.json").read_text())
    return cfg.get("student", {})


def _blank_store() -> mem.MemoryStore:
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    return mem.MemoryStore(root / "data" / "college-memory.json")


def _content_str(content: object) -> str:
    """Normalize an LLM response's `.content` (str | list of parts) to str."""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content or "")


def _shared_store(state: CollegeState) -> mem.MemoryStore:
    store = state.get("store")
    if isinstance(store, mem.MemoryStore):
        return store
    return _blank_store()


# ----------------------------------------------------------------- planner --
def planner(state: CollegeState) -> dict:
    store = _shared_store(state)
    profile = state.get("profile") or _profile()
    year = profile.get("graduation_year", 2027)
    today = datetime.now().strftime("%B %d, %Y")

    interests = store.top_interests(8)
    deadlines = store.upcoming_deadlines(60)
    coverage = store.data.get("coverage", {})
    uncovered = [k for k in interests if (coverage.get(k) or {}).get("last_researched") != mem.today()]

    from college_agents import prompts
    facts = []
    for t in uncovered[:3]:
        facts.extend(store.recent_facts(t, 4))
    known = [(f.get("topic"), f.get("fact")) for f in facts]

    fill = {
        "profile": prompts.profile_preamble(profile),
        "interests": json.dumps(interests),
        "deadlines": json.dumps([{"title": d["title"], "date": d["date"], "days_left": d.get("days_left")} for d in deadlines]),
        "open_questions": json.dumps(store.data.get("open_questions", [])),
        "coverage": json.dumps(coverage),
    }
    system = prompts.PLANNER_SYSTEM.format(year=year, date=today, profile=fill.pop("profile"))
    user = (f"Interests so far: {fill['interests']}\n"
                 f"Deadlines in memory: {fill['deadlines']}\n"
                 f"Open questions: {fill['open_questions']}\n"
                 f"Coverage: {fill['coverage']}\n"
                 f"Plan today's research questions for a {year} Texas senior (tech/STEM focus).")

    llm = get_model("research", temperature=0.1)
    raw = _content_str(llm.invoke(system + "\n\n" + user).content)
    try:
        questions = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip())["questions"]
    except (json.JSONDecodeError, KeyError, TypeError):
        questions = _regex_questions(raw)
    if not questions:
        questions = _fallback_questions(profile)
    return {"questions": questions[:6]}


def _regex_questions(raw: str) -> list[str]:
    return re.findall(r'"([^"]{15,})"', raw)


def _fallback_questions(profile: dict) -> list[str]:
    return [
        f"Fall {profile.get('graduation_year', 2027)} college application deadlines for Texas public universities (UT Austin, Texas A&M) STEM majors",
        "Current high-demand tech/STEM college majors and career trends for the class entering 2027",
        "STEM scholarships for Texas high school seniors closing in the next 90 days",
        "UT Austin Fall 2027 computer science admission requirements and test policy",
        "Test-optional vs required admissions policies 2026-2027 for top Texas universities",
    ]


# ------------------------------------------------------------- researchers --
def _make_researcher(role: str, system_tpl: str):
    def researcher(state: CollegeState) -> dict:
        profile = state.get("profile") or _profile()
        year = profile.get("graduation_year", 2027)
        today = datetime.now().strftime("%B %d, %Y")
        questions = state.get("questions", [])
        store = _shared_store(state)
        from college_agents import prompts, search

        config = json.loads((pathlib.Path(__file__).resolve().parent.parent / "data" / "college-config.json").read_text())
        limit = int(config.get("search", {}).get("max_results", 6))
        budget = int(config.get("search", {}).get("max_searches_per_agent", 4))

        system = system_tpl.format(year=year, date=today, profile=prompts.profile_preamble(profile))
        llm = get_model("research", temperature=0.2)

        briefs: list[tuple[str, str]] = []
        used = 0
        role_q = _pick_role_questions(role, questions)
        for q in role_q:
            if used >= budget:
                break
            results = search.web_search(q, limit)
            used += 1
            store.mark_coverage(q)
            if not results:
                briefs.append((q, "_No results found for this query._"))
                continue
            snippet_block = "\n".join(
                f"- [S{i}] {r.title}\n  URL: {r.url}\n  {r.snippet[:300]}" for i, r in enumerate(results[:limit])
            )
            prompt_user = (
                f"Today: {today}\n\n"
                f"RESEARCH QUESTION: {q}\n\n"
                f"The following are live web-search results (S0..S{len(results) - 1}). "
                f"Answer the question using ONLY these results; cite sources by URL. "
                f"Do not invent facts or deadlines. Be concise and concrete.\n\n"
                f"SEARCH RESULTS:\n{snippet_block}"
            )
            try:
                res = llm.invoke(system + "\n\n" + prompt_user)
                briefs.append((q, _content_str(res.content)))
            except Exception as exc:  # noqa: BLE001 - keep going if one question fails
                briefs.append((q, f"_Research error: {exc}._"))

        combined = _render_brief(role, briefs)
        _record_findings(store, role, combined)
        return {"briefs": {role: combined}, "coverage": [q for q, _ in briefs]}

    return researcher


def _pick_role_questions(role: str, questions: list[str]) -> list[str]:
    kw = {
        "studies": ["trend", "major", "degree", "program", "course", "career", "study", "stem"],
        "scholarships": ["scholar", "award", "grant", "financial", "funding"],
        "admissions": ["admission", "deadline", "application", "test", "requirement", "apply", "focus"],
    }[role]
    scored = sorted(questions, key=lambda q: -sum(1 for k in kw if k in q.casefold()))
    # return scored, then any that didn't match (keeps some diversity)
    matched = [q for q in scored if sum(1 for k in kw if k in q.casefold()) > 0]
    rest = [r for r in scored if r not in matched]
    return matched + rest


# ---------------------------------------------------------- findings -------------------------------------------------
def _record_findings(store: mem.MemoryStore, role: str, combined: str) -> None:
    """Extract structured facts/deadlines/scholarships from a researcher brief
    and persist them to memory, so the planner builds on them tomorrow."""
    if not combined.strip():
        return
    llm = get_model("research", temperature=0.0)
    prompt = (
        "Extract durable, verifiable findings from this research brief into JSON.\n\n"
        "Rules:\n"
        "- facts: list of {topic, fact, source} - one-line durable facts (policies, trends, requirements).\n"
        "- deadlines: list of {title, date (ISO YYYY-MM-DD), url} - only real dated deadlines mentioned.\n"
        "- scholarships: list of {name, amount, deadline (ISO YYYY-MM-DD or ''), url, eligibility} - only explicit named scholarships.\n"
        "- interests: list of strings - majors/programs the student showed keen interest in.\n"
        "- If a category has nothing, use an empty list. Return ONLY JSON.\n\n"
        f"ROLE: {role}\nBRIEF:\n{combined[:6000]}"
    )
    try:
        import json
        raw = _content_str(llm.invoke(prompt).content)
        parsed = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip())
    except Exception:  # noqa: BLE001 - extraction is best-effort
        return
    if not isinstance(parsed, dict):
        return
    for f in parsed.get("facts", []) or []:
        try:
            store.add_fact(str(f.get("topic", role)).strip(), str(f.get("fact", "")).strip(), str(f.get("source", "")))
        except Exception:  # noqa: BLE001
            continue
    for d in parsed.get("deadlines", []) or []:
        try:
            dl = str(d.get("date", "")).strip()
            if not dl:
                continue
            store.add_deadline(str(d.get("title", "")).strip(), dl, str(d.get("url", "")))
        except Exception:  # noqa: BLE001
            continue
    for s in parsed.get("scholarships", []) or []:
        try:
            store.add_scholarship(str(s.get("name", "")).strip(), str(s.get("amount", "")),
                                  str(s.get("deadline", "")).strip(), str(s.get("url", "")),
                                  str(s.get("eligibility", "")))
        except Exception:  # noqa: BLE001
            continue
    for i in parsed.get("interests", []) or []:
        try:
            store.add_interest(str(i).strip())
        except Exception:  # noqa: BLE001
            continue


def _render_brief(role: str, briefs: list[tuple[str, str]]) -> str:
    lines = [f"## {role.title()} Research"]
    for q, text in briefs:
        lines.append(f"\n### Q: {q}\n")
        lines.append(str(text).strip())
    return "\n".join(lines)


studies_researcher = _make_researcher(
    "studies",
    "You are a research analyst tracking CURRENT TRENDS in college studies for a tech/STEM student entering college in Fall {year}. Today is {date}.\n\n"
    "{profile}\n\n"
    "FOCUS: emerging/high-demand majors (AI, ML, data science, cybersecurity, quantum, biomedical engineering), program differences, employment outcomes, "
    "leading programs including Texas schools. Return a concise markdown brief with sections "
    "## Trends, ## Programs Worth Watching, ## Notes for This Student.",
)

scholarships_researcher = _make_researcher(
    "scholarships",
    "You are a scholarship researcher for a Texas high-school student entering college in Fall {year}. Today is {date}.\n\n{profile}\n\n"
    "FOCUS: scholarships open now and closing within ~90 days the student is ELIGIBLE for. Prioritize national STEM awards, UT Austin/Texas-specific awards, "
    "first-gen/income-based, and merit awards at target schools. Return a concise markdown brief: "
    "## New / Updated Scholarships, ## Approaching Deadlines, each with amount, deadline, eligibility, URL.",
)

admissions_researcher = _make_researcher(
    "admissions",
    "You are an admissions researcher for a Texas high-school student applying to college for Fall {year} entry. Today is {date}.\n\n{profile}\n\n"
    "FOCUS: requirements and REAL deadlines: application opening/closing, essay prompts, recommendations, transcripts/scores, test policy, "
    "application platforms (Common App, ApplyTexas). Return concise markdown: ## Requirements & Dates, ## What Needs Prep Now, ## Notes, with official URLs.",
)


# --------------------------------------------------------------- analyst ---
def analyst(state: CollegeState) -> dict:
    store = _shared_store(state)
    profile = state.get("profile") or _profile()
    year = profile.get("graduation_year", 2027)
    today = datetime.now().strftime("%B %d, %Y")

    briefs = state.get("briefs", {})
    studies = briefs.get("studies", "")
    scholarships = briefs.get("scholarships", "")
    admissions = briefs.get("admissions", "")

    from college_agents import prompts

    deadlines = store.upcoming_deadlines(90)
    scls = store.upcoming_scholarships(90)
    interests = store.top_interests(6)
    recent_facts = store.recent_facts(limit=6)

    inputs = {
        "studies_brief": studies,
        "scholarships_brief": scholarships,
        "admissions_brief": admissions,
        "deadlines": json.dumps([{k: d[k] for k in ("title", "date", "days_left", "url") if k in d} for d in deadlines]),
        "scholarships": json.dumps([{k: s[k] for k in ("name", "amount", "deadline", "days_left", "url") if k in s} for s in scls]),
        "interests": json.dumps(interests),
        "known_facts": json.dumps([[f.get("topic"), f.get("fact")] for f in recent_facts]),
    }
    system = prompts.ANALYST_SYSTEM.format(year=year, date=today, profile=prompts.profile_preamble(profile))
    user = (
        f"Researcher briefs:\n--- STUDIES ---\n{studies}\n--- SCHOLARSHIPS ---\n{scholarships}\n--- ADMISSIONS ---\n{admissions}\n\n"
        f"Memory deadlines: {inputs['deadlines']}\nMemory scholarships: {inputs['scholarships']}\n"
        f"Known interests: {inputs['interests']}\nKnown facts: {inputs['known_facts']}\n\n"
        f"Compile today's parent digest (markdown, actionable, brief)."
    )
    llm = get_model("research", temperature=0.3)
    digest = ""
    for attempt in range(2):  # retry once if the free model returns too little
        digest = _content_str(llm.invoke(system + "\n\n" + user).content).strip()
        if len(digest) >= 300:
            break
        if attempt == 0:
            user = (
                f"Researcher briefs:\n--- STUDIES ---\n{studies}\n--- SCHOLARSHIPS ---\n{scholarships}\n--- ADMISSIONS ---\n{admissions}\n\n"
                f"Memory deadlines: {inputs['deadlines']}\nMemory scholarships: {inputs['scholarships']}\n"
                f"Known interests: {inputs['interests']}\nKnown facts: {inputs['known_facts']}\n\n"
                f"The previous response was too short. Write a complete markdown digest (300+ words) covering "
                f"deadlines due soon, new findings, and suggested next steps - even if some briefs are thin."
            )

    store.log_digest(state.get("coverage", []))
    return {"digest": digest}


# -------------------------------------------------------------- workflow --
def build_graph():
    g = StateGraph(CollegeState)
    g.add_node("planner", planner)
    g.add_node("studies", studies_researcher)
    g.add_node("scholarships", scholarships_researcher)
    g.add_node("admissions", admissions_researcher)
    g.add_node("analyst", analyst)
    g.add_edge(START, "planner")
    g.add_edge("planner", "studies")
    g.add_edge("planner", "scholarships")
    g.add_edge("planner", "admissions")
    g.add_edge("studies", "analyst")
    g.add_edge("scholarships", "analyst")
    g.add_edge("admissions", "analyst")
    g.add_edge("analyst", END)
    return g.compile()


def run(profile: dict | None = None) -> dict:
    """Execute the full daily workflow; returns state incl. digest."""
    store = _blank_store()
    state_in: CollegeState = {"profile": profile or _profile(), "store": store}
    graph = build_graph()
    result: CollegeState = cast(CollegeState, graph.invoke(state_in))
    result.setdefault("digest", "")
    result.setdefault("coverage", [])
    # final memory maintenance (researchers already wrote to this shared store)
    store.prune()
    store.save()
    result["store"] = store
    return dict(result)