"""Web research tools for the college agents.

Primary: Tavily (free tier, needs TAVILY_API_KEY).
Fallback: DuckDuckGo HTML scraping (no key) when Tavily is absent or fails.
Both are exposed as plain functions usable inside LangGraph tools, and as
LangChain tools for tool-calling agents.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from college_agents.llm import _load_env


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""


def _tavily_key() -> str:
    _load_env()
    return os.environ.get("TAVILY_API_KEY", "").strip()


def _config() -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "data", "college-config.json")) as f:
        return json.load(f).get("search", {})


def tavily_search(query: str, max_results: int = 6) -> list[SearchResult]:
    key = _tavily_key()
    if not key:
        return []
    url = "https://api.tavily.com/search"
    body = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced" if len(query) > 80 else "basic",
        "include_answer": False,
        "include_images": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    results = []
    for item in data.get("results", []):
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", "")[:500],
            source="tavily",
        ))
    return results[:max_results]


def _ddg_scrape(query: str, max_results: int = 6) -> list[SearchResult]:
    """Free fallback: DuckDuckGo HTML endpoint, no API key required."""
    params = urllib.parse.urlencode({"q": query, "kl": "us-en", "kd": "-1"})
    url = f"https://html.duckduckgo.com/html/?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
    with urllib.request.urlopen(req, timeout=40) as r:
        page = r.read().decode("utf-8", errors="replace")
    results: list[SearchResult] = []
    # result blocks: <a rel="nofollow" class="result__a" href="...">Title</a>
    # snippet: <a class="result__snippet" ...>...</a>
    anchors = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    for i, (href, title) in enumerate(anchors[:max_results]):
        clean = re.sub(r"<[^>]+>", "", title)
        clean = html.unescape(clean).strip()
        if not clean:
            continue
        url2 = href.replace("//duckduckgo.com/l/?uddg=", "").split("&rut=")[0]
        try:
            url2 = urllib.parse.unquote(url2)
        except Exception:
            pass
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r"<[^>]+>", " ", snippets[i])).strip()[:400]
        results.append(SearchResult(title=clean, url=url2, snippet=snippet, source="duckduckgo"))
    return results[:max_results]


def web_search(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Search the web with Tavily, falling back to DuckDuckGo."""
    cfg = _config()
    limit = max_results or cfg.get("max_results", 6)
    if cfg.get("tavily", True):
        try:
            out = tavily_search(query, limit)
            if out:
                return out
        except Exception:
            pass
    if cfg.get("duckduckgo_fallback", True):
        try:
            time.sleep(1.0)  # be polite
            return _ddg_scrape(query, limit)
        except Exception:
            pass
    return []


# LangChain-tool friendly surface -------------------------------------------------

def web_search_tool(query: str, **_) -> str:
    """Run a web search and return a readable text summary of results."""
    results = web_search(query)
    if not results:
        return "No search results available."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}\n   URL: {r.url}\n   {r.snippet[:400]}")
    return "\n".join(lines)


def build_search_tools(max_searches: int = 4) -> list:
    from langchain_core.tools import tool

    @tool
    def search_web(query: str) -> str:
        """Search current websites for up-to-date information. Pass a focused, specific query."""
        return web_search_tool(query)

    @tool
    def search_deadlines(query: str) -> str:
        """Search for official deadlines, dates and requirements. Prioritizes .edu and official sites."""
        return web_search_tool(query + " official deadline site:.edu OR official")

    return [search_web, search_deadlines]