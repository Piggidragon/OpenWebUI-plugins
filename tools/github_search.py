"""
title: GitHub Global Search
author: Piggidragon
description: >
  Search the entire GitHub universe — public (and private) repos, code,
  issues, pull requests, commits, users, and topics — right from OpenWebUI.
  Works without a token (lower rate limits), or with a token for higher limits
  and access to your private repositories.
version: 1.0.1
"""

import json
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ═══════════════════════════════════════════════════════════════════
#  QUALIFIER REFERENCE (included so the model can see it)
# ═══════════════════════════════════════════════════════════════════

_QUALIFIER_HELP = """
GitHub Search Qualifiers — add these to your query for precision:

  **All types:**       `org:NAME`  `user:NAME`  `created:>2024-01-01`  `updated:>=2024-06-01`
  **Repositories:**    `language:python`  `stars:>100`  `forks:>=10`  `topic:machine-learning`  `is:private`
  **Code:**            `language:go`  `repo:owner/name`  `path:src/`  `extension:py`  `filename:config.yaml`
  **Issues / PRs:**    `type:issue`  `type:pr`  `state:open`  `label:bug`  `is:merged`  `author:USER`
  **Commits:**         `author:USER`  `committer:USER`  `repo:owner/name`  `merge:true`
  **Users:**           `type:user`  `type:org`  `repos:>10`  `followers:>=50`

Examples:
  `NeoForge registration language:java`     → Java code mentioning NeoForge
  `stars:>500 topic:ai language:python`     → popular Python AI repos
  `label:good-first-issue state:open`       → beginner-friendly open issues
  `author:torvalds repo:torvalds/linux`     → Linus' commits in linux.git
"""


class Tools:
    class Valves(BaseModel):
        """Global / admin settings — left empty on purpose."""
        pass

    class UserValves(BaseModel):
        GITHUB_TOKEN: str = Field(
            default="",
            description="Optional GitHub PAT for higher rate limits (30 vs 10 req/min) and private repo access. Leave empty to search public repos anonymously.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── helpers ────────────────────────────────────────────────────

    def _get_uv(self, __user__: Optional[dict]) -> "Tools.UserValves":
        if __user__ and "valves" in __user__:
            return __user__["valves"]
        return self.UserValves()

    def _headers(self, uv: "Tools.UserValves") -> dict:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "OpenWebUI-GitHub-Search",
        }
        if uv.GITHUB_TOKEN:
            h["Authorization"] = f"Bearer {uv.GITHUB_TOKEN}"
        return h

    # ═══════════════════════════════════════════════════════════════
    #  MAIN SEARCH FUNCTION
    # ═══════════════════════════════════════════════════════════════

    async def github_search(
        self,
        query: str,
        search_type: Literal[
            "repositories", "code", "issues", "commits", "users", "topics"
        ] = "repositories",
        per_page: int = 10,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Search across all of GitHub — code, repos, issues, PRs, commits, users, topics.

        Use this like a search engine for GitHub. It searches public repositories
        without a token, or all repos you have access to when a token is set.

        The `query` can be plain text OR include GitHub qualifiers
        (see _QUALIFIER_HELP for details).

        :param query: Search terms, optionally with GitHub qualifiers.
                      Examples: "NeoForge registration language:java",
                      "stars:>1000 topic:llm", "label:bug state:open repo:facebook/react"
        :param search_type: What to search. One of:
            - "repositories"  — find repos by name, description, topic, readme
            - "code"          — search actual source code across repos
            - "issues"        — issues AND pull requests (use type:issue or type:pr to narrow)
            - "commits"       — commit messages (add repo:owner/name for focus)
            - "users"         — GitHub users and organizations
            - "topics"        — repository topics
        :param per_page: Number of results (default: 10, max: 30)
        :return: Formatted search results with links, descriptions, and metadata.
        """

        uv = self._get_uv(__user__)
        per_page = min(per_page, 30)

        # ── build URL ─────────────────────────────────────────
        base = "https://api.github.com/search"

        # "topics" is a special search — it uses the repos endpoint internally
        if search_type == "topics":
            q = f"{query}+topic:{query}" if "topic:" not in query else query
            url = f"{base}/repositories?q={q}&per_page={per_page}&sort=stars&order=desc"
        else:
            url = f"{base}/{search_type}?q={query}&per_page={per_page}"

        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers(uv))

            if r.status_code == 403 and "rate limit" in (r.text or "").lower():
                return (
                    "⛔ **Rate limit exceeded.**\n\n"
                    "The anonymous GitHub Search API allows ~10 requests per minute. "
                    "Either wait a moment, or add your GitHub token in User Valves "
                    "→ GITHUB_TOKEN (account → valves) for 30 requests/minute."
                )

            if r.status_code == 422:
                return (
                    f"❌ **Invalid query.**\n\n"
                    f"GitHub could not parse: `{query}`\n\n"
                    f"💡 Make sure qualifiers use the correct syntax "
                    f"(e.g. `language:python`, `stars:>100`, `type:issue`).\n\n"
                    f"## Qualifier help\n{_QUALIFIER_HELP}"
                )

            r.raise_for_status()
            data = r.json()

            total = data.get("total_count", 0)
            if total == 0:
                return (
                    f"## No results for `{query}` (type: _{search_type}_)\n\n"
                    f"Try broadening your search or removing some qualifiers.\n\n"
                    f"ℹ️ When searching code, only the *default branch* is indexed.\n"
                    f"ℹ️ Forks are NOT indexed by default — add `fork:true` to include them."
                )

        # ── format results ────────────────────────────────────

        items = data["items"][:per_page]
        out_lines = [
            f"## GitHub {search_type.capitalize()} Search: `{query}`",
            f"**{total:,}** results  |  showing top {len(items)}",
            "",
        ]

        if search_type == "repositories":
            for i, repo in enumerate(items, 1):
                stars = f"⭐ {repo['stargazers_count']:,}"
                lang = repo.get("language") or ""
                lang_str = f"  🟡 {lang}" if lang else ""
                private = "🔒 " if repo.get("private") else ""
                desc = repo.get("description") or "*(no description)*"
                topics = ""
                if repo.get("topics"):
                    topics = "  `" + "` `".join(repo["topics"][:5]) + "`"
                out_lines.append(
                    f"**{i}. [{private}{repo['full_name']}]({repo['html_url']})**  {stars}{lang_str}\n"
                    f"   {desc[:200]}{topics}\n"
                    f"   Updated: {repo['updated_at'][:10]}  |  "
                    f"Forks: {repo.get('forks_count', 0):,}  |  "
                    f"Open issues: {repo.get('open_issues_count', 0):,}"
                )

        elif search_type == "code":
            for i, item in enumerate(items, 1):
                repo_name = item["repository"]["full_name"]
                path = item["path"]
                url = item["html_url"]
                out_lines.append(
                    f"**{i}. [`{repo_name}/{path}`]({url})**\n"
                    f"   Repository: [{repo_name}]({item['repository']['html_url']})  "
                    f"⭐ {item['repository'].get('stargazers_count', 0)}"
                )

        elif search_type == "issues":
            for i, item in enumerate(items, 1):
                kind = "🟢 PR" if "pull_request" in item else "🔴 Issue"
                state = f" [{item['state']}]"
                labels = ""
                if item.get("labels"):
                    labels = (
                        "  "
                        + " ".join(
                            f"`{l['name']}`" for l in item["labels"][:4]
                        )
                    )
                out_lines.append(
                    f"**{i}. [{kind} #{item['number']}]({item['html_url']})**"
                    f"{state} — {item['title'][:120]}\n"
                    f"   {item['repository_url'].split('/repos/')[-1]}  |  "
                    f"by [{item['user']['login']}]({item['user']['html_url']})  |  "
                    f"{item['created_at'][:10]}{labels}"
                )

        elif search_type == "commits":
            for i, item in enumerate(items, 1):
                sha = item["sha"][:7]
                msg = item["commit"]["message"].split("\n")[0][:100]
                author = (
                    item["author"]["login"]
                    if item.get("author")
                    else item["commit"]["author"]["name"]
                )
                out_lines.append(
                    f"**{i}. [`{sha}`]({item['html_url']})** {msg}\n"
                    f"   {item['repository']['full_name']}  |  "
                    f"by {author}  |  {item['commit']['author']['date'][:10]}"
                )

        elif search_type == "users":
            for i, user in enumerate(items, 1):
                kind = "🏢 Org" if user.get("type") == "Organization" else "👤 User"
                bio = user.get("bio") or "*(no bio)*"
                out_lines.append(
                    f"**{i}. [{kind} {user['login']}]({user['html_url']})**\n"
                    f"   {bio[:150]}\n"
                    f"   Repos: {user.get('public_repos', '?')}  |  "
                    f"Followers: {user.get('followers', '?')}  |  "
                    f"Location: {user.get('location') or '—'}"
                )

        elif search_type == "topics":
            # topic results are just repos tagged with the topic
            for i, repo in enumerate(items, 1):
                stars = f"⭐ {repo['stargazers_count']:,}"
                lang = repo.get("language") or ""
                lang_str = f"  🟡 {lang}" if lang else ""
                desc = repo.get("description") or "*(no description)*"
                out_lines.append(
                    f"**{i}. [{repo['full_name']}]({repo['html_url']})**  {stars}{lang_str}\n"
                    f"   {desc[:200]}\n"
                    f"   Updated: {repo['updated_at'][:10]}"
                )

        # ── add qualifier reference ────────────────────────────
        out_lines.append("")
        out_lines.append(
            "---\n💡 **Tip:** Refine your search with qualifiers: `language:python`, "
            "`stars:>100`, `label:bug`, `repo:owner/name`, `state:open`, etc."
        )

        return "\n".join(out_lines)


    # ═══════════════════════════════════════════════════════════════
    #  CONVENIENCE SHORTCUTS  (same logic, shorter names)
    # ═══════════════════════════════════════════════════════════════

    async def github_search_repos(
        self, query: str, per_page: int = 10, __user__: Optional[dict] = None
    ) -> str:
        """Search public/private GitHub repositories. Shortcut for github_search(type='repositories')."""
        return await self.github_search(query, "repositories", per_page, __user__)

    async def github_search_code(
        self, query: str, per_page: int = 10, __user__: Optional[dict] = None
    ) -> str:
        """Search source code across all of GitHub. Shortcut for github_search(type='code')."""
        return await self.github_search(query, "code", per_page, __user__)

    async def github_search_issues(
        self, query: str, per_page: int = 10, __user__: Optional[dict] = None
    ) -> str:
        """Search issues and pull requests across GitHub. Shortcut for github_search(type='issues')."""
        return await self.github_search(query, "issues", per_page, __user__)

    async def github_search_commits(
        self, query: str, per_page: int = 10, __user__: Optional[dict] = None
    ) -> str:
        """Search commit messages across GitHub. Shortcut for github_search(type='commits')."""
        return await self.github_search(query, "commits", per_page, __user__)

    async def github_search_users(
        self, query: str, per_page: int = 10, __user__: Optional[dict] = None
    ) -> str:
        """Search GitHub users and organizations. Shortcut for github_search(type='users')."""
        return await self.github_search(query, "users", per_page, __user__)