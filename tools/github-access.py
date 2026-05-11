"""
GitHub Integration Tool
─────────────────────────
Read repo contents, manage issues, pull requests, branches,
commits, and GitHub Actions workflows — directly from OpenWebUI.
Uses the GitHub REST API with your Personal Access Token.

UserValves (set under Account → Valves):
  • GITHUB_TOKEN           – Your GitHub PAT (ghp_… or github_pat_…)
  • ENABLE_CONTENT         – Toggle repo browsing / file reading
  • ENABLE_CONTENT_WRITE   – Toggle branch creation & file writes (always via PR)
  • ENABLE_ISSUES          – Toggle issue read / write / comments
  • ENABLE_PULL_REQUESTS   – Toggle PR read / write / reviews / comments
  • ENABLE_WORKFLOWS       – Toggle GitHub Actions workflow read / trigger

Code-writing workflow (enforced):
  create_branch → create_or_update_file → create_pull_request
  (No direct main writes, no merge function.)
"""

from pydantic import BaseModel, Field
from typing import Optional
import httpx
import base64
import json


class Tools:
    class Valves(BaseModel):
        """Global / admin settings – left empty on purpose."""

        pass

    class UserValves(BaseModel):
        GITHUB_TOKEN: str = Field(
            default="",
            description="GitHub Personal Access Token (classic or fine-grained)",
        )
        ENABLE_CONTENT: bool = Field(
            default=True,
            description="Enable tools: list repos, read files, search code",
        )
        ENABLE_CONTENT_WRITE: bool = Field(
            default=True,
            description="Enable tools: create branches, write/delete files (always via PR workflow)",
        )
        ENABLE_ISSUES: bool = Field(
            default=True,
            description="Enable tools: list, create, update, close, comment on issues",
        )
        ENABLE_PULL_REQUESTS: bool = Field(
            default=True,
            description="Enable tools: list, create PRs, request reviewers, comment (NO merge)",
        )
        ENABLE_WORKFLOWS: bool = Field(
            default=True,
            description="Enable tools: list workflows/runs, view logs, trigger, cancel, rerun",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── helpers ────────────────────────────────────────────────────

    def _get_valves(self, __user__: Optional[dict]) -> "Tools.UserValves":
        """Extract UserValves from the __user__ dict injected by OpenWebUI."""
        if __user__ and "valves" in __user__:
            return __user__["valves"]
        return self.UserValves()

    def _auth(self, uv: "Tools.UserValves") -> dict:
        t = uv.GITHUB_TOKEN
        if not t:
            raise ValueError(
                "❌ No GitHub token set. Go to Account → Valves "
                "and enter your PAT in GITHUB_TOKEN."
            )
        return {
            "Authorization": f"Bearer {t}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _guard(self, flag: bool, name: str) -> None:
        if not flag:
            raise ValueError(f"⛔ {name} is disabled in your User Valves.")

    # ═══════════════════════════════════════════════════════════════
    #  CONTENT – Reading
    # ═══════════════════════════════════════════════════════════════

    async def list_my_repos(self, __user__: Optional[dict] = None) -> str:
        """
        List your own GitHub repositories (most recently updated first).
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = "https://api.github.com/user/repos?sort=updated&per_page=30"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            repos = r.json()
            out = [
                f"{'🔒' if r['private'] else '🌐'} **{r['full_name']}**  ⭐{r['stargazers_count']}"
                for r in repos
            ]
            return "\n".join(out[:25]) + (
                f"\n\n… and {len(repos) - 25} more" if len(repos) > 25 else ""
            )

    async def list_user_repos(
        self, username: str, __user__: Optional[dict] = None
    ) -> str:
        """
        List public repositories of any GitHub user.

        :param username: GitHub username
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=30"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            repos = r.json()
            out = [f"🌐 **{r['full_name']}**  ⭐{r['stargazers_count']}" for r in repos]
            return (
                "\n".join(out[:25])
                if out
                else f"User '{username}' has no public repos."
            )

    async def get_repo(self, repo: str, __user__: Optional[dict] = None) -> str:
        """
        Get metadata about a single repository.

        :param repo: Repository as 'owner/name' (e.g. 'simeon/my-project')
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            d = r.json()
            return json.dumps(
                {
                    "full_name": d["full_name"],
                    "description": d.get("description"),
                    "private": d["private"],
                    "default_branch": d["default_branch"],
                    "stars": d["stargazers_count"],
                    "forks": d["forks_count"],
                    "open_issues": d["open_issues_count"],
                    "language": d.get("language"),
                    "topics": d.get("topics", []),
                    "created_at": d["created_at"],
                    "updated_at": d["updated_at"],
                    "html_url": d["html_url"],
                    "clone_url": d["clone_url"],
                },
                indent=2,
                ensure_ascii=False,
            )

    async def get_file(
        self, repo: str, path: str, ref: str = "main", __user__: Optional[dict] = None
    ) -> str:
        """
        Read a file or directory from a GitHub repository.

        :param repo: Repository as 'owner/name' (e.g. 'simeon/my-project')
        :param path: File path inside the repo (e.g. 'src/main.py')
        :param ref: Branch, tag or commit SHA (default: main)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            if r.status_code == 404:
                return f"❌ Not found: {repo}/{path}"
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                items = [
                    f"{'📁' if d['type'] == 'dir' else '📄'} {d['name']}" for d in data
                ]
                return f"**{repo}/{path}**\n" + "\n".join(items)
            content = base64.b64decode(data["content"]).decode(
                "utf-8", errors="replace"
            )
            lang = path.rsplit(".", 1)[-1] if "." in path else ""
            return (
                f"**{repo}/{path}** ({data['size']} bytes)\n```{lang}\n{content}\n```"
            )

    async def search_code(
        self, repo: str, query: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Search code in a GitHub repository.

        :param repo: Repository as 'owner/name'
        :param query: Search term (e.g. 'function login' or 'import os')
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/search/code?q={query}+repo:{repo}&per_page=10"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            data = r.json()
            if data["total_count"] == 0:
                return f"No results for '{query}' in {repo}."
            items = [f"- `{i['path']}`" for i in data["items"]]
            return f"Found **{data['total_count']}** results in {repo}:\n" + "\n".join(
                items
            )

    # ═══════════════════════════════════════════════════════════════
    #  CONTENT – Writing (Branch → File → PR workflow ONLY)
    # ═══════════════════════════════════════════════════════════════

    async def create_branch(
        self,
        repo: str,
        branch: str,
        from_branch: str = "main",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create a new branch from an existing branch (default: main).

        :param repo: Repository 'owner/name'
        :param branch: New branch name (e.g. 'feature/add-login')
        :param from_branch: Source branch to branch off from (default: main)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT_WRITE, "Content Write")
        ref_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{from_branch}"
        async with httpx.AsyncClient() as c:
            r = await c.get(ref_url, headers=self._auth(uv))
            r.raise_for_status()
            sha = r.json()["object"]["sha"]

            create_url = f"https://api.github.com/repos/{repo}/git/refs"
            payload = {"ref": f"refs/heads/{branch}", "sha": sha}
            r2 = await c.post(create_url, headers=self._auth(uv), json=payload)
            if r2.status_code == 422:
                return f"❌ Branch '{branch}' already exists in {repo}."
            r2.raise_for_status()
            return f"✅ Branch **{branch}** created in {repo} (from {from_branch})."

    async def create_or_update_file(
        self,
        repo: str,
        path: str,
        content: str,
        branch: str,
        message: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create or update a file on a branch (NOT on main — use via PR workflow).

        :param repo: Repository 'owner/name'
        :param path: File path inside the repo (e.g. 'docs/readme.md')
        :param content: New file content (plain text)
        :param branch: Branch to commit to (REQUIRED — new branch from create_branch)
        :param message: Commit message (auto-generated if empty)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT_WRITE, "Content Write")

        if not message:
            action = (
                "Update"
                if await self._file_exists(repo, path, branch, uv)
                else "Create"
            )
            message = f"{action} {path}"

        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        payload: dict = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }

        existing_sha = await self._file_sha(repo, path, branch, uv)
        if existing_sha:
            payload["sha"] = existing_sha

        async with httpx.AsyncClient() as c:
            r = await c.put(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            d = r.json()
            return (
                f"✅ **{d['content']['name']}** committed to `{branch}`\n"
                f"Commit: {d['commit']['sha'][:7]} — {d['commit']['message']}\n"
                f"URL: {d['content']['html_url']}"
            )

    async def delete_file(
        self,
        repo: str,
        path: str,
        branch: str,
        message: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Delete a file on a branch (NOT on main — use via PR workflow).

        :param repo: Repository 'owner/name'
        :param path: File path to delete
        :param branch: Branch to delete from
        :param message: Commit message (auto-generated if empty)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT_WRITE, "Content Write")

        sha = await self._file_sha(repo, path, branch, uv)
        if not sha:
            return f"❌ File '{path}' not found on branch '{branch}' in {repo}."

        if not message:
            message = f"Delete {path}"

        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        payload = {"message": message, "sha": sha, "branch": branch}

        async with httpx.AsyncClient() as c:
            r = await c.delete(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            return f"✅ **{path}** deleted from `{branch}`."

    # ── internal helpers ────────────────────────────────────────

    async def _file_exists(
        self, repo: str, path: str, ref: str, uv: "Tools.UserValves"
    ) -> bool:
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            return r.status_code == 200

    async def _file_sha(
        self, repo: str, path: str, ref: str, uv: "Tools.UserValves"
    ) -> str | None:
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            if r.status_code == 200:
                return r.json()["sha"]
            return None

    # ═══════════════════════════════════════════════════════════════
    #  BRANCHES – Reading
    # ═══════════════════════════════════════════════════════════════

    async def list_branches(self, repo: str, __user__: Optional[dict] = None) -> str:
        """
        List all branches in a repository.

        :param repo: Repository 'owner/name'
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}/branches?per_page=30"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            branches = r.json()
            if not branches:
                return f"No branches in {repo}."
            out = [f"🔀 **{b['name']}**  ({b['commit']['sha'][:7]})" for b in branches]
            return f"**{repo}** branches ({len(branches)}):\n" + "\n".join(out)

    # ═══════════════════════════════════════════════════════════════
    #  COMMITS – Reading
    # ═══════════════════════════════════════════════════════════════

    async def list_commits(
        self,
        repo: str,
        branch: str = "main",
        per_page: int = 20,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        List recent commits on a branch.

        :param repo: Repository 'owner/name'
        :param branch: Branch name (default: main)
        :param per_page: Number of commits (default: 20, max: 100)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}/commits?sha={branch}&per_page={min(per_page, 100)}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            commits = r.json()
            if not commits:
                return f"No commits on {branch} in {repo}."
            out = []
            for c in commits:
                sha = c["sha"][:7]
                msg = c["commit"]["message"].split("\n")[0]
                author = (
                    c["commit"]["author"]["name"]
                    if c.get("commit", {}).get("author")
                    else c["author"]["login"] if c.get("author") else "?"
                )
                out.append(f"`{sha}` **{msg}** — {author}")
            return f"**{repo}** ({branch}) last {len(out)} commits:\n" + "\n".join(out)

    async def get_commit(
        self, repo: str, sha: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Get detailed info about a single commit, including the diff.

        :param repo: Repository 'owner/name'
        :param sha: Commit SHA (full or short)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            c = r.json()
            files_out = []
            for f in c.get("files", [])[:20]:
                files_out.append(
                    f"  {f['status']:>7}  +{f['additions']:<4} -{f['deletions']:<4}  {f['filename']}"
                )
            return (
                f"## {c['sha'][:7]} {c['commit']['message'].split(chr(10))[0]}\n"
                f"**Author:** {c['commit']['author']['name']}  |  "
                f"**Date:** {c['commit']['author']['date']}\n"
                f"**URL:** {c['html_url']}\n\n"
                + ("Changed files:\n" + "\n".join(files_out) if files_out else "")
            )

    async def compare_branches(
        self, repo: str, base: str, head: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Compare two branches and show the diff summary.

        :param repo: Repository 'owner/name'
        :param base: Base branch (e.g. 'main')
        :param head: Head branch (e.g. 'feature/xyz')
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_CONTENT, "Content")
        url = f"https://api.github.com/repos/{repo}/compare/{base}...{head}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            d = r.json()
            status = d.get("status", "unknown")
            ahead = d.get("ahead_by", 0)
            behind = d.get("behind_by", 0)
            commits = d.get("commits", [])
            files = d.get("files", [])

            commit_lines = [
                f"  `{c['sha'][:7]}` {c['commit']['message'].split(chr(10))[0]}"
                for c in commits[:10]
            ]
            file_lines = [f"  {f['status']:>7}  {f['filename']}" for f in files[:15]]

            return (
                f"## {base} ← {head}\n"
                f"**Status:** {status}  |  Ahead: {ahead}  |  Behind: {behind}\n"
                + (
                    f"\nCommits ({len(commits)}):\n" + "\n".join(commit_lines)
                    if commit_lines
                    else ""
                )
                + ("\n…and more" if len(commits) > 10 else "")
                + (
                    f"\n\nFiles ({len(files)}):\n" + "\n".join(file_lines)
                    if file_lines
                    else ""
                )
                + ("\n…and more" if len(files) > 15 else "")
            )

    # ═══════════════════════════════════════════════════════════════
    #  ISSUES
    # ═══════════════════════════════════════════════════════════════

    async def list_issues(
        self,
        repo: str,
        state: str = "open",
        labels: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        List issues in a GitHub repository.

        :param repo: Repository 'owner/name'
        :param state: 'open', 'closed', or 'all' (default: open)
        :param labels: Comma-separated labels to filter by (optional)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page=25"
        if labels:
            url += f"&labels={labels}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            items = r.json()
            issues = [i for i in items if "pull_request" not in i]
            if not issues:
                return f"No {state} issues in {repo}."
            out = [
                f"#{i['number']} **{i['title']}**  [{i['state']}]  {i['user']['login']}"
                for i in issues
            ]
            return "\n".join(out)

    async def search_issues(
        self,
        repo: str,
        query: str,
        state: str = "open",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Search issues by text in title/body.

        :param repo: Repository 'owner/name'
        :param query: Search term
        :param state: 'open', 'closed', or 'all' (default: open)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = (
            f"https://api.github.com/search/issues"
            f"?q={query}+repo:{repo}+type:issue+state:{state}&per_page=15"
        )
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            data = r.json()
            if data["total_count"] == 0:
                return f"No issues matching '{query}' in {repo}."
            out = [
                f"#{i['number']} **{i['title']}**  [{i['state']}]  {i['user']['login']}"
                for i in data["items"]
            ]
            return f"Found **{data['total_count']}** issues:\n" + "\n".join(out)

    async def get_issue(
        self, repo: str, issue_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        Get full details of a single issue.

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            i = r.json()
            labels = ", ".join(l["name"] for l in i.get("labels", [])) or "none"
            assignees = ", ".join(a["login"] for a in i.get("assignees", [])) or "none"
            return (
                f"## #{i['number']} {i['title']}\n"
                f"**State:** {i['state']}  |  **Author:** {i['user']['login']}  |  "
                f"**Created:** {i['created_at']}\n"
                f"**Labels:** {labels}  |  **Assignees:** {assignees}\n"
                f"**URL:** {i['html_url']}\n\n"
                f"{i['body'] or '_(no description)_'}"
            )

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str = "",
        labels: str = "",
        assignees: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create a new issue.

        :param repo: Repository 'owner/name'
        :param title: Issue title
        :param body: Issue body (GitHub markdown)
        :param labels: Comma-separated labels (optional)
        :param assignees: Comma-separated GitHub usernames (optional)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues"
        payload: dict = {"title": title, "body": body}
        if labels:
            payload["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
        if assignees:
            payload["assignees"] = [
                a.strip() for a in assignees.split(",") if a.strip()
            ]
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            i = r.json()
            return f"✅ Issue **#{i['number']}** created: {i['title']}\n{i['html_url']}"

    async def update_issue(
        self,
        repo: str,
        issue_number: int,
        title: str = "",
        body: str = "",
        state: str = "",
        labels: str = "",
        assignees: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Update an issue (title, body, state, labels, assignees).

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number
        :param title: New title (omit to keep current)
        :param body: New body (omit to keep current)
        :param state: 'open' or 'closed' (omit to keep current)
        :param labels: Comma-separated labels (replaces existing; omit to keep)
        :param assignees: Comma-separated usernames (replaces existing; omit to keep)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
        payload: dict = {}
        if title:
            payload["title"] = title
        if body:
            payload["body"] = body
        if state:
            payload["state"] = state
        if labels:
            payload["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
        if assignees:
            payload["assignees"] = [
                a.strip() for a in assignees.split(",") if a.strip()
            ]
        if not payload:
            return "❌ No fields to update."
        async with httpx.AsyncClient() as c:
            r = await c.patch(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            i = r.json()
            return f"✅ Issue **#{i['number']}** updated.\n{i['html_url']}"

    async def close_issue(
        self, repo: str, issue_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        Close an issue.

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number to close
        """
        return await self.update_issue(
            repo, issue_number, state="closed", __user__=__user__
        )

    async def reopen_issue(
        self, repo: str, issue_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        Reopen a closed issue.

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number to reopen
        """
        return await self.update_issue(
            repo, issue_number, state="open", __user__=__user__
        )

    async def add_issue_comment(
        self, repo: str, issue_number: int, body: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Add a comment to an issue.

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number
        :param body: Comment text (GitHub markdown)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json={"body": body})
            r.raise_for_status()
            d = r.json()
            return f"✅ Comment added to **#{issue_number}**\n{d['html_url']}"

    async def list_issue_comments(
        self, repo: str, issue_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        List comments on an issue.

        :param repo: Repository 'owner/name'
        :param issue_number: Issue number
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_ISSUES, "Issues")
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments?per_page=20"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            comments = r.json()
            if not comments:
                return f"No comments on #{issue_number}."
            out = [
                f"**{c['user']['login']}** ({c['created_at']}):\n{c['body'][:200]}"
                + ("…" if len(c["body"]) > 200 else "")
                for c in comments
            ]
            return f"## Comments on #{issue_number}\n\n" + "\n\n".join(out)

    # ═══════════════════════════════════════════════════════════════
    #  PULL REQUESTS
    # ═══════════════════════════════════════════════════════════════

    async def list_pull_requests(
        self, repo: str, state: str = "open", __user__: Optional[dict] = None
    ) -> str:
        """
        List pull requests in a GitHub repository.

        :param repo: Repository 'owner/name'
        :param state: 'open', 'closed', or 'all' (default: open)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/pulls?state={state}&per_page=20"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            prs = r.json()
            if not prs:
                return f"No {state} PRs in {repo}."
            out = [
                f"#{p['number']} **{p['title']}**  [{p['state']}]  "
                f"{p['user']['login']}  ({p['head']['ref']} → {p['base']['ref']})"
                for p in prs
            ]
            return "\n".join(out)

    async def get_pull_request(
        self, repo: str, pr_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        Get full details of a single pull request.

        :param repo: Repository 'owner/name'
        :param pr_number: Pull request number
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            p = r.json()
            requested = (
                ", ".join(r["login"] for r in p.get("requested_reviewers", []))
                or "none"
            )
            return (
                f"## #{p['number']} {p['title']}\n"
                f"**State:** {p['state']}  |  **Mergeable:** {p.get('mergeable', '?')}"
                f"  |  **Author:** {p['user']['login']}\n"
                f"**Branch:** {p['head']['ref']} → {p['base']['ref']}\n"
                f"**Requested reviewers:** {requested}\n"
                f"**URL:** {p['html_url']}\n\n"
                f"{p['body'] or '_(no description)_'}"
            )

    async def get_pr_files(
        self, repo: str, pr_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        List files changed in a pull request with stats.

        :param repo: Repository 'owner/name'
        :param pr_number: Pull request number
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            files = r.json()
            if not files:
                return "No files changed."
            out = [
                f"{'+' + str(f['additions']):>5} {'-' + str(f['deletions']):>5}  {f['status']:>7}  {f['filename']}"
                for f in files
            ]
            return f"**#{pr_number}** changed files ({len(files)}):\n" + "\n".join(out)

    async def create_pull_request(
        self,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        draft: bool = False,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create a new pull request.

        :param repo: Repository 'owner/name'
        :param title: PR title
        :param head: Source branch name
        :param base: Target branch (default: main)
        :param body: PR description (markdown)
        :param draft: Create as draft PR (default: False)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/pulls"
        payload = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "draft": draft,
        }
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            p = r.json()
            return (
                f"✅ PR **#{p['number']}** created: {p['title']}\n"
                f"{p['html_url']}  {'(draft)' if draft else '(ready for review)'}"
            )

    async def request_reviewers(
        self,
        repo: str,
        pr_number: int,
        reviewers: str,
        team_reviewers: str = "",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Request reviewers for a pull request.

        :param repo: Repository 'owner/name'
        :param pr_number: Pull request number
        :param reviewers: Comma-separated GitHub usernames
        :param team_reviewers: Comma-separated team slugs (optional)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = (
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/requested_reviewers"
        )
        payload: dict = {
            "reviewers": [r.strip() for r in reviewers.split(",") if r.strip()],
        }
        if team_reviewers:
            payload["team_reviewers"] = [
                t.strip() for t in team_reviewers.split(",") if t.strip()
            ]
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json=payload)
            r.raise_for_status()
            p = r.json()
            requested = (
                ", ".join(r["login"] for r in p.get("requested_reviewers", []))
                or "none"
            )
            return f"✅ Reviewers requested for **#{pr_number}**: {requested}\n{p['html_url']}"

    async def add_pr_comment(
        self, repo: str, pr_number: int, body: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Add a general comment to a pull request.

        :param repo: Repository 'owner/name'
        :param pr_number: Pull request number
        :param body: Comment text (GitHub markdown)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json={"body": body})
            r.raise_for_status()
            d = r.json()
            return f"✅ Comment added to PR **#{pr_number}**\n{d['html_url']}"

    async def list_pr_comments(
        self, repo: str, pr_number: int, __user__: Optional[dict] = None
    ) -> str:
        """
        List comments on a pull request.

        :param repo: Repository 'owner/name'
        :param pr_number: Pull request number
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_PULL_REQUESTS, "Pull Requests")
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=20"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            comments = r.json()
            if not comments:
                return f"No comments on PR #{pr_number}."
            out = [
                f"**{c['user']['login']}** ({c['created_at']}):\n{c['body'][:200]}"
                + ("…" if len(c["body"]) > 200 else "")
                for c in comments
            ]
            return f"## Comments on PR #{pr_number}\n\n" + "\n\n".join(out)

    # ═══════════════════════════════════════════════════════════════
    #  WORKFLOWS (GitHub Actions)
    # ═══════════════════════════════════════════════════════════════

    async def list_workflows(self, repo: str, __user__: Optional[dict] = None) -> str:
        """
        List all GitHub Actions workflows in a repository.

        :param repo: Repository 'owner/name'
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/workflows?per_page=30"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            data = r.json()
            wfs = data.get("workflows", [])
            if not wfs:
                return f"No workflows in {repo}."
            out = [
                f"⚙️ **{w['name']}**  `{w['path']}`  [{w['state']}]  (ID: {w['id']})"
                for w in wfs
            ]
            return "\n".join(out)

    async def get_workflow(
        self, repo: str, workflow_id: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Get details of a single workflow.

        :param repo: Repository 'owner/name'
        :param workflow_id: Workflow ID or workflow file name (e.g. 'deploy.yml')
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            w = r.json()
            return json.dumps(
                {
                    "id": w["id"],
                    "name": w["name"],
                    "path": w["path"],
                    "state": w["state"],
                    "created_at": w["created_at"],
                    "updated_at": w["updated_at"],
                    "url": w["html_url"],
                },
                indent=2,
                ensure_ascii=False,
            )

    async def list_workflow_runs(
        self,
        repo: str,
        workflow_id: str = "",
        branch: str = "",
        status: str = "",
        per_page: int = 15,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        List recent workflow runs.

        :param repo: Repository 'owner/name'
        :param workflow_id: Filter by workflow ID or filename (optional)
        :param branch: Filter by branch (optional)
        :param status: Filter by status: 'completed', 'in_progress', 'queued' (optional)
        :param per_page: Number of runs (default: 15, max: 30)
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        if workflow_id:
            url = (
                f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/runs"
                f"?per_page={min(per_page, 30)}"
            )
        else:
            url = f"https://api.github.com/repos/{repo}/actions/runs?per_page={min(per_page, 30)}"
        if branch:
            url += f"&branch={branch}"
        if status:
            url += f"&status={status}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            data = r.json()
            runs = data.get("workflow_runs", [])
            if not runs:
                return "No workflow runs found."
            out = []
            for run in runs:
                icon = {"completed": "✅", "in_progress": "🔄", "queued": "⏳"}.get(
                    run["status"], "❓"
                )
                conclusion = (
                    f" → {run.get('conclusion', '')}" if run.get("conclusion") else ""
                )
                out.append(
                    f"{icon} **#{run['id']}** {run['name']}  "
                    f"[{run['status']}{conclusion}]  `{run['head_branch']}`  "
                    f"({run['created_at']})"
                )
            return f"**{repo}** workflow runs:\n" + "\n".join(out)

    async def get_workflow_run(
        self, repo: str, run_id: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Get detailed info about a single workflow run.

        :param repo: Repository 'owner/name'
        :param run_id: Run ID
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            run = r.json()
            return (
                f"## Run #{run['id']} — {run['name']}\n"
                f"**Workflow:** {run.get('workflow_id')}  |  "
                f"**Status:** {run['status']}  |  "
                f"**Conclusion:** {run.get('conclusion', 'N/A')}\n"
                f"**Branch:** {run['head_branch']}  |  "
                f"**Trigger:** {run['event']}\n"
                f"**Created:** {run['created_at']}  |  "
                f"**Updated:** {run['updated_at']}\n"
                f"**URL:** {run['html_url']}"
            )

    async def get_workflow_run_logs(
        self, repo: str, run_id: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Get logs of a workflow run (as zip download URL – best viewed in browser).

        :param repo: Repository 'owner/name'
        :param run_id: Run ID
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
        async with httpx.AsyncClient(follow_redirects=False) as c:
            r = await c.get(url, headers=self._auth(uv))
            r.raise_for_status()
            location = r.headers.get("Location", "")
            if not location:
                return "❌ No logs available."
            return f"📥 Logs download URL: {location}\n_(Opens in browser – a zip of all job logs)_"

    async def trigger_workflow_dispatch(
        self,
        repo: str,
        workflow_id: str,
        ref: str,
        inputs: str = "{}",
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Manually trigger a workflow_dispatch event.

        :param repo: Repository 'owner/name'
        :param workflow_id: Workflow ID or file name (e.g. 'deploy.yml')
        :param ref: Branch or tag to run on (e.g. 'main')
        :param inputs: JSON object of workflow inputs (default: '{}'), e.g. '{"env":"staging"}'
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/dispatches"
        try:
            parsed_inputs = json.loads(inputs)
        except json.JSONDecodeError:
            return '❌ Invalid JSON for inputs. Use format: \'{"key":"value"}\''
        payload = {"ref": ref, "inputs": parsed_inputs}
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv), json=payload)
            if r.status_code == 204:
                return f"🚀 Workflow **{workflow_id}** triggered on `{ref}`."
            r.raise_for_status()
            return f"✅ Triggered (unexpected status {r.status_code})."

    async def cancel_workflow_run(
        self, repo: str, run_id: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Cancel a running workflow.

        :param repo: Repository 'owner/name'
        :param run_id: Run ID to cancel
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/cancel"
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv))
            if r.status_code == 202:
                return f"🛑 Workflow run **#{run_id}** cancelled."
            r.raise_for_status()
            return f"✅ Cancelled (status {r.status_code})."

    async def rerun_workflow(
        self, repo: str, run_id: str, __user__: Optional[dict] = None
    ) -> str:
        """
        Rerun a failed workflow run.

        :param repo: Repository 'owner/name'
        :param run_id: Run ID to rerun
        """
        uv = self._get_valves(__user__)
        self._guard(uv.ENABLE_WORKFLOWS, "Workflows")
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/rerun"
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._auth(uv))
            if r.status_code == 201:
                return f"🔄 Workflow run **#{run_id}** rerunning."
            r.raise_for_status()
            return f"✅ Rerunning (status {r.status_code})."
