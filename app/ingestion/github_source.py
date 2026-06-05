"""GitHub ingestion source (spec §15.3).

:class:`GitHubSource` pulls a public profile's most relevant repositories via the
GitHub REST API (using :class:`httpx.AsyncClient`) and turns them into corpus
documents:

* one ``github_repo`` summary per repo (name, description, language, topics,
  stars);
* one ``github_readme`` per repo (decoded README);
* one ``github_commit`` per repo (recent commit history, one line per commit);
* up to ``github_max_source_files_per_repo`` ``github_source`` documents
  (representative source files).

Every network call is wrapped with :mod:`tenacity` for transient errors and is
rate-limit aware: when ``X-RateLimit-Remaining`` hits zero the source stops
issuing further requests for the remainder of the run and skips gracefully.
404/403/422 and similar terminal responses are logged and skipped, never raised,
so a single bad repo cannot abort the whole ingestion.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.ingestion.base import Source, slug
from app.rag.schemas import Document

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_API_VERSION = "2022-11-28"

# Path fragments that almost never carry signal worth embedding.
_EXCLUDED_PATH_FRAGMENTS = (
    "node_modules/",
    "vendor/",
    "dist/",
    "build/",
    ".min.",
    "/tests/",
    "test/",
    "__tests__/",
    "/.github/",
    "site-packages/",
)

# Path fragments that signal "primary" source we should prefer.
_PREFERRED_PATH_FRAGMENTS = ("src/", "app/", "lib/", "readme")

# Exceptions worth retrying: transient transport errors. HTTP status errors are
# inspected explicitly (we do not blindly retry 4xx).
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    httpx.TimeoutException,
)


class _RateLimitExhausted(Exception):
    """Raised internally when the GitHub rate limit is exhausted."""


class GitHubSource(Source):
    """Ingest a GitHub profile's repositories into corpus documents."""

    name = "github"

    def __init__(self, settings: Settings) -> None:
        """Initialize the source and build an authenticated HTTP client.

        Args:
            settings: Application settings carrying the GitHub token/username and
                the various per-repo limits.
        """
        self._settings = settings
        self._username = (settings.github_username or "").strip()

        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
            "User-Agent": "ai-persona-ingestion",
        }
        token = (settings.github_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.AsyncClient(
            base_url=_GITHUB_API_BASE,
            headers=headers,
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        self._attempts = max(1, int(settings.openai_max_retries) + 1)
        # Once True for the run, no further requests are issued.
        self._rate_limited = False

    # --------------------------------------------------------------- public API
    async def load(self) -> list[Document]:
        """Load documents for the configured GitHub user.

        Returns:
            A list of documents across all selected repositories. Empty if no
            username is configured or the API is entirely unreachable. Always
            closes the underlying HTTP client before returning.
        """
        if not self._username:
            logger.warning(
                "github_username is not configured; skipping GitHub ingestion"
            )
            await self._client.aclose()
            return []

        documents: list[Document] = []
        try:
            repos = await self._fetch_repos()
            if not repos:
                logger.warning("No repositories found for user %s", self._username)
                return []

            selected = self._select_repos(repos)
            logger.info(
                "Selected %d of %d repo(s) for user %s",
                len(selected),
                len(repos),
                self._username,
            )

            for repo in selected:
                if self._rate_limited:
                    logger.warning(
                        "GitHub rate limit exhausted; skipping remaining repos"
                    )
                    break
                try:
                    documents.extend(await self._documents_for_repo(repo))
                except _RateLimitExhausted:
                    self._rate_limited = True
                    logger.warning(
                        "GitHub rate limit exhausted while processing %s; "
                        "skipping remaining repos",
                        repo.get("full_name"),
                    )
                    break
                except Exception:  # pragma: no cover - per-repo isolation
                    logger.warning(
                        "Failed to ingest repo %s; skipping",
                        repo.get("full_name"),
                        exc_info=True,
                    )
        finally:
            await self._client.aclose()

        logger.info(
            "GitHub ingestion produced %d document(s) for %s",
            len(documents),
            self._username,
        )
        return documents

    # ------------------------------------------------------------ HTTP helpers
    async def _request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> httpx.Response | None:
        """Issue a GET request with retry + rate-limit handling.

        Args:
            path: API path (relative to the base URL) or absolute URL.
            params: Optional query parameters.
            accept: Optional ``Accept`` header override (e.g. raw content).

        Returns:
            The successful :class:`httpx.Response`, or ``None`` for terminal,
            skip-worthy responses (404, 403-not-rate-limit, 422, ...).

        Raises:
            _RateLimitExhausted: When the primary rate limit is exhausted.
        """
        if self._rate_limited:
            raise _RateLimitExhausted(path)

        headers = {"Accept": accept} if accept else None

        async def _do() -> httpx.Response:
            return await self._client.get(path, params=params, headers=headers)

        try:
            async for attempt in self._retrying():
                with attempt:
                    response = await _do()
        except _RETRYABLE_EXCEPTIONS:
            logger.warning("Transient error fetching %s; skipping", path, exc_info=True)
            return None

        self._note_rate_limit(response)

        if response.status_code == 200:
            return response

        if response.status_code == 403 and self._is_rate_limited(response):
            self._rate_limited = True
            raise _RateLimitExhausted(path)

        # 404 (missing readme/tree), 403 (forbidden), 422, etc. are non-fatal.
        logger.warning(
            "GitHub request %s returned HTTP %d; skipping",
            path,
            response.status_code,
        )
        return None

    def _retrying(self) -> AsyncRetrying:
        """Build a fresh tenacity controller for a single logical request."""
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=15.0),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        """Return True if ``response`` indicates the primary rate limit is hit."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        return remaining == "0"

    def _note_rate_limit(self, response: httpx.Response) -> None:
        """Log remaining rate-limit budget; flip the kill-switch when at zero."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is None:
            return
        try:
            remaining_int = int(remaining)
        except ValueError:
            return
        if remaining_int <= 0:
            self._rate_limited = True
            logger.warning("GitHub rate limit reached (remaining=%s)", remaining)
        elif remaining_int <= 5:
            logger.info("GitHub rate limit low (remaining=%s)", remaining_int)

    # ---------------------------------------------------------- repo selection
    async def _fetch_repos(self) -> list[dict[str, Any]]:
        """Fetch up to 100 of the user's repositories sorted by recent push."""
        response = await self._request(
            f"/users/{self._username}/repos",
            params={"sort": "pushed", "per_page": 100, "type": "owner"},
        )
        if response is None:
            return []
        try:
            data = response.json()
        except ValueError:
            logger.warning("Could not decode repos response for %s", self._username)
            return []
        return data if isinstance(data, list) else []

    def _select_repos(self, repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter forks (per setting) and take the top-N by stars then push date."""
        candidates = [
            repo
            for repo in repos
            if self._settings.github_include_forks or not repo.get("fork", False)
        ]
        candidates.sort(
            key=lambda repo: (
                int(repo.get("stargazers_count") or 0),
                str(repo.get("pushed_at") or ""),
            ),
            reverse=True,
        )
        return candidates[: max(0, int(self._settings.github_max_repos))]

    # ------------------------------------------------------- per-repo documents
    async def _documents_for_repo(self, repo: dict[str, Any]) -> list[Document]:
        """Build all documents (summary/readme/commits/source) for one repo."""
        full_name = str(repo.get("full_name") or "")
        if not full_name or "/" not in full_name:
            logger.debug("Skipping repo with malformed full_name: %r", full_name)
            return []
        owner, name = full_name.split("/", 1)
        html_url = str(repo.get("html_url") or f"{_GITHUB_API_BASE}/{full_name}")

        documents: list[Document] = [self._repo_summary_document(repo, full_name, html_url)]

        readme = await self._fetch_readme(owner, name, full_name, html_url)
        if readme is not None:
            documents.append(readme)

        commits = await self._fetch_commits(owner, name, full_name, html_url)
        if commits is not None:
            documents.append(commits)

        documents.extend(
            await self._fetch_source_files(owner, name, full_name, html_url, repo)
        )
        return documents

    def _repo_summary_document(
        self, repo: dict[str, Any], full_name: str, html_url: str
    ) -> Document:
        """Compose the ``github_repo`` summary document."""
        description = str(repo.get("description") or "").strip()
        language = str(repo.get("language") or "").strip()
        topics = repo.get("topics") or []
        topics = [str(t) for t in topics if t] if isinstance(topics, list) else []
        stars = int(repo.get("stargazers_count") or 0)
        forks = int(repo.get("forks_count") or 0)
        pushed_at = str(repo.get("pushed_at") or "")

        lines = [
            f"Repository: {full_name}",
            f"Description: {description}" if description else "Description: (none)",
            f"Primary language: {language}" if language else "Primary language: (unknown)",
            f"Topics: {', '.join(topics)}" if topics else "Topics: (none)",
            f"Stars: {stars}",
            f"Forks: {forks}",
            f"Last pushed: {pushed_at}" if pushed_at else "",
            f"URL: {html_url}",
        ]
        text = "\n".join(line for line in lines if line)

        metadata: dict[str, Any] = {
            "language": language,
            "stars": stars,
            "forks": forks,
        }
        if topics:
            metadata["topics"] = ", ".join(topics)

        return Document(
            id=f"github_repo:{slug(full_name)}",
            text=text,
            source_type="github_repo",
            source_id=full_name,
            title=f"{full_name} (repository)",
            url=html_url,
            metadata=metadata,
        )

    async def _fetch_readme(
        self, owner: str, name: str, full_name: str, html_url: str
    ) -> Document | None:
        """Fetch and decode the repo README, if present."""
        response = await self._request(f"/repos/{owner}/{name}/readme")
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            logger.warning("Could not decode README JSON for %s", full_name)
            return None

        content = self._decode_content(payload)
        if not content or not content.strip():
            return None

        readme_url = f"{html_url}#readme"
        return Document(
            id=f"github_readme:{slug(full_name)}",
            text=content,
            source_type="github_readme",
            source_id=full_name,
            title=f"{full_name} README",
            url=readme_url,
            metadata={"path": str(payload.get("path") or "README")},
        )

    @staticmethod
    def _decode_content(payload: dict[str, Any]) -> str:
        """Decode a GitHub ``content`` blob (base64) into text."""
        raw = payload.get("content")
        if not isinstance(raw, str) or not raw:
            return ""
        encoding = str(payload.get("encoding") or "base64")
        if encoding != "base64":
            return raw
        try:
            decoded = base64.b64decode(raw)
            return decoded.decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            logger.warning("Failed to base64-decode GitHub content blob")
            return ""

    async def _fetch_commits(
        self, owner: str, name: str, full_name: str, html_url: str
    ) -> Document | None:
        """Fetch recent commits and join them into one commit-history document."""
        # GitHub silently caps per_page at 100; clamp so a misconfigured value
        # >100 doesn't masquerade as a larger request (we don't paginate here).
        per_page = min(100, max(1, int(self._settings.github_max_commits_per_repo)))
        response = await self._request(
            f"/repos/{owner}/{name}/commits",
            params={"per_page": per_page},
        )
        if response is None:
            return None
        try:
            commits = response.json()
        except ValueError:
            logger.warning("Could not decode commits for %s", full_name)
            return None
        if not isinstance(commits, list) or not commits:
            return None

        lines: list[str] = []
        for entry in commits:
            line = self._format_commit_line(entry)
            if line:
                lines.append(line)
        if not lines:
            return None

        text = "\n".join(lines)
        return Document(
            id=f"github_commit:{slug(full_name)}",
            text=text,
            source_type="github_commit",
            source_id=full_name,
            title=f"{full_name} commit history",
            url=f"{html_url}/commits",
            metadata={"commit_count": len(lines)},
        )

    @staticmethod
    def _format_commit_line(entry: dict[str, Any]) -> str:
        """Format a single commit as ``"<sha7> <date> <author>: <subject>"``."""
        if not isinstance(entry, dict):
            return ""
        sha = str(entry.get("sha") or "")[:7]
        commit = entry.get("commit") or {}
        author = commit.get("author") or {}
        author_name = str(author.get("name") or "unknown")
        raw_date = str(author.get("date") or "")
        date = GitHubSource._format_commit_date(raw_date)
        message = str(commit.get("message") or "").strip()
        subject = message.splitlines()[0] if message else "(no message)"
        if not sha:
            return ""
        return f"{sha} {date} {author_name}: {subject}".strip()

    @staticmethod
    def _format_commit_date(raw_date: str) -> str:
        """Render an ISO commit timestamp as a ``YYYY-MM-DD`` date string."""
        if not raw_date:
            return ""
        try:
            parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).date().isoformat()
        except ValueError:
            return raw_date[:10]

    async def _fetch_source_files(
        self,
        owner: str,
        name: str,
        full_name: str,
        html_url: str,
        repo: dict[str, Any],
    ) -> list[Document]:
        """Select and fetch a handful of representative source files."""
        max_files = max(0, int(self._settings.github_max_source_files_per_repo))
        if max_files == 0:
            return []

        default_branch = str(repo.get("default_branch") or "main")
        tree = await self._fetch_tree(owner, name, full_name, default_branch)
        if not tree:
            return []

        candidates = self._select_source_paths(tree)
        documents: list[Document] = []
        for path in candidates:
            if len(documents) >= max_files:
                break
            if self._rate_limited:
                break
            document = await self._fetch_source_file(
                owner, name, full_name, html_url, default_branch, path
            )
            if document is not None:
                documents.append(document)
        return documents

    async def _fetch_tree(
        self, owner: str, name: str, full_name: str, branch: str
    ) -> list[dict[str, Any]]:
        """Fetch the recursive git tree for ``branch``."""
        response = await self._request(
            f"/repos/{owner}/{name}/git/trees/{quote(branch, safe='')}",
            params={"recursive": "1"},
        )
        if response is None:
            return []
        try:
            payload = response.json()
        except ValueError:
            logger.warning("Could not decode tree for %s", full_name)
            return []
        entries = payload.get("tree")
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def _select_source_paths(self, tree: list[dict[str, Any]]) -> list[str]:
        """Rank tree blobs and return candidate source-file paths, best first."""
        extensions = tuple(
            ext.lower() for ext in self._settings.github_source_extensions
        )
        max_bytes = int(self._settings.github_max_file_bytes)

        scored: list[tuple[int, int, str]] = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = str(entry.get("path") or "")
            if not path:
                continue
            lower = path.lower()
            if not lower.endswith(extensions):
                continue
            if any(fragment in lower for fragment in _EXCLUDED_PATH_FRAGMENTS):
                continue

            size = entry.get("size")
            try:
                size_int = int(size) if size is not None else 0
            except (TypeError, ValueError):
                size_int = 0
            if size_int > max_bytes:
                continue

            # Higher preference score sorts first; depth (fewer slashes) is a
            # tiebreaker so top-level/important files win.
            preference = sum(
                1 for fragment in _PREFERRED_PATH_FRAGMENTS if fragment in lower
            )
            depth = lower.count("/")
            scored.append((preference, -depth, path))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [path for _, _, path in scored]

    async def _fetch_source_file(
        self,
        owner: str,
        name: str,
        full_name: str,
        html_url: str,
        branch: str,
        path: str,
    ) -> Document | None:
        """Fetch a single source file's content as a document."""
        # Percent-encode the path so segments containing '#', '?', spaces, etc.
        # are not misparsed as URL fragment/query; keep '/' as the path separator.
        response = await self._request(
            f"/repos/{owner}/{name}/contents/{quote(path, safe='/')}",
            params={"ref": branch},
        )
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            logger.warning("Could not decode contents of %s in %s", path, full_name)
            return None
        if not isinstance(payload, dict):
            return None

        size = payload.get("size")
        try:
            size_int = int(size) if size is not None else 0
        except (TypeError, ValueError):
            size_int = 0
        if size_int > int(self._settings.github_max_file_bytes):
            logger.debug("Skipping %s (%d bytes exceeds cap)", path, size_int)
            return None

        content = self._decode_content(payload)
        if not content or not content.strip():
            return None

        file_url = str(payload.get("html_url") or f"{html_url}/blob/{branch}/{path}")
        composite_id = f"{full_name}:{path}"
        return Document(
            id=f"github_source:{slug(composite_id)}",
            text=content,
            source_type="github_source",
            source_id=composite_id,
            title=f"{full_name} {path}",
            url=file_url,
            metadata={"path": path, "bytes": size_int},
        )
