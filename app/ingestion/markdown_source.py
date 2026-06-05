"""Markdown knowledge-file ingestion source.

:class:`MarkdownSource` discovers hand-authored ``*.md`` files under the
configured knowledge directory (``settings.markdown_data_dir``, defaulting to
``data/``) and emits one :class:`~app.rag.schemas.Document` per file. These are
the persona's narrative knowledge files (``about.md``, ``experience.md``,
``portfolio.md``, ``projects.md``, ...).

Discovery is recursive but skips directories that hold *generated* artifacts or
non-markdown binaries — the ChromaDB store, the BM25 pickle, the Piper voice,
diagnostic dumps, and the resume PDF directory (handled by
:class:`~app.ingestion.resume.ResumeSource`). File reads are blocking, so they
run in a worker thread via :func:`asyncio.to_thread` to keep the event loop
free. Empty/whitespace-only files are skipped. Any error reading one file is
logged and skipped, so a single bad file can never abort ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import Settings
from app.ingestion.base import Source, slug
from app.rag.schemas import Document

logger = logging.getLogger(__name__)

#: Directory names (anywhere in the tree) that are never scanned for knowledge
#: markdown: generated stores, audio assets, diagnostics, and the resume dir
#: whose PDF is ingested by :class:`~app.ingestion.resume.ResumeSource`.
_EXCLUDED_DIR_NAMES = frozenset(
    {"chroma", "_diag_chroma", "bm25", "piper", "resume"}
)


def _project_root() -> Path:
    """Return the repository root (two levels up from this module)."""
    # app/ingestion/markdown_source.py -> app/ingestion -> app -> <project root>
    return Path(__file__).resolve().parents[2]


def _resolve_data_dir(settings: Settings) -> Path:
    """Resolve the markdown knowledge directory to an absolute path.

    A relative ``markdown_data_dir`` is anchored at the project root so the
    discovery result does not depend on the process's current directory.
    """
    configured = Path(settings.markdown_data_dir).expanduser()
    if configured.is_absolute():
        return configured
    return _project_root() / configured


def _is_excluded(path: Path, root: Path) -> bool:
    """Return True if ``path`` lives under any excluded directory below ``root``."""
    relative = path.relative_to(root)
    # Check directory components only (everything but the filename itself).
    return any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1])


def _discover_markdown_files(root: Path) -> list[Path]:
    """Return sorted ``*.md`` files under ``root``, skipping excluded dirs."""
    if not root.is_dir():
        logger.warning(
            "Markdown data dir %s does not exist; skipping markdown ingestion", root
        )
        return []
    files = [
        path
        for path in root.rglob("*.md")
        if path.is_file() and not _is_excluded(path, root)
    ]
    return sorted(files)


def _derive_title(text: str, path: Path) -> str:
    """Use the first markdown H1 (``# Heading``) as the title, else the filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return heading
    # Fall back to a humanized filename stem (e.g. "about" -> "About").
    return path.stem.replace("-", " ").replace("_", " ").strip().title() or path.name


class MarkdownSource(Source):
    """Ingest hand-authored markdown knowledge files into corpus documents."""

    name = "markdown"

    def __init__(self, settings: Settings) -> None:
        """Initialize the source.

        Args:
            settings: Application settings providing ``markdown_data_dir``.
        """
        self._settings = settings
        self._root = _resolve_data_dir(settings)

    async def load(self) -> list[Document]:
        """Discover and read every markdown file into a :class:`Document`.

        Returns:
            One document per non-empty markdown file (sorted by path). Files
            that are missing, unreadable, or blank are skipped with a log line.
        """
        paths = _discover_markdown_files(self._root)
        if not paths:
            logger.warning("No markdown files found under %s", self._root)
            return []

        logger.info("Discovered %d markdown file(s) under %s", len(paths), self._root)

        documents: list[Document] = []
        for path in paths:
            document = await self._load_file(path)
            if document is not None:
                documents.append(document)

        logger.info("Markdown ingestion produced %d document(s)", len(documents))
        return documents

    async def _load_file(self, path: Path) -> Document | None:
        """Read one markdown file into a document, or ``None`` if empty/unreadable."""
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Failed to read markdown file %s; skipping", path, exc_info=True)
            return None

        if not text.strip():
            logger.info("Markdown file %s is empty; skipping", path)
            return None

        # Stable, human-readable relative id (e.g. "about.md").
        rel_path = path.relative_to(self._root).as_posix()
        return Document(
            id=f"markdown:{slug(rel_path)}",
            text=text,
            source_type="markdown",
            source_id=rel_path,
            title=_derive_title(text, path),
            url=None,
            metadata={"path": rel_path, "filename": path.name},
        )
