"""Resume PDF ingestion source (spec §15.2).

:class:`ResumeSource` extracts text from a resume PDF using ``pypdf`` and emits
a single :class:`~app.rag.schemas.Document`. PDF parsing is blocking, so it runs
in a worker thread via :func:`asyncio.to_thread` to keep the event loop free.

The resume location is resolved (in priority order) from:

1. the ``RESUME_PATH`` environment variable, if set;
2. the first ``*.pdf`` found in the conventional ``data/resume/`` directory.

If no resume file can be found, the source logs a warning and returns ``[]`` so
ingestion can proceed with whatever other sources are available.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from app.config import Settings
from app.ingestion.base import Source, slug
from app.rag.schemas import Document

logger = logging.getLogger(__name__)

#: Environment variable that, when set, overrides resume discovery.
_RESUME_PATH_ENV = "RESUME_PATH"

#: Conventional directory (relative to the project root) holding the resume PDF.
_DEFAULT_RESUME_DIR = "data/resume"


def _project_root() -> Path:
    """Return the repository root (three levels up from this module)."""
    # app/ingestion/resume.py -> app/ingestion -> app -> <project root>
    return Path(__file__).resolve().parents[2]


def _discover_resume_path() -> Path | None:
    """Locate the resume PDF on disk.

    Returns:
        A path to the resume PDF, or ``None`` if none is configured/present.
    """
    env_value = os.environ.get(_RESUME_PATH_ENV, "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.is_file():
            return candidate
        logger.warning(
            "%s=%s does not point to an existing file; falling back to %s/",
            _RESUME_PATH_ENV,
            env_value,
            _DEFAULT_RESUME_DIR,
        )

    resume_dir = _project_root() / _DEFAULT_RESUME_DIR
    if not resume_dir.is_dir():
        return None

    pdfs = sorted(p for p in resume_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        return None
    if len(pdfs) > 1:
        logger.warning(
            "Multiple resume PDFs found in %s; using %s",
            resume_dir,
            pdfs[0].name,
        )
    return pdfs[0]


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    """Extract all text from a PDF (blocking; run in a thread).

    Args:
        path: Path to the PDF file.

    Returns:
        A ``(text, page_count)`` tuple. ``text`` is the per-page extracted text
        joined by blank lines. Pages that fail to extract are skipped.

    Raises:
        Exception: Any error raised by ``pypdf`` while opening/parsing the file
            is propagated to the caller, which handles it gracefully.
    """
    # Imported lazily so that importing this module never hard-requires pypdf
    # (e.g. in environments that only run the API and never ingest).
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # pragma: no cover - per-page extraction is best-effort
            logger.warning(
                "Failed to extract text from page %d of %s; skipping",
                page_number,
                path.name,
                exc_info=True,
            )
            continue
        page_text = page_text.strip()
        if page_text:
            page_texts.append(page_text)

    return "\n\n".join(page_texts), len(reader.pages)


class ResumeSource(Source):
    """Ingestion source that reads a resume PDF into a single document."""

    name = "resume"

    def __init__(self, settings: Settings) -> None:
        """Initialize the source.

        Args:
            settings: Application settings (used for persona naming in the
                document title).
        """
        self._settings = settings
        self._resume_path = _discover_resume_path()

    async def load(self) -> list[Document]:
        """Read the resume PDF and return it as one :class:`Document`.

        Returns:
            A single-element list with the resume document, or an empty list if
            the file is missing or yields no extractable text. Errors during
            parsing are logged and result in an empty list rather than an
            exception.
        """
        if self._resume_path is None:
            logger.warning(
                "No resume PDF found (set %s or place a *.pdf in %s/); "
                "skipping resume ingestion",
                _RESUME_PATH_ENV,
                _DEFAULT_RESUME_DIR,
            )
            return []

        path = self._resume_path
        logger.info("Loading resume from %s", path)

        try:
            text, page_count = await asyncio.to_thread(_extract_pdf_text, path)
        except Exception:
            logger.warning("Failed to read resume PDF %s; skipping", path, exc_info=True)
            return []

        if not text.strip():
            logger.warning("Resume PDF %s contained no extractable text; skipping", path)
            return []

        source_id = "resume"
        document = Document(
            id=f"resume:{slug(source_id)}",
            text=text,
            source_type="resume",
            source_id=source_id,
            title=f"{self._settings.persona_name} — Resume",
            url=None,
            metadata={"pages": page_count, "filename": path.name},
        )
        logger.info(
            "Loaded resume document (%d page(s), %d chars)", page_count, len(text)
        )
        return [document]
