"""Command-line entrypoint for offline corpus ingestion (spec §15.5).

Usage::

    python -m app.ingestion.run_ingest [--reset] [--sources resume,github]
                                       [--username USER]

Loads configuration from ``.env`` via :func:`app.config.get_settings`, builds
the requested :class:`~app.ingestion.base.Source` set, runs the
:class:`~app.ingestion.pipeline.IngestionPipeline`, and prints the JSON summary
to stdout. Exit code is non-zero on fatal failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from app.config import Settings, get_settings
from app.ingestion.base import Source
from app.ingestion.github_source import GitHubSource
from app.ingestion.markdown_source import MarkdownSource
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.resume import ResumeSource
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Mapping of CLI source names to their builders.
_SOURCE_BUILDERS = {
    "resume": ResumeSource,
    "markdown": MarkdownSource,
    "github": GitHubSource,
}
_DEFAULT_SOURCES = "resume,markdown,github"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="app.ingestion.run_ingest",
        description="Ingest the persona corpus (resume + GitHub) into the "
        "vector store and BM25 index.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing vector collection before ingesting.",
    )
    parser.add_argument(
        "--sources",
        default=_DEFAULT_SOURCES,
        help=(
            "Comma-separated source list to ingest "
            f"(choices: resume, markdown, github; default: {_DEFAULT_SOURCES})."
        ),
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Override settings.github_username for this run.",
    )
    return parser.parse_args(argv)


def _resolve_sources(names: str, settings: Settings) -> list[Source]:
    """Build the source objects named in the ``--sources`` argument.

    Args:
        names: Comma-separated source names.
        settings: Application settings passed to each source.

    Returns:
        The constructed sources, in the order requested.

    Raises:
        ValueError: If an unknown source name is requested.
    """
    requested = [part.strip().lower() for part in names.split(",") if part.strip()]
    if not requested:
        raise ValueError("--sources must name at least one source")

    sources: list[Source] = []
    for name in requested:
        builder = _SOURCE_BUILDERS.get(name)
        if builder is None:
            valid = ", ".join(sorted(_SOURCE_BUILDERS))
            raise ValueError(f"unknown source {name!r}; valid sources: {valid}")
        sources.append(builder(settings))
    return sources


async def main(argv: list[str] | None = None) -> dict:
    """Run ingestion and return the summary dict.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The pipeline summary dict.
    """
    args = _parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)

    if args.username:
        # Override the cached settings for this run only.
        settings.github_username = args.username
        logger.info("Overriding github_username -> %s", args.username)

    sources = _resolve_sources(args.sources, settings)
    logger.info(
        "Starting ingestion (sources=%s, reset=%s)",
        [source.name for source in sources],
        args.reset,
    )

    pipeline = IngestionPipeline(settings, sources=sources)
    summary = await pipeline.run(reset=args.reset)
    return summary


def _cli() -> int:
    """Synchronous wrapper for the console entrypoint. Returns an exit code."""
    try:
        summary = asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover - interactive abort
        logger.warning("Ingestion interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Ingestion failed")
        # Emit a structured error so callers parsing stdout get something useful.
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1

    print(json.dumps({"status": "ok", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
