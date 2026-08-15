"""Shared stdlib-logging setup for pipeline scripts.

Per CLAUDE.md's logging guideline: leveled console output via the stdlib
`logging` module rather than bare `print()`. Used by the labeling stage;
the pre-existing curation scripts (01-08) predate this and still use
print() + hand-built log-line lists, which is a separate cleanup.
"""
import logging
import sys

_CONFIGURED = False


def configure_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a single stderr handler, configured once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        # force=True: some dependencies (e.g. ete4's smartview module, pulled
        # in transitively via soamp.curation.taxonomy) call logging.basicConfig()
        # at import time. Without force=True, whichever import wins the race
        # silently keeps its own format/handlers since basicConfig() is a
        # documented no-op once the root logger already has handlers.
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
            force=True,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
