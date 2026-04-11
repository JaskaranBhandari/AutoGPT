"""Unified MCP Write tool that works in both E2B and non-E2B modes.

Replaces the CLI's built-in Write tool, which has no defence against output-token
truncation.  When the LLM generates a very large ``content`` argument the API
truncates the response mid-JSON and Ajv rejects it with the opaque
"'file_path' is a required property" error, losing the user's work.

This MCP tool:
- Detects partial truncation (content present but file_path missing)
- Detects complete truncation (empty args)
- Warns on large content that succeeded (>50K chars)
- In non-E2B mode: writes to the SDK working directory
- In E2B mode: delegates to the E2B sandbox write handler

The JSON schema places ``file_path`` FIRST so that truncation is more likely
to preserve the path (the API serialises properties in schema order).
"""

import json
import logging
import os
from typing import Any, Callable

from backend.copilot.context import get_sdk_cwd, is_allowed_local_path

logger = logging.getLogger(__name__)

# Inline content above this threshold triggers a warning — it survived this
# time but is dangerously close to the API output-token truncation limit.
_LARGE_CONTENT_WARN_CHARS = 50_000


def _mcp(text: str, *, error: bool = False) -> dict[str, Any]:
    if error:
        text = json.dumps({"error": text, "type": "error"})
    return {"content": [{"type": "text", "text": text}], "isError": error}


_PARTIAL_TRUNCATION_MSG = (
    "Your Write call was truncated (file_path missing but content "
    "was present). The content was too large for a single tool call. "
    "Write in chunks: use bash_exec with "
    "'cat > file << \"EOF\"\\n...\\nEOF' for the first section, "
    "'cat >> file << \"EOF\"\\n...\\nEOF' to append subsequent "
    "sections, then reference the file with "
    "@@agptfile:/path/to/file if needed."
)

_COMPLETE_TRUNCATION_MSG = (
    "Your Write call had empty arguments — this means your previous "
    "response was too long and the tool call was truncated by the API. "
    "Break your work into smaller steps. For large content, write "
    "section-by-section using bash_exec with "
    "'cat > file << \"EOF\"\\n...\\nEOF' and "
    "'cat >> file << \"EOF\"\\n...\\nEOF'."
)


def _check_truncation(file_path: str, content: str) -> dict[str, Any] | None:
    """Return an error response if the args look truncated, else ``None``."""
    if not file_path:
        if content:
            return _mcp(_PARTIAL_TRUNCATION_MSG, error=True)
        return _mcp(_COMPLETE_TRUNCATION_MSG, error=True)
    return None


async def _handle_write_non_e2b(args: dict[str, Any]) -> dict[str, Any]:
    """Write content to a file in the SDK working directory (non-E2B mode)."""
    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    truncation_err = _check_truncation(file_path, content)
    if truncation_err is not None:
        return truncation_err

    sdk_cwd = get_sdk_cwd()
    if not sdk_cwd:
        return _mcp("No SDK working directory available", error=True)

    # Resolve relative paths against SDK working directory
    if not os.path.isabs(file_path):
        resolved = os.path.normpath(os.path.join(sdk_cwd, file_path))
    else:
        resolved = os.path.normpath(file_path)

    # Validate path stays within allowed directories
    if not is_allowed_local_path(resolved, sdk_cwd):
        return _mcp(
            f"Path must be within the working directory: {os.path.basename(file_path)}",
            error=True,
        )

    try:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        return _mcp(f"Failed to write {resolved}: {exc}", error=True)

    msg = f"Successfully wrote to {resolved}"
    if len(content) > _LARGE_CONTENT_WARN_CHARS:
        logger.warning(
            "[Write] large inline content (%d chars) for %s",
            len(content),
            resolved,
        )
        msg += (
            f"\n\nWARNING: The content was very large ({len(content)} chars). "
            "Next time, write large files in sections using bash_exec with "
            "'cat > file << EOF ... EOF' and 'cat >> file << EOF ... EOF' "
            "to avoid output-token truncation."
        )
    return _mcp(msg)


async def _handle_write_e2b(args: dict[str, Any]) -> dict[str, Any]:
    """Write content to a file, delegating to the E2B sandbox."""
    from .e2b_file_tools import _handle_write_file

    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    truncation_err = _check_truncation(file_path, content)
    if truncation_err is not None:
        return truncation_err

    return await _handle_write_file(args)


def get_write_tool_handler(*, use_e2b: bool) -> Callable[..., Any]:
    """Return the appropriate Write handler for the current execution mode."""
    if use_e2b:
        return _handle_write_e2b
    return _handle_write_non_e2b


WRITE_TOOL_NAME = "Write"
WRITE_TOOL_DESCRIPTION = (
    "Write or create a file. Parent directories are created automatically. "
    "For large content (>2000 words), prefer writing in sections using "
    "bash_exec with 'cat > file' and 'cat >> file' instead."
)
WRITE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": (
                "The path to the file to write. "
                "Relative paths are resolved against the working directory."
            ),
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
        },
    },
    "required": ["file_path", "content"],
}
