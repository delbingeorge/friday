"""Local laptop tools Claude can call. Kept small: every definition is sent
(cached) with each request, and each tool call costs a round trip."""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from .files import FileAccess
from urllib.parse import urlparse


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


LOCAL_TOOLS = [
    _tool("open_app", "Open or switch to a macOS application by name, e.g. 'Safari', 'Spotify', 'Visual Studio Code'.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("open_url", "Open a web page in the default browser. Use for 'open YouTube', 'show me the docs for X', etc.",
          {"url": {"type": "string", "description": "Full http(s) URL"}}, ["url"]),
    _tool("set_volume", "Set the Mac's output volume.",
          {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"]),
    _tool("media", "Control Music or Spotify playback.",
          {"action": {"type": "string", "enum": ["play", "pause", "next", "previous"]},
           "app": {"type": "string", "enum": ["Spotify", "Music"]}}, ["action", "app"]),
    _tool("set_timer", "Start a countdown timer; Friday announces it out loud when it ends.",
          {"seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
           "label": {"type": "string"}}, ["seconds", "label"]),
    _tool("run_shortcut", "Run a user-defined macOS Shortcut by its exact name (reminders, smart home, messages...). "
          "Only use names the user has mentioned.",
          {"name": {"type": "string"}, "input": {"type": "string"}}, ["name"]),
]

_PROJECT = {"type": "string", "description": "Project name from list_projects (fuzzy: 'octane landing' works)"}
FILE_TOOLS = [
    _tool("list_projects", "List the user's local code projects (name and folder).", {}, []),
    _tool("project_overview", "Get an overview of a project: file tree, README/CLAUDE.md, manifest "
          "(package.json/pyproject), git branch, uncommitted changes and recent commits. "
          "Call this first when the user starts talking about a project.",
          {"project": _PROJECT}, ["project"]),
    _tool("find_files", "Find files by name fragment inside a project, or across the user's documents "
          "when no project is given.",
          {"pattern": {"type": "string", "description": "Part of the file name, e.g. 'auth' or '.tsx'"},
           "project": _PROJECT}, ["pattern"]),
    _tool("search_files", "Search file contents (case-insensitive, like grep) in a project, or across the "
          "user's documents when no project is given. Returns path:line:text matches.",
          {"query": {"type": "string"}, "project": _PROJECT,
           "file_glob": {"type": "string", "description": "Optional filter, e.g. '*.py' or 'src/**'"},
           "regex": {"type": "boolean", "description": "Treat query as a regular expression"}}, ["query"]),
    _tool("read_file", "Read a text file with line numbers (up to 250 lines per call). Path is relative "
          "to the project if one is given, otherwise absolute or ~/...",
          {"path": {"type": "string"}, "project": _PROJECT,
           "start_line": {"type": "integer", "minimum": 1},
           "max_lines": {"type": "integer", "minimum": 1, "maximum": 250}}, ["path"]),
    _tool("open_file", "Open a file (or reveal a folder) on screen so the user can look at it.",
          {"path": {"type": "string"}, "project": _PROJECT}, ["path"]),
]


class ToolRunner:
    def __init__(self, announce: Callable[[str], None], files: FileAccess | None = None, editor_app: str = ""):
        self._announce = announce  # speaks text out loud (used by timers)
        self.files = files
        self._editor_app = editor_app

    @property
    def definitions(self) -> list[dict]:
        return LOCAL_TOOLS + (FILE_TOOLS if self.files else [])

    def run(self, name: str, args: object) -> tuple[str, bool]:
        """Returns (result text, is_error)."""
        if not isinstance(args, dict):
            return "Invalid tool input.", True
        try:
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return f"Unknown tool {name}.", True
            return fn(**args), False
        except TypeError as e:  # wrong/missing arguments from a truncated or malformed input
            return f"Invalid arguments: {e}", True
        except subprocess.CalledProcessError as e:
            return (e.stderr or e.stdout or str(e)).strip()[:500], True
        except subprocess.TimeoutExpired:
            return "Timed out.", True
        except (PermissionError, LookupError, FileNotFoundError, ValueError) as e:
            return str(e), True
        except Exception as e:  # noqa: BLE001 - report any tool failure back to the model
            return f"Error: {e}", True

    @staticmethod
    def _run(cmd: list[str], timeout: float = 10) -> str:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()

    def _t_open_app(self, name: str) -> str:
        self._run(["open", "-a", str(name)])
        return f"Opened {name}."

    def _t_open_url(self, url: str) -> str:
        if urlparse(str(url)).scheme not in ("http", "https"):
            raise ValueError("only http(s) URLs are allowed")
        self._run(["open", str(url)])
        return "Opened."

    def _t_set_volume(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        self._run(["osascript", "-e", f"set volume output volume {level}"])
        return f"Volume set to {level}."

    def _t_media(self, action: str, app: str) -> str:
        verbs = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
        if action not in verbs or app not in ("Spotify", "Music"):
            raise ValueError("unsupported action or app")
        self._run(["osascript", "-e", f'tell application "{app}" to {verbs[action]}'])
        return "Done."

    def _t_set_timer(self, seconds: int, label: str) -> str:
        seconds = int(seconds)
        label = str(label)[:80]
        t = threading.Timer(seconds, self._announce, args=(f"Your {label} timer is done.",))
        t.daemon = True
        t.start()
        return f"Timer set for {seconds} seconds."

    def _t_run_shortcut(self, name: str, input: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "out.txt"
            cmd = ["shortcuts", "run", str(name), "--output-path", str(out_path)]
            if input:
                in_path = Path(d) / "in.txt"
                in_path.write_text(str(input))
                cmd += ["--input-path", str(in_path)]
            self._run(cmd, timeout=60)
            out = out_path.read_text(errors="replace").strip() if out_path.exists() else ""
        return out[:1000] or "Shortcut finished."

    # --- files & projects (read-only) -----------------------------------------
    def _t_list_projects(self) -> str:
        return self.files.list_projects()

    def _t_project_overview(self, project: str) -> str:
        return self.files.project_overview(project)

    def _t_find_files(self, pattern: str, project: str | None = None) -> str:
        return self.files.find_files(pattern, project)

    def _t_search_files(self, query: str, project: str | None = None, file_glob: str | None = None,
                        regex: bool = False) -> str:
        return self.files.search_files(query, project, file_glob, regex)

    def _t_read_file(self, path: str, project: str | None = None, start_line: int = 1,
                     max_lines: int = 250) -> str:
        return self.files.read_file(path, project, start_line, max_lines)

    def _t_open_file(self, path: str, project: str | None = None) -> str:
        return self.files.open_file(path, project, self._editor_app)
