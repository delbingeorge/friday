"""Read-only access to the user's projects and documents, for Claude's file tools.

Everything is confined to configured roots, secrets are refused, and outputs are
capped so a single tool call can't blow up the context (or the bill).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from difflib import get_close_matches
from pathlib import Path

IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".nuxt",
    "target", ".cache", ".idea", ".vscode", "coverage", ".turbo", ".pytest_cache", ".mypy_cache",
    "Pods", "DerivedData", ".gradle", "vendor", "models",
}
PROJECT_MARKERS = {".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml",
                   "build.gradle", "Gemfile", "composer.json", "Package.swift", "setup.py"}
SECRET_NAMES = ["*.env", ".env*", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks", "id_rsa*",
                "id_ed25519*", "id_ecdsa*", ".netrc", ".npmrc", ".pypirc", "*credentials*", "*secret*",
                "*.kdbx", "*.keychain*", "*.sqlite", "*.db"]
SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker", ".git", "Keychains"}

MAX_READ_LINES = 250
MAX_OUTPUT = 12000  # characters returned by any single tool


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [truncated, {len(text) - limit} more characters]"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class FileAccess:
    def __init__(self, project_roots: list[str], projects: dict[str, str], file_roots: list[str]):
        self.project_roots = [Path(p).expanduser().resolve() for p in project_roots]
        self.file_roots = [Path(p).expanduser().resolve() for p in file_roots]
        self.projects: dict[str, Path] = self._discover()
        for name, p in projects.items():  # explicit entries win
            self.projects[name] = Path(p).expanduser().resolve()

    # --- discovery & safety ---------------------------------------------------
    def _discover(self) -> dict[str, Path]:
        found: dict[str, Path] = {}

        def walk(d: Path, depth: int):
            try:
                entries = list(os.scandir(d))
            except OSError:
                return
            names = {e.name for e in entries}
            if depth > 0 and names & PROJECT_MARKERS:
                key = d.name if d.name not in found else f"{d.parent.name}/{d.name}"
                found[key] = d
                return  # don't treat sub-packages as separate projects
            if depth >= 3:
                return
            for e in entries:
                if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and e.name not in IGNORED_DIRS:
                    walk(Path(e.path), depth + 1)

        for root in self.project_roots:
            if root.is_dir():
                walk(root, 0)
        return dict(sorted(found.items()))

    @property
    def allowed_roots(self) -> list[Path]:
        return [*self.project_roots, *self.projects.values(), *self.file_roots]

    def _check(self, path: Path) -> Path:
        p = path.expanduser().resolve()
        if not any(p == r or p.is_relative_to(r) for r in self.allowed_roots):
            raise PermissionError(f"{p} is outside the folders Friday may read")
        if any(part in SECRET_DIRS for part in p.parts):
            raise PermissionError("that location holds credentials; not reading it")
        if any(fnmatch.fnmatch(p.name.lower(), pat) for pat in SECRET_NAMES):
            raise PermissionError(f"{p.name} looks like a secrets file; not reading it")
        return p

    def project(self, name: str) -> tuple[str, Path]:
        if name in self.projects:
            return name, self.projects[name]
        by_norm = {_norm(k): k for k in self.projects}
        if _norm(name) in by_norm:
            k = by_norm[_norm(name)]
            return k, self.projects[k]
        # Voice input: "octane landing" -> "octane-landing", "port folio" -> "portfolio".
        close = get_close_matches(_norm(name), list(by_norm), n=1, cutoff=0.6)
        if close:
            k = by_norm[close[0]]
            return k, self.projects[k]
        raise LookupError(f"No project called '{name}'. Known projects: {', '.join(self.projects)}")

    def _resolve(self, path: str, project: str | None) -> Path:
        if project:
            _, base = self.project(project)
            p = Path(path).expanduser()
            return self._check(p if p.is_absolute() else base / p)
        return self._check(Path(path))

    @staticmethod
    def _git(path: Path, *args: str) -> str:
        r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""

    # --- tools ----------------------------------------------------------------
    def list_projects(self) -> str:
        if not self.projects:
            return "No projects found. Add folders to project_roots or projects in config.toml."
        home = str(Path.home())
        return "\n".join(f"{k}: {str(p).replace(home, '~')}" for k, p in self.projects.items())

    def project_overview(self, name: str) -> str:
        key, root = self.project(name)
        out = [f"Project {key} at {root}"]

        tree, count = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            depth = len(rel.parts)
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
            if depth >= 2:
                dirnames[:] = []
            indent = "  " * depth
            if depth:
                tree.append(f"{'  ' * (depth - 1)}{rel.name}/")
            for f in sorted(filenames)[:25]:
                tree.append(f"{indent}{f}")
                count += 1
            if count > 120:
                tree.append("…")
                break
        out.append("Files (depth 2):\n" + "\n".join(tree))

        for doc in ("README.md", "README", "readme.md", "CLAUDE.md", "AGENTS.md"):
            f = root / doc
            if f.is_file():
                lines = f.read_text(errors="replace").splitlines()[:80]
                out.append(f"--- {doc} (first {len(lines)} lines) ---\n" + "\n".join(lines))

        pkg = root / "package.json"
        if pkg.is_file():
            try:
                d = json.loads(pkg.read_text())
                out.append("--- package.json ---\n" + json.dumps({
                    k: d.get(k) for k in ("name", "description", "scripts")
                } | {"dependencies": sorted(d.get("dependencies", {})),
                     "devDependencies": sorted(d.get("devDependencies", {}))}, indent=1))
            except (json.JSONDecodeError, OSError):
                pass
        for manifest in ("pyproject.toml", "Cargo.toml", "go.mod"):
            f = root / manifest
            if f.is_file():
                out.append(f"--- {manifest} ---\n" + "\n".join(f.read_text(errors="replace").splitlines()[:40]))

        if (root / ".git").exists():
            branch = self._git(root, "branch", "--show-current")
            status = self._git(root, "status", "--short")
            log = self._git(root, "log", "--oneline", "-8")
            out.append(f"--- git ---\nbranch: {branch}\nuncommitted changes:\n"
                       f"{chr(10).join(status.splitlines()[:20]) or '(none)'}\nrecent commits:\n{log}")
        return _clip("\n\n".join(out))

    def find_files(self, pattern: str, project: str | None = None) -> str:
        roots = [self.project(project)[1]] if project else self.file_roots
        hits: list[str] = []
        for root in roots:
            r = subprocess.run(
                ["rg", "--files", "--max-depth", "8", "--iglob", f"*{pattern}*",
                 *[a for d in IGNORED_DIRS for a in ("-g", f"!{d}/")], str(root)],
                capture_output=True, text=True, timeout=15,
            )
            hits += r.stdout.splitlines()
        hits = [h for h in hits if not any(fnmatch.fnmatch(Path(h).name.lower(), p) for p in SECRET_NAMES)]
        if not hits:
            return f"No files matching '{pattern}'."
        return _clip("\n".join(hits[:60]) + (f"\n… {len(hits) - 60} more" if len(hits) > 60 else ""))

    def search_files(self, query: str, project: str | None = None, file_glob: str | None = None,
                     regex: bool = False) -> str:
        roots = [self.project(project)[1]] if project else self.file_roots
        args = ["rg", "-n", "-i", "--max-count", "5", "--max-columns", "200", "--max-filesize", "1M"]
        if not regex:
            args.append("-F")
        if file_glob:
            args += ["-g", file_glob]
        for pat in SECRET_NAMES:
            args += ["-g", f"!{pat}"]
        r = subprocess.run([*args, "--", query, *map(str, roots)], capture_output=True, text=True, timeout=20)
        lines = r.stdout.splitlines()
        if not lines:
            return f"No matches for '{query}'."
        return _clip("\n".join(lines[:80]) + (f"\n… {len(lines) - 80} more matching lines" if len(lines) > 80 else ""))

    def read_file(self, path: str, project: str | None = None, start_line: int = 1,
                  max_lines: int = MAX_READ_LINES) -> str:
        p = self._resolve(path, project)
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found")
        if p.stat().st_size > 5_000_000:
            raise ValueError("file is larger than 5 MB")
        raw = p.read_bytes()
        if b"\0" in raw[:8192]:
            raise ValueError("binary file")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        n = max(1, min(int(max_lines), MAX_READ_LINES))
        chunk = lines[start - 1:start - 1 + n]
        body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
        end = start + len(chunk) - 1
        more = f" (file continues to line {len(lines)}; read again with start_line={end + 1})" if end < len(lines) else ""
        return _clip(f"{p} lines {start}-{end} of {len(lines)}{more}\n{body}")

    def open_file(self, path: str, project: str | None = None, app: str = "") -> str:
        p = self._resolve(path, project)
        cmd = ["open", "-a", app, str(p)] if app else ["open", "-t" if p.is_file() else "-R", str(p)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=10)
        return f"Opened {p.name}."
