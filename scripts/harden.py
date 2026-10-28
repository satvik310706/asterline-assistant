"""Fresh-clone, offline smoke, integrity, and secret-scan harness."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


EXPECTED = {
    "data/corpus.jsonl": "f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1",
    "eval/datasets/research.jsonl": "67d7e6e0ed5541a5ee8295b53c4978017af21f6a8fd19822cd16bd1ad0e7f417",
    "eval/datasets/memory.jsonl": "09d32b417dacc408d443f1d0a79b972dde84aee3e0ed317b2776034798cca68a",
}


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    os.environ["UV_CACHE_DIR"] = str(source / ".cache" / "uv-hardening")
    os.environ["ASTERLINE_DISABLE_TRACING"] = "1"
    os.environ["LANGSMITH_TRACING"] = "false"
    # Reuse the already-downloaded local embedding model for this verification;
    # a fresh user setup still performs the documented one-time download.
    os.environ["ASTERLINE_FASTEMBED_CACHE"] = str(source / ".cache" / "fastembed")
    # Keep all temporary clone/build output under the repository boundary.
    with tempfile.TemporaryDirectory(prefix=".asterline-clone-", dir=source) as temp:
        clone = Path(temp) / "repo"
        subprocess.run(["git", "clone", "--quiet", str(source), str(clone)], check=True)
        subprocess.run(["uv", "sync"], cwd=clone, check=True)
        subprocess.run(["uv", "run", "pytest", "-q"], cwd=clone, check=True)
        subprocess.run(["uv", "run", "asterline", "ask", "What is the Team plan price?", "--provider", "fake"], cwd=clone, check=True)
        for name, expected in EXPECTED.items():
            actual = hashlib.sha256((clone / name).read_bytes()).hexdigest()
            assert actual == expected, name
        secret_pattern = re.compile(r"sk-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|Bearer\s+[A-Za-z0-9._~+/=-]+", re.I)
        tracked = subprocess.check_output(["git", "ls-files"], cwd=clone, text=True).splitlines()
        synthetic_paths = {"eval/datasets/memory.jsonl", "eval/datasets/research.jsonl"}
        synthetic_prefixes = ("eval/final/", "eval/results/")
        scan_files = [name for name in tracked if name not in synthetic_paths and not name.startswith(synthetic_prefixes)]
        hits = [name for name in scan_files if name != ".env" and secret_pattern.search((clone / name).read_text(encoding="utf-8", errors="ignore"))]
        history = subprocess.check_output(["git", "log", "-p", "--all", "--", ".", ":!.env", ":!**/.env", ":!eval/datasets/memory.jsonl", ":!eval/datasets/research.jsonl", ":!eval/final/*", ":!eval/results/*"], cwd=clone, text=True, errors="ignore")
        assert not hits and not secret_pattern.search(history), (hits, "git history secret pattern")
    print("fresh clone hardening passed")


if __name__ == "__main__":
    main()
