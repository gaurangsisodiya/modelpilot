"""Every script answers --help without doing any work (no data, no network, no files written)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY_SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def clean_env(tmp_path: Path) -> dict[str, str]:
    """No API keys, and Hugging Face kept offline, so any real work would fail rather than download or call out."""
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_BASE_URL"))}
    return {**env, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


@pytest.mark.parametrize("script", PY_SCRIPTS, ids=lambda p: p.name)
def test_python_script_help(script: Path, tmp_path: Path):
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, env=clean_env(tmp_path),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage:")
    assert list(tmp_path.iterdir()) == [], "--help wrote files"


def test_live_benchmark_help(tmp_path: Path):
    r = subprocess.run(["bash", str(ROOT / "scripts" / "live_benchmark.sh"), "--help"], cwd=tmp_path,
                       env=clean_env(tmp_path), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "Costs money" in r.stdout
