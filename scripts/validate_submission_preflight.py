#!/usr/bin/env python3
"""
validate_submission_preflight.py — OpenEnv Submission Validator (Windows-compatible)

Python equivalent of validate-submission.sh

Usage:
    python scripts/validate_submission_preflight.py <ping_url> [repo_dir]

Arguments:
    ping_url   Your HuggingFace Space URL (e.g. https://venkat-023-openenv.hf.space)
    repo_dir   Path to your repo (default: current directory)

Examples:
    python scripts/validate_submission_preflight.py https://venkat-023-openenv.hf.space
    python scripts/validate_submission_preflight.py https://venkat-023-openenv.hf.space .
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Colours (disabled on non-TTY) ────────────────────────────────────────────
if sys.stdout.isatty():
    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BOLD   = "\033[1m"
    NC     = "\033[0m"
else:
    RED = GREEN = YELLOW = BOLD = NC = ""

PASS_COUNT = 0


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def passed(msg: str) -> None:
    global PASS_COUNT
    log(f"{GREEN}PASSED{NC} -- {msg}")
    PASS_COUNT += 1


def failed(msg: str) -> None:
    log(f"{RED}FAILED{NC} -- {msg}")


def hint(msg: str) -> None:
    print(f"  {YELLOW}Hint:{NC} {msg}")


def stop_at(step: str) -> None:
    print()
    print(f"{RED}{BOLD}Validation stopped at {step}.{NC} Fix the above before continuing.")
    sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <ping_url> [repo_dir]")
        print()
        print("  ping_url   Your HuggingFace Space URL (e.g. https://venkat-023-openenv.hf.space)")
        print("  repo_dir   Path to your repo (default: current directory)")
        sys.exit(1)

    ping_url = sys.argv[1].rstrip("/")
    repo_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(".").resolve()

    if not repo_dir.is_dir():
        print(f"Error: directory '{repo_dir}' not found")
        sys.exit(1)

    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{BOLD}  OpenEnv Submission Validator{NC}")
    print(f"{BOLD}========================================{NC}")
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {ping_url}")
    print()

    # ── Step 1: Ping HF Space ─────────────────────────────────────────────────
    log(f"{BOLD}Step 1/3: Pinging HF Space{NC} ({ping_url}/reset) ...")

    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{ping_url}/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            http_code = resp.status
    except urllib.error.HTTPError as e:
        http_code = e.code
    except Exception as e:
        http_code = 0
        log(f"  Connection error: {e}")

    if http_code == 200:
        passed("HF Space is live and responds to /reset")
    elif http_code == 0:
        failed("HF Space not reachable (connection failed or timed out)")
        hint("Check your network and that the Space is running.")
        hint(f"Try opening {ping_url} in your browser first.")
        hint("Make sure you're using the Space URL, e.g.: https://venkat-023-openenv.hf.space")
        stop_at("Step 1")
    else:
        failed(f"HF Space /reset returned HTTP {http_code} (expected 200)")
        hint("Make sure your Space is running and the URL is correct.")
        hint(f"Expected format: https://venkat-023-openenv.hf.space  (NOT huggingface.co/spaces/...)")
        stop_at("Step 1")

    # ── Step 2: Docker build ──────────────────────────────────────────────────
    log(f"{BOLD}Step 2/3: Running docker build{NC} ...")

    if not shutil.which("docker"):
        failed("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        stop_at("Step 2")

    if (repo_dir / "Dockerfile").is_file():
        docker_context = repo_dir
    elif (repo_dir / "server" / "Dockerfile").is_file():
        docker_context = repo_dir / "server"
    else:
        failed("No Dockerfile found in repo root or server/ directory")
        stop_at("Step 2")

    log(f"  Found Dockerfile in {docker_context}")

    try:
        result = subprocess.run(
            ["docker", "build", str(docker_context)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            passed("Docker build succeeded")
        else:
            failed("Docker build failed")
            print("\n".join(result.stderr.splitlines()[-20:]))
            stop_at("Step 2")
    except subprocess.TimeoutExpired:
        failed("Docker build timed out (600s)")
        stop_at("Step 2")
    except Exception as e:
        failed(f"Docker build error: {e}")
        stop_at("Step 2")

    # ── Step 3: openenv validate ──────────────────────────────────────────────
    log(f"{BOLD}Step 3/3: Running openenv validate{NC} ...")

    if not shutil.which("openenv"):
        failed("openenv command not found")
        hint("Install it: pip install openenv-core")
        stop_at("Step 3")

    try:
        result = subprocess.run(
            ["openenv", "validate"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            passed("openenv validate passed")
            if result.stdout.strip():
                log(f"  {result.stdout.strip()}")
        else:
            failed("openenv validate failed")
            print(result.stdout)
            print(result.stderr)
            stop_at("Step 3")
    except Exception as e:
        failed(f"openenv validate error: {e}")
        stop_at("Step 3")

    # ── All passed ────────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{GREEN}{BOLD}  All 3/3 checks passed!{NC}")
    print(f"{GREEN}{BOLD}  Your submission is ready to submit.{NC}")
    print(f"{BOLD}========================================{NC}")
    print()


if __name__ == "__main__":
    main()
