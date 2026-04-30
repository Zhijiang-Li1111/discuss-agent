"""CLI entry point for the multi-agent discussion framework."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

from discuss_agent.config import ConfigLoader
from discuss_agent.engine import DiscussionEngine


def _setup_logging(archive_dir: str | None = None) -> None:
    """Configure logging to both console and file."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — write to /tmp/discuss_agent_<timestamp>.log
    log_path = f"/tmp/discuss_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.info("Logging to %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent adversarial discussion."
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--resume",
        help="Path to existing discussion archive to resume from",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Number of additional rounds to run (used with --resume)",
    )
    parser.add_argument(
        "--guidance",
        help="Editorial guidance injected into agent prompts to steer the discussion",
    )
    args = parser.parse_args()

    if args.resume and args.rounds is None:
        print("Error: --rounds is required when using --resume", file=sys.stderr)
        sys.exit(1)

    if args.rounds is not None and not args.resume:
        print("Error: --rounds can only be used with --resume", file=sys.stderr)
        sys.exit(1)

    if args.rounds is not None and args.rounds < 1:
        print("Error: --rounds must be a positive integer", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = ConfigLoader.load(args.config)

    _setup_logging()
    logging.info("Config loaded: %s", args.config)

    engine = DiscussionEngine(config)
    result = asyncio.run(
        engine.run(
            resume_path=args.resume,
            extra_rounds=args.rounds,
            guidance=args.guidance,
        )
    )

    print(f"Discussion archived at: {result.archive_path}")
    if result.converged:
        print("Status: converged")
    elif result.terminated_by_error:
        print("Status: terminated by error")
    else:
        print("Status: max rounds reached without convergence")
