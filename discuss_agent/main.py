"""CLI entry point for the multi-agent discussion framework."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from discuss_agent.config import ConfigLoader
from discuss_agent.engine import DiscussionEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent adversarial discussion."
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = ConfigLoader.load(args.config)
    engine = DiscussionEngine(config)
    result = asyncio.run(engine.run())

    print(f"Discussion archived at: {result.archive_path}")
    if result.converged:
        print("Status: converged")
    elif result.terminated_by_error:
        print("Status: terminated by error")
    else:
        print("Status: max rounds reached without convergence")
