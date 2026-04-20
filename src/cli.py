"""
CLI Entry Point
================
Provides the ``pilot-readiness`` console command that wraps the
existing ``main.py`` pipeline with framework awareness.

Usage
-----
    pilot-readiness                         # Full pipeline
    pilot-readiness --quick-test            # Quick single-subject test
    pilot-readiness --config my_config.yaml # Custom configuration
    pilot-readiness --api                   # Launch REST API server
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Pilot Readiness AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pilot-readiness                              # Full pipeline
  pilot-readiness --quick-test                 # Single subject
  pilot-readiness --config my_config.yaml      # Custom config
  pilot-readiness --api --port 8000            # REST API server
  pilot-readiness --streaming --port 5000      # Streaming demo
        """,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with single subject (S2)")
    parser.add_argument("--dataset", type=str, default="wesad",
                        choices=["wesad", "swell"],
                        help="Dataset pipeline to run")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Use cached feature CSVs if available")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (use cached models)")
    parser.add_argument("--per-subject-norm", action="store_true",
                        help="Per-pilot Z-score normalization")
    parser.add_argument("--window-sec", type=int, default=60,
                        help="ECG window duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Window overlap fraction")
    parser.add_argument("--n-sessions", type=int, default=30,
                        help="Simulated MATB-II sessions per workload level")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # API mode
    parser.add_argument("--api", action="store_true",
                        help="Launch REST API server instead of pipeline")
    parser.add_argument("--streaming", action="store_true",
                        help="Launch streaming demo instead of pipeline")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    if args.api:
        # Launch REST API
        port = args.port or 8000
        from src.api.rest_api import run_server
        run_server(config=args.config, host=args.host, port=port)

    elif args.streaming:
        # Launch streaming demo
        port = args.port or 5000
        sys.argv = [
            "streaming_demo.py",
            "--dataset", args.dataset,
            "--host", args.host,
            "--port", str(port),
        ]
        from streaming_demo import main as streaming_main
        streaming_main()

    else:
        # Run standard pipeline
        from main import main as pipeline_main
        pipeline_main()


if __name__ == "__main__":
    main()
