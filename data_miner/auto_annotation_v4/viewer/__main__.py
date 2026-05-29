"""Entry point: python -m data_miner.auto_annotation_v4.viewer"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="auto_annotation_v4 pipeline result viewer (FastAPI, SQLite-backed)"
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="Job output directory (contains pipeline.db). "
             "E.g. output/auto_annotation_v4/datatang_val_detect",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Fallback directory to look up source images (defaults to the "
             "image_dir recorded in job_info).",
    )
    parser.add_argument(
        "--port", type=int, default=8091, help="Port to serve on (default: 8091)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir)
    if not job_dir.exists():
        print(f"Job directory not found: {job_dir}", file=sys.stderr)
        sys.exit(1)

    db_path = job_dir / "pipeline.db"
    if not db_path.exists():
        print(f"pipeline.db not found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    image_dir = Path(args.image_dir) if args.image_dir else None

    from data_miner.auto_annotation_v4.viewer.app import create_app

    app = create_app(job_dir, image_dir)

    print(f"Starting v4 viewer at http://localhost:{args.port}")
    print(f"Job dir:   {job_dir}")
    print(f"DB:        {db_path}")
    print(f"Image dir: {image_dir or '(from job_info)'}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
