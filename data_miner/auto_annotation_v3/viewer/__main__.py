"""Entry point: python -m data_miner.auto_annotation_v3.viewer"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="auto_annotation_v3 pipeline result viewer (FastAPI)"
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="Job output directory (e.g. output/auto_annotation_v3/person_only_sample)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Fallback directory to look up source images by stem",
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

    ckpt_dir = job_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"No checkpoints found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    image_dir = Path(args.image_dir) if args.image_dir else None

    from data_miner.auto_annotation_v3.viewer.app import create_app

    app = create_app(job_dir, image_dir)

    print(f"Starting v3 viewer at http://localhost:{args.port}")
    print(f"Job dir:   {job_dir}")
    print(f"Image dir: {image_dir}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
