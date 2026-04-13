"""Entry point: python -m data_miner.auto_annotation_v2.viewer_fast"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fast pipeline result viewer (FastAPI)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/auto_annotation_v2",
        help="Pipeline output directory (default: output/auto_annotation_v2)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image path (will use its directory for lookup)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to serve on (default: 8090)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    if args.image:
        img_path = Path(args.image)
        if image_dir is None:
            image_dir = img_path.parent

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    ckpt_dir = output_dir / ".checkpoints"
    if not ckpt_dir.exists():
        print(f"No checkpoint data found in {ckpt_dir}")
        sys.exit(1)

    from data_miner.auto_annotation_v2.viewer_fast.app import create_app

    app = create_app(output_dir, image_dir)

    print(f"Starting fast viewer at http://localhost:{args.port}")
    print(f"Output dir: {output_dir}")
    print(f"Image dir: {image_dir}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
