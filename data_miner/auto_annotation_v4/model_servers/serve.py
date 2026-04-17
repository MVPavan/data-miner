"""Unified model server launcher.

Usage:
    # Single model
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --model grounding_dino --port 3001 --gpu cuda:0

    # All enabled models from config
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --config path/to/servers.yaml --all

    # Specific models from config
    python -m data_miner.auto_annotation_v4.model_servers.serve \\
        --config path/to/servers.yaml --models grounding_dino sam3_dart
"""

from __future__ import annotations

from ..configs.enums import DetectorName

_SERVER_REGISTRY: dict[DetectorName, type] = {}


def _get_registry() -> dict[DetectorName, type]:
    """Lazy import to avoid loading all model deps at startup."""
    if not _SERVER_REGISTRY:
        from .grounding_dino import GDINOApi
        from .falcon import FalconApi
        from .sam3_dart import SAM3DartApi
        from .owlvit2 import OWLv2Api
        from .omdet_turbo import OmDetTurboApi

        _SERVER_REGISTRY.update({
            DetectorName.GROUNDING_DINO: GDINOApi,
            DetectorName.FALCON: FalconApi,
            DetectorName.SAM3_DART: SAM3DartApi,
            DetectorName.OWLVIT2: OWLv2Api,
            DetectorName.OMDET_TURBO: OmDetTurboApi,
        })
    return _SERVER_REGISTRY


def main() -> None:
    import argparse
    import logging
    import signal
    import subprocess
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Launch detector model servers")
    parser.add_argument("--model", type=str, help="Single model to launch")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, help="Path to user config YAML")
    parser.add_argument("--all", action="store_true", help="Launch all enabled")
    parser.add_argument("--models", nargs="+", help="Specific models to launch")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout", type=float, default=0.05)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Single model mode
    # ------------------------------------------------------------------
    if args.model:
        import litserve as ls

        registry = _get_registry()
        name = DetectorName(args.model)
        api_cls = registry[name]
        api = api_cls()
        server = ls.LitServer(
            api,
            accelerator="gpu",
            devices=[args.gpu],
            max_batch_size=args.max_batch_size,
            batch_timeout=args.batch_timeout,
        )
        server.run(port=args.port)
        return

    # ------------------------------------------------------------------
    # Multi-model mode from config
    # ------------------------------------------------------------------
    if args.config:
        from ..configs.loader import load_config

        cfg = load_config(user_config=args.config)
        enabled = cfg.servers.enabled_detectors()

        if args.models:
            enabled = {
                DetectorName(m): enabled[DetectorName(m)]
                for m in args.models
            }

        procs: list[subprocess.Popen] = []
        for det_name, det_cfg in enabled.items():
            cmd = [
                sys.executable, "-m",
                "data_miner.auto_annotation_v4.model_servers.serve",
                "--model", det_name.value,
                "--port", str(det_cfg.port),
                "--gpu", det_cfg.gpu,
                "--max-batch-size", str(det_cfg.max_batch_size),
            ]
            print(f"Launching {det_name.value} on port {det_cfg.port} ({det_cfg.gpu})")
            procs.append(subprocess.Popen(cmd))

        # Wait and handle shutdown
        def _shutdown(sig, frame):
            for p in procs:
                p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        for p in procs:
            p.wait()
        return

    # ------------------------------------------------------------------
    # No mode selected
    # ------------------------------------------------------------------
    parser.print_help()


if __name__ == "__main__":
    main()
