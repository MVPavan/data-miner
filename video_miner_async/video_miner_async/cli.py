"""
CLI Module

Rich command-line interface for the video mining pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from .config import (
    PipelineConfig,
    DownloadConfig,
    ExtractionConfig,
    FilterConfig,
    DeduplicationConfig,
    DetectionConfig,
    DetectorType,
    SamplingStrategy,
    FilterModel,
)
from .constants import DINO_MODELS
from .pipeline import VideoPipeline

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def setup_file_logging(log_file: Path, level: str = "INFO") -> logging.FileHandler:
    """
    Add file handler to root logger for persistent logging.
    
    Args:
        log_file: Path to log file
        level: Log level string (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        The file handler (for cleanup if needed)
    """
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Format for file (more detailed than console)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(file_handler)
    
    return file_handler


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    Video Miner v3 - High-performance video mining pipeline.
    
    Generate large-scale computer vision datasets from YouTube videos.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


# =============================================================================
# NOTE: The following commands have been removed. Use 'run-config' instead:
#
#   - 'run' command: Use 'run-config' with a YAML file
#   - 'search' command: Use 'search.enabled: true' in YAML config
#   - 'download/filter/deduplication/detection': Use 'stages' in YAML config
#
# Example:
#   video-miner run-config pipeline.yaml
#
# =============================================================================


# =============================================================================
# Registry Command Group
# =============================================================================

@main.group()
def registry() -> None:
    """Manage the video registry."""
    pass


@registry.command("status")
@click.option("--registry-file", "-r", type=click.Path(), default="./video_registry.yaml",
              help="Path to video registry file")
@click.pass_context
def registry_status(ctx: click.Context, registry_file: str) -> None:
    """Show registry status and statistics."""
    from .registry import VideoRegistry
    from rich.table import Table
    
    reg = VideoRegistry.load_or_create(Path(registry_file))
    stats = reg.get_statistics()
    
    console.print("\n[bold]Video Registry Status[/]")
    console.print(f"  File: {registry_file}")
    console.print(f"  Total videos: {stats['total_videos']}")
    console.print(f"  Keywords: {', '.join(stats['keywords']) if stats['keywords'] else 'None'}")
    console.print(f"  Last updated: {stats['updated']}")
    
    # Status breakdown table
    if stats["by_status"]:
        table = Table(title="Videos by Status")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        for status, count in stats["by_status"].items():
            table.add_row(status, str(count))
        
        console.print(table)
    else:
        console.print("\n[dim]No videos in registry.[/]")


@registry.command("summary")
@click.option("--registry-file", "-r", type=click.Path(), default="./video_registry.yaml",
              help="Path to video registry file")
@click.pass_context
def registry_summary(ctx: click.Context, registry_file: str) -> None:
    """Show pipeline stage summary with frame counts."""
    from .registry import VideoRegistry
    from rich.table import Table
    from rich.panel import Panel
    
    reg = VideoRegistry.load_or_create(Path(registry_file))
    summary = reg.get_stage_summary()
    
    console.print("\n[bold cyan]Pipeline Stage Summary[/]")
    console.print(f"  Registry: {registry_file}")
    console.print(f"  Total Videos: {summary['total_videos']}")
    
    # Video stages table
    table = Table(title="Video Processing Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Videos", justify="right", style="green")
    table.add_column("Progress", justify="right", style="yellow")
    
    total = summary["total_videos"]
    stages = [
        ("Downloaded", summary["downloaded"]),
        ("Extracted", summary["extracted"]),
        ("Filtered", summary["filtered"]),
        ("Deduplicated", summary["deduplicated"]),
        ("Detected", summary["detected"]),
    ]
    
    for stage_name, count in stages:
        pct = f"{count / max(total, 1) * 100:.0f}%" if total > 0 else "0%"
        table.add_row(stage_name, str(count), pct)
    
    if summary["failed"] > 0:
        table.add_row("[red]Failed[/]", f"[red]{summary['failed']}[/]", "")
    
    console.print(table)
    
    # Frame statistics
    frames = summary["frames"]
    console.print("\n[bold]Frame Statistics[/]")
    console.print(f"  Frames extracted:    {frames['extracted']:,}")
    console.print(f"  Frames after filter: {frames['after_filter']:,}")
    console.print(f"  Unique frames:       {frames['unique']:,}")
    console.print(f"  Total detections:    {summary['detections']:,}")
    
    # Filter efficiency
    if frames["extracted"] > 0:
        filter_rate = (1 - frames["after_filter"] / frames["extracted"]) * 100
        console.print(f"\n  Filter reduction: {filter_rate:.1f}%")
    if frames["after_filter"] > 0:
        dedup_rate = (1 - frames["unique"] / frames["after_filter"]) * 100
        console.print(f"  Dedup reduction:  {dedup_rate:.1f}%")


@registry.command("export")
@click.option("--registry-file", "-r", type=click.Path(), default="./video_registry.yaml",
              help="Path to video registry file")
@click.option("--status", "-s", type=click.Choice(["pending", "complete", "failed", "all"]),
              default="pending", help="Filter by status")
@click.option("--output", "-o", type=click.Path(), default="urls.txt",
              help="Output file for URLs")
@click.pass_context
def registry_export(
    ctx: click.Context,
    registry_file: str,
    status: str,
    output: str,
) -> None:
    """Export video URLs from registry."""
    from .registry import VideoRegistry, VideoStatus
    
    reg = VideoRegistry.load_or_create(Path(registry_file))
    
    if status == "all":
        urls = [v.url for v in reg.videos.values()]
    else:
        status_enum = VideoStatus(status)
        urls = reg.get_urls_by_status(status_enum)
    
    # Write to file
    with open(output, "w") as f:
        for url in urls:
            f.write(url + "\n")
    
    console.print(f"[green]Exported {len(urls)} URLs to {output}[/]")


@registry.command("list")
@click.option("--registry-file", "-r", type=click.Path(), default="./video_registry.yaml",
              help="Path to video registry file")
@click.option("--status", "-s", type=click.Choice(["pending", "complete", "failed", "all"]),
              default="all", help="Filter by status")
@click.option("--limit", "-n", type=int, default=20, help="Max videos to show")
@click.pass_context
def registry_list(
    ctx: click.Context,
    registry_file: str,
    status: str,
    limit: int,
) -> None:
    """List videos in the registry."""
    from .registry import VideoRegistry, VideoStatus
    from rich.table import Table
    
    reg = VideoRegistry.load_or_create(Path(registry_file))
    
    if status == "all":
        videos = list(reg.videos.values())[:limit]
    else:
        status_enum = VideoStatus(status)
        videos = reg.get_by_status(status_enum)[:limit]
    
    if not videos:
        console.print("[dim]No videos found.[/]")
        return
    
    table = Table(title=f"Videos (showing {len(videos)})")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Title", max_width=40)
    table.add_column("Status", style="yellow")
    table.add_column("Keyword", style="dim")
    
    for v in videos:
        table.add_row(
            v.video_id,
            (v.title[:37] + "...") if v.title and len(v.title) > 40 else (v.title or "N/A"),
            v.status.value,
            v.source_keyword or "",
        )
    
    console.print(table)


# =============================================================================
# YAML Config Commands
# =============================================================================

# =============================================================================
# Pipeline Execution Helpers (DRY)
# =============================================================================

def _get_config_maps():
    """Get enum mapping dictionaries for config conversion."""
    from .config import DetectorType, SamplingStrategy, FilterModel
    
    detector_map = {
        "dino-x": DetectorType.DINO_X,
        "moondream3": DetectorType.MOONDREAM3,
        "florence2": DetectorType.FLORENCE2,
        "grounding-dino": DetectorType.GROUNDING_DINO,
    }
    
    filter_model_map = {
        "siglip2-so400m": FilterModel.SIGLIP2_SO400M,
        "siglip2-giant": FilterModel.SIGLIP2_GIANT,
    }
    
    sampling_map = {
        "interval": SamplingStrategy.INTERVAL,
        "time": SamplingStrategy.TIME_BASED,
        "keyframe": SamplingStrategy.KEYFRAME,
    }
    
    return detector_map, filter_model_map, sampling_map


def _build_pipeline_config(cfg, urls: list[str], output_dir: Path):
    """Build PipelineConfig from OmegaConf configuration."""
    from .config import (
        PipelineConfig, DownloadConfig, ExtractionConfig,
        FilterConfig, DeduplicationConfig, DetectionConfig,
    )
    from .constants import DINO_MODELS
    
    detector_map, filter_model_map, sampling_map = _get_config_maps()
    
    return PipelineConfig(
        urls=urls,
        classes=list(cfg.classes),
        output_dir=output_dir,
        stages=list(cfg.stages),
        device_map=cfg.device.get("device_map", "auto"),
        use_fp16=cfg.device.get("use_fp16", True),
        download=DownloadConfig(
            force=cfg.download.get("force", False),
            output_dir=output_dir / "videos",
        ),
        extraction=ExtractionConfig(
            force=cfg.extraction.get("force", False),
            input_dir=Path(cfg.extraction.input_dir) if cfg.extraction.get("input_dir") else None,
            output_dir=output_dir / "frames_raw",
            strategy=sampling_map.get(cfg.extraction.strategy),
            interval_frames=cfg.extraction.interval_frames,
            max_frames_per_video=cfg.extraction.get("max_frames") or None,
            max_workers=cfg.extraction.get("max_workers", 4),
        ),
        filter=FilterConfig(
            force=cfg.filter.get("force", False),
            input_dir=Path(cfg.filter.input_dir) if cfg.filter.get("input_dir") else None,
            output_dir=output_dir / "frames_filtered",
            model=filter_model_map.get(cfg.filter.model),
            threshold=cfg.filter.threshold,
            batch_size=cfg.filter.batch_size,
        ),
        deduplication=DeduplicationConfig(
            force=cfg.deduplication.get("force", False),
            input_dir=Path(cfg.deduplication.input_dir) if cfg.deduplication.get("input_dir") else None,
            output_dir=output_dir / "frames_deduplicated",
            threshold=cfg.deduplication.threshold,
            batch_size=cfg.deduplication.batch_size,
            k_neighbors=cfg.deduplication.get("k_neighbors", 50),
            use_siglip=(cfg.deduplication.model == "siglip"),
            dino_model_id=DINO_MODELS.get(cfg.deduplication.get("dino_variant", "dinov2-base")),
        ),
        detection=DetectionConfig(
            force=cfg.detection.get("force", False),
            input_dir=Path(cfg.detection.input_dir) if cfg.detection.get("input_dir") else None,
            output_dir=output_dir / "detections",
            detector=detector_map.get(cfg.detection.model),
            confidence_threshold=cfg.detection.threshold,
            batch_size=cfg.detection.batch_size,
        ),
    )


def _setup_pipeline_execution(cfg, log_suffix: str = ""):
    """
    Common setup for pipeline execution.
    
    Returns:
        tuple: (output_dir, registry, urls, file_handler) or None if nothing to do
    """
    from .registry import VideoRegistry
    from .modules.downloader import gather_input_urls
    from .config_loader import save_config
    
    output_dir = Path(cfg.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    file_handler = None
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("save_to_file", True):
        log_name = f"pipeline{log_suffix}.log"
        log_file = Path(logging_cfg.get("log_file", output_dir / log_name))
        log_level = logging_cfg.get("level", "INFO")
        file_handler = setup_file_logging(log_file, log_level)
        console.print(f"[dim]Logging to: {log_file}[/]")
        logging.info(f"Pipeline started - project: {cfg.project.name}")
    
    # Initialize registry
    registry_path = Path(cfg.project.registry_file)
    registry = VideoRegistry.load_or_create(registry_path)
    
    # Execute search if enabled
    if cfg.get("search", {}).get("enabled", False):
        from .search import execute_search_stage
        
        console.print(f"\n[bold cyan]Executing Search Stage...[/]")
        found, added = execute_search_stage(cfg, registry)
        console.print(f"[green]Search complete:[/] {found} found, {added} new videos added")
        console.print(f"[dim]Registry: {registry_path} ({len(registry.videos)} total)[/]")
    
    # Gather URLs
    urls = gather_input_urls(cfg, registry)
    
    # Save resolved config
    config_output_path = output_dir / "config_used.yaml"
    save_config(cfg, config_output_path)
    console.print(f"[dim]Config saved to: {config_output_path}[/]")
    
    return output_dir, registry, urls, file_handler


def _cleanup_file_handler(file_handler) -> None:
    """Cleanup file handler to avoid duplicates in batch mode."""
    if file_handler:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


# =============================================================================
# Pipeline Execution Functions
# =============================================================================

def _execute_pipeline(cfg) -> None:
    """Execute the synchronous pipeline."""
    
    setup_result = _setup_pipeline_execution(cfg)
    output_dir, registry, urls, file_handler = setup_result
    
    try:
        # Check if we have work to do
        if not urls and (not cfg.stages or len(cfg.stages) == 0):
            console.print("\n[bold green]Search-only mode complete![/]")
            return
        
        if not urls and "download" in cfg.stages:
            console.print("[yellow]No URLs to process. Use search or provide URLs for download stage.[/]")
            return
        
        pipeline_config = _build_pipeline_config(cfg, urls, output_dir)
        
        # Run pipeline
        from .pipeline import VideoPipeline
        
        console.print(f"\n[bold green]Starting pipeline:[/] {cfg.project.name}")
        console.print(f"  Output: {output_dir}")
        console.print(f"  Classes: {', '.join(cfg.classes)}")
        console.print(f"  Stages: {', '.join(cfg.stages)}")
        
        pipeline = VideoPipeline(pipeline_config, registry=registry)
        result = pipeline.run()
        
        console.print("\n[bold green]Pipeline Complete![/]")
        console.print(f"  Videos downloaded: {result.videos_downloaded}")
        console.print(f"  Frames extracted: {result.frames_extracted}")
        console.print(f"  Frames after filter: {result.frames_filtered}")
        console.print(f"  Unique frames: {result.frames_deduplicated}")
        console.print(f"  Detections: {result.detections_found}")
        
    finally:
        _cleanup_file_handler(file_handler)
        
        # Clear GPU cache for next config run
        from .utils.device import clear_gpu_cache
        clear_gpu_cache()


def _execute_async_pipeline(cfg) -> None:
    """Execute the async pipeline with stage-level parallelism."""
    import asyncio
    from .async_pipeline import AsyncPipelineOrchestrator, AsyncPipelineConfig
    
    console.print("\n[bold cyan]Using Async Pipeline[/] (stage-level parallelism)")
    
    setup_result = _setup_pipeline_execution(cfg, log_suffix="_async")
    output_dir, registry, urls, file_handler = setup_result
    
    try:
        # Check if we have work to do
        if not urls and (not cfg.stages or len(cfg.stages) == 0):
            console.print("\n[bold green]Search-only mode complete![/]")
            return
        
        if not urls and "download" in cfg.stages:
            console.print("[yellow]No URLs to process. Use search or provide URLs for download stage.[/]")
            return
        
        pipeline_config = _build_pipeline_config(cfg, urls, output_dir)
        
        # Build async config
        async_cfg = cfg.get("async_pipeline", {})
        async_config = AsyncPipelineConfig(
            download_workers=async_cfg.get("download_workers", 2),
            extract_workers=async_cfg.get("extract_workers", 2),
            filter_workers=async_cfg.get("filter_workers", 1),
            max_thread_workers=async_cfg.get("max_thread_workers", 8),
            download_queue_size=async_cfg.get("download_queue_size", 4),
            extract_queue_size=async_cfg.get("extract_queue_size", 4),
            filter_queue_size=async_cfg.get("filter_queue_size", 2),
            cleanup_raw_frames=async_cfg.get("cleanup_raw_frames", False),
            cleanup_videos=async_cfg.get("cleanup_videos", False),
        )
        
        console.print(f"\n[bold green]Starting async pipeline:[/] {cfg.project.name}")
        console.print(f"  Output: {output_dir}")
        console.print(f"  Classes: {', '.join(cfg.classes)}")
        console.print(f"  Stages: {', '.join(cfg.stages)}")
        console.print(f"  URLs: {len(urls)}")
        console.print(f"  Workers: download={async_config.download_workers}, "
                     f"extract={async_config.extract_workers}, "
                     f"filter={async_config.filter_workers}")
        
        # Create orchestrator and run
        orchestrator = AsyncPipelineOrchestrator(
            pipeline_config=pipeline_config,
            async_config=async_config,
            registry=registry,
        )
        
        result = asyncio.run(orchestrator.run(urls))
        
        console.print("\n[bold green]Async Pipeline Complete![/]")
        console.print(f"  Videos downloaded: {result.downloaded}")
        console.print(f"  Videos extracted: {result.extracted}")
        console.print(f"  Videos filtered: {result.filtered}")
        console.print(f"  Unique frames: {result.deduplicated}")
        console.print(f"  Detections: {result.detected}")
        console.print(f"  Failed: {result.failed}")
        console.print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
        
    finally:
        _cleanup_file_handler(file_handler)
        
        # Clear GPU cache for next config run
        from .utils.device import clear_gpu_cache
        clear_gpu_cache()


@main.command("run-config")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Validate config without running pipeline")
@click.option("--async", "use_async", is_flag=True, help="Use async pipeline for parallel processing")
@click.pass_context
def run_config(ctx: click.Context, config_paths: tuple[str], dry_run: bool, use_async: bool) -> None:
    """
    Run pipeline from YAML configuration file(s).
    
    Accepts multiple files or directories. Files are processed in the order given.
    Directories are expanded to all .yaml files within them (sorted by name).
    The config file is merged with config/default.yaml.
    
    Use --async for stage-level parallel processing (recommended for 50+ videos).
    
    Example:
        video-miner run-config pipeline.yaml
        video-miner run-config pipeline.yaml --async
        video-miner run-config phase1.yaml phase2.yaml
        video-miner run-config ./configs/
    """
    from .config_loader import load_config, validate_config, print_config
    
    if not config_paths:
        console.print("[red]Error: Please provide at least one config file or directory.[/]")
        console.print("Usage: video-miner run-config [OPTIONS] CONFIG_PATHS...")
        return
    
    # Collect all config files in order
    config_files = []
    for p_str in config_paths:
        path = Path(p_str)
        if path.is_dir():
            dir_configs = sorted(list(path.glob("*.yaml"))) + sorted(list(path.glob("*.yml")))
            if not dir_configs:
                console.print(f"[yellow]Warning: No .yaml files found in {path}[/]")
            else:
                config_files.extend(dir_configs)
                console.print(f"[dim]Added {len(dir_configs)} configs from {path}[/]")
        else:
            config_files.append(path)
            
    if not config_files:
        console.print("[yellow]No configuration files found to process.[/]")
        return
        
    if len(config_files) > 1:
        console.print(f"[bold green]Batch Mode:[/] Queued {len(config_files)} configuration files")
    
    # Iterate and execute
    for i, cfg_file in enumerate(config_files, 1):
        if len(config_files) > 1:
            console.print(f"\n[bold magenta]Processing Config {i}/{len(config_files)}: {cfg_file.name}[/]")
            
        console.print(f"[bold blue]Loading configuration:[/] {cfg_file}")
        
        try:
            # Load and merge configuration
            cfg = load_config(cfg_file)
            
            # Validate
            is_valid, errors = validate_config(cfg)
            if not is_valid:
                console.print(f"[bold red]Configuration errors in {cfg_file.name}:[/]")
                for err in errors:
                    console.print(f"  • {err}")
                if len(config_files) > 1:
                    continue # Skip invalid configs in batch mode
                raise click.Abort()
            
            console.print("[green]Configuration validated successfully[/]")
            
            if dry_run:
                console.print("\n[bold]Resolved Configuration:[/]")
                console.print(print_config(cfg))
                continue
            
            # Execute with selected pipeline
            if use_async:
                _execute_async_pipeline(cfg)
            else:
                _execute_pipeline(cfg)
            
        except Exception as e:
            console.print(f"[red]Error processing {cfg_file.name}: {e}[/]")
            if len(config_files) == 1:
                raise click.Abort()


@main.command("validate-config")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--show", is_flag=True, help="Show resolved configuration")
@click.pass_context
def validate_config_cmd(ctx: click.Context, config_file: str, show: bool) -> None:
    """
    Validate a configuration file without running.
    
    Example:
        video-miner validate-config pipeline.yaml --show
    """
    from .config_loader import load_config, validate_config, print_config
    
    console.print(f"[bold blue]Validating:[/] {config_file}")
    
    try:
        cfg = load_config(config_file)
        is_valid, errors = validate_config(cfg)
        
        if is_valid:
            console.print("[bold green]✓ Configuration is valid[/]")
        else:
            console.print("[bold red]✗ Configuration has errors:[/]")
            for err in errors:
                console.print(f"  • {err}")
        
        if show:
            console.print("\n[bold]Resolved Configuration:[/]")
            console.print(print_config(cfg))
            
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/]")
        raise click.Abort()


if __name__ == "__main__":
    main()

# TODO: can video download and filtering happen in parallel?
# TODO: can existing dedup handle 100 videos each of 1000 frames?