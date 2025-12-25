"""Logging setup using Loguru + Rich for detection_metrics."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel


# Global console instance for Rich output
console = Console()


def setup_logging(
    log_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Configure Loguru with colored console + optional file output.
    
    Args:
        log_file: Optional path to log file (with rotation).
        verbose: If True, set log level to DEBUG.
        quiet: If True, set log level to WARNING.
    """
    # Remove default handler
    logger.remove()
    
    # Determine log level
    if quiet:
        level = "WARNING"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    # Console handler (colored)
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )
    
    # File handler (plain text, with rotation)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level="DEBUG",  # Log everything to file
        )


def create_progress() -> Progress:
    """
    Create a Rich progress bar with standard columns.
    
    Returns:
        Progress: Configured Rich progress bar.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_results_table(title: str, columns: list[str]) -> Table:
    """
    Create a Rich table for displaying results.
    
    Args:
        title: Table title.
        columns: List of column names.
    
    Returns:
        Table: Configured Rich table.
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col)
    return table


def print_summary_panel(title: str, content: str) -> None:
    """
    Print a Rich panel with summary information.
    
    Args:
        title: Panel title.
        content: Panel content.
    """
    console.print(Panel(content, title=title, border_style="green"))


# Re-export commonly used items
__all__ = [
    "logger",
    "console",
    "setup_logging",
    "create_progress",
    "create_results_table",
    "print_summary_panel",
    "Progress",
    "Table",
]
