"""Per-stage-per-image checkpoint system with atomic writes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .contracts import StageName


class CheckpointManager:
    """Manages per-stage-per-image checkpoints for crash recovery.

    Layout:
        {checkpoint_dir}/{image_stem}/{stage_name}.json
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _stage_path(self, image_stem: str, stage: StageName) -> Path:
        return self._dir / image_stem / f"{stage.value}.json"

    def exists(self, image_stem: str, stage: StageName) -> bool:
        return self._stage_path(image_stem, stage).is_file()

    def save(
        self,
        image_stem: str,
        stage: StageName,
        data: BaseModel | list[BaseModel] | dict[str, Any],
    ) -> Path:
        """Atomically write checkpoint data as JSON."""
        path = self._stage_path(image_stem, stage)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, BaseModel):
            payload = data.model_dump(mode="json")
        elif isinstance(data, list):
            payload = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in data
            ]
        else:
            payload = data

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=f".{stage.value}_"
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return path

    def load(self, image_stem: str, stage: StageName) -> Any:
        """Load checkpoint data from JSON."""
        path = self._stage_path(image_stem, stage)
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_as(
        self, image_stem: str, stage: StageName, model: type[BaseModel]
    ) -> BaseModel:
        """Load checkpoint and validate into a Pydantic model."""
        data = self.load(image_stem, stage)
        return model.model_validate(data)

    def load_list_as(
        self, image_stem: str, stage: StageName, model: type[BaseModel]
    ) -> list[BaseModel]:
        """Load checkpoint as a list of Pydantic models."""
        data = self.load(image_stem, stage)
        if not isinstance(data, list):
            raise TypeError(f"Expected list, got {type(data).__name__}")
        return [model.model_validate(item) for item in data]

    def all_stages_complete(self, image_stem: str, stages: list[StageName]) -> bool:
        """Check if all given stages have checkpoints for this image."""
        return all(self.exists(image_stem, stage) for stage in stages)

    def clear_image(self, image_stem: str) -> None:
        """Remove all checkpoints for an image."""
        image_dir = self._dir / image_stem
        if image_dir.is_dir():
            for f in image_dir.iterdir():
                f.unlink(missing_ok=True)
            image_dir.rmdir()
