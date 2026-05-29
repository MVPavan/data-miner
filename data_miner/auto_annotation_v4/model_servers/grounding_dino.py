"""LitAPI wrapper for GroundingDINO.  All inference logic in GDINOModel."""

from .base import DetectorServerBase, run_server
from ..models.grounding_dino import GDINOModel


class GDINOApi(DetectorServerBase):
    """GroundingDINO server -- delegates entirely to GDINOModel."""

    model_id = "IDEA-Research/grounding-dino-base"
    # Per-image prompt fan-out chunk size. Default 1 pairs with
    # max_batch_size=8 to keep the Swin pass at (8,3,H,W) ~11 GiB peak.
    # Settable via serve.py --prompt-chunk-size for tuning.
    prompt_chunk_size: int = 1

    def setup(self, device: str) -> None:
        self.model = GDINOModel()
        self.model.load(device, self.model_id,
                        prompt_chunk_size=self.prompt_chunk_size)


if __name__ == "__main__":
    run_server(GDINOApi, default_port=3001,
               default_model_id="IDEA-Research/grounding-dino-base")
