"""LitAPI wrapper for GroundingDINO.  All inference logic in GDINOModel."""

from .base import DetectorServerBase, run_server
from ..models.grounding_dino import GDINOModel


class GDINOApi(DetectorServerBase):
    """GroundingDINO server -- delegates entirely to GDINOModel."""

    model_id = "IDEA-Research/grounding-dino-base"

    def setup(self, device: str) -> None:
        self.model = GDINOModel()
        self.model.load(device, self.model_id)


if __name__ == "__main__":
    run_server(GDINOApi, default_port=3001,
               default_model_id="IDEA-Research/grounding-dino-base")
