"""LitAPI wrapper for Falcon-Perception.  All inference logic in FalconModel."""

from .base import DetectorServerBase, run_server
from ..models.falcon import FalconModel


class FalconApi(DetectorServerBase):
    """Falcon-Perception server -- delegates entirely to FalconModel."""

    model_id = "tiiuae/Falcon-Perception"

    def setup(self, device: str) -> None:
        self.model = FalconModel()
        self.model.load(device, self.model_id)


if __name__ == "__main__":
    run_server(FalconApi, default_port=3002,
               default_model_id="tiiuae/Falcon-Perception")
