"""LitAPI wrapper for OmDet-Turbo.  All inference logic in OmDetTurboModel."""

from .base import DetectorServerBase, run_server
from ..models.omdet_turbo import OmDetTurboModel


class OmDetTurboApi(DetectorServerBase):
    """OmDet-Turbo server -- delegates entirely to OmDetTurboModel."""

    model_id = "omlab/omdet-turbo-swin-tiny-hf"

    def setup(self, device: str) -> None:
        self.model = OmDetTurboModel()
        self.model.load(device, self.model_id)


if __name__ == "__main__":
    run_server(OmDetTurboApi, default_port=3005,
               default_model_id="omlab/omdet-turbo-swin-tiny-hf")
