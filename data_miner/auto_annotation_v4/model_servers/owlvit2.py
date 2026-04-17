"""LitAPI wrapper for OWLv2 (OWL-ViT v2).  All inference logic in OWLv2Model."""

from .base import DetectorServerBase, run_server
from ..models.owlvit2 import OWLv2Model


class OWLv2Api(DetectorServerBase):
    """OWLv2 server -- delegates entirely to OWLv2Model."""

    model_id = "google/owlv2-base-patch16-ensemble"

    def setup(self, device: str) -> None:
        self.model = OWLv2Model()
        self.model.load(device, self.model_id)


if __name__ == "__main__":
    run_server(OWLv2Api, default_port=3004,
               default_model_id="google/owlv2-base-patch16-ensemble")
