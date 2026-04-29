"""Label Studio ML backend for the aa_v4 manual review system.

Pure protocol adapter: turns LS predict() calls into SAM 3.1 HTTP requests
(text→detect, click→mask) and DB lookups (cached proposals on task open).
The backend does not load any model; all inference flows through the SAM 3.1
LitServe server on port 3014.

Public surface:
    ManualReviewerMLBackend  – LabelStudioMLBase subclass (server.py)
    dispatch                 – pure routing function (routes.py)
"""

from manual_reviewer.ml_backend.routes import (
    batch_proposals,
    dispatch,
    smart_click,
    smart_text,
    visual_prompt,
)

__all__ = [
    "batch_proposals",
    "dispatch",
    "smart_click",
    "smart_text",
    "visual_prompt",
]
