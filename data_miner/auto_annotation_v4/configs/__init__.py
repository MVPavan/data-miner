"""auto_annotation_v4.configs — configuration package.

Re-exports enums, wire types, contracts, Pydantic settings models, and the
config loader so consumers can do a single flat import::

    from data_miner.auto_annotation_v4.configs import (
        Stage, DetectResult, AutoAnnotationV4Config, load_config,
    )
"""

from .enums import *  # noqa: F401,F403
from .wire import *  # noqa: F401,F403
from .contracts import *  # noqa: F401,F403
from .settings import *  # noqa: F401,F403
from .loader import load_config, compute_config_hash, default_config_path
